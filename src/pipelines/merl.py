from .ml import train_ml_model
from src.models.mixed_effects.linear_mixed_model import fit_lmm_safe
from ..utils.pipeline_helpers import (
    combine_predictions,
    update_targets,
    apply_random_effects,
    build_prediction_df,
    compute_residuals,
    to_numpy,
)
from ..utils.seeding import set_seed
from .losses import compute_all_losses
import numpy as np
import joblib
from pathlib import Path


class MERLPipeline:

    def __init__(
        self,
        ml_model_fn,
        ml_config,
        lmm_fn=None,
        lmm_config=None,
        loss="mse",
        n_iter=5,
        use_ml=True,
        use_lmm=True,
        early_stopping=True,
        tol=1e-4,
        ml_mode="reinit",
        tensorboard_logger=None,
        seed=None,
    ):
        """Initialize Mixed ML + LMM iterative pipeline."""

        self.ml_model_fn = ml_model_fn
        self.ml_config = ml_config
        self.lmm_fn = lmm_fn
        self.lmm_config = lmm_config

        self.loss = loss
        self.n_iter = int(n_iter)

        self.use_ml = use_ml
        self.use_lmm = use_lmm

        self.early_stopping = early_stopping
        self.tol = float(tol)

        self.ml_mode = ml_mode
        self.tb = tensorboard_logger

        self._persistent_model = None

        self.seed = seed

        if seed is not None:
            set_seed(seed)

    def run(
        self,
        X_train,
        y_train,
        train_df,
        X_val=None,
        y_val=None,
        val_df=None,
        X_test=None,
        y_test=None,
        test_df=None,
        group_var="subject_id",
        fold=None,
    ):
        """Run the full iterative mixed ML + LMM training loop."""

        y_dict, dfs, y_current, extra_data = self._initialize_data(
            y_train,
            y_val,
            y_test,
            train_df,
            val_df,
            test_df,
        )

        history = []
        pred_history = []
        model_history = []
        re_history = []
        lmm_metrics_history = []

        prev_loss = None

        # ==================================================
        # Calculate baselines once
        #
        # IMPORTANT:
        # All baseline parameters are calculated from the
        # original training outcomes only.
        # ==================================================

        baselines = self._calculate_baselines(
            y_dict=y_dict,
            train_df=train_df,
            dfs=dfs,
            group_var=group_var,
        )

        # ==================================================
        # Iterative ML + LMM loop
        # ==================================================

        for t in range(self.n_iter):

            print(f"\n=== Iteration {t} ===")

            # --------------------------------------------------
            # 1. ML training step
            # --------------------------------------------------

            model, preds = self._run_ml_step(
                X_train,
                X_val,
                X_test,
                y_current,
                extra_data,
                t,
                fold,
            )

            model_history.append(model)

            # --------------------------------------------------
            # 2. Residual computation
            # --------------------------------------------------

            residuals = self._compute_residuals(
                y_dict,
                preds,
            )

            # --------------------------------------------------
            # 3. LMM fitting step
            # --------------------------------------------------

            random_effects, lmm_metrics = self._run_lmm_step(
                train_df,
                residuals,
                re_history,
                lmm_metrics_history,
                t,
            )

            re_history.append(random_effects)
            lmm_metrics_history.append(lmm_metrics)

            # --------------------------------------------------
            # 4. Target update
            # --------------------------------------------------

            y_current = self._update_targets(
                y_dict,
                dfs,
                random_effects,
                group_var,
            )

            # --------------------------------------------------
            # 5. Full prediction assembly
            # --------------------------------------------------

            full_preds = self._combine_predictions(
                preds,
                dfs,
                random_effects,
                group_var,
            )

            contributions = self._compute_prediction_contributions(
                preds,
                full_preds,
            )

            # --------------------------------------------------
            # 6. Loss computation
            # --------------------------------------------------

            metrics = self._compute_metrics(
                y_dict,
                preds,
                full_preds,
            )

            history.append(metrics)

            # --------------------------------------------------
            # 7. Prediction dataframe construction
            # --------------------------------------------------

            pred_history.append(
                self._build_prediction_history(
                    y_dict,
                    preds,
                    dfs,
                    random_effects,
                    group_var,
                )
            )

            # --------------------------------------------------
            # 8. Logging
            # --------------------------------------------------

            self._log_metrics(
                metrics,
                fold,
                t,
                model,
                contributions,
            )

            # --------------------------------------------------
            # 9. Early stopping
            # --------------------------------------------------

            if self._check_convergence(
                metrics,
                prev_loss,
            ):
                break

            prev_loss = metrics["full_train"]

            # --------------------------------------------------
            # 10. Baselines
            #
            # Baselines were already calculated once before
            # the loop. They must not be recalculated using
            # updated targets.
            # --------------------------------------------------

        # ==================================================
        # Final output
        # ==================================================

        return self._build_output(
            history,
            pred_history,
            model_history,
            re_history,
            baselines,
            lmm_metrics_history
        )


    # =====================================================
    # ONE-HOT ENCODING
    # ====================================================
    def _one_hot_encode_features(
        self,
        dfs,
        feature_names,
    ):
        """
        One-hot encode categorical features using the training
        dataframe as the reference for the category mapping.

        Parameters
        ----------
        dfs : dict
            Dictionary containing train/val/test DataFrames.

        feature_names : list[str]
            Features to one-hot encode.

        Returns
        -------
        dict
            Dictionary containing one-hot encoded numpy arrays
            for train/val/test.
        """

        if not feature_names:
            return {
                "train": None,
                "val": None,
                "test": None,
            }

        train_df = dfs["train"]

        if train_df is None:
            raise ValueError(
                "Training dataframe is required for "
                "one-hot encoding."
            )

        # --------------------------------------------------
        # Check that all requested features exist
        # --------------------------------------------------

        for feature in feature_names:

            if feature not in train_df.columns:
                raise ValueError(
                    f"Missing one-hot encoding feature "
                    f"'{feature}' in training dataframe."
                )

            for split in ["val", "test"]:

                if (
                    dfs[split] is not None
                    and feature not in dfs[split].columns
                ):
                    raise ValueError(
                        f"Missing one-hot encoding feature "
                        f"'{feature}' in {split} dataframe."
                    )

        # --------------------------------------------------
        # Fit categories using training data only
        # --------------------------------------------------

        category_maps = {}

        for feature in feature_names:

            categories = (
                train_df[feature]
                .dropna()
                .unique()
                .tolist()
            )

            category_maps[feature] = categories

        # --------------------------------------------------
        # Encode each split
        # --------------------------------------------------

        encoded_data = {}

        for split in ["train", "val", "test"]:

            df = dfs[split]

            if df is None:
                encoded_data[split] = None
                continue

            encoded_columns = []

            for feature in feature_names:

                values = df[feature]

                categories = category_maps[feature]

                # Create one column per category
                for category in categories:

                    encoded_columns.append(
                        (
                            values == category
                        ).astype(
                            np.float32
                        ).to_numpy()
                    )

            if encoded_columns:

                encoded_data[split] = np.column_stack(
                    encoded_columns
                ).astype(np.float32)

            else:

                encoded_data[split] = None

        return encoded_data

    # ======================================================
    # DATA INITIALIZATION
    # ======================================================

    def _initialize_data(
        self,
        y_train,
        y_val,
        y_test,
        train_df,
        val_df,
        test_df,
    ):
        """Prepare targets, dataframes, and optional transformer features."""

        y_dict = {
            "train": to_numpy(y_train),
            "val": to_numpy(y_val) if y_val is not None else None,
            "test": to_numpy(y_test) if y_test is not None else None,
        }

        self.train_mean = np.mean(
            y_dict["train"]
        )

        dfs = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        y_current = {
            k: (v.copy() if v is not None else None)
            for k, v in y_dict.items()
        }

        # ==================================================
        # EXTRA FEATURES
        # ==================================================

        extra_data = {
            "train": None,
            "val": None,
            "test": None,
        }

        # --------------------------------------------------
        # 1. Existing extra transformer features
        # --------------------------------------------------

        extra_feature_names = self.ml_config.get(
            "extra_transformer_features",
            None,
        )

        if extra_feature_names:

            for split, df in dfs.items():

                if df is None:
                    continue

                missing_features = [
                    feature
                    for feature in extra_feature_names
                    if feature not in df.columns
                ]

                if missing_features:
                    raise ValueError(
                        f"Missing extra transformer features in "
                        f"{split} dataframe: {missing_features}"
                    )

                extra_data[split] = (
                    df[
                        extra_feature_names
                    ]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

        # --------------------------------------------------
        # 2. Optional one-hot encoding
        # --------------------------------------------------

        use_one_hot = self.ml_config.get(
            "one_hot_encoding",
            False,
        )

        one_hot_features = self.ml_config.get(
            "one_hot_encoding_features",
            None,
        )

        if use_one_hot:

            if not one_hot_features:
                raise ValueError(
                    "one_hot_encoding is True, but "
                    "one_hot_encoding_features is empty."
                )

            one_hot_data = self._one_hot_encode_features(
                dfs,
                one_hot_features,
            )

            # --------------------------------------------------
            # Combine existing extra features + one-hot features
            # --------------------------------------------------

            for split in [
                "train",
                "val",
                "test",
            ]:

                existing = extra_data[split]
                encoded = one_hot_data[split]

                if existing is None:
                    extra_data[split] = encoded

                elif encoded is None:
                    extra_data[split] = existing

                else:
                    extra_data[split] = np.concatenate(
                        [
                            existing,
                            encoded,
                        ],
                        axis=1,
                    )

        return (
            y_dict,
            dfs,
            y_current,
            extra_data,
        )

    # ======================================================
    # BASELINES
    # ======================================================

    def _calculate_baselines(
        self,
        y_dict,
        train_df,
        dfs,
        group_var,
    ):
        """
        Calculate three baselines using training data only.

        Baselines:

        1. global_baseline
           Always predict the overall mean outcome from
           the training set.

        2. group_baseline
           Always predict the participant/group mean outcome
           from the training set.

        3. last_outcome_baseline
           Always predict the participant's last observed
           outcome in the training set, determined by the
           largest timestamp `t`.

        For participants that are not present in the
        training data, the global training mean is used
        as fallback.

        Returns
        -------
        dict
            Contains both predictions and losses for each
            baseline and each split.
        """

        y_train = np.asarray(
            y_dict["train"]
        )

        # ==================================================
        # 1. GLOBAL BASELINE
        # ==================================================

        global_mean = np.mean(
            y_train
        )

        global_predictions = {}

        for split in [
            "train",
            "val",
            "test",
        ]:

            if dfs[split] is None:

                global_predictions[split] = None

            else:

                global_predictions[split] = np.full(
                    len(dfs[split]),
                    global_mean,
                    dtype=float,
                )

        # ==================================================
        # 2. GROUP MEAN BASELINE
        # ==================================================

        train_baseline_df = train_df.copy()

        train_baseline_df["_baseline_y"] = y_train

        group_means = (
            train_baseline_df
            .groupby(group_var)["_baseline_y"]
            .mean()
            .to_dict()
        )

        group_predictions = {}

        for split in [
            "train",
            "val",
            "test",
        ]:

            if dfs[split] is None:

                group_predictions[split] = None
                continue

            groups = dfs[split][
                group_var
            ].to_numpy()

            group_predictions[split] = np.array(
                [
                    group_means.get(
                        group,
                        global_mean,
                    )
                    for group in groups
                ],
                dtype=float,
            )

        # ==================================================
        # 3. LAST OUTCOME BASELINE
        # ==================================================

        # Sort the training data by participant and time.
        #
        # This makes "last outcome" explicitly mean the
        # outcome at the largest value of `t`.
        train_baseline_df = (
            train_baseline_df
            .sort_values(
                [group_var, "t"]
            )
        )

        last_outcomes = (
            train_baseline_df
            .groupby(group_var)["_baseline_y"]
            .last()
            .to_dict()
        )

        last_outcome_predictions = {}

        for split in [
            "train",
            "val",
            "test",
        ]:

            if dfs[split] is None:

                last_outcome_predictions[split] = None
                continue

            groups = dfs[split][
                group_var
            ].to_numpy()

            last_outcome_predictions[split] = np.array(
                [
                    last_outcomes.get(
                        group,
                        global_mean,
                    )
                    for group in groups
                ],
                dtype=float,
            )

        # ==================================================
        # Store all baseline predictions
        # ==================================================

        baseline_predictions = {
            "global_baseline": global_predictions,
            "group_baseline": group_predictions,
            "last_outcome_baseline": last_outcome_predictions,
        }

        # ==================================================
        # Calculate baseline losses
        # ==================================================

        baseline_losses = {}

        for (
            baseline_name,
            predictions,
        ) in baseline_predictions.items():

            baseline_losses[baseline_name] = {}

            for split in [
                "train",
                "val",
                "test",
            ]:

                y_true = y_dict[split]
                y_pred = predictions[split]

                if (
                    y_true is None
                    or y_pred is None
                ):

                    baseline_losses[
                        baseline_name
                    ][split] = None

                    continue

                baseline_losses[
                    baseline_name
                ][split] = self._compute_single_loss(
                    y_true,
                    y_pred,
                )

        # ==================================================
        # Return predictions + losses
        # ==================================================

        return {
            "global_baseline": {
                "predictions": global_predictions,
                "loss": baseline_losses[
                    "global_baseline"
                ],
            },

            "group_baseline": {
                "predictions": group_predictions,
                "loss": baseline_losses[
                    "group_baseline"
                ],
            },

            "last_outcome_baseline": {
                "predictions": last_outcome_predictions,
                "loss": baseline_losses[
                    "last_outcome_baseline"
                ],
            },
        }

    # ======================================================
    # SINGLE LOSS
    # ======================================================

    def _compute_single_loss(
        self,
        y_true,
        y_pred,
    ):
        """
        Calculate the configured loss for a single set
        of predictions.

        Currently supports:
            - mse
            - mae
        """

        y_true = np.asarray(
            y_true
        )

        y_pred = np.asarray(
            y_pred
        )

        if self.loss == "mse":

            return float(
                np.mean(
                    (
                        y_true
                        - y_pred
                    ) ** 2
                )
            )

        elif self.loss == "mae":

            return float(
                np.mean(
                    np.abs(
                        y_true
                        - y_pred
                    )
                )
            )
        
        elif self.loss == "rmse":
                
                return float(
                    np.sqrt(
                        np.mean(
                            (
                                y_true
                                - y_pred
                            ) ** 2
                        )
                    )
                )

        else:

            raise ValueError(
                f"Unsupported baseline loss: "
                f"{self.loss}. "
                f"Add it to _compute_single_loss()."
            )

    # ======================================================
    # ML STEP
    # ======================================================

    def _run_ml_step(
        self,
        X_train,
        X_val,
        X_test,
        y_current,
        extra_data,
        t,
        fold,
    ):
        """Train or fine-tune ML model and return predictions."""

        if not self.use_ml:

            preds = {
                "train": np.full_like(
                    y_current["train"],
                    self.train_mean,
                    dtype=float,
                ),

                "val": (
                    np.full_like(
                        y_current["val"],
                        self.train_mean,
                        dtype=float,
                    )
                    if y_current["val"] is not None
                    else None
                ),

                "test": (
                    np.full_like(
                        y_current["test"],
                        self.train_mean,
                        dtype=float,
                    )
                    if y_current["test"] is not None
                    else None
                ),
            }

            return None, preds

        init_model = None

        if (
            self.ml_mode == "finetune"
            and self._persistent_model is not None
        ):
            init_model = self._persistent_model

        model, preds = train_ml_model(
            self.ml_model_fn,
            self.ml_config,
            X_train,
            y_current["train"],
            X_val,
            y_current["val"],
            X_test,

            # Optional transformer features
            X_train_extra=extra_data["train"],
            X_val_extra=extra_data["val"],
            X_test_extra=extra_data["test"],

            # Logger
            logger=self.tb,

            init_model=init_model,
            iteration=t,
            fold=fold,
            seed=self.seed,
        )

        self._persistent_model = model

        return model, preds

    # ======================================================
    # RESIDUALS
    # ======================================================

    def _compute_residuals(
        self,
        y_dict,
        preds,
    ):
        """Compute ML residuals for LMM fitting."""

        if not self.use_ml:

            return (
                y_dict["train"]
                - self.train_mean
            )

        return compute_residuals(
            y_dict["train"],
            preds["train"],
        )

    # ======================================================
    # LMM STEP
    # ======================================================

    def _run_lmm_step(
        self,
        train_df,
        residuals,
        re_history,
        lmm_metrics_history,
        t,
    ):
        """Fit linear mixed model on residuals."""

        if not self.use_lmm:
            return None, None

        return fit_lmm_safe(
            self.lmm_fn,
            self.lmm_config,
            train_df,
            residuals,
            re_history,
            lmm_metrics_history,
            t,
        )

    # ======================================================
    # TARGET UPDATE
    # ======================================================

    def _update_targets(
        self,
        y_dict,
        dfs,
        random_effects,
        group_var,
    ):
        """Remove random effects from targets."""

        if not self.use_lmm:
            return y_dict

        return {
            split: update_targets(
                y_dict[split],
                dfs[split],
                random_effects,
                group_var,
            )
            for split in [
                "train",
                "val",
                "test",
            ]
        }

    # ======================================================
    # PREDICTION COMBINATION
    # ======================================================

    def _combine_predictions(
        self,
        preds,
        dfs,
        random_effects,
        group_var,
    ):
        """Combine ML predictions with LMM corrections."""

        # --------------------------------------------------
        # ML + LMM
        # --------------------------------------------------

        if self.use_ml and self.use_lmm:

            return combine_predictions(
                preds,
                dfs,
                random_effects,
                group_var,
            )

        # --------------------------------------------------
        # ML only
        # --------------------------------------------------

        if self.use_ml and not self.use_lmm:

            return preds

        # --------------------------------------------------
        # LMM only
        # --------------------------------------------------

        if not self.use_ml and self.use_lmm:

            full_preds = {}

            for split in [
                "train",
                "val",
                "test",
            ]:

                if dfs[split] is None:

                    full_preds[split] = None
                    continue

                re = apply_random_effects(
                    dfs[split],
                    random_effects,
                    group_var,
                )

                full_preds[split] = (
                    self.train_mean
                    + re
                )

            return full_preds

        raise ValueError(
            "At least one of use_ml or use_lmm must be True."
        )

    # ======================================================
    # METRICS
    # ======================================================

    def _compute_metrics(
        self,
        y_dict,
        preds,
        full_preds,
    ):
        """Compute ML-only and full-model losses."""

        return compute_all_losses(
            y_dict,
            preds,
            full_preds,
            self.loss,
        )

    # ======================================================
    # PREDICTION HISTORY
    # ======================================================

    def _build_prediction_history(
        self,
        y_dict,
        preds,
        dfs,
        random_effects,
        group_var,
    ):
        """Build prediction DataFrames for all splits."""

        return {
            split: (
                build_prediction_df(
                    y_true=y_dict[split],
                    ml_pred=preds[split],
                    df=dfs[split],
                    random_effects=random_effects,
                    group_var=group_var,
                    apply_re_fn=apply_random_effects,
                )
                if dfs[split] is not None
                else None
            )
            for split in [
                "train",
                "val",
                "test",
            ]
        }

    # ======================================================
    # PREDICTION CONTRIBUTIONS
    # ======================================================

    def _compute_prediction_contributions(
        self,
        ml_preds,
        full_preds,
    ):
        """
        Estimate percentage contribution of:

        - ML prediction
        - Random effects correction

        based on absolute magnitudes.
        """

        contributions = {}

        for split in [
            "train",
            "val",
            "test",
        ]:

            if (
                ml_preds.get(split) is None
                or full_preds.get(split) is None
            ):
                contributions[split] = None
                continue

            ml = np.asarray(
                ml_preds[split]
            )

            full = np.asarray(
                full_preds[split]
            )

            re = full - ml

            ml_mag = np.mean(
                np.abs(ml)
            )

            re_mag = np.mean(
                np.abs(re)
            )

            total = (
                ml_mag
                + re_mag
                + 1e-8
            )

            contributions[split] = {
                "fixed_effect_pct": (
                    100
                    * ml_mag
                    / total
                ),

                "random_effect_pct": (
                    100
                    * re_mag
                    / total
                ),
            }

        return contributions

    # ======================================================
    # MODE
    # ======================================================

    def _get_mode(self):

        if self.use_ml and self.use_lmm:
            return "merl"

        elif self.use_ml:
            return "ml_only"

        elif self.use_lmm:
            return "lmm_only"

        else:
            return "none"

    # ======================================================
    # LOGGING
    # ======================================================

    def _log_metrics(
        self,
        metrics,
        fold,
        iteration,
        model,
        contributions=None,
    ):
        """Print + TensorBoard logging."""

        mode = self._get_mode()

        if mode == "merl":

            train_label = (
                f"MERL Train {self.loss}"
            )

            val_label = (
                f"MERL Val {self.loss}"
            )

            ml_train_label = (
                f"ML Train {self.loss}"
            )

            ml_val_label = (
                f"ML Val {self.loss}"
            )

            print(
                f"{ml_train_label}:   "
                f"{metrics['ml_train']:.6f}"
            )

            print(
                f"{train_label}: "
                f"{metrics['full_train']:.6f}"
            )

            if metrics["ml_val"] is not None:

                print(
                    f"{ml_val_label}:     "
                    f"{metrics['ml_val']:.6f}"
                )

                print(
                    f"{val_label}:   "
                    f"{metrics['full_val']:.6f}"
                )

        elif mode == "ml_only":

            train_label = (
                f"ML Train {self.loss}"
            )

            val_label = (
                f"ML Val {self.loss}"
            )

            print(
                f"{train_label}: "
                f"{metrics['full_train']:.6f}"
            )

            if metrics["full_val"] is not None:

                print(
                    f"{val_label}:   "
                    f"{metrics['full_val']:.6f}"
                )

        elif mode == "lmm_only":

            train_label = (
                f"LMM Train {self.loss}"
            )

            val_label = (
                f"LMM Val {self.loss}"
            )

            print(
                f"{train_label}: "
                f"{metrics['full_train']:.6f}"
            )

            if metrics["full_val"] is not None:

                print(
                    f"{val_label}:   "
                    f"{metrics['full_val']:.6f}"
                )

        else:

            print(
                "No active model component."
            )

        # ==================================================
        # TensorBoard logging
        # ==================================================

        if self.tb is None or model is None:
            return

        fold_name = f"fold_{fold}"

        # --------------------------------------------------
        # Train losses
        # --------------------------------------------------

        self.tb.add_scalars(
            f"{model.model_name}/ml_train",
            {
                fold_name: metrics["ml_train"]
            },
            iteration,
        )

        self.tb.add_scalars(
            f"{model.model_name}/full_train",
            {
                fold_name: metrics["full_train"]
            },
            iteration,
        )

        # --------------------------------------------------
        # Validation losses
        # --------------------------------------------------

        if metrics["ml_val"] is not None:

            self.tb.add_scalars(
                f"{model.model_name}/ml_val",
                {
                    fold_name: metrics["ml_val"]
                },
                iteration,
            )

            self.tb.add_scalars(
                f"{model.model_name}/full_val",
                {
                    fold_name: metrics["full_val"]
                },
                iteration,
            )

        # --------------------------------------------------
        # Contribution logging
        # --------------------------------------------------

        if contributions is not None:

            train_contrib = contributions.get(
                "train"
            )

            if train_contrib is not None:

                self.tb.add_scalars(
                    f"{model.model_name}/prediction_contributions/train",
                    {
                        f"{fold_name}_fixed_effects":
                            train_contrib[
                                "fixed_effect_pct"
                            ],

                        f"{fold_name}_random_effects":
                            train_contrib[
                                "random_effect_pct"
                            ],
                    },
                    iteration,
                )

            val_contrib = contributions.get(
                "val"
            )

            if val_contrib is not None:

                self.tb.add_scalars(
                    f"{model.model_name}/prediction_contributions/val",
                    {
                        f"{fold_name}_fixed_effects":
                            val_contrib[
                                "fixed_effect_pct"
                            ],

                        f"{fold_name}_random_effects":
                            val_contrib[
                                "random_effect_pct"
                            ],
                    },
                    iteration,
                )

    # ======================================================
    # CONVERGENCE
    # ======================================================

    def _check_convergence(
        self,
        metrics,
        prev_loss,
    ):
        """Check early stopping condition."""

        if (
            not self.early_stopping
            or prev_loss is None
        ):
            return False

        return (
            abs(
                prev_loss
                - metrics["full_train"]
            )
            < self.tol
        )

    # ======================================================
    # OUTPUT
    # ======================================================

    def _build_output(
        self,
        history,
        pred_history,
        model_history,
        re_history,
        baselines,
        lmm_metrics_history
    ):
        """Package final outputs."""

        return {
            "history": history,

            "predictions_history": (
                pred_history
            ),

            "ml_models_history": (
                model_history
            ),

            "random_effects_history": (
                re_history
            ),

            "final_model": (
                model_history[-1]
                if model_history
                else None
            ),

            "final_random_effects": (
                re_history[-1]
                if re_history
                else None
            ),

            "final_lmm_metrics": (
                lmm_metrics_history[-1]
                if lmm_metrics_history
                else None
            ),

            "baselines": baselines,
        }
    
    def save_output(self, output, path):
        """
        Save the complete pipeline output.

        Parameters
        ----------
        output : dict
            Output returned by pipeline.run().
        path : str or Path
            Destination file.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            output,
            path,
            compress=3,
        )

        print(
            f"Saved MERL results to: {path}"
        )
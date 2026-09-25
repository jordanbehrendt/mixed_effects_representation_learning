import torch
import numpy as np
from ..utils.seeding import set_seed


def train_ml_model(
    ml_model_fn,
    ml_config,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    X_train_extra=None,
    X_val_extra=None,
    X_test_extra=None,
    logger=None,
    init_model=None,
    iteration=0,
    fold=None,
    seed=None,
):
    """Train an ML model and return the fitted model with predictions."""

    if seed is not None:
        set_seed(seed)

    # --------------------------------------------------
    # SAFE CONVERSION
    # --------------------------------------------------
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    if X_val is not None:
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

    if X_test is not None:
        X_test = np.asarray(X_test, dtype=np.float32)

    if X_train_extra is not None:
        X_train_extra = np.asarray(
            X_train_extra,
            dtype=np.float32,
        )

    if X_val_extra is not None:
        X_val_extra = np.asarray(
            X_val_extra,
            dtype=np.float32,
        )

    if X_test_extra is not None:
        X_test_extra = np.asarray(
            X_test_extra,
            dtype=np.float32,
        )

    # Initialize model (either from scratch or reuse existing one)
    if init_model is not None:
        model = init_model
    else:
        model = ml_model_fn(ml_config)

    # -----------------------------
    # FIT MODEL
    # -----------------------------
    fit_output = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_train_extra=X_train_extra,
        X_val_extra=X_val_extra,
        config=ml_config.get("params"),
        logger=logger,
        iteration=iteration,
        fold=fold,
    )

    best_model = fit_output.get("best_model", model)

    # -----------------------------
    # GENERATE PREDICTIONS
    # -----------------------------
    preds = {
        "train": best_model.predict(
            X_train,
            extra_features=X_train_extra,
        ),

        "val": (
            best_model.predict(
                X_val,
                extra_features=X_val_extra,
            )
            if X_val is not None
            else None
        ),

        "test": (
            best_model.predict(
                X_test,
                extra_features=X_test_extra,
            )
            if X_test is not None
            else None
        ),
    }

    return best_model, preds
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml


def find_project_root(start_path: Path) -> Path:
    """Find and return the project root by locating the 'scripts' directory."""
    for parent in start_path.resolve().parents:
        if (parent / "scripts").exists():
            return parent

    raise RuntimeError("Project root not found")


def _load_dataset(
    dataset_path,
    project_root,
    yaml_file,
    seed=42,
    experiment_name=None,
    output_dir=None,
):
    """
    Load the dataset and build the MERL pipeline.

    This function contains the logic shared by all splitting strategies.

    Returns
    -------
    latent_df : pd.DataFrame
        Latent/participant-level metadata.

    X : torch.Tensor
        Input data with shape (N, T, F).

    y : torch.Tensor
        Outcome values.

    config : dict
        Loaded YAML configuration.

    pipeline :
        Pipeline constructed from the YAML configuration.
    """

    dataset_path = Path(dataset_path)
    project_root = Path(project_root)

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    latent_df = pd.read_parquet(
        dataset_path / "latent.parquet"
    )

    outcome_df = pd.read_parquet(
        dataset_path / "outcome.parquet"
    )

    X_npy = np.load(
        dataset_path / "time_series.npy"
    )

    # Original shape:
    # (N, F, T)
    #
    # Transformer expects:
    # (N, T, F)

    X = torch.tensor(
        X_npy,
        dtype=torch.float32,
    ).permute(0, 2, 1)

    y = torch.tensor(
        outcome_df["y"].values,
        dtype=torch.float32,
    )

    latent_df = latent_df.reset_index(drop=True)
    outcome_df = outcome_df.reset_index(drop=True)

    # --------------------------------------------------
    # 2. CHECK DATA ALIGNMENT
    # --------------------------------------------------

    if not (
        len(latent_df)
        == X.shape[0]
        == len(y)
    ):
        raise ValueError(
            "Mismatch in dataset alignment: "
            f"latent_df={len(latent_df)}, "
            f"X={X.shape[0]}, "
            f"y={len(y)}"
        )

    # --------------------------------------------------
    # 3. LOAD CONFIGURATION
    # --------------------------------------------------

    yaml_path = Path(yaml_file)

    if not yaml_path.is_absolute():
        yaml_path = project_root / yaml_path

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"YAML configuration not found: {yaml_path}"
        )

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # If output_dir is provided, override the log_dir in the configuration. Same for experiment_name.
    if output_dir is not None:
        config["pipeline"]["merl"]["log_dir"] = output_dir

    if experiment_name is not None:
        config["pipeline"]["merl"]["description"] = experiment_name

    # --------------------------------------------------
    # 4. LOAD PIPELINE
    # --------------------------------------------------

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.pipelines.build_pipeline import build_pipeline

    pipeline = build_pipeline(
        config,
        seed=seed,
    )

    return latent_df, X, y, config, pipeline


def _create_split_data(
    latent_df,
    X,
    y,
    train_idx,
    val_idx,
    test_idx,
    config,
):
    """
    Create tensors and dataframes from split indices.

    This logic is shared by all splitting strategies.
    """

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    # --------------------------------------------------
    # 1. CREATE TENSOR SPLITS
    # --------------------------------------------------

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # --------------------------------------------------
    # 2. CREATE DATAFRAME SPLITS
    # --------------------------------------------------

    train_df = (
        latent_df
        .loc[train_idx]
        .reset_index(drop=True)
        .copy()
    )

    val_df = (
        latent_df
        .loc[val_idx]
        .reset_index(drop=True)
        .copy()
    )

    test_df = (
        latent_df
        .loc[test_idx]
        .reset_index(drop=True)
        .copy()
    )

    # Add outcome to each dataframe
    train_df["y"] = y_train.numpy()
    val_df["y"] = y_val.numpy()
    test_df["y"] = y_test.numpy()

    # --------------------------------------------------
    # 3. PACKAGE DATA
    # --------------------------------------------------

    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,

        # Useful for inspecting the split
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,

        # Configuration
        "config": config,
    }

    return data


def _last_measurement_participant_split(
    df,
    group_col="z",
    time_col="t",
    seed=42,
):
    """
    Create a participant-level 50/50 split using only the
    final measurement of each participant.

    For each participant:

        all measurements except the final one -> train
        final measurement                    -> validation OR test

    Approximately 50% of participants are assigned to validation
    and the remaining participants to test.

    The split is reproducible through `seed`.
    """

    rng = np.random.default_rng(seed)

    train_idx = []
    last_measurements = []

    # --------------------------------------------------
    # 1. COLLECT PARTICIPANTS
    # --------------------------------------------------

    for _, group in df.groupby(group_col):

        group = group.sort_values(time_col)
        idx = group.index.to_numpy()

        # Participants with at least one measurement
        if len(idx) == 0:
            continue

        # All measurements except the final one -> train
        train_idx.extend(idx[:-1])

        # Keep only the final measurement
        last_measurements.append(idx[-1])

    # --------------------------------------------------
    # 2. RANDOMIZE PARTICIPANTS
    # --------------------------------------------------

    rng.shuffle(last_measurements)

    # --------------------------------------------------
    # 3. SPLIT FINAL MEASUREMENTS 50/50
    # --------------------------------------------------

    n_participants = len(last_measurements)

    n_val = n_participants // 2

    # If odd, randomly decide which split gets the extra participant
    if n_participants % 2 == 1:
        if rng.random() < 0.5:
            n_val += 1

    val_idx = last_measurements[:n_val]
    test_idx = last_measurements[n_val:]

    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


def load_dataset_last_measurement_participant_split(
    dataset_path,
    project_root,
    yaml_file,
    seed=42,
    group_col="z",
    time_col="t",
    experiment_name=None,
    output_dir=None,
):
    """
    Load the dataset using a participant-level 50/50 split
    of the final measurement.

    For each participant:

        measurements[:-1] -> train

    The final measurement is assigned entirely to either
    validation or test.

    Approximately 50% of participants are assigned to validation
    and 50% to test.

    The split is reproducible through `seed`.
    """

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    latent_df, X, y, config, pipeline = _load_dataset(
        dataset_path=dataset_path,
        project_root=project_root,
        yaml_file=yaml_file,
        seed=seed,
        experiment_name=experiment_name,
        output_dir=output_dir
    )

    # --------------------------------------------------
    # 2. CREATE SPLIT
    # --------------------------------------------------

    train_idx, val_idx, test_idx = (
        _last_measurement_participant_split(
            latent_df,
            group_col=group_col,
            time_col=time_col,
            seed=seed,
        )
    )

    # --------------------------------------------------
    # 3. CREATE DATA OBJECT
    # --------------------------------------------------

    data = _create_split_data(
        latent_df=latent_df,
        X=X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        config=config,
    )

    return data, pipeline
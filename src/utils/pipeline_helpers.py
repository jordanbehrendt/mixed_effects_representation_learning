import numpy as np
import pandas as pd


def build_prediction_df(
    y_true,
    ml_pred,
    df,
    random_effects,
    group_var,
    apply_re_fn,
):
    """Build a DataFrame with true values, ML predictions, random effects, and full predictions."""

    if y_true is None or ml_pred is None:
        return None  # Nothing to build

    y_true = np.asarray(y_true, dtype=float)   # Ensure NumPy array
    ml_pred = np.asarray(ml_pred, dtype=float) # Ensure NumPy array

    n = len(y_true)

    # -----------------------------
    # random effects contribution
    # -----------------------------
    if random_effects is not None and df is not None:
        re = apply_re_fn(df, random_effects, group_var)  # Compute random effects
    else:
        re = np.full(n, np.nan)  # No random effects available

    # -----------------------------
    # full prediction
    # -----------------------------
    if np.isnan(re).all():
        full_pred = ml_pred  # Only ML prediction
    else:
        full_pred = ml_pred + re  # ML + random effects

    # -----------------------------
    # dataframe
    # -----------------------------
    out = pd.DataFrame({
        "y_true": y_true,
        "ml_pred": ml_pred,
        "random_effect": re,
        "full_pred": full_pred,
    })

    out.index.name = "sample_index"  # Label index

    return out


def apply_random_effects(df, random_effects, group_var):
    """Compute random effects contribution for each row based on group membership."""
    if random_effects is None:
        return np.zeros(len(df))  # No effects → zeros

    df = df.copy()  # Avoid modifying original
    df = df.merge(random_effects, on=group_var, how="left")  # Join effects

    re = np.zeros(len(df), dtype=float)  # Initialize effects

    if "re_(Intercept)" in df.columns:
        re += df["re_(Intercept)"].astype(float).values  # Add intercept effect

    for col in random_effects.columns:
        if col.startswith("re_") and col != "re_(Intercept)":
            var = col.replace("re_", "")
            if var in df.columns:
                re += df[col].astype(float).values * df[var].astype(float).values  # Add slope effect

    return re


def combine_predictions(preds, dfs, random_effects, group_var):
    """Combine ML predictions with random effects for each data split."""
    full = {}

    for split in ["train", "val", "test"]:
        if preds[split] is None:
            full[split] = None  # Skip missing split
            continue

        if random_effects is None:
            full[split] = preds[split]  # No adjustment
        else:
            re = apply_random_effects(dfs[split], random_effects, group_var)  # Compute effects
            full[split] = preds[split] + re  # Add effects

    return full


def update_targets(y, df, random_effects, group_var):
    """Adjust targets by removing random effects."""
    if y is None:
        return None  # No targets

    if random_effects is None:
        return y  # No adjustment needed

    re = apply_random_effects(df, random_effects, group_var)  # Compute effects
    return y - re  # Remove effects


def to_numpy(y):
    """Convert input to a NumPy array of floats."""
    return np.asarray(y, dtype=float)


def compute_residuals(y_true, y_pred):
    """Return residuals (y_true - y_pred) as a NumPy array."""
    return to_numpy(y_true) - to_numpy(y_pred)
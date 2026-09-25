import numpy as np


def mse(y, pred):
    """Compute mean squared error between targets and predictions."""
    if y is None or pred is None:
        return None  # Handle missing inputs
    return float(np.mean((y - pred) ** 2))  # MSE


def compute_all_losses(y_dict, ml_preds, full_preds, loss="mse"):
    """Compute losses for all splits using the specified loss function."""

    # Select loss function
    if loss == "mse":
        loss_fn = mse
    elif loss == "rmse":
        loss_fn = lambda y, pred: np.sqrt(mse(y, pred))
    else:
        raise ValueError(f"Unsupported loss: {loss}")  # Unknown loss

    losses = {}

    for split in ["train", "val", "test"]:
        losses[f"ml_{split}"] = loss_fn(y_dict[split], ml_preds[split])       # ML-only loss
        losses[f"full_{split}"] = loss_fn(y_dict[split], full_preds[split])  # Full model loss

    return losses
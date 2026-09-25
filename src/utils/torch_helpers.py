import torch
import torch.nn as nn

def get_device(config):
    """Return torch device based on config."""

    device_cfg = config.get("device", "auto")

    # AUTO
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CPU
    if device_cfg == "cpu":
        return torch.device("cpu")

    # CUDA (accept cuda:0, cuda:1, cuda:5, etc.)
    if isinstance(device_cfg, str) and "cuda" in device_cfg:
        if torch.cuda.is_available():
            return torch.device(device_cfg)
        return torch.device("cpu")

    # GPU index as integer (optional convenience)
    if isinstance(device_cfg, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_cfg}")
        return torch.device("cpu")

    raise ValueError(f"Unknown device config: {device_cfg}")

def get_loss(name: str):
    """Return loss function by name."""
    name = name.lower()

    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.SmoothL1Loss()

    raise ValueError(f"Unknown loss: {name}")


def get_optimizer(name: str, params, model_parameters):
    """Return optimizer instance."""
    name = name.lower()

    if name == "adam":
        return torch.optim.Adam(model_parameters, **params)
    elif name == "adamw":
        return torch.optim.AdamW(model_parameters, **params)
    elif name == "sgd":
        return torch.optim.SGD(model_parameters, **params)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model_parameters, **params)

    raise ValueError(f"Unknown optimizer: {name}")
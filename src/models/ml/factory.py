from src.models.ml.torch_models import TorchModelWrapper
from src.models.ml.merl_biosignal_transformer import MERLBiosignalTransformer

def biosignal_transformer_model_fn(input_dim, config):
    """Factory function to create a multimodal biosignal Transformer."""

    return MERLBiosignalTransformer(
        input_dim=input_dim,
        extra_dim=config.get("extra_dim", 0),

        # Transformer parameters
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        num_layers=config.get("num_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 256),

        # Regularization / activation
        dropout=config.get("dropout", 0.1),
        activation=config.get("activation", "gelu"),

        # Pooling
        pooling=config.get("pooling", "cls"),

        # Sequence length
        max_seq_len=config.get("max_seq_len", 2000),

        # Modality-specific CNN parameters
        conv_channels=config.get("conv_channels", 32),
        conv_kernel_size=config.get("conv_kernel_size", 9),

        # Temporal downsampling before Transformer
        downsample_factor=config.get("downsample_factor", 2),

        # Use signal branch
        use_signal_branch=config.get("use_signal_branch", True),
    )

def get_ml_model(config):
    """Return a configured ML model wrapper based on the given config."""
    name = config["name"]
    framework = config.get("framework", "sklearn")
    params = config.get("params", {})

    if framework == "torch":
        
        if name == "biosignal_transformer":
            return TorchModelWrapper(model_fn=biosignal_transformer_model_fn, config=params, model_name=name)

    raise ValueError(f"Unknown model: {name} with framework: {framework}")  # Invalid config error
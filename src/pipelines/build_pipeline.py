from src.pipelines.merl import MERLPipeline
from src.models.mixed_effects.adapter import lmm_adapter
from src.models.ml.factory import get_ml_model
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def build_pipeline(config, seed=None):
    """Build and return a configured MERL pipeline."""

    pipe_cfg = config["pipeline"]["merl"]  # Pipeline configuration
    ml_cfg = config["model"]["ml"]            # ML model configuration
    lmm_cfg = config["model"]["lmm"]          # LMM configuration
    description = config["pipeline"]["merl"].get("description", "Test")

    tb = None
    if pipe_cfg.get("tensorboard", False):
        log_dir = pipe_cfg.get("log_dir", "runs/merl")
        log_dir = log_dir / f"{description}_{int(time.time())}"
        tb = SummaryWriter(log_dir=log_dir)  # Initialize TensorBoard logger

    pipeline = MERLPipeline(
        ml_model_fn=get_ml_model,                  # ML model factory
        ml_config=ml_cfg,
        lmm_fn=lmm_adapter,                        # LMM adapter function
        lmm_config=lmm_cfg,
        loss=pipe_cfg.get("loss", "mse"),          # Loss function name
        n_iter=pipe_cfg.get("n_iter", 5),          # Number of iterations
        use_ml=pipe_cfg.get("use_ml", True),       # Enable ML component
        use_lmm=pipe_cfg.get("use_lmm", True),     # Enable LMM component
        early_stopping=pipe_cfg.get("early_stopping", True),  # Early stopping flag
        tol=pipe_cfg.get("tol", 1e-4),             # Convergence tolerance
        ml_mode=pipe_cfg.get("ml_init_mode", "reinit"),  # ML init strategy
        tensorboard_logger=tb,                      # TensorBoard writer
        seed=seed                                   # Random seed for reproducibility
    )

    return pipeline  # Return configured pipeline
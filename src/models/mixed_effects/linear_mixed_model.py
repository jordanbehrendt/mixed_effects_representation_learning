import subprocess
import pandas as pd
import os
import tempfile
from src.utils import helpers
from pathlib import Path

def fit_lmm_safe(lmm_fn, lmm_config, train_df, residuals, history, lmm_metrics_history, iteration):
    """Safely fit LMM and fall back to previous random effects if fitting fails."""

    df = train_df.copy()
    df["residual"] = residuals  # Add residuals as target

    try:
        outputs = lmm_fn(
            data=df,
            **lmm_config
        )
        return outputs["random_effects"], outputs["metrics"]  # Return estimated random effects

    except RuntimeError as e:
        print(f"LMM failed at iter {iteration}: {e}")  # Log failure

        if history:
            print("→ Reusing previous random effects")
            return history[-1], lmm_metrics_history[-1]  # Fallback to last successful estimate

        return None, None  # No fallback available

def fit_lmm(
    script_path: str,
    data: pd.DataFrame,
    fixed_formula: str,
    group_var: str,
    random_formula: str = "1",
    random_effects_covariance: str = "diagonal",
    residual_covariance: str = "independence",
    time_var: str = "time",
    estimation_method: str = "REML",
) -> dict:
    """
    Fit a linear mixed model (LMM) using an R backend (nlme).

    This function interfaces with an R script via subprocess,
    enabling advanced mixed-effects modeling while maintaining
    a Python-based workflow.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing all variables used in the model.
    fixed_formula : str
        Fixed-effects formula in R syntax (e.g., "y ~ x1 + x2").
    group_var : str
        Grouping variable for random effects (e.g., subject ID).
    random_formula : str, optional
        Random-effects structure in R syntax (default: "1",
        i.e., random intercept only).
    cov_structure : str, optional
        Covariance structure for within-group errors.
        Supported options include:
        - "independence"
        - "AR1" (autoregressive)
        - "CS" (compound symmetry)
    time_var : str, optional
        Time variable required for time-dependent covariance
        structures such as AR(1).
    estimation_method : str, optional
        Estimation method used by the model:
        - "REML" (default)
        - "ML"

    Returns
    -------
    dict
        Dictionary containing:
        - fixed_effects : pd.DataFrame
        - random_effects : pd.DataFrame
            Indexed by group variable.
        - metrics : dict
            Model fit statistics.

    Raises
    ------
    RuntimeError
        If the R script execution fails.
    FileNotFoundError
        If the R script or expected output files are missing.
    ValueError
        If input data is invalid.

    Notes
    -----
    - Uses a temporary directory to exchange data with R.
    - Requires R and the `nlme` package to be installed.
    - The R script must output:
        * fixed_effects.csv
        * random_effects.csv
        * metrics.csv
    - Error messages from R are propagated for easier debugging.
    """

    if data is None or data.empty:
        raise ValueError("Input `data` must be a non-empty DataFrame.")  # Validate input

    PROJECT_ROOT = helpers.find_project_root(Path(__file__))  # Locate project root
    script_path = os.path.join(PROJECT_ROOT, script_path)     # Resolve full script path

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"R script not found: {script_path}")  # Validate script

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.csv")

        # ------------------------------------------------------------------
        # Write input data
        # ------------------------------------------------------------------
        data.to_csv(input_path, index=False)  # Save data for R

        # ------------------------------------------------------------------
        # Call R script
        # ------------------------------------------------------------------
        result = subprocess.run(
            [
                "Rscript",
                script_path,
                input_path,
                fixed_formula,
                group_var,
                random_formula,
                random_effects_covariance,
                residual_covariance,
                time_var,
                estimation_method,
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )

        # ------------------------------------------------------------------
        # Handle R errors explicitly
        # ------------------------------------------------------------------
        if result.returncode != 0:
            raise RuntimeError(f"R script failed:\n{result.stderr}")  # Propagate R error

        # ------------------------------------------------------------------
        # Load outputs
        # ------------------------------------------------------------------
        fixed_path = os.path.join(tmpdir, "fixed_effects.csv")
        random_path = os.path.join(tmpdir, "random_effects.csv")
        metrics_path = os.path.join(tmpdir, "metrics.csv")

        if not all(os.path.exists(p) for p in [fixed_path, random_path, metrics_path]):
            raise FileNotFoundError(
                "One or more expected output files were not generated by the R script."
            )  # Ensure outputs exist

        fixed = pd.read_csv(fixed_path)    # Load fixed effects
        random = pd.read_csv(random_path)  # Load random effects
        metrics = pd.read_csv(metrics_path)  # Load metrics

        return {
            "fixed_effects": fixed,
            "random_effects": random.set_index(group_var),  # Index by group
            "metrics": metrics.to_dict(orient="records")[0],  # Convert to dict
        }
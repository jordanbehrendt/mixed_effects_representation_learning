# Mixed-Effects Representation Learning (MERL)

Code for **Mixed-Effects Machine Learning: Integrating Random-Effects into
Representation Learning for Longitudinal Multimodal Data**.

MERL combines a multimodal CNN–Transformer with a linear mixed-effects
model to separate population-level representation learning from
participant-specific random effects in longitudinal, multimodal
(biosignal) prediction tasks. This repository contains the implementation
of MERL, the neural network and mixed-effects baselines used in the paper
and the scripts needed to reproduce the reported experiments.

## Repository Structure

```
.
├── run_iclr_experiments.py     # Main entry point: runs all 15 paper experiments
├── environment.yml             # Conda environment specification
├── model_yaml/                 # Experiment configurations
│   ├── ml.yaml                 #   NN baseline (no participant identifiers)
│   ├── ml_ohe.yaml             #   NN baseline (one-hot participant identifiers)
│   └── merl.yaml               #   MERL (NN + linear mixed-effects model)
├── scripts/
│   └── linear_mixed_model.R    # Fits the LMM step of MERL (called via subprocess)
├── src/
│   ├── models/
│   │   ├── ml/                 # CNN–Transformer architecture and model factory
│   │   └── mixed_effects/      # Linear mixed-effects model + adapter to the R script
│   ├── pipelines/               # MERL alternating optimization pipeline, plain-ML pipeline, losses
│   └── utils/                   # Dataset loading, experiment runner, seeding, helper functions
└── data/                       # Simulated datasets (see "Data" below)
```

## Environment Setup

Dependencies for the Python side are listed in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate merl_env
```

The mixed-effects component of MERL is fit in R (`scripts/linear_mixed_model.R`,
using the `nlme` package) and is called from Python via `subprocess`. This
means, in addition to the conda environment above, you need a working
**R installation with the `nlme` package** available on your system `PATH`
(e.g. `install.packages("nlme")` in R). `nlme` is part of R's base
distribution in most installations, but make sure `Rscript` is callable
from the shell the experiments run in.

## Data

Each experimental condition has its own dataset directory containing:

- `latent.parquet` — participant-/assessment-level generative features and metadata
- `outcome.parquet` — the simulated outcome `y`
- `time_series.npy` — the simulated multimodal biosignals (ECG, PPG, respiration, EMG₁, EMG₂), shaped `(N, F, T)`

The paper considers five outcome/variance conditions, corresponding to five
dataset directories:

```
fixed_effects_outcome/
random_effects_equal_variance_outcome/
random_effects_heterogeneous_variance_outcome/
mixed_effects_equal_variance_outcome/
mixed_effects_heterogeneous_variance_outcome/
```

**`time_series.npy` is shared across conditions and is not stored separately
per directory in this release**. Before running any experiments, copy the shared
`time_series.npy` file into **each** of the five dataset directories listed
above, so that every directory contains all three files
(`latent.parquet`, `outcome.parquet`, `time_series.npy`).

## Running the Experiments

`run_iclr_experiments.py` runs all 15 experiments reported in the paper
(3 methods × 5 outcome/variance conditions), each across 5 random seeds
(`{42, 123, 456, 789, 101112}`).

Before running it, open the script and fill in the four `TODO` paths at the
top:

```python
yaml_dir = "..."        # path to the model_yaml/ directory (trailing slash), e.g. "model_yaml/"
main_output_dir = "..." # directory where experiment outputs should be written (trailing slash)
dataset_dir = "..."     # parent directory containing the five dataset folders above (trailing slash)
PROJECT_ROOT = "..."    # absolute path to the root of this repository
```

Notes:
- `PROJECT_ROOT` is added to `sys.path` so that the `src` package can be
  imported and is also used to resolve the relative `r_script` path in the
  YAML configs (`scripts/linear_mixed_model.R`) — it should point to the
  repository root (the directory containing `src/`, `scripts/` and
  `model_yaml/`).
- `yaml_dir` and `dataset_dir` are concatenated with filenames using plain
  string concatenation (e.g. `yaml_dir + "ml.yaml"`), so make sure they end
  with a trailing `/`.

Once the paths are set and the shared `time_series.npy` has been copied
into each dataset directory, run:

```bash
python run_iclr_experiments.py
```

This runs the experiments sequentially; execution stops if an experiment
fails. For each of the 15 experiments, outputs for every seed are written
to `main_output_dir/<experiment_name>/output_seed<seed>/`.

### Running a single experiment

To run a subset of experiments (e.g. for debugging or partial reruns), you
can import and call `run_experiment` directly instead of executing the full
script:

```python
from src.utils.run_experiment import run_experiment

run_experiment(
    PROJECT_ROOT,
    seeds=[42],
    experiment_name="mixed_effects_equal_variance_merl",
    dataset_path=dataset_dir + "mixed_effects_equal_variance_outcome",
    yaml_file=yaml_dir + "merl.yaml",
    output_dir=main_output_dir,
)
```

### Experiment configurations

The three YAML files in `model_yaml/` correspond to the three prediction
methods compared in the paper and are applied to each of the five dataset
conditions:

| Config | Method | Description |
|---|---|---|
| `ml.yaml` | NN | CNN–Transformer only, no participant identifiers, no random effects (`use_lmm: false`) |
| `ml_ohe.yaml` | NN$_{\text{OHE}}$ | Same architecture, with a one-hot participant identifier concatenated before the regression head |
| `merl.yaml` | MERL | CNN–Transformer alternated with a linear mixed-effects model (`use_lmm: true`) |

The mixed-effects specification (grouping variable, random-effects formula,
covariance structure, REML estimation) is set under the `lmm:` key in each
YAML file and the neural network architecture/training hyperparameters are
set under `model.ml.params`, matching the values reported in the paper's
appendix. The training device is also configurable there
(`model.ml.params.device`, e.g. `cuda:0` or `cpu`) — the provided configs
default to `cuda:0`, so adjust this per YAML file if you are running without
a GPU or on a different device.

## Outputs

For each `(experiment, seed)` combination, the pipeline saves the trained
model, predictions and metrics (including estimated random-effects
variance components, where applicable) to
`main_output_dir/<experiment_name>/output_seed<seed>/`. If
`pipeline.merl.tensorboard: true` (as set in the provided configs),
TensorBoard logs are additionally written for the MERL alternating
optimization procedure.

## Citation

If you use this code, please cite the paper (see the ICLR 2027 submission
for full details).
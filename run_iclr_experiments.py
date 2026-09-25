from pathlib import Path
import sys

# ============================================================
# Project setup
# ============================================================
seeds = [42, 123, 456, 789, 101112]
yaml_dir = # TODO
main_output_dir = # TODO
dataset_dir = # TODO
PROJECT_ROOT = # TODO

sys.path.append(str(PROJECT_ROOT))

from src.utils.run_experiment import run_experiment


# ============================================================
# Dataset directories
# ============================================================

fe_data_dir = f"{dataset_dir}fixed_effects_outcome"

re_equal_variance_data_dir = (
    f"{dataset_dir}random_effects_equal_variance_outcome"
)

re_heterogeneous_variance_data_dir = (
    f"{dataset_dir}random_effects_heterogeneous_variance_outcome"
)

me_equal_variance_data_dir = (
    f"{dataset_dir}mixed_effects_equal_variance_outcome"
)

me_heterogeneous_variance_data_dir = (
    f"{dataset_dir}mixed_effects_heterogeneous_variance_outcome"
)


# ============================================================
# Fixed effects experiments
# ============================================================

fe_ml = {
    "experiment_name": "fixed_effects_ml",
    "dataset_path": fe_data_dir,
    "yaml_file": yaml_dir + "ml.yaml",
    "output_dir": main_output_dir + "fixed_effects_ml/",
}

fe_ml_ohe = {
    "experiment_name": "fixed_effects_ml_ohe",
    "dataset_path": fe_data_dir,
    "yaml_file": yaml_dir + "ml_ohe.yaml",
    "output_dir": main_output_dir + "fixed_effects_ml_ohe/",
}

fe_merl = {
    "experiment_name": "fixed_effects_merl",
    "dataset_path": fe_data_dir,
    "yaml_file": yaml_dir + "merl.yaml",
    "output_dir": main_output_dir + "fixed_effects_merl/",
}


# ============================================================
# Random effects — equal variance
# ============================================================

re_equal_ml = {
    "experiment_name": "random_effects_equal_variance_ml",
    "dataset_path": re_equal_variance_data_dir,
    "yaml_file": yaml_dir + "ml.yaml",
    "output_dir": main_output_dir + "random_effects_equal_variance_ml/",
}

re_equal_ml_ohe = {
    "experiment_name": "random_effects_equal_variance_ml_ohe",
    "dataset_path": re_equal_variance_data_dir,
    "yaml_file": yaml_dir + "ml_ohe.yaml",
    "output_dir": main_output_dir + "random_effects_equal_variance_ml_ohe/",
}

re_equal_merl = {
    "experiment_name": "random_effects_equal_variance_merl",
    "dataset_path": re_equal_variance_data_dir,
    "yaml_file": yaml_dir + "merl.yaml",
    "output_dir": main_output_dir + "random_effects_equal_variance_merl/",
}


# ============================================================
# Random effects — heterogeneous variance
# ============================================================

re_heterogeneous_ml = {
    "experiment_name": "random_effects_heterogeneous_variance_ml",
    "dataset_path": re_heterogeneous_variance_data_dir,
    "yaml_file": yaml_dir + "ml.yaml",
    "output_dir": (
        main_output_dir
        + "random_effects_heterogeneous_variance_ml/"
    ),
}

re_heterogeneous_ml_ohe = {
    "experiment_name": "random_effects_heterogeneous_variance_ml_ohe",
    "dataset_path": re_heterogeneous_variance_data_dir,
    "yaml_file": yaml_dir + "ml_ohe.yaml",
    "output_dir": (
        main_output_dir
        + "random_effects_heterogeneous_variance_ml_ohe/"
    ),
}

re_heterogeneous_merl = {
    "experiment_name": "random_effects_heterogeneous_variance_merl",
    "dataset_path": re_heterogeneous_variance_data_dir,
    "yaml_file": yaml_dir + "merl.yaml",
    "output_dir": (
        main_output_dir
        + "random_effects_heterogeneous_variance_merl/"
    ),
}


# ============================================================
# Mixed effects — equal variance
# ============================================================

me_equal_ml = {
    "experiment_name": "mixed_effects_equal_variance_ml",
    "dataset_path": me_equal_variance_data_dir,
    "yaml_file": yaml_dir + "ml.yaml",
    "output_dir": main_output_dir + "mixed_effects_equal_variance_ml/",
}

me_equal_ml_ohe = {
    "experiment_name": "mixed_effects_equal_variance_ml_ohe",
    "dataset_path": me_equal_variance_data_dir,
    "yaml_file": yaml_dir + "ml_ohe.yaml",
    "output_dir": main_output_dir + "mixed_effects_equal_variance_ml_ohe/",
}

me_equal_merl = {
    "experiment_name": "mixed_effects_equal_variance_merl",
    "dataset_path": me_equal_variance_data_dir,
    "yaml_file": yaml_dir + "merl.yaml",
    "output_dir": main_output_dir + "mixed_effects_equal_variance_merl/",
}


# ============================================================
# Mixed effects — heterogeneous variance
# ============================================================

me_heterogeneous_ml = {
    "experiment_name": "mixed_effects_heterogeneous_variance_ml",
    "dataset_path": me_heterogeneous_variance_data_dir,
    "yaml_file": yaml_dir + "ml.yaml",
    "output_dir": (
        main_output_dir
        + "mixed_effects_heterogeneous_variance_ml/"
    ),
}

me_heterogeneous_ml_ohe = {
    "experiment_name": "mixed_effects_heterogeneous_variance_ml_ohe",
    "dataset_path": me_heterogeneous_variance_data_dir,
    "yaml_file": yaml_dir + "ml_ohe.yaml",
    "output_dir": (
        main_output_dir
        + "mixed_effects_heterogeneous_variance_ml_ohe/"
    ),
}

me_heterogeneous_merl = {
    "experiment_name": "mixed_effects_heterogeneous_variance_merl",
    "dataset_path": me_heterogeneous_variance_data_dir,
    "yaml_file": yaml_dir + "merl.yaml",
    "output_dir": (
        main_output_dir
        + "mixed_effects_heterogeneous_variance_merl/"
    ),
}


# ============================================================
# Experiments
# ============================================================

experiments = [
    fe_ml,
    fe_ml_ohe,
    fe_merl,
    re_equal_ml,
    re_equal_ml_ohe,
    re_equal_merl,
    re_heterogeneous_ml,
    re_heterogeneous_ml_ohe,
    re_heterogeneous_merl,
    me_equal_ml,
    me_equal_ml_ohe,
    me_equal_merl,
    me_heterogeneous_ml,
    me_heterogeneous_ml_ohe,
    me_heterogeneous_merl,
]


# ============================================================
# Sequential execution
# ============================================================

def run_experiments(experiments):
    """
    Run all experiments sequentially.

    Each experiment must finish before the next experiment starts.
    If an experiment fails, execution stops and the remaining
    experiments are not started.
    """

    total_experiments = len(experiments)

    for i, experiment in enumerate(experiments, start=1):

        experiment_name = experiment["experiment_name"]

        print()
        print("=" * 80)
        print(
            f"STARTING EXPERIMENT {i}/{total_experiments}: "
            f"{experiment_name}"
        )
        print("=" * 80, flush=True)

        try:
            run_experiment(
                PROJECT_ROOT,
                seeds,
                **experiment,
            )

        except Exception as e:
            print()
            print("=" * 80)
            print(
                f"FAILED EXPERIMENT {i}/{total_experiments}: "
                f"{experiment_name}"
            )
            print(f"Error: {repr(e)}")
            print("=" * 80, flush=True)

            # Stop here. Do not start the remaining experiments.
            raise

        print()
        print("=" * 80)
        print(
            f"COMPLETED EXPERIMENT {i}/{total_experiments}: "
            f"{experiment_name}"
        )
        print("=" * 80, flush=True)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("STARTING MERL EXPERIMENTS")
    print(f"Number of experiments: {len(experiments)}")
    print(f"Seeds: {seeds}")
    print("=" * 80, flush=True)

    run_experiments(experiments)

    print()
    print("=" * 80)
    print("ALL MERL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 80, flush=True)
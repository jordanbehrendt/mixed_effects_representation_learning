from pathlib import Path
import sys


def run_experiment(
    project_root,
    seeds,
    experiment_name,
    dataset_path,
    yaml_file,
    output_dir,
):
    """
    Run an experiment for one or more random seeds and save the outputs.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root directory of the project. This directory is added to sys.path
        so that project-specific imports work.

    seeds : list[int]
        List of random seeds to run.

    experiment_name : str
        Name of the experiment. A directory with this name will be created
        inside output_dir.

    dataset_path : str or pathlib.Path
        Path to the dataset.

    yaml_file : str
        Path to the YAML experiment configuration, relative to project_root.

    output_dir : str or pathlib.Path
        Root directory where experiment outputs will be saved.

    Returns
    -------
    dict
        Dictionary containing the experiment configuration and outputs for
        every seed.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    project_root = Path(project_root)
    output_dir = Path(output_dir)

    # Make project modules importable
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Import after adding project_root to sys.path
    from src.utils.helpers import load_dataset_last_measurement_participant_split

    # Create experiment output directory
    experiment_output_dir = output_dir / experiment_name
    experiment_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\n{'=' * 80}\n"
        f"Experiment: {experiment_name}\n"
        f"{'=' * 80}"
    )

    experiment_outputs = []

    # ------------------------------------------------------------------
    # Run all seeds
    # ------------------------------------------------------------------

    for seed in seeds:

        print(
            f"\nRunning seed {seed} "
            f"for {yaml_file}"
        )

        # --------------------------------------------------------------
        # Load data and build pipeline
        # --------------------------------------------------------------

        data, pipeline = load_dataset_last_measurement_participant_split(
            dataset_path=dataset_path,
            project_root=project_root,
            yaml_file=yaml_file,
            seed=seed,
            experiment_name=experiment_name,
            output_dir=output_dir,   
        )

        # --------------------------------------------------------------
        # Run pipeline
        # --------------------------------------------------------------

        outputs = pipeline.run(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            train_df=data["train_df"],
            val_df=data["val_df"],
            test_df=data["test_df"],
            group_var="z",
            fold=0,
        )

        # --------------------------------------------------------------
        # Save output
        # --------------------------------------------------------------

        seed_output_path = (
            experiment_output_dir
            / f"output_seed{seed}"
        )

        experiment_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pipeline.save_output(
            output=outputs,
            path=seed_output_path,
        )

        print(
            f"Saved output for seed {seed} to "
            f"{seed_output_path}"
        )

        # Store seed together with results
        experiment_outputs.append(
            {
                "seed": seed,
                "outputs": outputs,
            }
        )

    # ------------------------------------------------------------------
    # Return experiment results
    # ------------------------------------------------------------------

    return {
        "experiment_name": experiment_name,
        "project_root": str(project_root),
        "dataset_path": str(dataset_path),
        "yaml_file": yaml_file,
        "seeds": seeds,
        "output_dir": str(experiment_output_dir),
        "outputs": experiment_outputs,
    }
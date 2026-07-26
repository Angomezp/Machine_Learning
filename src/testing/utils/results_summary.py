from pathlib import Path
import csv
import json


def update_results_summary(
    training_dir: str | Path,
    testing_dir: str | Path,
    metrics: dict,
    checkpoint_info: dict,
    threshold: float,
) -> None:
    """
    Actualiza el resumen global de experimentos.
    """

    training_dir = Path(training_dir)

    testing_dir = Path(testing_dir)

    # Carpeta del experimento (baseline, undersampling_1_10, etc.)
    experiment_dir = training_dir.parent

    summary_file = experiment_dir.parent / "results_summary.csv"

    config = checkpoint_info["config"]

    row = {

        "experiment": experiment_dir.name,

        "architecture": config["architecture"],

        "batch_size": config["batch_size"],

        "learning_rate": config["learning_rate"],

        "optimizer": config["optimizer"],

        "scheduler": config["scheduler"],

        "epochs": config["epochs"],

        "best_epoch": checkpoint_info["epoch"],

        "monitor_metric": checkpoint_info["monitor_metric"],

        "threshold": threshold,

        "loss": metrics["loss"],

        "precision": metrics["precision"],

        "recall": metrics["recall"],

        "f1": metrics["f1"],

        "roc_auc": metrics["roc_auc"],

    }

    write_header = not summary_file.exists()

    with open(
        summary_file,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    print()
    print(f"✓ Results summary updated -> {summary_file}")
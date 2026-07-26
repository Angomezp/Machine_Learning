from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

from sklearn.metrics import ConfusionMatrixDisplay

def plot_training_history(
    history_path: Path,
    threshold_results_path: Path,
    validation_predictions_path: Path,
    output_dir: Path,
)-> None:
    """
    Genera todas las figuras del entrenamiento.

    Parameters
    ----------
    history_path
        Ruta al training_history.json.

    threshold_results_path
        Ruta al threshold_results.json.

    validation_predictions_path
        Ruta al validation_predictions.npz.

    output_dir
        Carpeta donde se almacenarán las figuras.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # LOAD TRAINING HISTORY
    with open(history_path, "r") as f:

        history = json.load(f)

    # LOAD THRESHOLD RESULTS
    with open(threshold_results_path, "r") as f:

        threshold_results = json.load(f)

    validation_predictions = np.load(validation_predictions_path)
    labels = validation_predictions["labels"]

    probabilities = validation_predictions["probabilities"]

    # TRAINING CURVES
    plot_loss(
        history,
        output_dir,
    )

    plot_metrics(
        history,
        output_dir,
    )

    plot_learning_rate(
        history,
        output_dir,
    )

    # THRESHOLD ANALYSIS
    plot_threshold_curve(
        threshold_results,
        output_dir,
    )

    # ROC
    plot_roc_curve(
        labels,
        probabilities,
        output_dir,
    )

    # PRECISION-RECALL
    plot_precision_recall_curve(
        labels,
        probabilities,
        output_dir,
    )

    # CONFUSION MATRIX
    plot_confusion_matrix(
        labels,
        probabilities,
        threshold_results["best_threshold"],
        output_dir,
    )

def plot_loss(
    history: dict,
    output_dir: Path,
):

    epochs = history["epoch"]

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "loss.png",
        dpi=300,
    )

    plt.close()



def plot_metrics(
    history: dict,
    output_dir: Path,
):

    epochs = history["epoch"]

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        history["precision"],
        label="Precision",
    )

    plt.plot(
        epochs,
        history["recall"],
        label="Recall",
    )

    plt.plot(
        epochs,
        history["f1"],
        label="F1",
    )

    plt.plot(
        epochs,
        history["roc_auc"],
        label="ROC-AUC",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Score")

    plt.ylim(0,1)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "metrics.png",
        dpi=300,
    )

    plt.close()


def plot_learning_rate(
    history: dict,
    output_dir: Path,
):

    epochs = history["epoch"]

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        history["learning_rate"],
    )

    plt.xlabel("Epoch")

    plt.ylabel("Learning Rate")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "learning_rate.png",
        dpi=300,
    )

    plt.close()

def plot_threshold_curve(
    threshold_results: dict,
    output_dir: Path,
):
    """
    Grafica Precision, Recall y F1 en función del threshold.
    """

    history = threshold_results["history"]

    plt.figure(figsize=(8,5))

    plt.plot(
        history["threshold"],
        history["precision"],
        label="Precision",
    )

    plt.plot(
        history["threshold"],
        history["recall"],
        label="Recall",
    )

    plt.plot(
        history["threshold"],
        history["f1"],
        label="F1",
    )

    plt.axvline(
        threshold_results["best_threshold"],
        color="red",
        linestyle="--",
        label=f'Best = {threshold_results["best_threshold"]:.2f}',
    )

    plt.xlabel("Threshold")

    plt.ylabel("Metric")

    plt.ylim(0, 1)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "threshold_curve.png",
        dpi=300,
    )

    plt.close()

def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
):
    """
    Curva ROC.
    """

    fpr, tpr, _ = roc_curve(
        labels,
        probabilities,
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=300,
    )

    plt.close()

def plot_precision_recall_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
):
    """
    Curva Precision-Recall.
    """

    precision, recall, _ = precision_recall_curve(
        labels,
        probabilities,
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=300,
    )

    plt.close()


def plot_confusion_matrix(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_dir,
):
    """
    Grafica la matriz de confusión mostrando conteos y porcentajes.
    """

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    cm = confusion_matrix(
        labels,
        predictions,
    )

    # Porcentajes
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)


    # Figura
    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "0 - No Deforestation",
            "1 - Deforestation",
        ],
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    # Reemplazar texto por Conteo + %

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            disp.text_[i, j].set_text(
                f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)"
            )

            disp.text_[i, j].set_fontsize(12)

    # Estética
    ax.set_title("Confusion Matrix")

    ax.grid(False)

    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
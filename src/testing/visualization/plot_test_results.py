from pathlib import Path

from ...training.visualization.plot_training_history import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
)



def plot_test_results(
    labels,
    probabilities,
    threshold: float,
    output_dir: str | Path,
) -> None:
    """
    Genera todas las figuras correspondientes
    al conjunto de test.

    Parameters
    ----------
    labels
        Etiquetas reales.

    probabilities
        Probabilidades predichas por el modelo.

    threshold
        Threshold óptimo encontrado durante
        validación.

    output_dir
        Directorio del experimento.
    """

    output_dir = Path(output_dir)

    figures_dir = output_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ####################################################################
    # ROC
    ####################################################################

    plot_roc_curve(

        labels=labels,

        probabilities=probabilities,

        output_dir=figures_dir,

    )

    ####################################################################
    # PRECISION-RECALL
    ####################################################################

    plot_precision_recall_curve(

        labels=labels,

        probabilities=probabilities,

        output_dir=figures_dir,

    )

    ####################################################################
    # CONFUSION MATRIX
    ####################################################################

    plot_confusion_matrix(

        labels=labels,

        probabilities=probabilities,

        threshold=threshold,

        output_dir=figures_dir,

    )

    print()

    print("✓ Test figures generated")

    print(f"  • {figures_dir/'test_roc_curve.png'}")

    print(f"  • {figures_dir/'test_precision_recall_curve.png'}")

    print(f"  • {figures_dir/'test_confusion_matrix.png'}")
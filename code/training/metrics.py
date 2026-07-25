from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import numpy as np


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calcula las métricas de clasificación.

    Parameters
    ----------
    labels
        Etiquetas reales.

    probabilities
        Probabilidades predichas.

    threshold
        Umbral de clasificación.

    Returns
    -------
    dict
        Diccionario con las métricas.
    """

    # BINARIZE PREDICTIONS
    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    # METRICS
    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    # RETURN
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
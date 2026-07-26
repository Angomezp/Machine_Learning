from sklearn.metrics import (
    confusion_matrix,
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

    ###############################################################
    # BINARIZAR PREDICCIONES
    ###############################################################

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    ###############################################################
    # DEBUG (AÑADIR ESTE BLOQUE)
    ###############################################################

    print("\n" + "=" * 60)
    print("Prediction statistics")
    print("=" * 60)

    print(f"Threshold            : {threshold:.2f}")
    print(f"Min probability      : {probabilities.min():.4f}")
    print(f"Max probability      : {probabilities.max():.4f}")
    print(f"Mean probability     : {probabilities.mean():.4f}")

    print(f"Predicted positives  : {predictions.sum()}")
    print(f"Real positives       : {labels.sum()}")

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
    ).ravel()

    print("\nConfusion Matrix")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    ###############################################################
    # MÉTRICAS
    ###############################################################

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

    ###############################################################
    # RETORNO
    ###############################################################

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
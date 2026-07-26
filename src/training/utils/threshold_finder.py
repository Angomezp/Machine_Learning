from pathlib import Path

import json
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


class ThresholdFinder:
    """
    Busca el threshold que maximiza el F1 sobre el conjunto
    de validación.
    """

    def __init__(
        self,
        output_dir: Path,
        start: float = 0.01,
        stop: float = 0.99,
        step: float = 0.01,
    ) -> None:

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.thresholds = np.arange(
            start,
            stop + step,
            step,
        )

    def search(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict:

        history = {

            "threshold": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        best_threshold = 0.50

        best_f1 = -1.0

        best_precision = 0.0

        best_recall = 0.0

        for threshold in self.thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(np.uint8)

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

            history["threshold"].append( float(threshold) )

            history["precision"].append( float(precision) )

            history["recall"].append( float(recall) )

            history["f1"].append( float(f1) )

            if f1 > best_f1:

                best_f1 = f1

                best_threshold = threshold

                best_precision = precision

                best_recall = recall

        results = {

            "best_threshold": float(best_threshold),

            "best_f1": float(best_f1),

            "precision": float(best_precision),

            "recall": float(best_recall),

            "history": history,

        }

        self.save(results)

        return results

    def save(
        self,
        results: dict,
    ) -> None:

        file = self.output_dir / "threshold_results.json"

        with open(
            file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                results,
                f,
                indent=4,
            )
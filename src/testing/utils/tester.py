from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...training.utils.metrics import compute_metrics


class Tester:
    """
    Evaluador del modelo sobre el conjunto de test.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str,
        threshold: float,
    ) -> None:

        self.model = model.to(device)

        self.test_loader = test_loader

        self.criterion = criterion

        self.device = device

        self.threshold = threshold

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _move_to_device(
        self,
        batch: dict[str, torch.Tensor],
    ):
        """
        Mueve un batch al dispositivo.
        """

        static = batch["static"].to(self.device)

        temporal = batch["temporal"].to(self.device)

        label = batch["label"].float().to(self.device)

        return static, temporal, label

    def evaluate(
        self,
    ) -> dict:
        """
        Evalúa el modelo.
        """

        self.model.eval()

        running_loss = 0.0

        all_labels = []

        all_probabilities = []

        with torch.no_grad():

            progress = tqdm(
                self.test_loader,
                desc="Testing",
                leave=False,
            )

            for batch in progress:

                static, temporal, label = self._move_to_device(batch)

                logits = self.model(
                    static,
                    temporal,
                )

                loss = self.criterion(
                    logits.squeeze(1),
                    label,
                )

                running_loss += loss.item()

                probabilities = torch.sigmoid(
                    logits.squeeze(1)
                )

                all_labels.extend(
                    label.cpu().numpy()
                )

                all_probabilities.extend(
                    probabilities.cpu().numpy()
                )

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        test_loss = running_loss / len(
            self.test_loader
        )

        metrics = compute_metrics(
            labels=np.asarray(all_labels),
            probabilities=np.asarray(all_probabilities),
            threshold=self.threshold,
        )

        metrics["loss"] = test_loss

        metrics["labels"] = np.asarray(all_labels)

        metrics["probabilities"] = np.asarray(
            all_probabilities
        )

        return metrics

    def save_results(
        self,
        metrics: dict,
    ) -> None:
        """
        Guarda las métricas finales.
        """

        results = {

            "loss": float(metrics["loss"]),

            "precision": float(metrics["precision"]),

            "recall": float(metrics["recall"]),

            "f1": float(metrics["f1"]),

            "roc_auc": float(metrics["roc_auc"]),

            "threshold": float(self.threshold),

        }

        with open(
            self.output_dir / "test_results.json",
            "w",
        ) as f:

            json.dump(
                results,
                f,
                indent=4,
            )

    def save_predictions(
        self,
        metrics: dict,
    ) -> None:
        """
        Guarda las probabilidades obtenidas en test.
        """

        np.savez_compressed(

            self.output_dir /
            "test_predictions.npz",

            labels=metrics["labels"],

            probabilities=metrics[
                "probabilities"
            ],
        )
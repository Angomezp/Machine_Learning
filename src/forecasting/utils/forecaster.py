from pathlib import Path

import torch
import numpy as np

from tqdm import tqdm


class Forecaster:
    """
    Ejecuta inferencia del modelo sobre datos futuros.

    Genera:
        - Probabilidades de deforestación
        - Predicciones binarias usando threshold
    """

    def __init__(
        self,
        model,
        forecast_loader,
        device,
        threshold: float,
    ) -> None:

        self.model = model
        self.forecast_loader = forecast_loader
        self.device = device
        self.threshold = threshold


    def _move_to_device(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mueve los datos al dispositivo.

        Parameters
        ----------
        batch:
            Batch generado por el Dataset.

        Returns
        -------
        tuple
            static y temporal.
        """

        static = batch["static"].to( self.device )

        temporal = batch["temporal"].to( self.device ) 

        return static, temporal


    def forecast(self) -> dict:

        self.model.eval()

        all_probabilities = []

        all_coordinates = []

        all_geo_coordinates = []

        all_sample_ids = []

        with torch.no_grad():

            progress = tqdm(
                self.forecast_loader,
                desc="Forecasting",
                leave=False,
            )

            for batch in progress:

                static, temporal = self._move_to_device(batch)

                all_coordinates.append(
                    batch["coordinates"].numpy()
                )

                all_geo_coordinates.append(
                    batch["geo_coordinates"].numpy()
                )

                all_sample_ids.append(
                    batch["sample_id"].numpy()
                )

                logits = self.model(
                    static,
                    temporal,
                )

                probabilities = torch.sigmoid(
                    logits.squeeze(1)
                )

                all_probabilities.append(
                    probabilities.cpu().numpy()
                )

        probabilities = np.concatenate(
            all_probabilities
        )

        coordinates = np.concatenate(
            all_coordinates
        )

        geo_coordinates = np.concatenate(
            all_geo_coordinates
        )

        sample_ids = np.concatenate(
            all_sample_ids
        )

        predictions = (
            probabilities >= self.threshold
        ).astype(np.uint8)

        return {

            "probabilities": probabilities,

            "predictions": predictions,

            "coordinates": coordinates,

            "geo_coordinates": geo_coordinates,

            "sample_ids": sample_ids,

            "metadata": self.forecast_loader.dataset.metadata,

            "threshold": self.threshold,

        }
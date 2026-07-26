from pathlib import Path

import torch
import torch.nn as nn


class Checkpoint:
    """
    Guarda y carga checkpoints completos del entrenamiento.
    """

    def __init__(
        self,
        output_dir: str | Path,
        filename: str = "best_model.pt",
    ) -> None:

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path = self.output_dir / filename

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: dict,
        monitor_metric: str,
        config: dict,
    ) -> None:
        """
        Guarda un checkpoint completo.
        """

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict()
                    if scheduler is not None
                    else None,

                "metrics": metrics,

                "monitor_metric": monitor_metric,

                "config": config,
            },
            self.path,
        )

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        device: str | torch.device = "cpu",
    ) -> dict:
        """
        Carga un checkpoint.

        Returns
        -------
        dict
            Diccionario con toda la información almacenada.
        """

        checkpoint = torch.load(
            self.path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        return checkpoint

    def exists(self) -> bool:
        """
        Indica si existe un checkpoint.
        """

        return self.path.exists()
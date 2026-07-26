from pathlib import Path

import torch
import torch.nn as nn


class Checkpoint:
    """
    Guarda y carga checkpoints del entrenamiento.
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
        epoch: int,
        metrics: dict,
    ) -> None:
        """
        Guarda un checkpoint.
        """

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            self.path,
        )

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> dict:
        """
        Carga un checkpoint.

        Returns
        -------
        dict
            Información almacenada.
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

        return checkpoint

    def exists( self, ) -> bool:
        return self.path.exists()
import torch
import torch.nn as nn

from .blocks.static_branch import StaticBranch
from .blocks.temporal_branch import TemporalBranch
from .blocks.fusion_head import FusionHead


class MDFNet(nn.Module):
    """
    Mini Deep Forest Network (MDFNet)

    Entradas
    --------
    static:
        (B, 2, H, W)

    temporal:
        (B, 8, T, H, W)

    Salida
    -------
    logits:
        (B, 1)
    """

    def __init__(self) -> None:

        super().__init__()

        # Branches
        self.static_branch = StaticBranch()
        self.temporal_branch = TemporalBranch()

        # Classification Head
        self.head = FusionHead()

    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        static
            (B,2,H,W)

        temporal
            (B,8,T,H,W)

        Returns
        -------
        Tensor
            (B,1)
        """

        
        # Static Features
        static_features = self.static_branch(static)

        # Temporal Features
        temporal_features = self.temporal_branch(temporal)

        # Feature Fusion
        features = torch.cat(
            (
                static_features,
                temporal_features,
            ),
            dim=1,
        )

        # Classification
        logits = self.head(features)

        return logits
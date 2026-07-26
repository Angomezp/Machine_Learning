import torch
import torch.nn as nn

from .SPP import SpatialPyramidPooling


class FusionHead(nn.Module):
    """
    Cabeza de clasificación.

    Fusiona las características espaciales provenientes de las
    ramas estática y temporal.

    Entrada
    --------
    (B, 32, H, W)

    Salida
    -------
    (B, 1)
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dim: int = 16,
        dropout: float = 0.20,
    ) -> None:

        super().__init__()

        # Spatial Fusion
        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(hidden_dim),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(hidden_dim),

            nn.ReLU(inplace=True),
        )

        # Global Pooling
        self.pool = SpatialPyramidPooling( levels=(1, 2, 4) )

        # Classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                336,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout( p=dropout, ),

            nn.Linear( hidden_dim, 1, ),

        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (B,32,H,W)

        Returns
        -------
        Tensor
            Shape (B,1)
        """

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x
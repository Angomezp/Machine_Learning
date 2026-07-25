import torch
import torch.nn as nn


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
        self.spatial_fusion = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d( 32 ),

            nn.ReLU( inplace=True ),

        )

        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d( output_size=1 )

        # Classifier
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32,
                hidden_dim,
            ),

            nn.ReLU( inplace=True ),

            nn.Dropout( p=dropout ),

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

        x = self.spatial_fusion(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x
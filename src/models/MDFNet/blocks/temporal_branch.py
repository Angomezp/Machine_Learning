import torch
import torch.nn as nn


class TemporalBranch(nn.Module):
    """
    Rama temporal del modelo.

    Procesa la secuencia temporal mediante convoluciones 3D.

    Entrada
    --------
    (B, 8, T, H, W)

    Salida
    -------
    (B, 16, H, W)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 16,
    ) -> None:

        super().__init__()

        # Feature Extraction
        self.feature_extractor = nn.Sequential(

            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),

            nn.BatchNorm3d( out_channels ),

            nn.ReLU( inplace=True ),

        )

        # Temporal Aggregation
        self.temporal_reduction = nn.Sequential(

            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),

            nn.BatchNorm3d( out_channels ),

            nn.ReLU( inplace=True ),

        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (B, 8, 3, H, W)

        Returns
        -------
        Tensor
            Shape (B, 16, H, W)
        """

        # Space-Time Features
        x = self.feature_extractor(x)

        # Learned Temporal Aggregation
        x = self.temporal_reduction(x)

        # Remove Temporal Dimension
        x = x.squeeze(2)

        return x
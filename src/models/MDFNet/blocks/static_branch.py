import torch
import torch.nn as nn


class StaticBranch(nn.Module):
    """
    Rama estática del modelo.

    Procesa las variables estáticas:
        - Tree Cover
        - Gain

    Entrada
    -------
    (B, 2, H, W)

    Salida
    ------
    (B, 16, H, W)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 16,
        kernel_size: int = 3,
    ) -> None:

        super().__init__()

        padding = kernel_size // 2

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),

            nn.BatchNorm2d( out_channels ),

            
            nn.ReLU( inplace=True ),

             nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            (B, 2, H, W)

        Returns
        -------
        Tensor
            (B, 16, H, W)
        """

        return self.features(x)
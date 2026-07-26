import torch
import torch.nn as nn


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP).

    Aplica AdaptiveMaxPooling a múltiples escalas y concatena
    las características obtenidas.

    Niveles por defecto:
        1×1
        2×2
        4×4

    Entrada
    -------
    (B, C, H, W)

    Salida
    ------
    (B, C * (1² + 2² + 4²))
    """

    def __init__(
        self,
        levels=(1, 2, 4),
    ):
        super().__init__()

        self.levels = levels

        self.pools = nn.ModuleList(
            [
                nn.AdaptiveMaxPool2d(level)
                for level in levels
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = []

        # Pooling en cada nivel

        for pool in self.pools:

            pooled = pool(x)

            pooled = pooled.flatten(start_dim=1)

            features.append(pooled)

        # Concatenar todos los niveles
        return torch.cat(features, dim=1)

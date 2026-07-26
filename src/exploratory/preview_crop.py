"""
preview_crop.py

Realiza un recorte de prueba sobre un archivo GeoTIFF de Global Forest Change
y muestra el resultado para verificar que el área de estudio es correcta.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform

from ..config import (
    CENTER_LAT,
    CENTER_LON,
    WIDTH_KM,
    HEIGHT_KM,
    RAW_DIR,
    PREVIEW_INPUT_FILE,
    PREVIEW_OUTPUT_FILE,
)

# ============================================================
# Utilidades
# ============================================================

def bounding_box(
    center_lat: float,
    center_lon: float,
    width_km: float,
    height_km: float,
):
    """
    Calcula un bounding box alrededor de un punto central.
    """

    delta_lat = height_km / 111.32

    delta_lon = width_km / (
        111.32 * math.cos(math.radians(center_lat))
    )

    left = center_lon - delta_lon / 2
    right = center_lon + delta_lon / 2

    bottom = center_lat - delta_lat / 2
    top = center_lat + delta_lat / 2

    return left, bottom, right, top


def normalize(image):

    image = image.astype(np.float32)

    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)

    image = np.clip(image, p2, p98)

    image = (image - p2) / (p98 - p2)

    return image


# ============================================================
# Bounding Box
# ============================================================

left, bottom, right, top = bounding_box(
    CENTER_LAT,
    CENTER_LON,
    WIDTH_KM,
    HEIGHT_KM,
)

print("=" * 60)
print("Área de estudio")
print("=" * 60)
print(f"Centro : ({CENTER_LAT:.6f}, {CENTER_LON:.6f})")
print(f"Ancho  : {WIDTH_KM} km")
print(f"Alto   : {HEIGHT_KM} km")
print()

print("Bounding Box")
print(f"Left   : {left}")
print(f"Right  : {right}")
print(f"Bottom : {bottom}")
print(f"Top    : {top}")

# ============================================================
# Recorte
# ============================================================

with rasterio.open(PREVIEW_INPUT_FILE) as src:

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        src.transform,
    )

    crop = src.read(window=window)

    profile = src.profile.copy()

    profile.update(
        width=crop.shape[2],
        height=crop.shape[1],
        transform=window_transform(
            window,
            src.transform,
        ),
    )

    with rasterio.open(
        PREVIEW_OUTPUT_FILE,
        "w",
        **profile,
    ) as dst:

        dst.write(crop)

print()
print(f"Recorte guardado en:\n{PREVIEW_OUTPUT_FILE}")

# ============================================================
# Visualización
# ============================================================

# Bandas oficiales del producto LAST
#
# Banda 1 -> Red
# Banda 2 -> NIR
# Banda 3 -> SWIR1
# Banda 4 -> SWIR2

red = normalize(crop[0])

nir = normalize(crop[1])

swir = normalize(crop[2])

rgb = np.dstack((red, nir, swir))



plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("Preview del recorte")
plt.axis("off")
plt.tight_layout()
plt.show()

print()
print(f"Tamaño del recorte: {crop.shape[2]} x {crop.shape[1]} píxeles")
print(f"Número de bandas : {crop.shape[0]}")
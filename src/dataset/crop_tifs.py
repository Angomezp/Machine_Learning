"""
crop_tif.py

Recorta todos los GeoTIFF utilizando una única ventana definida
sobre un raster de referencia.

Todos los archivos resultantes quedan perfectamente alineados.
"""

import math
from pathlib import Path

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from ..config import (
    CENTER_LAT,
    CENTER_LON,
    WIDTH_KM,
    HEIGHT_KM,
    RAW_DIR,
    CROPPED_DIR,
    PADDING_METERS,
    REFERENCE_FILE,
)



# ==========================================================
# Utilidades
# ==========================================================

def make_odd(value: int) -> int:
    """Devuelve el entero impar más cercano por exceso."""
    if value % 2 == 0:
        return value + 1
    return value


# ==========================================================
# Raster de referencia
# ==========================================================

with rasterio.open(REFERENCE_FILE) as ref:

    print("=" * 70)
    print("RASTER DE REFERENCIA")
    print("=" * 70)

    print(f"Archivo : {REFERENCE_FILE.name}")
    print(f"CRS     : {ref.crs}")
    print(f"Bounds  : {ref.bounds}")
    print(f"Resolución (°): {ref.res}")

    row, col = ref.index(CENTER_LON, CENTER_LAT)

    print()
    print(f"Centro geográfico : ({CENTER_LAT}, {CENTER_LON})")
    print(f"Centro en píxeles : ({row}, {col})")

    # ---------------------------------------------
    # Resolución en metros
    # ---------------------------------------------

    meters_per_degree_lat = 111320.0

    meters_per_degree_lon = (
        111320.0 *
        math.cos(math.radians(CENTER_LAT))
    )

    pixel_width_m = abs(ref.res[0]) * meters_per_degree_lon
    pixel_height_m = abs(ref.res[1]) * meters_per_degree_lat

    padding_x_pixels = math.ceil( PADDING_METERS / pixel_width_m )

    padding_y_pixels = math.ceil( PADDING_METERS / pixel_height_m )



    print()
    print(f"Resolución aproximada:")
    print(f"   X = {pixel_width_m:.2f} m/pixel")
    print(f"   Y = {pixel_height_m:.2f} m/pixel")

    width_pixels = math.ceil(  WIDTH_KM * 1000 / pixel_width_m )
    height_pixels = math.ceil( HEIGHT_KM * 1000 / pixel_height_m )

    # Agregar padding a ambos lados
    width_pixels += 2 * padding_x_pixels
    height_pixels += 2 * padding_y_pixels

    width_pixels = make_odd(width_pixels)
    height_pixels = make_odd(height_pixels)

    print()
    print("Área solicitada")
    print(f"   {WIDTH_KM} x {HEIGHT_KM} km")

    print()
    print(f"Padding: {PADDING_METERS} m")
    print(f"Padding X: {padding_x_pixels} píxeles")
    print(f"Padding Y: {padding_y_pixels} píxeles")

    print()
    print("Área equivalente")
    print(f"   {width_pixels} x {height_pixels} píxeles")



    real_width = width_pixels * pixel_width_m / 1000
    real_height = height_pixels * pixel_height_m / 1000

    print()
    print("Área real")
    print(f"   {real_width:.3f} x {real_height:.3f} km")

    window = Window(
        col_off=col - width_pixels // 2,
        row_off=row - height_pixels // 2,
        width=width_pixels,
        height=height_pixels,
    )

    reference_transform = window_transform(
        window,
        ref.transform,
    )


# ==========================================================
# Recorte
# ==========================================================

print()
print("=" * 70)
print("RECORTANDO TIFF")
print("=" * 70)

for tif in sorted(RAW_DIR.glob("*.tif")):

    print(f"\n{tif.name}")

    with rasterio.open(tif) as src:

        crop = src.read(window=window)

        if crop.shape[1] != height_pixels or crop.shape[2] != width_pixels:
            raise RuntimeError(
                f"{tif.name}: tamaño inesperado {crop.shape}"
            )

        profile = src.profile.copy()

        profile.update(
            width=width_pixels,
            height=height_pixels,
            transform=reference_transform,
        )

        output = CROPPED_DIR / tif.name

        with rasterio.open(output, "w", **profile) as dst:
            dst.write(crop)

        print(
            f"✓ {crop.shape[2]} x {crop.shape[1]} "
            f"({crop.shape[0]} bandas)"
        )

print()
print("=" * 70)
print("FINALIZADO")
print("=" * 70)
print(f"Todos los TIFF fueron recortados con una ventana de {width_pixels} × {height_pixels} píxeles.")
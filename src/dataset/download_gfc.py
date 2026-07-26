import rasterio

from ..config import *
from .utils.utils import build_url, download_file, get_tile

tile = get_tile(CENTER_LAT, CENTER_LON)

print("=" * 60)
print(f"Tile: {tile}")
print("=" * 60)

# ----------------------------------------------------------
# Descarga productos estáticos y ultimo año de loss
# ----------------------------------------------------------

for product in STATIC_PRODUCTS:

    filename = RAW_DIR / f"{PRODUCT_NAMES[product]}.tif"

    if filename.exists():
        print(f"{filename.name} ya existe.")
        continue

    # Se utiliza la ultima versión disponible únicamente
    year = GFC_YEARS[-1]

    url = build_url(
        product=product,
        tile=tile,
        year=year,
    )
    print(f"URL: {url}")
    download_file(url, filename)
    with rasterio.open(filename) as src:
        print(src.bounds)

# ----------------------------------------------------------
# Descarga productos temporales sin ultimo año de loss
# ----------------------------------------------------------

for year in GFC_YEARS:

    print(f"\n===== GFC {year} =====")

    for product in TEMPORAL_PRODUCTS:

        filename = RAW_DIR / (
            f"{PRODUCT_NAMES[product]}_{year}.tif"
        )

        if filename.exists():
            print(f"{filename.name} ya existe.")
            continue

        url = build_url(
            product=product,
            tile=tile,
            year=year,
        )
        print(f"URL: {url}")
        download_file(url, filename)
        with rasterio.open(filename) as src:
            print(src.bounds)

print("\nDescarga finalizada.")
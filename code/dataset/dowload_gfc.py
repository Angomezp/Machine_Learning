from config import *
from utils import *

tile = get_tile(CENTER_LAT, CENTER_LON)

print("=" * 60)
print(f"Tile: {tile}")
print("=" * 60)

# ----------------------------------------------------------
# Descarga productos estáticos
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

    download_file(url, filename)

# ----------------------------------------------------------
# Descarga productos temporales
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

        download_file(url, filename)

print("\nDescarga finalizada.")
import math
from pathlib import Path

import requests

from config import GFC_VERSIONS

BASE_URL = (
    "https://storage.googleapis.com/"
    "earthenginepartners-hansen/"
)


def get_version(year: int) -> str:

    if year not in GFC_VERSIONS:
        raise ValueError(
            f"No existe versión para GFC {year}"
        )

    return GFC_VERSIONS[year]


def get_tile(lat: float, lon: float) -> str:
    """
    Devuelve el nombre del tile utilizado por
    Global Forest Change (Hansen).
    """

    if lat >= 0:
        lat_tile = (math.floor(lat / 10) + 1) * 10
        ns = "N"
    else:
        lat_tile = math.ceil(lat / 10) * 10
        ns = "S"

    if lon >= 0:
        lon_tile = math.floor(lon / 10) * 10
        ew = "E"
    else:
        lon_tile = abs(math.floor(lon / 10) * 10)
        ew = "W"

    return f"{lat_tile:02d}{ns}_{lon_tile:03d}{ew}"


def build_url(
    product: str,
    tile: str,
    year: int
) -> str:

    version = get_version(year)

    return (
        f"{BASE_URL}"
        f"GFC-{year}-{version}/"
        f"Hansen_GFC-{year}-{version}_{product}_{tile}.tif"
    )


def download_file(
    url: str,
    destination: Path
):

    print(f"Descargando {destination.name}")

    response = requests.get(
        url,
        stream=True
    )

    response.raise_for_status()

    with open(destination, "wb") as f:

        for chunk in response.iter_content(1024 * 1024):

            if chunk:
                f.write(chunk)

    print("✓ Finalizado")


def file_exists(path: Path) -> bool:
    """
    Verifica si el archivo ya existe.
    """
    return path.exists()
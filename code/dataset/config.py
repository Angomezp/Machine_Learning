from pathlib import Path
# ============================================================
# Rutas del proyecto
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

for directory in (DATA_DIR, RAW_DIR, CROPPED_DIR, PREPROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = DATA_DIR

# ============================================================
# Área de estudio
# ============================================================

CENTER_LAT = 1.62
CENTER_LON = -74.35

WIDTH_KM = 10
HEIGHT_KM = 10

# ============================================================
# Global Forest Change
# ============================================================



GFC_YEARS = [
    2023,
    2024,
    2025,
]

GFC_VERSIONS = {
    2023: "v1.11",
    2024: "v1.12",
    2025: "v1.13",
}

STATIC_PRODUCTS = [
    "treecover2000",
    "gain",
    "datamask",
]

TEMPORAL_PRODUCTS = [
    "lossyear",
    "last",
]

PRODUCT_NAMES = {
    "treecover2000": "tc2000",
    "gain": "gain",
    "datamask": "mask",
    "lossyear": "loss",
    "last": "last",
}
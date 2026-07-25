from pathlib import Path
# ============================================================
# Rutas del proyecto
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped"

DATASET_DIR = DATA_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "gfc_dataset_2025.h5"

SPLIT_DIR = DATASET_DIR
SPLIT_PATH = DATASET_DIR / "split_target2025_undersample10_seed42.npz"  # cambiar nombre respectivo

MODELS_OUTPUT_DIR = BASE_DIR.parent / "models" / "MDFNet" / "checkpoints"

for directory in (DATA_DIR, RAW_DIR, CROPPED_DIR, DATASET_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = DATA_DIR

# ============================================================
# Área de estudio
# ============================================================

CENTER_LAT = 3.30  # grados
CENTER_LON = -71.55 # grados

WIDTH_KM = 10
HEIGHT_KM = 10

# Margen adicional alrededor del área
PADDING_METERS = 500

# ============================================================
# Global Forest Change
# ============================================================

GFC_YEARS = [
    2022,
    2023,
    2024,
    2025,
]

GFC_VERSIONS = {
    2022: "v1.10",
    2023: "v1.11",
    2024: "v1.12",
    2025: "v1.13",
}

STATIC_PRODUCTS = [
    "treecover2000",
    "gain",
    "datamask",
    "lossyear", # Si cambia con el tiempo, pero solo necesitamos el último año de pérdida. (Con ese construimos todos los recent_loss)
]

TEMPORAL_PRODUCTS = [
    "last",
]

PRODUCT_NAMES = {
    "treecover2000": "tc2000",
    "gain": "gain",
    "datamask": "mask",
    "lossyear": "loss",
    "last": "last",
}

# Hyperparámetros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 1e-3
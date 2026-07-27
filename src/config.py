from pathlib import Path
# ============================================================
# Rutas del proyecto
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent    # ROOT - Machine_Learning

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped"
DATASET_DIR = DATA_DIR / "dataset"
PREVIEW_DIR = DATA_DIR / "preview"

FORECAST_DATASET_NAME = "gfc_forecast_dataset_2026.h5"  # cambiar nombre respectivo
DATASET_NAME = "gfc_dataset_2025.h5"  # cambiar nombre respectivo
DATASET_PATH = DATASET_DIR / DATASET_NAME 

SPLIT_DIR = DATASET_DIR
SPLIT_PATH = DATASET_DIR / "split_target2025_undersample10_seed42.npz"  # cambiar nombre respectivo

MODELS_OUTPUT_DIR = BASE_DIR / "models" / "MDFNet" 

OUTPUT_DIR = DATA_DIR

PREVIEW_INPUT_FILE = RAW_DIR / "last_2025.tif"  # cambiar nombre respectivo
PREVIEW_OUTPUT_FILE = PREVIEW_DIR / "preview_last_2025.tif"

for directory in (DATA_DIR, RAW_DIR, CROPPED_DIR, DATASET_DIR, SPLIT_DIR, MODELS_OUTPUT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# Área de estudio
# ============================================================

CENTER_LAT = 3.30  # grados
CENTER_LON = -71.55 # grados

WIDTH_KM = 15
HEIGHT_KM = 15

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

# Archivo de referencia para recortar todos los GeoTIFF
REFERENCE_FILE = RAW_DIR / "last_2025.tif"  # cambiar nombre respectivo


# Hyperparámetros de entrenamiento
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_DELTA = 1e-4
EARLY_STOPPING_MONITOR = "roc_auc"   # (loss, f1, roc_auc, precision, recall en validacion )
EARLY_STOPPING_MODE = "max"  # (min, max) dependiendo de la métrica a monitorear

BEST_MODEL_METRIC = "roc_auc"  # (loss, f1, roc_auc, precision, recall  en validacion )
BEST_MODEL_MODE = "max"  # (min, max) dependiendo de la métrica a monitorear


EXPERIMENTS = [

    {
        "name": "baseline",
        "undersampling": None,
    },

    {
        "name": "undersampling_1_15",
        "undersampling": 15,
    },

    {
        "name": "undersampling_1_10",
        "undersampling": 10,
    },

]
from pathlib import Path

import h5py
import numpy as np

from ..config import DATASET_PATH,SPLIT_PATH



###############################################################################
# ABRIR ARCHIVOS
###############################################################################

print("=" * 80)
print("ABRIENDO SPLIT")
print("=" * 80)

split = np.load(SPLIT_PATH)

with h5py.File(DATASET_PATH, "r") as h5:

    labels = h5["label"][:]

###############################################################################
# METADATA
###############################################################################

print("\nMETADATA\n")

for key in split.files:

    if key.endswith("_indices"):
        continue

    print(f"{key:25}: {split[key]}")

###############################################################################
# ÍNDICES
###############################################################################

original_train = split["original_train_indices"]

train = split["train_indices"]

validation = split["validation_indices"]

test = split["test_indices"]

###############################################################################
# FUNCIÓN AUXILIAR
###############################################################################

def print_split(name: str, indices: np.ndarray):

    split_labels = labels[indices]

    positives = int(split_labels.sum())

    negatives = len(indices) - positives

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    print(f"Samples        : {len(indices):,}")

    print(f"Positivos      : {positives:,}")

    print(f"Negativos      : {negatives:,}")

    print(
        f"Positive ratio : "
        f"{100 * positives / len(indices):.4f}%"
    )

    if positives > 0:

        print(
            f"Negative ratio : "
            f"1:{negatives / positives:.2f}"
        )

###############################################################################
# SPLITS
###############################################################################

print("\n" + "=" * 80)
print("ORIGINAL SPLIT")
print("=" * 80)

print_split(
    "Original Train",
    original_train
)

print_split(
    "Validation",
    validation
)

print_split(
    "Test",
    test
)

###############################################################################
# TRAIN FINAL
###############################################################################

print("\n" + "=" * 80)
print("FINAL TRAIN")
print("=" * 80)

print_split(
    "Train (After Undersampling)",
    train
)

###############################################################################
# UNDERSAMPLING
###############################################################################

print("\n" + "=" * 80)
print("UNDERSAMPLING")
print("=" * 80)

removed = len(original_train) - len(train)

original_labels = labels[original_train]

final_labels = labels[train]

original_pos = int(original_labels.sum())
original_neg = len(original_train) - original_pos

final_pos = int(final_labels.sum())
final_neg = len(train) - final_pos

removed_percentage = (
    100 * removed / len(original_train)
)

kept_percentage = (
    100 * len(train) / len(original_train)
)

print(f"Original train samples : {len(original_train):,}")
print(f"Final train samples    : {len(train):,}")

print()

print(
    f"Removed samples        : "
    f"{removed:,} ({removed_percentage:.2f}%)"
)

print(
    f"Remaining samples      : "
    f"{len(train):,} ({kept_percentage:.2f}%)"
)

print()

print(f"Original negatives     : {original_neg:,}")
print(f"Final negatives        : {final_neg:,}")

print()

print(f"Original positives     : {original_pos:,}")
print(f"Final positives        : {final_pos:,}")

###############################################################################
# CONSISTENCIA
###############################################################################

print("\n" + "=" * 80)
print("CONSISTENCY CHECK")
print("=" * 80)

print(
    "Train subset of original :",
    set(train.tolist()).issubset(
        set(original_train.tolist())
    )
)

print(
    "Train ∩ Validation :",
    len(
        set(train.tolist()).intersection(
            validation.tolist()
        )
    )
)

print(
    "Train ∩ Test :",
    len(
        set(train.tolist()).intersection(
            test.tolist()
        )
    )
)

print(
    "Validation ∩ Test :",
    len(
        set(validation.tolist()).intersection(
            test.tolist()
        )
    )
)

print()

print(
    "Validation unchanged :",
    len(validation) == int(
        split["validation_ratio"] * len(labels)
    )
)

print(
    "Test unchanged :",
    len(test)
    == len(labels)
    - len(original_train)
    - int(split["validation_ratio"] * len(labels))
)

###############################################################################
# PRIMEROS ÍNDICES
###############################################################################

print("\n" + "=" * 80)
print("FIRST INDICES")
print("=" * 80)

print("Original Train :", original_train[:10])

print("Train          :", train[:10])

print("Validation     :", validation[:10])

print("Test           :", test[:10])

###############################################################################
# RESUMEN
###############################################################################

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Dataset samples : {len(labels):,}")

print()

print(f"Original Train  : {len(original_train):,}")
print(f"Final Train     : {len(train):,}")

print(f"Validation      : {len(validation):,}")
print(f"Test            : {len(test):,}")

print()

print(
    f"Training subset : "
    f"{100 * len(train) / len(original_train):.2f}% "
    f"del train original"
)

print(
    f"Removed negatives : "
    f"{len(original_train) - len(train):,}"
)
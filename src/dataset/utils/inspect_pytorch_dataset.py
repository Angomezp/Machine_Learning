from pathlib import Path

from torch.utils.data import DataLoader

from pytorch_dataset import PyTorchDataset

from config import DATASET_PATH, SPLIT_PATH


def inspect_dataset(name: str):

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    dataset = PyTorchDataset(
        dataset_path=DATASET_PATH,
        split_path=SPLIT_PATH,
        split=name,
    )

    ####################################################################
    # Tamaño
    ####################################################################

    print(f"\nNúmero de muestras : {len(dataset):,}")

    ####################################################################
    # Primera muestra
    ####################################################################

    sample = dataset[0]

    print("\nPrimera muestra")

    print(
        f"Static      : {sample['static'].shape}"
    )

    print(
        f"Temporal    : {sample['temporal'].shape}"
    )

    print(
        f"Label       : {sample['label']}"
    )

    print(
        f"Coordinates : {sample['coordinates']}"
    )

    print(
        f"Geo         : {sample['geo_coordinates']}"
    )

    print(
        f"Sample ID   : {sample['sample_id']}"
    )

    ####################################################################
    # DataLoader
    ####################################################################

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nBatch")

    print(
        f"Static      : {batch['static'].shape}"
    )

    print(
        f"Temporal    : {batch['temporal'].shape}"
    )

    print(
        f"Label       : {batch['label'].shape}"
    )

    print(
        f"Coordinates : {batch['coordinates'].shape}"
    )

    print(
        f"Geo         : {batch['geo_coordinates'].shape}"
    )

    print(
        f"Sample ID   : {batch['sample_id'].shape}"
    )

    ####################################################################
    # Validaciones
    ####################################################################

    assert sample["static"].shape == (2, 17, 17)

    assert sample["temporal"].shape == (8, 3, 17, 17)

    assert batch["static"].shape == (32, 2, 17, 17)

    assert batch["temporal"].shape == (32, 8, 3, 17, 17)

    print("\n✓ Dataset correcto")

    dataset.close()


if __name__ == "__main__":

    inspect_dataset("train")

    inspect_dataset("validation")

    inspect_dataset("test")

    print("\n" + "=" * 80)
    print("TODO CORRECTO")
    print("=" * 80)
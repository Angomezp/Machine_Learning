from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset.pytorch_dataset import PyTorchDataset

from models.MDFNet.MDFNet import MDFNet

from training.trainer import Trainer

from dataset.dataset_splitter import DatasetSplitter

from dataset.config import DATASET_PATH, SPLIT_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODELS_OUTPUT_DIR


def load_split( split_path: str | Path, ) -> DatasetSplitter:
    """
    Carga el archivo de split.
    """
    return  np.load(split_path, allow_pickle=True) 

def create_datasets(
    dataset_path: Path,
    split: dict,
):
    """
    Crea los datasets de entrenamiento y validación.
    """

    train_dataset = PyTorchDataset(
        split_path=SPLIT_PATH,
        dataset_path=dataset_path,

        split = "train",

    )

    validation_dataset = PyTorchDataset(
        split_path=SPLIT_PATH,
        dataset_path=dataset_path,

        split = "validation",

    )

    return train_dataset, validation_dataset

def create_dataloaders(
    train_dataset,
    validation_dataset,
):
    """
    Crea los DataLoaders.
    """

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

    )

    return train_loader, validation_loader


def create_model():

    return MDFNet()

def create_trainer(
    model,
    train_loader,
    validation_loader,
):
    """
    Construye el Trainer.
    """

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

    )

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        validation_loader=validation_loader,

        criterion=criterion,

        optimizer=optimizer,

        device=device,

        epochs=EPOCHS,

        output_dir= MODELS_OUTPUT_DIR

    )

    return trainer


def main() -> None:

    print("\n" + "=" * 70)
    print("FORESTNET TRAINING")
    print("=" * 70)

    # SPLIT
    split = load_split( SPLIT_PATH )

    # DATASETS
    train_dataset, validation_dataset = create_datasets(
        DATASET_PATH,
        split,
    )
    # DATALOADERS
    train_loader, validation_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
    )

    # MODEL
    model = create_model()

    # TRAINER
    trainer = create_trainer(
        model,
        train_loader,
        validation_loader,
    )

    # TRAIN
    history = trainer.fit()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    return history


if __name__ == "__main__":
    main()
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .visualization.plot_training_history import plot_training_history

from ..models.MDFNet.MDFNet import MDFNet

from ..config import DATASET_PATH, SPLIT_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODELS_OUTPUT_DIR

from ..dataset.dataset_splitter import DatasetSplitter
from ..dataset.utils.pytorch_dataset import PyTorchDataset

from .utils.trainer import Trainer


def load_split( split_path: str | Path, ) -> DatasetSplitter:
    """
    Carga el archivo de split.
    """
    return  np.load(split_path, allow_pickle=True) 

def create_datasets(
    dataset_path: Path,
    split_path: str | Path,
):
    """
    Crea los datasets de entrenamiento y validación.
    """

    train_dataset = PyTorchDataset(
        split_path=split_path,
        dataset_path=dataset_path,

        split = "train",

    )

    validation_dataset = PyTorchDataset(
        split_path=split_path,
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
    output_dir: str | Path,
):
    """
    Construye el Trainer.
    """

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

    )

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    configuration = {

        "architecture": "MDFNet",

        "batch_size": BATCH_SIZE,

        "epochs": EPOCHS,

        "learning_rate": LEARNING_RATE,

        "optimizer": "Adam",

        "scheduler": "ReduceLROnPlateau",

        "criterion": "BCEWithLogitsLoss",

    }

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        validation_loader=validation_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        epochs=EPOCHS,

        output_dir= output_dir,

        config = configuration,

    )

    return trainer


def main(
    output_dir: str | Path,
    split_path: str | Path
 ) -> None:

    print("\n" + "=" * 70)
    print("FORESTNET TRAINING")
    print("=" * 70)

    # SPLIT
    split = load_split( split_path )

    # DATASETS
    train_dataset, validation_dataset = create_datasets(
        DATASET_PATH,
        split_path,
    )
    # DATALOADERS
    train_loader, validation_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
    )

    # MODEL
    model = create_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(model)
    print()
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Total parameters     : {total_params:,}")
    print("=" * 70)

    # TRAINER
    trainer = create_trainer(
        model,
        train_loader,
        validation_loader,
        output_dir,
    )

    # TRAIN
    history = trainer.fit()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    return {

        "history": history,

        "output_dir": Path(output_dir),

    }


if __name__ == "__main__":
    main(output_dir=MODELS_OUTPUT_DIR,
         split_path=SPLIT_PATH,)
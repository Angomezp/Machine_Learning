from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .early_stopping import EarlyStopping
from .checkpoint import Checkpoint

class Trainer:
    """
    Clase encargada del entrenamiento del modelo.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        output_dir: str | Path = "checkpoints",
    ) -> None:

        # MODEL
        self.model = model.to(device)

        # DATA
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        # TRAINING

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = device

        self.epochs = epochs

        # CHECKPOINTS

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.checkpoint = Checkpoint(
            self.output_dir
        )

        # HISTORY
        self.history = {

            "train_loss": [],

            "validation_loss": [],

            "precision": [],

            "recall": [],

            "f1": [],

            "roc_auc": [],
        }

        ####################################################################
        # EARLY STOPPING
        ####################################################################

        self.early_stopping = EarlyStopping(
            patience=10,
            min_delta=0.0,
        )
        # BEST MODEL

        self.best_validation_loss = float("inf")

        self.best_epoch = -1

    def _move_to_device(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mueve un batch al dispositivo.

        Parameters
        ----------
        batch
            Batch proveniente del DataLoader.

        Returns
        -------
        tuple
            static, temporal, label
        """

        static = batch["static"].to(self.device)

        temporal = batch["temporal"].to(self.device)

        label = batch["label"].float().to(self.device)

        return static, temporal, label

    def _train_epoch(
        self,
        epoch: int,
    ) -> float:
        """
        Ejecuta una época de entrenamiento.

        Parameters
        ----------
        epoch
            Número de época.

        Returns
        -------
        float
            Pérdida promedio de entrenamiento.
        """

        # TRAIN MODE
        self.model.train()

        running_loss = 0.0

        # LOOP

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
        )

        for batch in progress:

            # DATA
            static, temporal, label = self._move_to_device(batch)

            #RESET GRADIENTS
            self.optimizer.zero_grad()

            # FORWARD PASS THROUGH THE MODEL
            logits = self.model(
                static,
                temporal,
            )

            # LOSS COMPUTATION
            loss = self.criterion(
                logits.squeeze(1),
                label,
            )

            # BACKPROPAGATION
            loss.backward()

            # UPDATE PARAMETERS
            self.optimizer.step()

            # HISTORY
            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        
        # CALCULATE EPOCH LOSS
        epoch_loss = running_loss / len(self.train_loader)

        return epoch_loss

    def _validate(
        self,
    ) -> dict:
        """
        Evalúa el modelo sobre el conjunto de validación.

        Returns
        -------
        dict
            Diccionario con loss y métricas.
        """

        # SET EVALUATION MODE
        self.model.eval()

        running_loss = 0.0

        all_labels = []

        all_probabilities = []

        # VALIDATION LOOP
        with torch.no_grad():

            progress = tqdm(
                self.validation_loader,
                desc="Validation",
                leave=False,
            )

            for batch in progress:

                # DATA
                static, temporal, label = self._move_to_device(batch)

                # LOGITS            
                logits = self.model(
                    static,
                    temporal,
                )

                # LOSS COMPUTATION
                loss = self.criterion(
                    logits.squeeze(1),
                    label,
                )

                running_loss += loss.item()

                # PROBABILITIES
                probabilities = torch.sigmoid( logits.squeeze(1) )

                # STORE LABELS AND PROBABILITIES
                all_probabilities.extend( probabilities.cpu().numpy() )

                all_labels.extend( label.cpu().numpy() )

                progress.set_postfix( loss=f"{loss.item():.4f}" )

        # CALCULATE VALIDATION LOSS
        validation_loss = running_loss / len( self.validation_loader )

        # COMPUTE METRICS
        metrics = compute_metrics(
            labels=np.asarray(all_labels),
            probabilities=np.asarray(all_probabilities),
        )

        metrics["loss"] = validation_loss
        metrics["labels"] = np.asarray(all_labels)
        metrics["probabilities"] = np.asarray(all_probabilities)

        return metrics

    def fit(
        self,
    ) -> dict:
        """
        Entrena el modelo durante todas las épocas.
        """

        print("\n" + "=" * 70)
        print("START TRAINING")
        print("=" * 70)

        print(f"Device              : {self.device}")
        print(f"Epochs              : {self.epochs}")
        print(f"Training batches    : {len(self.train_loader)}")
        print(f"Validation batches  : {len(self.validation_loader)}")

        print("=" * 70)

        # TRAINING LOOP
        for epoch in range(1, self.epochs + 1):

            # TRAINING
            train_loss = self._train_epoch( epoch )

            # VALIDATION
            validation_metrics = self._validate()

            # SAVE BEST MODEL
            if validation_metrics["loss"] < self.best_validation_loss:

                self.best_validation_loss = validation_metrics["loss"]

                self.best_epoch = epoch

                self.checkpoint.save(

                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=validation_metrics,

                )

                print("✓ Best model saved")

            # EARLY STOPPING
            if self.early_stopping( validation_metrics["loss"] ):

                print()
                print( f"Early stopping at epoch {epoch}" )

                break

            # HISTORY
            self.history["train_loss"].append( train_loss )

            self.history["validation_loss"].append( validation_metrics["loss"] )

            self.history["precision"].append( validation_metrics["precision"] ) 

            self.history["recall"].append( validation_metrics["recall"] )

            self.history["f1"].append( validation_metrics["f1"] )

            self.history["roc_auc"].append( validation_metrics["roc_auc"] )

            #LOG
            print(
                f"Epoch {epoch:03d}/{self.epochs} | "

                f"Train Loss: {train_loss:.4f} | "

                f"Validation Loss: {validation_metrics['loss']:.4f} | "

                f"Precision: {validation_metrics['precision']:.4f} | "

                f"Recall: {validation_metrics['recall']:.4f} | "

                f"F1: {validation_metrics['f1']:.4f} | "

                f"ROC-AUC: {validation_metrics['roc_auc']:.4f}"
            )

        return self.history
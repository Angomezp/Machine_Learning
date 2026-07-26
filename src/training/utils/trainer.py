from pathlib import Path

import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..visualization.plot_training_history import plot_training_history

from ...config import (
    BEST_MODEL_METRIC,
    BEST_MODEL_MODE, 
    EARLY_STOPPING_DELTA, 
    EARLY_STOPPING_MODE, 
    EARLY_STOPPING_MONITOR, 
    EARLY_STOPPING_PATIENCE
)

from .threshold_finder import ThresholdFinder
from .metrics import compute_metrics
from .early_stopping import EarlyStopping
from .checkpoint import Checkpoint

class Trainer:
    """
    Clase encargada del entrenamiento del modelo.
    """

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        output_dir,
        config,
    ):

        # MODEL
        self.model = model.to(device)

        # DATA
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        # TRAINING

        self.criterion = criterion

        self.optimizer = optimizer
        self.scheduler = scheduler

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
        self.threshold_finder = ThresholdFinder(
            output_dir=self.output_dir,
        )

        self.config = config

        # HISTORY
        self.history = {

            "epoch": [],

            "train_loss": [],

            "validation_loss": [],

            "precision": [],

            "recall": [],

            "f1": [],

            "roc_auc": [],

            "learning_rate": [],

            "best_metric": BEST_MODEL_METRIC,

            "best_mode": BEST_MODEL_MODE,

        }

        # EARLY STOPPING
        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_DELTA,
            mode=EARLY_STOPPING_MODE,
        )
        # BEST MODEL

        self.best_metric = float("inf") if BEST_MODEL_MODE == "min" else float("-inf")

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

    def _save_validation_predictions(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> Path:
        """
        Guarda las etiquetas reales y probabilidades obtenidas
        sobre el conjunto de validación.
        """

        output_path = self.output_dir / "validation_predictions.npz"

        np.savez_compressed(
            output_path,
            labels=labels,
            probabilities=probabilities,
        )

        return output_path

    def _save_best_model(
        self,
        epoch: int,
        metrics: dict,
        mode: str,
        metric_name: str,
    ) -> None:
        """
        Guarda el mejor modelo basado en la métrica especificada.

        Parameters
        ----------
        epoch
            Número de época.
        metrics
            Diccionario con métricas de validación.
        mode
            "min" si se busca minimizar la métrica, "max" si se busca maximizarla.
        metric_name
            Nombre de la métrica a monitorear.
        """

        if mode == "min":
            is_best = metrics[metric_name] < self.best_metric
        else:
            is_best = metrics[metric_name] > self.best_metric

        if is_best:
            self.best_metric = metrics[metric_name]
            self.best_epoch = epoch

            self.checkpoint.save(

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                epoch=epoch,

                metrics=metrics,

                monitor_metric=metric_name,

                config=self.config,

            )

            print(f"✓ Best model saved at epoch {epoch} with {metric_name}: {metrics[metric_name]:.4f}")


    def _save_training_history(self) -> None:
        """
        Guarda el historial completo del entrenamiento.
        """

        history_path = self.output_dir / "training_history.json"

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )
    
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

            # UPDATE SCHEDULER
            self.scheduler.step(
                validation_metrics["loss"]
            )
            # UPDATE HISTORY
            self.history["epoch"].append(epoch)
            
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )

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


            checkpoint_metrics = validation_metrics.copy()

            checkpoint_metrics.pop("labels")

            checkpoint_metrics.pop("probabilities")
            # SAVE BEST MODEL
            self._save_best_model(
                epoch=epoch,
                metrics=checkpoint_metrics,
                mode=BEST_MODEL_MODE,
                metric_name=BEST_MODEL_METRIC,
            )
            
            # EARLY STOPPING
            if self.early_stopping( validation_metrics[EARLY_STOPPING_MONITOR] ):

                print()
                print( f"Early stopping at epoch {epoch}" )

                break


        threshold_results = self.threshold_finder.search(
            labels=validation_metrics["labels"],
            probabilities=validation_metrics["probabilities"],
        )

        print( f"Best Threshold : {threshold_results['best_threshold']:.2f}"  )

        print( f"Best F1        : {threshold_results['best_f1']:.4f}" 
        )

        # SAVE IN HISTORY

        self.history["best_threshold"] = threshold_results["best_threshold"]
        self.history["best_threshold_f1"] = threshold_results["best_f1"]
        self.history["best_threshold_precision"] = threshold_results["precision"]
        self.history["best_threshold_recall"] = threshold_results["recall"]

        self.history["requested_epochs"] = self.epochs

        self.history["epochs_completed"] = len(self.history["epoch"])
        self._save_training_history()

        validation_predictions_path = self._save_validation_predictions(
            labels=validation_metrics["labels"],
            probabilities=validation_metrics["probabilities"],
        )

        plot_training_history(
                history_path=self.output_dir / "training_history.json",
                threshold_results_path=self.output_dir / "threshold_results.json",
                validation_predictions_path=validation_predictions_path,
                output_dir=self.output_dir / "figures",
            )
        return self.history
from pathlib import Path
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ...config import (
    DATASET_PATH,
    BATCH_SIZE,
)

from ...dataset.utils.pytorch_dataset import PyTorchDataset

from ...models.MDFNet.MDFNet import MDFNet

from ...training.utils.checkpoint import Checkpoint

from .tester import Tester
from .results_summary import update_results_summary
from ..visualization.plot_test_results import plot_test_results

class TestingPipeline:
    """
    Pipeline completo de evaluación.

    Se encarga de:

        - Crear Dataset
        - Crear DataLoader
        - Construir el modelo
        - Cargar el checkpoint
        - Leer el threshold óptimo
        - Ejecutar el Tester
    """

    def __init__(
        self,
        training_dir,
        testing_dir,
        split_path ,
    ) -> None:

        self.training_dir = Path(training_dir)

        self.testing_dir = Path(testing_dir)

        self.split_path = Path(split_path)

        self.training_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.model = None

        self.test_dataset = None

        self.test_loader = None

        self.threshold = None

        self.checkpoint_info = None

    
    # DATASET
    def create_dataset(self) -> None:
        """
        Construye el Test Dataset.
        """

        self.test_dataset = PyTorchDataset(

            split_path=self.split_path,

            dataset_path=DATASET_PATH,

            split="test",

        )

    
    # DATALOADER
    def create_dataloader(self) -> None:
        """
        Construye el Test DataLoader.
        """

        self.test_loader = DataLoader(

            self.test_dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=4,

            pin_memory=True,

        )

    
    # MODEL
    def create_model(self) -> None:
        """
        Construye MDFNet.
        """

        self.model = MDFNet().to(self.device)

    
    # CHECKPOINT
    def load_checkpoint(self) -> None:
        """
        Carga el mejor modelo.
        """

        checkpoint = Checkpoint(
            self.training_dir
        )

        self.checkpoint_info = checkpoint.load(

            model=self.model,

            device=self.device,

        )

    
    # THRESHOLD
    def load_best_threshold(self) -> None:
        """
        Lee el threshold óptimo encontrado
        durante validación.
        """

        threshold_path = (

            self.training_dir

            / "threshold_results.json"

        )

        with open(
            threshold_path,
            "r",
        ) as f:

            threshold_results = json.load(f)

        self.threshold = threshold_results[
            "best_threshold"
        ]

    
    # TESTER
    def create_tester(self) -> Tester:
        """
        Construye el Tester.
        """

        return Tester(

            model=self.model,

            test_loader=self.test_loader,

            criterion=self.criterion,

            device=self.device,

            output_dir=self.testing_dir,

            threshold=self.threshold,

        )
        
    # EVALUATION
    def evaluate(self) -> dict:
        """
        Ejecuta la evaluación completa.

        Returns
        -------
        dict
            Métricas calculadas sobre el conjunto de test.
        """

        tester = self.create_tester()

        metrics = tester.evaluate()

        tester.save_results(metrics)

        tester.save_predictions(metrics)

        return metrics

    
    # RUN PIPELINE
    

    def run(self) -> dict:
        """
        Ejecuta el pipeline completo de testing.

        Returns
        -------
        dict
            Métricas finales obtenidas en test.
        """

        print("\n" + "=" * 70)
        print("FORESTNET TESTING")
        print("=" * 70)

        #
        # DATASET
        #
        print("Creating Test Dataset...")
        self.create_dataset()

        #
        # DATALOADER
        #
        print("Creating Test DataLoader...")
        self.create_dataloader()

        #
        # MODEL
        #
        print("Building MDFNet...")
        self.create_model()

        total_params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        print("=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)

        print(self.model)

        print()

        print(
            f"Trainable parameters : "
            f"{trainable_params:,}"
        )

        print(
            f"Total parameters     : "
            f"{total_params:,}"
        )

        print("=" * 70)

        #
        # CHECKPOINT
        #
        print("Loading Best Checkpoint...")
        self.load_checkpoint()

        print(
            f"Checkpoint epoch : "
            f"{self.checkpoint_info['epoch']}"
        )

        print(
            f"Optimized metric : "
            f"{self.checkpoint_info['monitor_metric']}"
        )

        #
        # THRESHOLD
        #
        print("Loading Best Threshold...")
        self.load_best_threshold()

        print(
            f"Best threshold : "
            f"{self.threshold:.4f}"
        )

        #
        # TEST
        #
        print("Evaluating...")

        metrics = self.evaluate()

        print()

        print("=" * 70)
        print("TEST RESULTS")
        print("=" * 70)

        print(
            f"Loss      : {metrics['loss']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['recall']:.4f}"
        )

        print(
            f"F1        : {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : {metrics['roc_auc']:.4f}"
        )

        print("=" * 70)

        plot_test_results(

            labels=metrics["labels"],

            probabilities=metrics["probabilities"],

            threshold=self.threshold,

            output_dir=self.testing_dir,

        )

        update_results_summary(

            training_dir=self.training_dir,

            testing_dir=self.testing_dir,

            metrics=metrics,

            checkpoint_info=self.checkpoint_info,

            threshold=self.threshold,

        )

        print()
        print(f"Testing results saved in: {self.testing_dir}")

        return metrics
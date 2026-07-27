from pathlib import Path
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ...dataset.utils.forecast_dataset import ForecastDataset
from ...config import (
    CROPPED_DIR,
    DATASET_DIR,
    BATCH_SIZE,
    FORECAST_DATASET_NAME,
)


from ...dataset.utils.pytorch_dataset import PyTorchDataset

from ...models.MDFNet.MDFNet import MDFNet

from ...training.utils.checkpoint import Checkpoint


from .forecaster import Forecaster
from .forecast_writter import ForecastWriter
from .visualization import ForecastVisualizer



class ForecastPipeline:
    """
    Pipeline completo de forecast 2026.

    Responsabilidades:

        - Crear dataset de forecast
        - Crear DataLoader
        - Construir modelo
        - Cargar checkpoint entrenado
        - Cargar threshold óptimo
        - Ejecutar inferencia
        - Guardar resultados
        - Generar visualizaciones
    """


    def __init__(
        self,
        training_dir,
        forecast_dir,
        split_path,
    ) -> None:


        self.training_dir = Path( training_dir )

        self.forecast_dir = Path( forecast_dir )


        self.split_path = Path( split_path )

        self.forecast_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None

        self.forecast_dataset = None

        self.forecast_loader = None

        self.threshold = None

        self.checkpoint_info = None

    # DATASET
    def create_dataset(self):

        self.forecast_dataset = ForecastDataset(
            dataset_path=DATASET_DIR / FORECAST_DATASET_NAME,
        )

    # DATALOADER
    def create_dataloader(self):

        """
        Construye DataLoader.
        """

        self.forecast_loader = DataLoader(

            self.forecast_dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=0,

            pin_memory=True,

        )
    # MODEL
    def create_model(self):

        """
        Construye MDFNet.
        """
        self.model = MDFNet().to( self.device )

    # CHECKPOINT
    def load_checkpoint(self):

        """
        Carga el mejor modelo entrenado.
        """
        checkpoint = Checkpoint(
            self.training_dir
        )

        self.checkpoint_info = checkpoint.load( 
            model=self.model,
            device=self.device,
        )

    # THRESHOLD
    def load_threshold(self):

        """
        Carga threshold óptimo encontrado
        durante validación.
        """

        threshold_path = ( self.training_dir / "threshold_results.json" )

        with open(
            threshold_path,
            "r",
        ) as file:

            threshold_results = json.load( file )

        self.threshold = threshold_results[ "best_threshold" ]

    # FORECASTER
    def create_forecaster(self):

        return Forecaster(
            model=self.model,
            forecast_loader=self.forecast_loader,
            device=self.device,
            threshold=self.threshold,
        )

    # RUN
    def run(self):

        print("\n" + "=" * 70)
        print("MDFNet FORECAST 2026")
        print("=" * 70)

        print("Creating forecast dataset...")
        self.create_dataset()

        print("Creating forecast dataloader...")
        self.create_dataloader()

        print("Building MDFNet...")
        self.create_model()

        print("Loading checkpoint...")
        self.load_checkpoint()

        print(
            f"Checkpoint epoch: "
            f"{self.checkpoint_info['epoch']}"
        )

        print("Loading threshold...")
        self.load_threshold()

        print(f"Threshold: {self.threshold:.4f}")

        print("Running forecast...")

        forecaster = self.create_forecaster()
        results = forecaster.forecast()

        metadata = self.forecast_dataset.metadata

        ####################################################
        # ESCRIBIR RESULTADOS
        ####################################################

        writer = ForecastWriter(
            output_dir=self.forecast_dir,
            forecast_dataset=self.forecast_dataset,
        )

        writer.save_predictions(results)

        writer.save_summary(
            results,
            metadata={
                "checkpoint_epoch": self.checkpoint_info["epoch"],
                "architecture": self.checkpoint_info["config"]["architecture"],
                "forecast_year": metadata["forecast_year"],
            },
        )

        ####################################################
        # VISUALIZACIONES
        ####################################################

        visualizer = ForecastVisualizer(
            self.forecast_dir,
            rgb_path = CROPPED_DIR / "last_2025.tif",
        )

        visualizer.generate_all(results)

        print()
        print("=" * 70)
        print("FORECAST COMPLETED")
        print("=" * 70)

        return results
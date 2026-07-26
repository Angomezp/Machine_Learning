import h5py
from sklearn.model_selection import train_test_split
from enum import Enum
import numpy as np
from pathlib import Path

from ..config import DATASET_PATH, SPLIT_DIR

class SamplingStrategy(Enum):

    NONE = "none"
    UNDERSAMPLE = "undersample"


class DatasetSplitter:

    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        train_ratio: float = 0.60,
        validation_ratio: float = 0.20,
        test_ratio: float = 0.20,
        sampling: SamplingStrategy = SamplingStrategy.UNDERSAMPLE,
        negative_ratio: int = 10,
        random_state: int = 42,
    ):

        self.dataset_path = dataset_path
        self.output_dir = output_dir

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        self.sampling = sampling
        self.negative_ratio = negative_ratio

        self.random_state = random_state

        self.labels = None

        self.num_samples = None
        self.num_positive = None
        self.num_negative = None

        self.target_year = None
        self.patch_size = None

        self.original_train_indices = None
        self.train_indices = None
        self.validation_indices = None
        self.test_indices = None


    def build(
        self,
        filename: str | None = None
    ) -> None:

        self._load_labels()

        self._train_validation_test_split()

        self._undersample_train()

        self._save_split(filename=filename)


    def _load_labels(self) -> None:
        """
        Carga las etiquetas y la metadata necesaria del dataset HDF5.

        No carga los tensores de imágenes, únicamente la información
        requerida para realizar el split del dataset.
        """

        print("\n" + "=" * 60)
        print("Cargando labels...")
        print("=" * 60)

        with h5py.File(self.dataset_path, "r") as h5:

            # Labels
            self.labels = h5["label"][:]

            # Número total de muestras
            self.num_samples = len(self.labels)

            # Metadata (opcional pero útil posteriormente)
            self.target_year = int(h5.attrs["target_year"])
            self.patch_size = int(h5.attrs["patch_size"])

            self.num_positive = int(h5.attrs["num_positive"])
            self.num_negative = int(h5.attrs["num_negative"])

        print(f"Target year : {self.target_year}")
        print(f"Patch size  : {self.patch_size}")
        print()
        print(f"Total       : {self.num_samples:,}")
        print(f"Positivos   : {self.num_positive:,}")
        print(f"Negativos   : {self.num_negative:,}")
        print(
            f"Balance     : "
            f"{100 * self.num_positive / self.num_samples:.2f}% positivos"
        )

    def _train_validation_test_split(self) -> None:
        """
        Divide el dataset en entrenamiento, validación y prueba
        utilizando un muestreo estratificado.

        La división se realiza en dos pasos:

            1. Train (60%) y Restante (40%)
            2. Validation (20%) y Test (20%)

        El undersampling se aplicará posteriormente únicamente sobre
        el conjunto de entrenamiento.
        """

        print("\n" + "=" * 60)
        print("Generando split Train / Validation / Test...")
        print("=" * 60)

        indices = np.arange(self.num_samples)

        # TRAIN (60%) - RESTANTE (40%)
        train_idx, remaining_idx, train_labels, remaining_labels = train_test_split(
            indices,
            self.labels,
            train_size=self.train_ratio,
            stratify=self.labels,
            random_state=self.random_state,
            shuffle=True,
        )

        # VALIDATION (20%) - TEST (20%)

        validation_fraction = ( self.validation_ratio / (self.validation_ratio + self.test_ratio) )

        validation_idx, test_idx = train_test_split(
            remaining_idx,
            train_size=validation_fraction,
            stratify=remaining_labels,
            random_state=self.random_state,
            shuffle=True,
        )

        # Guardar índices
        self.train_indices = np.sort(train_idx)

        self.validation_indices = np.sort(validation_idx)

        self.test_indices = np.sort(test_idx)

        # Debug
        print(f"Train      : {len(self.train_indices):,}")
        print(f"Validation : {len(self.validation_indices):,}")
        print(f"Test       : {len(self.test_indices):,}")

        print()

        print(
            f"Total      : "
            f"{len(self.train_indices)+len(self.validation_indices)+len(self.test_indices):,}"
        )

    def _undersample_train(self) -> None:
        """
        Aplica undersampling únicamente al conjunto de entrenamiento.

        Conserva todos los ejemplos positivos y selecciona aleatoriamente
        una cantidad de negativos determinada por:

            negativos = positivos * negative_ratio

        Validation y Test permanecen sin modificaciones.
        """

        # Guardar train original
        self.original_train_indices = self.train_indices.copy()

        if self.sampling != SamplingStrategy.UNDERSAMPLE:

            print("\nNo se aplicará undersampling.")

            return

        print("\n" + "=" * 60)
        print("Aplicando undersampling...")
        print("=" * 60)

        # Separar positivos y negativos

        train_labels = self.labels[self.train_indices]

        positive_indices = self.train_indices[train_labels == 1]

        negative_indices = self.train_indices[train_labels == 0]

        num_positive = len(positive_indices)
        num_negative = len(negative_indices)

        print(f"Positivos originales : {num_positive:,}")
        print(f"Negativos originales : {num_negative:,}")

        # Número de negativos a conservar
        desired_negatives = num_positive * self.negative_ratio

        desired_negatives = min(
            desired_negatives,
            num_negative
        )

        # Muestreo aleatorio
        rng = np.random.default_rng(self.random_state)

        selected_negatives = rng.choice(
            negative_indices,
            size=desired_negatives,
            replace=False
        )

        # Construir nuevo conjunto de entrenamiento
        train_indices = np.concatenate(

            [
                positive_indices,
                selected_negatives
            ]

        )

        rng.shuffle(train_indices)

        self.train_indices = train_indices

        # Estadísticas
        train_labels = self.labels[self.train_indices]

        positives = int(train_labels.sum())
        negatives = len(train_labels) - positives

        print()

        print("Después del undersampling")

        print(f"Samples   : {len(self.train_indices):,}")
        print(f"Positivos : {positives:,}")
        print(f"Negativos : {negatives:,}")

        print(
            f"Ratio : 1:{negatives / positives:.1f}"
        )

        print(
            f"Positive percentage : "
            f"{100 * positives / len(self.train_indices):.2f}%"
        )

    def _split_name(self) -> str:

        sampling = self.sampling.value

        if sampling != SamplingStrategy.NONE.value:
            sampling = f"{sampling}{self.negative_ratio}"

        return (
            f"split_"
            f"target{self.target_year}_"
            f"{sampling}_"
            f"seed{self.random_state}.npz"
        )
    
    def _save_split(
        self,
        filename: str |None = None
    ) -> None:
        """
        Guarda los índices de entrenamiento, validación y prueba.

        El dataset HDF5 original nunca se modifica.
        """

        print("\n" + "=" * 60)
        print("Guardando split...")
        print("=" * 60)

        output_dir = self.output_dir

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if filename is None:
            output_file = output_dir / self._split_name()
        else:
            output_file = output_dir / filename

        np.savez_compressed(

            output_file,
            original_train_indices=self.original_train_indices,
            train_indices=self.train_indices,
            validation_indices=self.validation_indices,
            test_indices=self.test_indices,
            train_ratio=self.train_ratio,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            sampling=self.sampling.value,
            negative_ratio=self.negative_ratio,
            random_state=self.random_state,
            target_year=self.target_year,
            
        )

        print(f"Archivo : {output_file}")

        print()

        print(f"Train      : {len(self.train_indices):,}")

        print(f"Validation : {len(self.validation_indices):,}")

        print(f"Test       : {len(self.test_indices):,}")

        print("\n✓ Split guardado correctamente.")


if __name__ == "__main__":
    DS = DatasetSplitter(
        dataset_path= DATASET_PATH,
        output_dir= SPLIT_DIR,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
        sampling= SamplingStrategy.UNDERSAMPLE,
        negative_ratio=10,
        random_state=42,
    )

    DS.build()

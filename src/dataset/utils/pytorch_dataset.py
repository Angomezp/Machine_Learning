import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PyTorchDataset(Dataset):

    def __init__(
        self,
        dataset_path: Path,
        split_path: Path,
        split: str = "train",
        transform=None,
    ):
        """
        Dataset de PyTorch para el dataset GFC.

        Parameters
        ----------
        dataset_path : Path
            Ruta al dataset HDF5.

        split_path : Path
            Ruta al archivo NPZ con los índices.

        split : str
            Uno de:
                - "train"
                - "validation"
                - "test"

        transform
            Transformaciones opcionales.
        """

        self.dataset_path = Path(dataset_path)

        self.split_path = Path(split_path)

        self.split = split.lower()

        self.transform = transform

        # Objetos cargados posteriormente
        self.h5 = None

        self.indices = None

        # Cargar información
        self._load_split()

    
    def _load_split(self) -> None:
        """
        Carga los índices correspondientes al split solicitado.

        Los índices se leen desde el archivo NPZ generado por
        DatasetSplitter.
        """

    
        print(f"[Worker {os.getpid()}] cargando split...")

        # Verificar split solicitado
        valid_splits = {
            "train": "train_indices",
            "validation": "validation_indices",
            "test": "test_indices",
        }

        if self.split not in valid_splits:

            raise ValueError(
                f"Split '{self.split}' no válido. "
                f"Debe ser uno de: {list(valid_splits.keys())}"
            )

        # Cargar información del NPZ
        with np.load(self.split_path) as split_data:

            self.indices = split_data[valid_splits[self.split]]

            self.target_year = int(
                split_data["target_year"]
            )

            self.sampling = str(
                split_data["sampling"]
            )

            self.random_state = int(
                split_data["random_state"]
            )

    def _open_dataset(self) -> None:
        """
        Abre el dataset HDF5.

        El archivo permanece abierto durante toda la vida del Dataset
        para permitir acceso aleatorio eficiente a las muestras.
        """

        # Ya está abierto

        if self.h5 is not None:
            return

        print(f"[Worker {os.getpid()}] Opening dataset...")

        # Abrir HDF5
        self.h5 = h5py.File(
            self.dataset_path,
            "r"
        )

        # Verificar datasets obligatorios

        required_datasets = [
            "static",
            "temporal",
            "label",
            "coordinates",
            "geo_coordinates",
            "sample_id",
        ]

        for dataset in required_datasets:

            if dataset not in self.h5:

                raise KeyError(
                    f"No se encontró el dataset '{dataset}'."
                )

        # Referencias a datasets
        self.static = self.h5["static"]

        self.temporal = self.h5["temporal"]

        self.label = self.h5["label"]

        self.coordinates = self.h5["coordinates"]

        self.geo_coordinates = self.h5["geo_coordinates"]

        self.sample_id = self.h5["sample_id"]

        # Metadata
        self.patch_size = int(
            self.h5.attrs["patch_size"]
        )

        self.num_samples = int(
            self.h5.attrs["num_samples"]
        )

        self.target_year = int(
            self.h5.attrs["target_year"]
        )


    def __len__(self) -> int:
        """
        Número de muestras del split.
        """

        return len(self.indices)

    def __getitem__(
        self,
        idx: int,
    ):
        """
        Obtiene una muestra del dataset.

        Parameters
        ----------
        idx : int
            Índice dentro del split.

        Returns
        -------
        dict
            Diccionario con la información de la muestra.
        """
        if self.h5 is None:
            self._open_dataset()
        
        # Índice real dentro del HDF5
        sample_idx = int(self.indices[idx])

        # Leer datos
        static = self.static[sample_idx]

        temporal = self.temporal[sample_idx]

        label = self.label[sample_idx]

        coordinates = self.coordinates[sample_idx]

        geo_coordinates = self.geo_coordinates[sample_idx]

        sample_id = self.sample_id[sample_idx]

        
        # Tensorización
        static = torch.from_numpy(
            static
        ).float()

        temporal = torch.from_numpy(
            temporal
        ).float()

        label = torch.tensor(
            label,
            dtype=torch.float32,
        )

        coordinates = torch.from_numpy(
            coordinates
        ).int()

        geo_coordinates = torch.from_numpy(
            geo_coordinates
        ).double()

        sample_id = torch.tensor(
            sample_id,
            dtype=torch.int32,
        )

        # Transformaciones opcionales

        if self.transform is not None:

            static, temporal = self.transform(
                static,
                temporal,
            )

        # Retorno
        return {
            "static": static,
            "temporal": temporal,
            "label": label,
            "coordinates": coordinates,
            "geo_coordinates": geo_coordinates,
            "sample_id": sample_id,
        }
    
    def close(self) -> None:
        """
        Cierra el archivo HDF5.
        """

        if self.h5 is not None:

            self.h5.close()

            self.h5 = None

            self.static = None

            self.temporal = None

            self.label = None

            self.coordinates = None

            self.geo_coordinates = None

            self.sample_id = None

    def __del__(self):
        """
        Destructor para asegurar que el archivo HDF5 se cierre al eliminar
        el objeto.
        """

        self.close()
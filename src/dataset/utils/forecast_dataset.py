from pathlib import Path

import h5py
import rasterio
import torch
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    """
    Dataset para inferencia (forecast).

    No contiene labels.
    """

    def __init__(
        self,
        dataset_path: str | Path,
    ):

        self.dataset_path = Path(dataset_path)

        self.h5 = h5py.File(
            self.dataset_path,
            "r",
        )

        # ------------------------
        # Metadata
        # ------------------------

        self.num_samples = int(
            self.h5.attrs["num_samples"]
        )

        self.forecast_year = int(
            self.h5.attrs["forecast_year"]
        )

        self.height = int(
            self.h5.attrs["height"]
        )

        self.width = int(
            self.h5.attrs["width"]
        )

        self.transform = rasterio.Affine(
            *self.h5.attrs["transform"]
        )

        self.crs = self.h5.attrs["crs"]

        self.bounds = tuple(
            self.h5.attrs["bounds"]
        )

        # ------------------------
        # Datasets
        # ------------------------

        self.static = self.h5["static"]

        self.temporal = self.h5["temporal"]

        self.coordinates = self.h5["coordinates"]

        self.geo_coordinates = self.h5["geo_coordinates"]

        self.sample_id = self.h5["sample_id"]

    def __len__(self):

        return len(self.static)

    def __getitem__(self, index):

        return {

            "static": torch.from_numpy(
                self.static[index]
            ).float(),

            "temporal": torch.from_numpy(
                self.temporal[index]
            ).float(),

            "coordinates": torch.from_numpy(
                self.coordinates[index]
            ).long(),

            "geo_coordinates": torch.from_numpy(
                self.geo_coordinates[index]
            ).double(),

            "sample_id": int(
                self.sample_id[index]
            ),

        }

    @property
    def metadata(self):

        return {
            "forecast_year": self.forecast_year,
            "height": self.height,
            "width": self.width,
            "transform": self.transform,
            "crs": self.crs,
            "bounds": self.bounds,
        }
    def close(self):

        if hasattr(self, "h5"):
            self.h5.close()


    def __del__(self):

        try:
            self.close()
        except Exception:
            pass
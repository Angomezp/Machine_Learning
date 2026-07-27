from pathlib import Path

import numpy as np
import h5py
import rasterio
from time import perf_counter

from ..config import (
    CROPPED_DIR,
    PRODUCT_NAMES, 
    DATASET_DIR,
    FORECAST_DATASET_NAME,
)

class GFCForecastDatasetBuilder:
    """
    Construye el dataset de predicción a partir de los TIFF
    recortados del Global Forest Change.
    No posee labels.
    """

    def __init__(
        self,
        data_dir: Path,
        patch_size: int = 17,
        target_year: int = 2026,
    ):
        self.data_dir = Path(data_dir)
        self.output_file = DATASET_DIR / FORECAST_DATASET_NAME

        self.patch_size = patch_size
        self.radius = patch_size // 2

        self.forecast_year = target_year
        self.temporal_years = [target_year - 3, target_year - 2, target_year-1]
        self.recent_loss = {}
        self.valid_pixels = []

        # Variables estáticas
        self.treecover = None
        self.gain = None
        self.datamask = None
        self.lossyear = None

        # Variables temporales
        self.last = {}

        # Información espacial
        self.transform = None
        self.crs = None
        self.height = None
        self.width = None

    def build(self):

        self._load_images()

        self._build_recent_loss()

        self._build_valid_pixels()

        self._build_dataset()

    def _load_raster(
        self,
        filename: str
    ) -> np.ndarray:

        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:

            image = src.read()

            if self.transform is None:

                self.transform = src.transform
                self.crs = src.crs
                self.height = src.height
                self.width = src.width
                self.bounds = src.bounds

            return image

    def _load_images(self):

        print("=" * 60)
        print("Cargando imágenes...")
        print("=" * 60)

        self.treecover = self._load_raster(PRODUCT_NAMES["treecover2000"] + ".tif")[0]

        self.gain = self._load_raster(PRODUCT_NAMES["gain"] + ".tif")[0]

        self.datamask = self._load_raster(PRODUCT_NAMES["datamask"] + ".tif")[0]

        # Convertimos la máscara a valores binarios (0 y 1) 0= agua o sin datos, 1 = tierra
        self.datamask = (self.datamask == 1).astype(np.uint8)

        self.lossyear = self._load_raster(PRODUCT_NAMES["lossyear"] + ".tif")[0]

        for year in self.temporal_years:

            self.last[year] = self._load_raster(
                f"last_{year}.tif"
            )

        print("\nResumen")

        print(f"Resolución : {self.width} x {self.height}")
        print(f"CRS        : {self.crs}")

        print("\nVariables estáticas")

        print(" treecover :", self.treecover.shape)
        print(" gain      :", self.gain.shape)
        print(" datamask  :", self.datamask.shape)
        print(" lossyear  :", self.lossyear.shape)

        print("\nVariables temporales")

        for year in self.temporal_years:

            print( f" last_{year} : {self.last[year].shape}" )

    def _build_recent_loss(self):

        """
        Construye las cuatro variables recent_loss para cada año
        temporal (target year - 1, target year - 2, target year - 3).
        """

        print("\n" + "=" * 60)
        print("Construyendo variables recent_loss...")
        print("=" * 60)

        self.recent_loss = {}

        for year in self.temporal_years:

            self.recent_loss[year] = self._build_recent_loss_year(year)

            print(f"\nAño {year}")

            for name, layer in self.recent_loss[year].items():

                print( f"  {name:<15}: {np.count_nonzero(layer):>8} píxeles" )
    def _build_recent_loss_year(
        self,
        year: int
    ) -> dict:

        """
        Construye las cuatro capas recent_loss para un año dado.

        Parameters
        ----------
        year : int

        Returns
        -------
        dict
            Diccionario con las cuatro capas binarias.
        """

        loss = self.lossyear
        year_code = year - 2000

        recent_loss1 = (
            (loss >= year_code - 1) &
            (loss <= year_code)
        )

        recent_loss2 = (
            (loss >= year_code - 4) &
            (loss <= year_code - 2)
        )

        recent_loss3 = (
            (loss >= year_code - 7) &
            (loss <= year_code - 5)
        )

        recent_loss4 = (
            (loss >= 1) &
            (loss <= year_code - 8)
        )

        return {
            "recent_loss1": recent_loss1.astype(np.uint8),
            "recent_loss2": recent_loss2.astype(np.uint8),
            "recent_loss3": recent_loss3.astype(np.uint8),
            "recent_loss4": recent_loss4.astype(np.uint8),
        }

    def _extract_static_patch(
        self,
        row: int,
        col: int
    ) -> np.ndarray:

        """
        Extrae el tensor estático alrededor de un píxel.

        Returns
        -------
        np.ndarray
            Tensor de forma (2, patch_size, patch_size)
        """

        r = self.radius

        treecover = self.treecover[
            row-r:row+r+1,
            col-r:col+r+1
        ]

        gain = self.gain[
            row-r:row+r+1,
            col-r:col+r+1
        ]

        static = np.stack(
            [ treecover, gain ],
            axis=0 
        )

        return static.astype(np.float32)
    
    def _build_valid_pixels(self):

        """
        Construye la lista de píxeles válidos para generar patches.
        """

        print("\n" + "=" * 60)
        print("Buscando píxeles válidos...")
        print("=" * 60)

        self.valid_pixels = []

        r = self.radius

        row_min = r
        row_max = self.height - r

        col_min = r
        col_max = self.width - r

        for row in range(row_min, row_max):

            for col in range(col_min, col_max):

                # Solo tierra
                if self.datamask[row, col] == 0:
                    continue

                # Solo píxeles con cobertura arbórea de mas de 30%
                if self.treecover[row, col] < 30:
                    continue

                self.valid_pixels.append((row, col))

        print(f"Píxeles válidos: {len(self.valid_pixels):,}")

    def _extract_temporal_patch(
        self,
        row: int,
        col: int
    ) -> np.ndarray:
        """
        Extrae el tensor temporal de un píxel.

        Returns
        -------
        np.ndarray
            Tensor de forma (8, 3, patch_size, patch_size)
        """

        r = self.radius

        temporal = []

        for year in self.temporal_years:

            rl = self.recent_loss[year]

            yearly_tensor = np.stack(
                [
                    self.last[year][0,
                        row-r:row+r+1,
                        col-r:col+r+1
                    ],

                    self.last[year][1,
                        row-r:row+r+1,
                        col-r:col+r+1
                    ],

                    self.last[year][2,
                        row-r:row+r+1,
                        col-r:col+r+1
                    ],

                    self.last[year][3,
                        row-r:row+r+1,
                        col-r:col+r+1
                    ],

                    rl["recent_loss1"][
                        row-r:row+r+1,
                        col-r:col+r+1
                    ],

                    rl["recent_loss2"][
                        row-r:row+r+1,
                        col-r:col+r+1
                    ],

                    rl["recent_loss3"][
                        row-r:row+r+1,
                        col-r:col+r+1
                    ],

                    rl["recent_loss4"][
                        row-r:row+r+1,
                        col-r:col+r+1
                    ]

                ],
                axis=0

            )

            temporal.append(yearly_tensor)

        temporal = np.stack(
            temporal,
            axis=1
        )

        return temporal.astype(np.float32)

    

    def _create_sample(
        self,
        row: int,
        col: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construye una muestra del dataset.

        Parameters
        ----------
        row : int
            Fila del píxel central.

        col : int
            Columna del píxel central.

        Returns
        -------
        tuple
            ( static_tensor, temporal_tensor)
        """

        static = self._extract_static_patch( row, col )

        temporal = self._extract_temporal_patch( row, col )

        return ( static, temporal, )

    def _build_dataset(self):

        print("\n" + "=" * 70)
        print("Construyendo dataset de forecast...")
        print("=" * 70)

        total_samples = len(self.valid_pixels)

        print(f"Muestras: {total_samples:,}")

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with h5py.File(self.output_file, "w") as h5:
     
            # DATASETS    
            static_ds = h5.create_dataset(
                "static",
                shape=(total_samples, 2, self.patch_size, self.patch_size),
                dtype=np.float32
            )

            temporal_ds = h5.create_dataset(
                "temporal",
                shape=(
                    total_samples,
                    8,
                    len(self.temporal_years),
                    self.patch_size,
                    self.patch_size
                ),
                dtype=np.float32
            )

            coordinate_ds = h5.create_dataset(
                "coordinates",
                shape=(total_samples, 2),
                dtype=np.int32
            )

            geocoordinate_ds = h5.create_dataset(
                "geo_coordinates",
                shape=(total_samples, 2),
                dtype=np.float64
            )

            sample_id_ds = h5.create_dataset(
                "sample_id",
                shape=(total_samples,),
                dtype=np.int32
            )

            
            # CONSTRUCCIÓN
            

            t0 = perf_counter()

            for i, (row, col) in enumerate(self.valid_pixels):

                static, temporal = self._create_sample( row, col )

                static_ds[i] = static

                temporal_ds[i] = temporal

                coordinate_ds[i] = (row, col)

                lon, lat = rasterio.transform.xy(
                    self.transform,
                    row,
                    col,
                    offset="center"
                )

                geocoordinate_ds[i] = (
                    lon,
                    lat
                )

                sample_id_ds[i] = i

                if (i+1) % 1000 == 0:

                    elapsed = perf_counter() - t0

                    print(
                        f"{i+1:6,}/{total_samples:,}"
                        f"  {elapsed:.1f}s"
                        f"  {(i+1)/elapsed:.1f} muestras/s"
                    )

            
            # METADATA
            h5.attrs["dataset_type"] = "forecast"
            h5.attrs["forecast_year"] = self.forecast_year
            h5.attrs["forecast"] = True
            h5.attrs["has_labels"] = False
            h5.attrs["temporal_years"] = self.temporal_years

            h5.attrs["patch_size"] = self.patch_size
            h5.attrs["radius"] = self.radius

            h5.attrs["height"] = self.height
            h5.attrs["width"] = self.width

            h5.attrs["transform"] = tuple(self.transform)

            h5.attrs["crs"] = self.crs.to_string()

            h5.attrs["bounds"] = (
                self.bounds.left,
                self.bounds.bottom,
                self.bounds.right,
                self.bounds.top,
            )

            h5.attrs["num_samples"] = len(self.valid_pixels)

            h5.attrs["num_static_channels"] = 2
            h5.attrs["num_temporal_channels"] = 8
            h5.attrs["num_timesteps"] = 3


            h5.attrs["static_features"] = [
                "treecover",
                "gain"
            ]

            h5.attrs["temporal_features"] = [
                "red",
                "nir",
                "swir1",
                "swir2",
                "recent_loss1",
                "recent_loss2",
                "recent_loss3",
                "recent_loss4"
            ]

        # RESUMEN
        print("\n" + "=" * 70)
        print("Forecast dataset construido")
        print("=" * 70)

        print(f"Archivo            : {self.output_file}")
        print(f"Muestras           : {total_samples:,}")
        print(f"Año      : {self.forecast_year}")

        print("\nDatasets almacenados en el archivo HDF5 en " + str(self.output_file))

if __name__ == "__main__":
    builder = GFCForecastDatasetBuilder(
        data_dir=CROPPED_DIR,
        target_year=2026,
    )

    builder.build()
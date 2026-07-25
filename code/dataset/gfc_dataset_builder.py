from pathlib import Path

import numpy as np
import h5py
import rasterio
from time import perf_counter

from config import (
    CROPPED_DIR,
    PRODUCT_NAMES, 
    DATASET_DIR
    )

class GFCDatasetBuilder:
    """
    Construye el dataset de entrenamiento a partir de los TIFF
    recortados del Global Forest Change.
    """

    def __init__(
        self,
        data_dir: Path,
        patch_size: int = 17,
        target_year: int = 2025,
        filename: str = "gfc_dataset_{target_year}.h5"
    ):
        self.data_dir = Path(data_dir)
        self.output_file = DATASET_DIR / filename.format(target_year=target_year)

        self.patch_size = patch_size
        self.radius = patch_size // 2

        self.target_year = target_year
        self.temporal_years = [2022, 2023, 2024]
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

        #self._debug()

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
        temporal (2022, 2023 y 2024).
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

    def _create_label(
        self,
        row: int,
        col: int
    ) -> np.uint8:
        """
        Construye la etiqueta para un píxel.

        Parameters
        ----------
        row : int
            Fila del píxel.

        col : int
            Columna del píxel.

        Returns
        -------
        np.uint8
            1 si el píxel fue deforestado exactamente en el año objetivo,
            0 en caso contrario.
        """

        target_code = self.target_year - 2000

        return np.uint8( int(self.lossyear[row, col]) == target_code )
    

    def _create_sample(
        self,
        row: int,
        col: int
    ) -> tuple[np.ndarray, np.ndarray, np.uint8]:
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
            ( static_tensor, temporal_tensor, label )
        """

        static = self._extract_static_patch( row, col )

        temporal = self._extract_temporal_patch( row, col )

        label = self._create_label( row, col )

        return ( static, temporal, label )

    def _build_dataset(self):

        print("\n" + "=" * 70)
        print("Construyendo dataset...")
        print("=" * 70)

        total_samples = len(self.valid_pixels)

        print(f"Muestras: {total_samples:,}")

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with h5py.File(self.output_file, "w") as h5:

            ####################################################################
            # DATASETS
            ####################################################################

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

            label_ds = h5.create_dataset(
                "label",
                shape=(total_samples,),
                dtype=np.uint8
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

            ####################################################################
            # CONSTRUCCIÓN
            ####################################################################

            positives = 0
            t0 = perf_counter()

            for i, (row, col) in enumerate(self.valid_pixels):

                static, temporal, label = self._create_sample( row, col )

                static_ds[i] = static

                temporal_ds[i] = temporal

                label_ds[i] = label

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

                positives += int(label)

                if (i + 1) % 5000 == 0 or (i + 1) == total_samples:

                    print(
                        f"{i + 1:>8,} / {total_samples:,}"
                    )
                if (i+1) % 1000 == 0:

                    elapsed = perf_counter() - t0

                    print(
                        f"{i+1:6,}/{total_samples:,}"
                        f"  {elapsed:.1f}s"
                        f"  {(i+1)/elapsed:.1f} muestras/s"
                    )

            ####################################################################
            # METADATA
            ####################################################################

            negatives = total_samples - positives

            h5.attrs["target_year"] = self.target_year
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

            h5.attrs["num_positive"] = positives
            h5.attrs["num_negative"] = negatives
            h5.attrs["positive_ratio"] = positives / total_samples

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

        ########################################################################
        # RESUMEN
        ########################################################################

        print("\n" + "=" * 70)
        print("Dataset construido")
        print("=" * 70)

        print(f"Archivo            : {self.output_file}")

        print(f"Muestras           : {total_samples:,}")

        print(f"Positivos          : {positives:,}")

        print(f"Negativos          : {negatives:,}")

        print(f"Positive ratio     : {100 * positives / total_samples:.2f}%")

        print(f"Negative ratio     : {100 * negatives / total_samples:.2f}%")

        if positives > 0:

            print(
                f"Desbalance         : "
                f"1:{negatives / positives:.2f}"
            )

        print("\nDatasets almacenados en el archivo HDF5 en " + str(self.output_file))

    def _debug(self):

        print("\n" + "=" * 80)
        print("DEBUG GFCDatasetBuilder")
        print("=" * 80)

        ####################################################################
        # Información general
        ####################################################################

        print("\n[1] INFORMACIÓN GENERAL\n")

        print(f"Target year : {self.target_year}")
        print(f"Patch size  : {self.patch_size}")
        print(f"Radius      : {self.radius}")

        print(f"\nImagen")

        print(f"Height : {self.height}")
        print(f"Width  : {self.width}")

        print(f"\nTemporal years : {self.temporal_years}")

        print(f"\nPixeles válidos : {len(self.valid_pixels):,}")

        ####################################################################
        # Variables estáticas
        ####################################################################

        print("\n" + "=" * 80)
        print("[2] VARIABLES ESTÁTICAS")
        print("=" * 80)

        print(
            f"Treecover  min={self.treecover.min()} "
            f"max={self.treecover.max()} "
            f"mean={self.treecover.mean():.2f}"
        )

        print(
            f"Gain unique : {np.unique(self.gain)}"
        )

        print(
            f"Datamask unique : "
            f"{np.unique(self.datamask, return_counts=True)}"
        )

        ####################################################################
        # Distribución de lossyear
        ####################################################################

        print("\n" + "=" * 80)
        print("[3] LOSSYEAR")
        print("=" * 80)

        values, counts = np.unique(
            self.lossyear,
            return_counts=True
        )


        for value, count in zip(values, counts):
            value = int(value)
            if value == 0:
                year = "No Loss"
            else:
                year = 2000 + value

            print(
                f"{year:>8} : {count:>8,}"
            )

        ####################################################################
        # Balance de clases
        ####################################################################

        print("\n" + "=" * 80)
        print("[4] BALANCE DEL DATASET")
        print("=" * 80)

        target_code = self.target_year - 2000

        positives = np.count_nonzero( self.lossyear == target_code )

        negatives = len(self.valid_pixels) - positives

        total = positives + negatives

        print(f"Target year : {self.target_year}")

        print(f"\nPositivos : {positives:,}")

        print(f"Negativos : {negatives:,}")

        print(f"Total     : {total:,}")

        print(
            f"\nPositive ratio : "
            f"{100*positives/total:.4f}%"
        )

        print(
            f"Negative ratio : "
            f"{100*negatives/total:.4f}%"
        )

        print(
            f"\nDesbalance : 1 positivo cada "
            f"{negatives / positives:.1f} negativos"
        )

        ####################################################################
        # Primera muestra positiva
        ####################################################################

        print("\n" + "=" * 80)
        print("[5] EJEMPLO POSITIVO")
        print("=" * 80)

        positive_found = False

        for row, col in self.valid_pixels:

            if int(self.lossyear[row, col]) == target_code:

                positive_found = True

                static, temporal, label = self._create_sample(
                    row,
                    col
                )

                print(f"Pixel ({row},{col})")

                print(f"Label : {label}")

                print(f"Static : {static.shape}")

                print(f"Temporal : {temporal.shape}")

                break

        if not positive_found:

            print("No existen muestras positivas.")

        ####################################################################
        # Primera muestra negativa
        ####################################################################

        print("\n" + "=" * 80)
        print("[6] EJEMPLO NEGATIVO")
        print("=" * 80)

        for row, col in self.valid_pixels:

            if int(self.lossyear[row, col]) != target_code:

                static, temporal, label = self._create_sample(
                    row,
                    col
                )

                print(f"Pixel ({row},{col})")

                print(f"Label : {label}")

                print(f"Static : {static.shape}")

                print(f"Temporal : {temporal.shape}")

                break

        print("\n" + "=" * 80)
        print("DEBUG FINALIZADO")
        print("=" * 80)


builder = GFCDatasetBuilder(
    data_dir=CROPPED_DIR
)

builder.build()
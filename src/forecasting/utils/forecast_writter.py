from pathlib import Path
import json

import numpy as np
import rasterio


class ForecastWriter:
    """
    Guarda los resultados del forecast.
    """

    def __init__(
        self,
        output_dir: Path,
        forecast_dataset,
    ):

        self.output_dir = Path(output_dir)
        self.dataset = forecast_dataset

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_predictions(
        self,
        results: dict,
    ):

        np.save(
            self.output_dir / "probabilities.npy",
            results["probabilities"],
        )

        np.save(
            self.output_dir / "predictions.npy",
            results["predictions"],
        )

        np.save(
            self.output_dir / "coordinates.npy",
            results["coordinates"],
        )

        print("✓ Predicciones guardadas.")

    def _build_raster(
        self,
        values,
        dtype=np.float32,
    ):

        raster = np.full(
            (
                self.dataset.height,
                self.dataset.width,
            ),
            np.nan,
            dtype=dtype,
        )

        coordinates = self.dataset.coordinates

        for value, (row, col) in zip(values, coordinates):

            raster[row, col] = value

        return raster

    def save_probability_map(
        self,
        results,
    ):

        raster = self._build_raster(
            results["probabilities"],
            dtype=np.float32,
        )

        output = self.output_dir / "forecast_probability_2026.tif"

        with rasterio.open(

            output,

            "w",

            driver="GTiff",

            height=self.dataset.height,

            width=self.dataset.width,

            count=1,

            dtype=np.float32,

            crs=self.dataset.crs,

            transform=self.dataset.transform,

            nodata=np.nan,

        ) as dst:

            dst.write(
                raster,
                1,
            )

        print("✓ Probability map generado.")

    def save_binary_map(
        self,
        results,
    ):

        raster = self._build_raster(
            results["predictions"],
            dtype=np.uint8,
        )

        output = self.output_dir / "forecast_binary_2026.tif"

        with rasterio.open(

            output,

            "w",

            driver="GTiff",

            height=self.dataset.height,

            width=self.dataset.width,

            count=1,

            dtype=np.uint8,

            crs=self.dataset.crs,

            transform=self.dataset.transform,

            nodata=255,

        ) as dst:

            dst.write(
                raster,
                1,
            )

        print("✓ Binary map generado.")

    def save_summary(
        self,
        results,
        metadata=None,
    ):

        summary = {

            "prediction_year": self.dataset.forecast_year,

            "threshold": float(results["threshold"]),

            "pixels": int(len(results["predictions"])),

            "predicted_deforestation":

                int(results["predictions"].sum()),

        }

        if metadata is not None:

            summary.update(metadata)

        with open(

            self.output_dir / "forecast_summary.json",

            "w",

        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )

        print("✓ Summary guardado.")
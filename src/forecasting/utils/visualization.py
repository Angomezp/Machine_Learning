from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

class ForecastVisualizer:
    """
    Genera visualizaciones del forecast 2026.
    """
    def __init__(
        self,
        output_dir: str | Path,
        rgb_path,
    ) -> None:

        self.output_dir = Path(
            output_dir
        )
        self.rgb_path = Path(rgb_path)

        self.figure_dir = (
            self.output_dir
            /
            "figures"
        )

        self.figure_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    def save_probability_raster(
        self,
        probabilities: np.ndarray,
        coordinates: np.ndarray,
        metadata: dict,
    ):
        """
        Reconstruye un raster con la probabilidad de deforestación.
        """

        height = metadata["height"]
        width = metadata["width"]

        raster = np.full(
            (height, width),
            -9999.0,
            dtype=np.float32,
        )

        for probability, (row, col) in zip(probabilities, coordinates):
            raster[row, col] = probability

        output = self.output_dir / "forecast_probability_2026.tif"

        with rasterio.open(
            output,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=metadata["crs"],
            transform=metadata["transform"],
            nodata=-9999.0,
        ) as dst:

            dst.write(raster, 1)

        print(f"✓ Saved {output}")

    def plot_probability_histogram(
        self,
        probabilities: np.ndarray,
    ) -> None:
        """
        Histograma de probabilidades predichas.
        """

        plt.figure(
            figsize=(8,5)
        )


        plt.hist(
            probabilities,
            bins=50,
            edgecolor="black",
        )


        plt.xlabel(
            "Probability of deforestation"
        )

        plt.ylabel(
            "Number of samples"
        )


        plt.title(
            "Forecast 2026 Probability Distribution"
        )


        plt.grid(
            alpha=0.3
        )


        plt.tight_layout()


        path = (
            self.figure_dir
            /
            "forecast_probability_histogram.png"
        )


        plt.savefig(
            path,
            dpi=300,
        )

        plt.close()


        print(
            f"✓ Saved {path}"
        )
    def save_binary_raster(
        self,
        predictions,
        coordinates,
        metadata,
    ):

        height = metadata["height"]
        width = metadata["width"]

        raster = np.full(
            (height, width),
            255,
            dtype=np.uint8,
        )

        for prediction, (row, col) in zip(predictions, coordinates):
            raster[row, col] = prediction

        output = self.output_dir / "forecast_binary_2026.tif"

        with rasterio.open(
            output,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=metadata["crs"],
            transform=metadata["transform"],
            nodata=255,
        ) as dst:

            dst.write(raster, 1)

        print(f"✓ Saved {output}")

    def plot_binary_distribution(
        self,
        predictions: np.ndarray,
    ) -> None:
        """
        Distribución de clases predichas.
        """


        counts = np.bincount(
            predictions,
            minlength=2,
        )

        labels = [
            "No Deforestation",
            "Deforestation",
        ]


        labels = [
            "No Deforestation",
            "Deforestation",
        ]


        plt.figure(
            figsize=(6,5)
        )


        plt.bar(
            labels,
            counts,
        )


        plt.ylabel(
            "Number of samples"
        )


        plt.title(
            "Forecast 2026 Binary Predictions"
        )


        for index, value in enumerate(counts):

            plt.text(
                index,
                value,
                f"{value:,}",
                ha="center",
                va="bottom",
            )


        plt.tight_layout()


        path = (
            self.figure_dir
            /
            "forecast_binary_distribution.png"
        )


        plt.savefig(
            path,
            dpi=300,
        )

        plt.close()


        print(
            f"✓ Saved {path}"
        )


    def plot_probability_distribution(
        self,
        probabilities: np.ndarray,
        threshold: float,
    ) -> None:
        """
        Visualiza probabilidades separadas por threshold.
        """


        positive = probabilities[
            probabilities >= threshold
        ]


        negative = probabilities[
            probabilities < threshold
        ]


        plt.figure(
            figsize=(8,5)
        )


        plt.scatter(
            range(len(negative)),
            negative,
            s=5,
            label="No deforestation",
        )


        plt.scatter(
            range(len(positive)),
            positive,
            s=5,
            label="Deforestation",
        )


        plt.axhline(
            threshold,
            linestyle="--",
            label=f"Threshold {threshold:.3f}",
        )


        plt.xlabel(
            "Sample index"
        )


        plt.ylabel(
            "Probability"
        )


        plt.title(
            "Forecast 2026 Probabilities"
        )


        plt.legend()


        plt.grid(
            alpha=0.3
        )


        plt.tight_layout()


        path = (
            self.figure_dir
            /
            "forecast_probability_distribution.png"
        )


        plt.savefig(
            path,
            dpi=300,
        )


        plt.close()


        print(
            f"✓ Saved {path}"
        )

    def plot_probability_map(
        self,
        probabilities,
        coordinates,
        metadata,
    ):

        image = np.full(
            (metadata["height"], metadata["width"]),
            np.nan,
        )

        for p, (row, col) in zip(probabilities, coordinates):

            image[row, col] = p

        image = np.ma.masked_invalid(image)

        plt.figure(figsize=(10,10))

        plt.imshow(
            image,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        plt.colorbar(
            label="Probability"
        )

        plt.title("Forecast 2026")

        plt.tight_layout()

        plt.savefig(
            self.figure_dir/"forecast_probability_map.png",
            dpi=300,
        )

        plt.close()

    def plot_probability_overlay(
        self,
        rgb: np.ndarray,
        probabilities: np.ndarray,
        coordinates: np.ndarray,
        threshold: float,
        metadata: dict,
    ):
        """
        Superpone las probabilidades de deforestación sobre la imagen RGB.

        Los píxeles con probabilidad menor al threshold no se muestran.
        Los demás se colorean en una escala de rojo cuya intensidad aumenta
        conforme aumenta la probabilidad.
        """
        with rasterio.open(rgb) as src:

            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

        # normalizar para visualizar
        rgb = rgb.astype(np.float32)

        rgb -= rgb.min()

        rgb /= rgb.max()

        height, width = rgb.shape[:2]

        overlay = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        alpha = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        for probability, (row, col) in zip(probabilities, coordinates):

            if probability < threshold:
                continue

            # Reescala [threshold,1] -> [0,1]
            value = (probability - threshold) / (1.0 - threshold)

            overlay[row, col] = value
            alpha[row, col] = value

        fig, ax = plt.subplots(figsize=(12, 12))

        # Imagen del bosque
        ax.imshow(rgb)

        # Probabilidades
        im = ax.imshow(
            overlay,
            cmap="Reds",
            alpha=alpha,
            vmin=0,
            vmax=1,
        )

        cbar = plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        cbar.set_label("Probability of deforestation")

        plt.title(
            f"Forecast {metadata['forecast_year']}"
        )

        ax.axis("off")

        plt.tight_layout()

        output = (
            self.figure_dir
            / "forecast_probability_overlay.png"
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"✓ Saved {output}")

    def generate_all(
        self,
        results: dict,
    ) -> None:

        probabilities = results["probabilities"]
        predictions = results["predictions"]
        threshold = results["threshold"]

        coordinates = results["coordinates"]
        metadata = results["metadata"]

        self.plot_probability_histogram(
            probabilities
        )

        self.plot_binary_distribution(
            predictions
        )

        self.plot_probability_distribution(
            probabilities,
            threshold,
        )

        self.save_probability_raster(
            probabilities,
            coordinates,
            metadata,
        )

        self.save_binary_raster(
            predictions,
            coordinates,
            metadata,
        )

        self.plot_probability_map(
            probabilities,
            coordinates,
            metadata,
        )

        self.plot_probability_overlay(
            self.rgb_path,
            probabilities,
            coordinates,
            threshold,
            metadata
        )
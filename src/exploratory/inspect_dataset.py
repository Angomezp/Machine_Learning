import h5py
import numpy as np
from ..config import DATASET_PATH, GFC_YEARS

TARGET_YEAR = GFC_YEARS[-1]


def inspect_dataset(dataset_path: str):

    print("=" * 80)
    print("ABRIENDO DATASET")
    print("=" * 80)

    with h5py.File(dataset_path, "r") as h5:

        ####################################################################
        # Metadata
        ####################################################################

        print("\nMETADATA\n")

        for key, value in h5.attrs.items():
            print(f"{key:25}: {value}")

        ####################################################################
        # Datasets
        ####################################################################

        print("\n" + "=" * 80)
        print("DATASETS")
        print("=" * 80)

        for name in h5.keys():

            ds = h5[name]

            print(f"\n{name}")
            print(f"  Shape : {ds.shape}")
            print(f"  Dtype : {ds.dtype}")

        ####################################################################
        # Balance
        ####################################################################

        print("\n" + "=" * 80)
        print("BALANCE")
        print("=" * 80)

        labels = h5["label"][:]

        positives = int(labels.sum())
        negatives = len(labels) - positives

        print(f"Total      : {len(labels):,}")
        print(f"Positivos  : {positives:,}")
        print(f"Negativos  : {negatives:,}")
        print(f"Ratio      : {100*positives/len(labels):.5f}%")

        ####################################################################
        # Primer sample
        ####################################################################

        print("\n" + "=" * 80)
        print("PRIMER SAMPLE")
        print("=" * 80)

        static = h5["static"][0]
        temporal = h5["temporal"][0]
        label = h5["label"][0]
        coordinate = h5["coordinates"][0]
        geo = h5["geo_coordinates"][0]

        print(f"Static shape     : {static.shape}")
        print(f"Temporal shape   : {temporal.shape}")
        print(f"Label            : {label}")
        print(f"Pixel            : {coordinate}")
        print(f"Geo              : {geo}")

        ####################################################################
        # Variables estáticas
        ####################################################################

        print("\nVariables estáticas")

        print(
            f"Treecover min={static[0].min():.1f} "
            f"max={static[0].max():.1f}"
        )

        print(
            f"Gain unique={np.unique(static[1])}"
        )

        ####################################################################
        # Variables temporales
        ####################################################################

        print("\nVariables temporales")

        years = h5.attrs["temporal_years"]

        for t, year in enumerate(years):

            print(f"\nAño {year}")

            red = temporal[0, t]
            nir = temporal[1, t]
            swir1 = temporal[2, t]
            swir2 = temporal[3, t]

            rl1 = temporal[4, t]
            rl2 = temporal[5, t]
            rl3 = temporal[6, t]
            rl4 = temporal[7, t]

            print(
                f"Red      min={red.min():6.1f} "
                f"max={red.max():6.1f} "
                f"mean={red.mean():6.2f}"
            )

            print(
                f"NIR      min={nir.min():6.1f} "
                f"max={nir.max():6.1f} "
                f"mean={nir.mean():6.2f}"
            )

            print(
                f"SWIR1    min={swir1.min():6.1f} "
                f"max={swir1.max():6.1f} "
                f"mean={swir1.mean():6.2f}"
            )

            print(
                f"SWIR2    min={swir2.min():6.1f} "
                f"max={swir2.max():6.1f} "
                f"mean={swir2.mean():6.2f}"
            )

            print()

            print(
                f"RecentLoss1 "
                f"sum={int(rl1.sum())} "
                f"unique={np.unique(rl1)}"
            )

            print(
                f"RecentLoss2 "
                f"sum={int(rl2.sum())} "
                f"unique={np.unique(rl2)}"
            )

            print(
                f"RecentLoss3 "
                f"sum={int(rl3.sum())} "
                f"unique={np.unique(rl3)}"
            )

            print(
                f"RecentLoss4 "
                f"sum={int(rl4.sum())} "
                f"unique={np.unique(rl4)}"
            )

if __name__ == "__main__":

    inspect_dataset(DATASET_PATH)
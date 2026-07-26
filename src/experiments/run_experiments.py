from pathlib import Path

from ..config import (
    MODELS_OUTPUT_DIR,
    DATASET_PATH,
    EXPERIMENTS,
)

from ..training.train import main as train

from ..testing.utils.testing_pipeline import TestingPipeline

from ..dataset.dataset_splitter import (
    DatasetSplitter,
    SamplingStrategy,
)

def create_directories(experiment_name: str):

    experiment_dir = Path(MODELS_OUTPUT_DIR) / experiment_name

    training_dir = experiment_dir / "training"

    testing_dir = experiment_dir / "testing"

    split_path = experiment_dir / "split.npz"

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    testing_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return experiment_dir, training_dir, testing_dir, split_path


def create_split(
    experiment_dir: Path,
    undersampling,
):

    if undersampling is None:

        sampling = SamplingStrategy.NONE

        negative_ratio = 10

    else:

        sampling = SamplingStrategy.UNDERSAMPLE

        negative_ratio = undersampling

    splitter = DatasetSplitter(

        dataset_path=DATASET_PATH,

        output_dir=experiment_dir,

        train_ratio=0.60,

        validation_ratio=0.20,

        test_ratio=0.20,

        sampling=sampling,

        negative_ratio=negative_ratio,

        random_state=42,

    )

    splitter.build(
        filename="split.npz",
    )


def run_experiment(experiment):

    print("\n" + "=" * 90)
    print(f"RUNNING {experiment['name']}")
    print("=" * 90)

    (
        experiment_dir,
        training_dir,
        testing_dir,
        split_path,
    ) = create_directories(
        experiment["name"]
    )

    
    # SPLIT
    print("\nCreating dataset split...")

    create_split(

        experiment_dir=experiment_dir,

        undersampling=experiment["undersampling"],

    )

    # TRAIN
    print("\nStarting training...")

    train(

        output_dir=training_dir,

        split_path=split_path,

    )

    # TEST
    print("\nStarting testing...")

    pipeline = TestingPipeline(

        training_dir=training_dir,

        testing_dir=testing_dir,

        split_path=split_path,

    )

    pipeline.run()

    print("\nExperiment finished.")
    print("=" * 90)


def main():

    print("\n")
    print("=" * 90)
    print("FORESTNET EXPERIMENTS")
    print("=" * 90)

    for experiment in EXPERIMENTS:

        run_experiment(experiment)

    print("\n")
    print("=" * 90)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 90)


if __name__ == "__main__":

    main()
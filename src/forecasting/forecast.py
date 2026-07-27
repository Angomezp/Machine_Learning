from pathlib import Path

from .utils.forecast_pipeline import ForecastPipeline

from ..config import MODELS_OUTPUT_DIR

def create_forecast_paths(
    experiment_name: str,
):
    """
    Crea las rutas necesarias para el forecast.
    """

    experiment_dir = Path(MODELS_OUTPUT_DIR) / experiment_name

    training_dir = experiment_dir / "training" 

    forecast_dir = experiment_dir / "forecast"

    split_path = experiment_dir / "split.npz"

    return (
        training_dir,
        forecast_dir,
        split_path,
    )

def run_forecast(
    experiment_name: str,
):
    """
    Ejecuta forecast para un experimento.
    """

    print("\n" + "=" * 90)

    print( f"FORECAST EXPERIMENT: {experiment_name}" )

    print("=" * 90)
    (
        training_dir,
        forecast_dir,
        split_path,

    ) = create_forecast_paths(
        experiment_name
    )

    pipeline = ForecastPipeline(

        training_dir=training_dir,

        forecast_dir=forecast_dir,

        split_path=split_path,

    )

    results = pipeline.run()

    print()
    print( "Forecast finished successfully" ) 
    return results



def main():

    experiment_name = "baseline" 
    run_forecast( experiment_name )


if __name__ == "__main__":
    main()
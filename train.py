import logging
import time
from src.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    import argparse
    import gin
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        help="Path to the config file",
        default="configs/classifier.gin",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
        default="DEBUG",
    )
    args = parser.parse_args()

    # Log the start time
    time_start = time.time()

    # Load the gin configuration
    if not pathlib.Path(args.cfg).exists():
        raise FileNotFoundError(f"Config file {args.cfg} not found.")
    gin.parse_config_file(args.cfg)

    # Create a directory for the outputs (model_name)
    models_dir = gin.query_parameter("%MODELS_DIR")
    model_name = gin.query_parameter("%NAME")
    out_dir = f"{models_dir}/{model_name}"

    # Create the directory for all results if it doesn't exist
    if not pathlib.Path(models_dir).exists():
        pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)

    # If the output directory already exists, add a timestamp to the name
    if pathlib.Path(out_dir).exists():
        out_dir = out_dir + f'_{time.strftime("%Y%m%d-%H%M%S")}'

    # Create the output directory for the model
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Configure logger
    logging.basicConfig(
        level=args.log_level,
        handlers=[
            logging.FileHandler(f"{out_dir}/console.log"),
            logging.StreamHandler(),
        ],
    )

    # Initialize the training pipeline
    pipeline = TrainingPipeline()

    # Split the data
    pipeline.prepare_data()

    # Train the model
    pipeline.train()

    # Evaluate the model
    pipeline.evaluate()

    # Log time
    time_elapsed = time.time() - time_start
    logging.info(
        f"Training completed in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"Training completed in {round(time_elapsed / 60, 2)} minutes."
    )

    # Dump operative config
    gin_path = f"{out_dir}/operative_config.gin"
    with open(gin_path, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Config saved to {gin_path}")

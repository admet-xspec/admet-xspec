"""
Main script to run the training pipeline.

It is configured via a gin configuration file (see configs/train.gin for an example).
The training pipeline includes:
- Preparing and splitting the data (if train and test paths are not provided explicitly)
- Training the model (using set hyperparameters or optimizing them via cross-validation)
- Evaluating the model on the test set
- Saving the trained model and evaluation metrics to the output directory
"""

import logging
import time
import tempfile
from src.training_pipeline import TrainingPipeline
import argparse
import gin
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        help="Path to the config file",
        default="configs/train.gin",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
        default="DEBUG",
    )
    args = parser.parse_args()

    # Configure logger
    temp_log_file = tempfile.NamedTemporaryFile(delete=False)
    logging.basicConfig(
        level=args.log_level,
        handlers=[
            logging.FileHandler(temp_log_file.name),
            logging.StreamHandler(),
        ],
    )

    time_start = time.time()

    # Load the gin configuration
    if not pathlib.Path(args.cfg).exists():
        raise FileNotFoundError(f"Config file {args.cfg} not found.")
    gin.parse_config_file(args.cfg)

    # Initialize the training pipeline
    pipeline = TrainingPipeline()

    # Create the output directory for the model
    pipeline.out_dir.mkdir(parents=True, exist_ok=True)

    # Split the data if train and test paths are not provided explicitly (default behavior)
    if not (pipeline.train_path and pipeline.test_path):
        pipeline.prepare_data()
    else:
        logging.info(
            f"Using explicit train and test datasets: {pipeline.train_path}, {pipeline.test_path}"
        )

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
    gin_path = pipeline.out_dir / "operative_config.gin"
    with open(gin_path, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Config saved to {gin_path}")

    # Move the temporary log file to the output directory
    temp_log_file.close()
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    for handler in handlers:
        handler.close()
        root_logger.removeHandler(handler)
    pathlib.Path(temp_log_file.name).rename(pipeline.out_dir / "training.log")

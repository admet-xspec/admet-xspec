"""
Script to run inference using a pre-trained model and save predictions and metrics.

It is configured via a gin configuration file (see configs/predict.gin for an example).
The inference pipeline includes:
- Loading the pre-trained model
- Making predictions on the provided dataset
- Saving the predictions to an output CSV file
- Optionally computing and saving evaluation metrics if true labels are available
"""

import gin
import logging
from src.inference_pipeline import InferencePipeline
import tempfile
import argparse
import pathlib
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        help="Path to the config file",
        default="configs/predict.gin",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
        default="DEBUG",
    )
    args = parser.parse_args()

    time_start = time.time()

    # Load the gin configuration
    if not pathlib.Path(args.cfg).is_file():
        raise FileNotFoundError(f"Config file {args.cfg} not found.")
    gin.parse_config_file(args.cfg)

    # Configure logger
    temp_log_file = tempfile.NamedTemporaryFile(delete=False)
    logging.basicConfig(
        level=args.log_level,
        handlers=[
            logging.FileHandler(temp_log_file.name),
            logging.StreamHandler(),
        ],
    )

    # Initialize the inference pipeline
    pipeline = InferencePipeline()

    # Run the prediction
    results = pipeline.predict()

    # Compute and save metrics if possible
    if pipeline.can_compute_metrics():
        logging.info("Labels detected in the data file, computing metrics")
        metrics = pipeline.evaluate()

    # Log time
    time_elapsed = time.time() - time_start
    logging.info(
        f"Prediction completed in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"Prediction completed in {round(time_elapsed / 60, 2)} minutes."
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

"""
Main script to run the training pipeline.

It is configured via a gin configuration file (see configs/RFF_AChE_ECFP_human.gin for an example).
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
from datetime import datetime
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
        format="%(message)s",
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

    # Run the training pipeline
    pipeline.run()

    # Log time
    time_elapsed = time.time() - time_start
    logging.info(
        f"TrainingPipeline finished in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"TrainingPipeline finished in {round(time_elapsed / 60, 2)} minutes."
    )

    config_str = gin.operative_config_str()
    timestamp = datetime.now().strftime("%d_%H_%M_%S")
    logging.info(f"Dumping config and logs with timestamp {timestamp}.")

    # Dump operative config
    pipeline.dump_logs(config_str, f'training_config_{timestamp}.gin')

    temp_log_file.close()
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    for handler in handlers:
        handler.close()
        root_logger.removeHandler(handler)
    with open(temp_log_file.name, "r") as f:
        log_contents = f.read()

    # Dump logs
    pipeline.dump_logs(log_contents, f'training_log_{timestamp}.txt')


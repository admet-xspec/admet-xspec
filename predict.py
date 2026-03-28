"""Run inference from a gin configuration."""

import gin
import logging
from src.inference_pipeline import InferencePipeline
import tempfile
import argparse
import pathlib
import time

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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
        format="%(message)s",
        handlers=[
            logging.FileHandler(temp_log_file.name),
            logging.StreamHandler(),
        ],
    )

    # Route pipeline log dumping to this runtime log file.
    gin.bind_parameter("InferencePipeline.logfile", temp_log_file.name)

    # Initialize and run the inference pipeline
    pipeline = InferencePipeline()
    results = pipeline.run()

    # Log time
    time_elapsed = time.time() - time_start
    logging.info(
        f"Prediction completed in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"Prediction completed in {round(time_elapsed / 60, 2)} minutes."
    )

    logging.info(f"Predictions file: {results}")

    # Refresh dumped logs so runtime timing above is included in the persisted log file.
    pipeline.data_interface.dump_logs(pipeline.out_dir)

    temp_log_file.close()
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    for handler in handlers:
        handler.close()
        root_logger.removeHandler(handler)

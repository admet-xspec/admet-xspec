from src.processing_pipeline import ProcessingPipeline
import logging
import time
import tempfile
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
        default="INFO",
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

    pipeline = ProcessingPipeline()
    pipeline.run()

    # Log time
    time_elapsed = time.time() - time_start
    logging.info(
        f"ProcessingPipeline finished in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"ProcessingPipeline finished in {round(time_elapsed / 60, 2)} minutes."
    )

    # Save log file with timestamp
    log_directory = pathlib.Path("./logs")
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = pathlib.Path(f"processing_{timestamp}.log")
    pathlib.Path(temp_log_file.name).rename(log_directory / log_filename)

import logging
import time
import glob
from pathlib import Path
import pandas as pd

from src.mgmt_pipeline import ManagementPipeline

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
        default="configs/train.gin",
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

    logs_dir = Path("data/preprocessing/logs")
    logging_path = logs_dir / "console.log"
    
    if not logging_path.exists():
        logging_path.parent.mkdir(parents=True, exist_ok=True)
        logging_path.touch(exist_ok=True)

    # Configure logger
    logging.basicConfig(
        level=args.log_level,
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler(),
        ],
    )

    # Initialize the management pipeline
    pipeline = ManagementPipeline()

    pipeline.run()
    
    # Log time
    time_elapsed = time.time() - time_start
    logging.info(
        f"Manual processing completed in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"Manual processing completed in {round(time_elapsed / 60, 2)} minutes."
    )

    # Dump operative config
    gin_path = f"data/preprocessing/logs/operative_config.gin"
    Path(gin_path).parent.mkdir(parents=True, exist_ok=True)
    Path(gin_path).touch(exist_ok=True)

    with open(gin_path, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Config saved to {gin_path}")

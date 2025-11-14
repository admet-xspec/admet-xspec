from src.processing_pipeline import ProcessingPipeline
import logging
from datetime import datetime
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

    pipeline = ProcessingPipeline()
    pipeline.run()

    # Log time
    time_elapsed = time.time() - time_start
    logging.info(
        f"ProcessingPipeline finished in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"ProcessingPipeline finished in {round(time_elapsed / 60, 2)} minutes."
    )

    config_str = gin.operative_config_str()

    temp_log_file.close()
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    for handler in handlers:
        handler.close()
        root_logger.removeHandler(handler)
    with open(temp_log_file.name, "r") as f:
        log_contents = f.read()

    # Dump logs and operative config
    timestamp = datetime.now().strftime('%d_%H_%M_%S')

    # Dump operative config
    if pipeline.do_train_test_split:
        # If doing train-test split, dump to a split-specific subdirectory
        pipeline.dump_logs_to_data_dir(config_str, f'operative_config_{timestamp}.gin', dump_to_split_subdir=True)
        pipeline.dump_logs_to_data_dir(log_contents, f'processing_{timestamp}.log', dump_to_split_subdir=True)
    else:
        # Otherwise, dump to the dataset directory / logs
        pipeline.dump_logs_to_data_dir(config_str, f'operative_config_{timestamp}.gin')
        pipeline.dump_logs_to_data_dir(log_contents, f'processing_{timestamp}.log')
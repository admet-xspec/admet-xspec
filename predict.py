import gin
import logging
from src.inference_pipeline import InferencePipeline

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        help="Path to the config file",
        default="configs/.gin",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
        default="DEBUG",
    )
    args = parser.parse_args()

    # Load the gin configuration
    if not pathlib.Path(args.cfg).is_file():
        raise FileNotFoundError(f"Config file {args.cfg} not found.")
    gin.parse_config_file(args.cfg)
    logging.basicConfig(level=logging.INFO)

    data_path = pathlib.Path(gin.query_parameter("predict.data_path"))
    task_name = gin.query_parameter("predict.task_name")

    # Configure logger
    logging.basicConfig(
        level=args.log_level,
        handlers=[
            logging.FileHandler(f"{data_path.parent}/{task_name}_console.log"),
            logging.StreamHandler(),
        ],
    )

    # Initialize the inference pipeline
    pipeline = InferencePipeline()

    # Run the prediction
    pipeline.predict()

    # Dump operative config
    gin_path = f"{out_dir}/operative_config.gin"
    with open(gin_path, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Config saved to {gin_path}")
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
    results_path = pipeline.out_dir / "predictions.csv"
    results.to_csv(results_path, index=False)

    # Compute and save metrics if possible
    if pipeline.can_compute_metrics():
        metrics = pipeline.get_metrics()

        metrics_path = pipeline.out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            import json

            json.dump(metrics, f)
        logging.info(f"Metrics saved to {metrics_path}")

    # Dump operative config
    gin_path = pipeline.out_dir / "operative_config.gin"
    with open(gin_path, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Config saved to {gin_path}")

    # Move the temporary log file to the output directory
    final_log_path = pipeline.out_dir / "predict.log"
    pathlib.Path(temp_log_file.name).rename(final_log_path)
    temp_log_file.close()

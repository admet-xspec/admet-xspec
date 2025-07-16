import pandas as pd
import gin
import logging
import pickle
import argparse
from src.featurizer.featurizer import (
    EcfpFeaturizer,
    RdkitFeaturizer,
    RdkitEcfpFeaturizer,
)
from src.predictor.base import PredictorBase
from src.predictor.scikit import SvmClassifier, SvmRegressor, RfClassifier, RfRegressor
from src.utils import clean_smiles, get_nice_class_name

SMILES_COLUMN = ["smiles", "molecule"]


@gin.configurable
def predict(
    data_path: str | None = None,
    model_path: str | None = None,
    task_name: str | None = None,
):

    data = pd.read_csv(data_path)
    logging.info(f"Loaded data from {data_path}")

    # Parse the SMILES column
    smiles_list = []
    for col in SMILES_COLUMN:
        if col in data.columns:
            smiles_list = data[col].tolist()
            break
    if not smiles_list:
        raise ValueError("No valid SMILES column found in the data.")

    # Clean the SMILES strings
    processed_smiles = clean_smiles(smiles_list)
    logging.info(f"Cleaned {len(smiles_list)} SMILES strings")
    if len(processed_smiles) != len(smiles_list):
        logging.warning(
            f"{len(smiles_list) - len(processed_smiles)} SMILES strings could not be prepared"
        )

    # Load the model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    logging.info(f"Loaded model {get_nice_class_name(model)} from {model_path}")

    # Make inferences
    y_pred = model.predict(smiles_list)
    logging.info(f"Predicted {len(y_pred)} values for task '{task_name}'")

    # Add predictions to the DataFrame
    if model.task_type == "classifier":
        data[f"{task_name}_class"] = [0 if pred < 0.5 else 1 for pred in y_pred]
        data[f"{task_name}_class_probability"] = [float(pred) for pred in y_pred]

    elif model.task_type == "regressor":
        data[f"{task_name}_pred"] = y_pred

    else:
        raise ValueError(f"Unknown task type: {model.task_type}")

    # Save the predictions to a new CSV file
    output_path = data_path.replace(".csv", f"_{task_name}.csv")
    data.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")


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

    # Run the prediction
    predict()

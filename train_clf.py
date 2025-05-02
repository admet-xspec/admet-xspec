import pandas as pd
from src.utils import clean_smiles, get_nice_class_name
import logging
from src.predictor.chemprop import ChempropBinaryClassifier, ChempropRegressor
from src.predictor.scikit import (
    SvmClassifier,
    SvmRegressor,
    RfClassifier,
    RfRegressor,
    PredictorBase,
)
from src.featurizer.fingerprint import EcfpFeaturizer, FeaturizerBase
from sklearn.model_selection import train_test_split
import time
import json
import gin


@gin.configurable()
def train(
    data_path: str,
    predictor: PredictorBase,
    featurizer: FeaturizerBase | None,
    random_state: int = 42,
    test_size: float = 0.2,
    strafity_test: bool = False,
    model_name="model",
    out_dir="models",
):
    logging.basicConfig(level=logging.DEBUG)

    # Load data
    df = pd.read_csv(data_path)

    # Try to sanitize the data
    pre_cleaning_length = len(df)
    df.columns = df.columns.str.lower()
    df["smiles"] = clean_smiles(df["smiles"])
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    new_length = len(df)
    if pre_cleaning_length != new_length:
        logging.info(f"Dropped {pre_cleaning_length - new_length} invalid SMILES")
    logging.info(f"Dataset size: {new_length}")
    logging.info(f"Predictor: {get_nice_class_name(predictor.model)}")

    # Perform a train-test split
    X, y = df["smiles"], df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if strafity_test else None,
    )

    # set the working directory
    predictor.set_working_dir(f"{out_dir}/{model_name}")

    if hasattr(predictor, "inject_featurizer"):
        # If the predictor has an inject_featurizer method, use it
        predictor.inject_featurizer(featurizer)
    else:
        # ingore the featurizer
        logging.info(
            f"Model {get_nice_class_name(predictor)} uses internal featurizer - ignoring {get_nice_class_name(featurizer)}."
        )

    # train (either use hyperparameters provided in the predictor .gin config file directly, or
    #        conduct hyperparameter optimization over distributions given in the same .gin config file)
    predictor.train(X_train, y_train)

    # test
    y_pred = predictor.infer(X_test)

    # save the model
    predictor.save(out_dir)

    # dump operative config
    gin_path = f"{out_dir}/operative_config.gin"
    with open(gin_path, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Config saved to {gin_path}")

    # save metrics
    metrics_path = f"{out_dir}/metrics.json"
    metrics_dict = predictor.calc_metrics(y_test, y_pred)
    with open(metrics_path, "w") as f:
        json.dump(
            metrics_dict,
            f,
        )
        logging.info(f"Metrics saved to {metrics_path}")
        logging.info(metrics_dict)


if __name__ == "__main__":
    import argparse
    import gin
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        default="configs/config.gin",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
        default="INFO",
    )
    args = parser.parse_args()

    # Load the gin configuration
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    gin.parse_config_file(args.config)

    # Create a directory for the outputs (model_name)
    models_dir = gin.query_parameter("%MODELS_DIR")
    model_name = gin.query_parameter("%NAME")
    out_dir = f"{models_dir}/{model_name}"

    # If the directory already exists, add a timestamp to the name
    if os.path.isdir(out_dir):
        out_dir = out_dir + f'_{time.strftime("%Y%m%d-%H%M%S")}'

    os.mkdir(out_dir)

    # Configure logger
    logging.basicConfig(
        level=args.log_level,
        handlers=[
            logging.FileHandler(f"{out_dir}/console.log"),
            logging.StreamHandler(),
        ],
    )

    # Train the model
    train(out_dir=out_dir)

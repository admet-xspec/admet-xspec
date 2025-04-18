import os

import pandas as pd
from src.utils import clean_smiles
import logging
from src.predictor.chemprop import ChempropBinaryClassifier, ChempropRegressor
from src.predictor.scikit import (
    SvmClassifier,
    SvmRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    PredictorBase,
)
from src.featurizer.fingerprint import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import time
import json
import gin


@gin.configurable()
def train(
    data_path: str,
    predictor: PredictorBase,
    featurizer: FeaturizerBase | None,
    random_state: int = 42,
    cv: int = 1,
    test_size: float = 0.2,
    strafity_test: bool = False,
    model_name = 'model',
    out_dir = 'models'
):
    logging.basicConfig(level=logging.DEBUG)

    df = pd.read_csv(data_path)

    featurizer = EcfpFeaturizer(radius=2, n_bits=2048, count=False)

    pre_cleaning_length = len(df)
    df["SMILES"] = clean_smiles(df["SMILES"])
    df = df.dropna(subset=["SMILES"]).reset_index(drop=True)
    new_length = len(df)
    if pre_cleaning_length != new_length:
        logging.info(f"Dropped {pre_cleaning_length - new_length} invalid SMILES")

    logging.info(f"Dataset size: {new_length}")
    logging.info(f"Predictor: {predictor.name()}")

    # Perform a train-test split
    X, y = df["SMILES"], df["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if strafity_test else None,
    )

    if hasattr(predictor, "inject_featurizer"):
        # If the predictor has an inject_featurizer method, use it
        logging.info(
            f"Injecting {featurizer.name()} featurizer into {predictor.name()}"
        )
        predictor.inject_featurizer(featurizer)
    else:
        # ingore the featurizer
        logging.info(
            f"Model {predictor.name()} uses internal featurizer - ignoring {featurizer.name()}."
        )

    # training
    predictor.train(X_train, y_train)

    # testing
    y_pred = predictor.predict(X_test)

    # calculate ROC_AUC and accuracy
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())

    logging.info(f"ROC AUC: {round(roc_auc, 3)}")
    logging.info(f"Accuracy: {round(accuracy, 3)}")
    logging.info(f"Precision: {round(precision, 3)}")
    logging.info(f"Recall: {round(recall, 3)}")

    # save the model
    predictor.save(out_dir)
    logging.info(f"Model saved to {out_dir}")

    # dump operative config
    gin_path = f"{out_dir}/operative_config.gin"
    with open(gin_path, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Config saved to {gin_path}")

    # save metrics
    metrics_path = f"{out_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
            },
            f,
        )


if __name__ == "__main__":
    import argparse
    import gin
    import os

    parser = argparse.ArgumentParser(description="Train a predictor")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        default="configs/train_clf.gin",
    )
    args = parser.parse_args()

    # Load the gin configuration
    gin.parse_config_file(args.config)

    # Get model name
    models_dir = gin.query_parameter("%MODELS_DIR")
    model_name = gin.query_parameter("%MODEL").wrapped

    # Create a directory for the outputs (model_name + timestamp)
    out_dir = f'{models_dir}/{model_name}_{time.strftime("%Y%m%d-%H%M%S")}'
    os.mkdir(out_dir)

    # Save logs to a file
    logger = logging.getLogger(__name__)
    FileOutputHandler = logging.FileHandler('console.log')
    logger.addHandler(FileOutputHandler)

    # Train the model
    train(out_dir=out_dir)

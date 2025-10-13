import pandas as pd
from src.utils import SmilesCleaner, get_nice_class_name
import logging
from src.predictor.predictor_base import PredictorBase
from src.data.featurizer import (
    FeaturizerBase,
)
from src.data.split import DataSplitterBase
import json
import gin
import time
from pathlib import Path


@gin.configurable()
class TrainingPipeline:

    def __init__(
        self,
        data_path: Path | str,
        splitter: DataSplitterBase,
        predictor: PredictorBase,
        featurizer: FeaturizerBase,
        model_name: str,
        out_dir: Path | str = "models",
        test_size: float = 0.2,
        stratify: bool = True,
        train_path: Path | str = None,
        test_path: Path | str = None,
    ):

        self.data_path = Path(data_path)
        self.splitter = splitter
        self.predictor = predictor
        self.out_dir = Path(out_dir) / model_name
        # If the output directory already exists, add a timestamp to the name
        if self.out_dir.exists():
            self.out_dir = self.out_dir.with_name(
                f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}'
            )
        self.test_size = test_size
        self.stratify = stratify

        # If the predictor has an inject_featurizer method, invoke it
        if hasattr(predictor, "inject_featurizer"):
            predictor.inject_featurizer(featurizer)
        else:
            # ingore the featurizer
            logging.info(
                f"Model {get_nice_class_name(predictor)} uses internal featurizer - ignoring {get_nice_class_name(featurizer)}."
            )

        self.train_path = Path(train_path) if train_path else None
        self.test_path = Path(test_path) if test_path else None

    def prepare_data(self):
        """
        Prepares the data for training and evaluation.
        -> Loads the data from a CSV file.
        -> Cleans the SMILES strings.
        -> Splits the data into training and testing sets.
        -> Saves the training and testing sets to CSV files.
        """

        # Load the data
        df = pd.read_csv(self.data_path)

        # Sanitize the data
        cleaner = SmilesCleaner()
        pre_cleaning_length = len(df)
        df.columns = df.columns.str.lower()
        df["smiles"] = df["smiles"].apply(cleaner.clean)
        df.dropna(subset=["smiles"], inplace=True)
        if pre_cleaning_length != len(df):
            logging.info(f"Dropped {pre_cleaning_length - len(df)} invalid SMILES")
        logging.info(f"Dataset size: {len(df)}")

        # Perform a train-test split
        X_train, X_test, y_train, y_test = self.splitter.split(df["smiles"], df["y"])
        logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Save the splits to a designated subdir
        self.train_path = (
            self.data_path.parent / self.splitter.get_cache_key() / "train.csv"
        )
        self.test_path = (
            self.data_path.parent / self.splitter.get_cache_key() / "test.csv"
        )

        self.train_path.parent.mkdir(parents=True, exist_ok=True)
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

        df_train = pd.DataFrame({"smiles": X_train, "y": y_train})
        df_test = pd.DataFrame({"smiles": X_test, "y": y_test})

        df_train.to_csv(self.train_path, index=False)
        df_test.to_csv(self.test_path, index=False)

        logging.info(f"Train data saved to {self.train_path}")
        logging.info(f"Test data saved to {self.test_path}")

    def train(self):
        """
        Trains the model and saves the parameters.
        """

        if self.train_path is None:
            raise ValueError(
                "The dataset has not been split yet. Use prepare_data method first"
            )

        # Load the data and parse X, y columns
        X_train, y_train = self._parse_data(self.train_path)

        # train (either use hyperparameters provided in the predictor .gin config file directly or
        #        conduct hyperparameter optimization over distributions given in the same .gin config file)
        self.predictor.train(X_train, y_train)

        # save the trained model
        self.predictor.save(self.out_dir)

    def evaluate(self):
        """
        Evaluates model performance on a holdout test set and saves the metrics to a file.
        """

        if self.test_path is None:
            raise ValueError(
                "The dataset has not been split yet. Use prepare_data method first"
            )

        X_test, y_test = self._parse_data(self.test_path)

        # evaluate the model
        metrics_dict = self.predictor.evaluate(X_test, y_test)
        logging.info(f"Metrics: {metrics_dict}")

        # save metrics
        metrics_path = self.out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                metrics_dict,
                f,
            )
            logging.info(f"Metrics saved to {metrics_path}")

    def _parse_data(self, csv_path):
        data = pd.read_csv(csv_path)
        logging.debug(f"Reading data from {csv_path}")
        if "smiles" not in data.columns:
            raise ValueError("No 'smiles' column detected in the data .csv")
        if "y" not in data.columns:
            raise ValueError("No 'y' column detected in the data .csv")
        return data["smiles"], data["y"]

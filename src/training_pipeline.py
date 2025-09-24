import pandas as pd
from src.utils import clean_smiles, get_nice_class_name
import logging
from src.predictor.predictor_base import PredictorBase
from src.data.featurizer import (
    FeaturizerBase,
)
from src.data.split import DataSplitterBase
import json
import gin
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
    ):

        self.data_path = Path(data_path)
        self.splitter = splitter
        self.predictor = predictor
        self.model_name = model_name
        self.out_dir = out_dir
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

        self.train_path = None
        self.test_path = None

    def prepare_data(self):
        """
        Prepares the data for training and evaluation.
        -> Loads the data from a CSV file.
        -> Cleans the SMILES strings.
        -> Splits the data into training and testing sets.
        -> Saves the training and testing sets to CSV files.
        """

        SMILES_COL = ["smiles", "SMILES", "molecule"]
        TARGET_COL = ["y", "Y", "label", "LABEL"]

        # Load the data
        df = pd.read_csv(self.data_path)

        # Check if the necessary columns are present
        if not any(col in df.columns for col in SMILES_COL):
            raise ValueError(
                f"Input CSV must contain one of the following columns for SMILES: {SMILES_COL}"
            )
        if not any(col in df.columns for col in TARGET_COL):
            raise ValueError(
                f"Input CSV must contain one of the following columns for target variable: {TARGET_COL}"
            )

        # Rename the columns to standard names
        df = df.rename(
            columns={col: "smiles" for col in SMILES_COL if col in df.columns}
        )
        df = df.rename(columns={col: "y" for col in TARGET_COL if col in df.columns})

        # Sanitize the data
        pre_cleaning_length = len(df)
        df.columns = df.columns.str.lower()
        df["smiles"] = clean_smiles(df["smiles"])
        df = df.dropna(subset=["smiles"]).reset_index(drop=True)
        if pre_cleaning_length != len(df):
            logging.info(f"Dropped {pre_cleaning_length - len(df)} invalid SMILES")
        logging.info(f"Dataset size: {len(df)}")

        # Perform a train-test split
        X_train, X_test, y_train, y_test = self.splitter.split(df["smiles"], df["y"])
        logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Save the splits to a designated subdir
        save_path_train = self.data_path / self.splitter.get_cache_key() / "train.csv"
        save_path_test = self.data_path / self.splitter.get_cache_key() / "test.csv"

        df_train = pd.DataFrame([X_train, y_train], columns=["smiles", "y"])
        df_test = pd.DataFrame([X_test, y_test], columns=["smiles", "y"])

        df_train.to_csv(save_path_train)
        df_test.to_csv(save_path_test)

    def train(self):
        """
        Trains the model and saves the parameters.
        """

        if self.train_path is None:
            raise ValueError(
                "The dataset has not been split yet. Use prepare_data method first."
            )

        # Load the data and parse X, y columns
        X_train, y_train = self._parse_data(self.train_path)

        # Set the working directory (to which the results and model parameters will be saved)
        self.predictor.set_working_dir(f"{self.out_dir}/{self.model_name}")

        # train (either use hyperparameters provided in the predictor .gin config file directly, or
        #        conduct hyperparameter optimization over distributions given in the same .gin config file)
        self.predictor.train(X_train, y_train)

        # save the traned model
        self.predictor.save()

    def evaluate(self):
        """
        Evaluates model performance on a holdout test set and saves the metrics to a file.
        """

        if self.test_path is None:
            raise ValueError(
                "The dataset has not been split yet. Use prepare_data method first."
            )

        X_test, y_test = self._parse_data(self.test_path)

        # evaluate the model
        metrics_dict = self.predictor.evaluate(X_test, y_test)
        logging.info(f"Metrics: {metrics_dict}")

        # save metrics
        metrics_path = f"{self.out_dir}/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                metrics_dict,
                f,
            )
            logging.info(f"Metrics saved to {metrics_path}")

    def _parse_data(self, csv_path):
        data = pd.read_csv(csv_path)
        if "smiles" not in data.columns:
            raise ValueError("""No 'smiles' column detected in the data .csv""")
        if "y" not in data.columns:
            raise ValueError("""No 'y' column detected in the data .csv""")
        return data["smiles"], data["y"]

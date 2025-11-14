import pandas as pd
from src.utils import (
    get_nice_class_name,
    log_markdown_table
)
import logging
from src.predictor.predictor_base import PredictorBase
from src.data.featurizer import (
    FeaturizerBase,
)
from src.data.data_interface import DataInterface
import json
import gin
from datetime import datetime
from pathlib import Path


@gin.configurable()
class TrainingPipeline:

    def __init__(
        self,
        data_interface: DataInterface,
        model_name: str | None = None,
        predictor: PredictorBase | None = None,
        featurizer: FeaturizerBase | None = None,
        models_output_dir: Path | str = "models",
        train_sets: list[str] | None = None,  # friendly_name
        test_sets: list[str] | None = None,  # friendly_name
        refit_on_full_data: bool = False,
        optimize_hyperparameters: bool = False,
    ):

        self.model_name = model_name
        self.data_interface = data_interface
        self.predictor = predictor

        self.train_sets = train_sets
        self.test_sets = test_sets
        self.optimize_hyperparameters = optimize_hyperparameters

        self.model_output_dir = (
                Path(models_output_dir)
                / f"{self.model_name}_{datetime.now().strftime('%d_%H_%M_%S')}"
        )

        # whether to refit the model on the full dataset after evaluation (train + test)
        self.refit_on_full_data = refit_on_full_data

        # inject featurizer into predictor if applicable
        self.try_inject_featurizer(featurizer)

    def run(self):
        # load data
        train_df = pd.concat(self.load_datasets(self.train_sets))
        test_df = pd.concat(self.load_datasets(self.test_sets))

        logging.info(f"Training dataset size: {len(train_df)}")
        logging.info(f"Test dataset size: {len(test_df)}")

        # parse X, y columns
        # we assume that the normalized datasets loaded from data_interface all have "smiles" and "y" columns
        X_train, y_train = train_df["smiles"], train_df["y"]
        X_test, y_test = test_df["smiles"], test_df["y"]

        # train
        if self.optimize_hyperparameters:
            if hasattr(self.predictor, "train_optimize"):
                self.predictor.train_optimize(X_train, y_train)
            else:
                logging.warning(
                    f"Model {get_nice_class_name(self.predictor)} does not support hyperparameter optimization - skipping."
                )
        else:
            self.predictor.train(X_train, y_train)

        # save the trained model
        self.save_model()

        # evaluate on test
        metrics = self.evaluate(X_test, y_test)
        self.save_metrics(metrics)

        # Refit on the entire dataset (train + test) if specified in the gin config
        if self.refit_on_full_data:
            logging.info("Refitting the model on the entire dataset (train + test).")
            self.train(
                pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
            )
            self.save_model(name="model_refit")

    def load_datasets(self, friendly_names: list[str]) -> list[pd.DataFrame]:
        """
        Loads datasets by their friendly names using the data interface.
        """
        dataset_dfs = [
            self.data_interface.get_by_friendly_name(friendly_name)
            for friendly_name in friendly_names
        ]
        return dataset_dfs

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> dict:
        """
        Evaluates model performance on a holdout test set and saves the metrics to a file.
        """
        # evaluate the model
        metrics_dict = self.predictor.evaluate(X_test, y_test)
        # log metrics
        logging.info(f"Metrics: {metrics_dict}")
        # and in a more copy-paste friendly format
        logging.info("Metrics (markdown):")
        log_markdown_table(metrics_dict)
        return metrics_dict

    def save_metrics(self, metrics: dict) -> None:
        """
        Saves the evaluation metrics to a JSON file.
        """
        metrics_path = self.model_output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                metrics,
                f,
            )
        logging.info(f"Metrics saved to {metrics_path}")

    def save_model(self, name: str = "model") -> None:
        """
        Saves the trained model to the output directory.
        """
        self.predictor.save(self.model_output_dir, name="model")
        logging.info(f"Model saved to {self.model_output_dir}")

    def dump_logs(self, config_str: str, filename: str):
        """
        Dumps the logs or operative config to the output directory.
        """
        path = self.model_output_dir / filename
        with open(path, "w") as f:
            f.write(config_str)

    def try_inject_featurizer(self, featurizer: FeaturizerBase | None):
        """
        If a featurizer is provided and the predictor does not already have one,
        inject the featurizer into the predictor.
        """
        # If the predictor has an inject_featurizer method, invoke it
        if hasattr(self.predictor, "inject_featurizer"):
            self.predictor.inject_featurizer(featurizer)
        else:
            # ingore the featurizer
            logging.warning(
                f"Model {get_nice_class_name(self.predictor)} uses internal featurizer - ignoring {get_nice_class_name(featurizer)}."
            )
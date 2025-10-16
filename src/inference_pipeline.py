import pandas as pd
import gin
import logging
import pickle
from src.utils import (
    get_clean_smiles,
    get_nice_class_name,
    log_markdown_table,
    parse_smiles_from_messy_csv,
    parse_targets_from_messy_csv,
)
from pathlib import Path
import json


@gin.configurable
class InferencePipeline:

    def __init__(
        self,
        model_path: Path | str,
        data_path: Path | str,
        task_name,
    ):

        self.model = self._load_model(model_path)
        self.data_path = Path(data_path)
        self.out_dir = self.data_path.parent / task_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        self.data = self._prepare_data(data_path)

    def predict(self):

        # Select only valid SMILES for prediction
        valid_data = self.data[self.data["is_valid"]]

        # Inference
        y_pred = self.model.predict(valid_data["smiles"].tolist())
        if hasattr(self.model, "classify"):
            y_class = self.model.classify(y_pred)
        else:
            y_class = None

        pred_df = pd.DataFrame(
            {"original_index": valid_data["original_index"], "pred": y_pred}
        )

        if y_class is not None:
            pred_df["pred_class"] = y_class

        # Merge predictions back to the original data
        result_df = pd.merge(self.data, pred_df, on="original_index", how="left")
        result_df = result_df.drop(columns=["original_index", "is_valid"])

        # Save results
        results_path = self.out_dir / "predictions.csv"
        result_df.to_csv(results_path, index=False)
        logging.info(f"Predictions saved to {results_path}")

    def evaluate(self):
        valid_data = self.data[self.data["is_valid"]]
        X = valid_data["smiles"].tolist()
        y_true = valid_data["y"].tolist()
        # TODO: Make this more robust, as model.evaluate() method uses predict() internally
        metrics = self.model.evaluate(X, y_true)
        logging.info(f"Evaluation metrics: {metrics}")

        # Log metrics in markdown format
        logging.info("Metrics (markdown):")
        log_markdown_table(metrics)

        # Save metrics
        metrics_path = self.out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def can_compute_metrics(self):
        if "y" not in self.data.columns:
            logging.debug("No target column 'y' in the data - cannot compute metrics.")
            return False
        return True

    def _load_model(self, model_path):
        """
        Loads the model from a pickle
        """
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        logging.info(f"Loaded {get_nice_class_name(model)} model from {model_path}")

        return model

    def _prepare_data(self, data_path):
        """
        Loads the data and cleans it up in preparation for inference
        The inference pipeline expects a .csv file with at least one column named 'smiles'.
        Optionally, a second column named 'y' can be present for evaluation purposes.
        :return:
        """
        logging.info(f"Loading data from {data_path}")
        smiles_col = parse_smiles_from_messy_csv(data_path)
        target_col = parse_targets_from_messy_csv(data_path)

        # Clean the SMILES strings
        processed_smiles = smiles_col.apply(get_clean_smiles)

        df = pd.DataFrame(
            {
                "original_smiles": smiles_col,
                "original_index": processed_smiles.index,
                "smiles": processed_smiles,
                "is_valid": processed_smiles.notnull(),
            }
        )

        if target_col is not None:
            df["y"] = target_col

        df["comment"] = df["is_valid"].apply(
            lambda x: "" if x else "SMILES failed in preparation"
        )
        num_invalid = df["is_valid"].value_counts().get(False, 0)
        if num_invalid > 0:
            logging.warning(f"{num_invalid} SMILES strings couldn't have been parsed")

        return df

import pandas as pd
import gin
import logging
import pickle
from src.utils import SmilesCleaner, get_nice_class_name, get_metric_callable
from pathlib import Path


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
        self.smiles_column_names = ["smiles", "SMILES", "molecule"]
        self.y_column_names = ["y", "Y", "target", "Target"]

        # Load and prepare data
        self.data = self._prepare_data(data_path)
        self.preds = None

    def predict(self):

        # Select only valid SMILES for prediction
        valid_data = self.data[self.data["is_valid"]]

        # Inference
        y_pred = self.model.predict(valid_data["smiles"].tolist())
        if hasattr(self.model, "classify"):
            y_class = self.model.classify(y_pred)
            self.preds = y_class
        else:
            y_class = None
            self.preds = y_pred

        pred_df = pd.DataFrame(
            {"original_index": valid_data["original_index"], "pred": y_pred}
        )

        if y_class is not None:
            pred_df["pred_class"] = y_class

        # Merge predictions back to the original data
        result_df = pd.merge(self.data, pred_df, on="original_index", how="left")
        result_df = result_df.drop(columns=["original_index", "is_valid"])
        return result_df

    def get_metrics(self):
        valid_data = self.data[self.data["is_valid"]]
        y_true = valid_data["y"].tolist()
        metrics_dict = {}
        for m in self.model.evaluation_metrics:
            metrics_dict[m] = get_metric_callable(m)(y_true, self.preds)
        return metrics_dict

    def can_compute_metrics(self):
        valid_data = self.data[self.data["is_valid"]]
        if valid_data.empty:
            logging.warning("No valid SMILES in the data - cannot compute metrics.")
            return False
        if "y" not in valid_data.columns:
            logging.warning(
                "No target column 'y' in the data - cannot compute metrics."
            )
            return False
        if not hasattr(self.model, "evaluation_metrics"):
            logging.warning("The model does not have any defined evaluation metrics.")
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
        :return:
        """
        data = pd.read_csv(data_path)
        logging.info(f"Loaded data from {data_path}")

        # Parse the SMILES column
        smiles_col = None
        for col in self.smiles_column_names:
            if col in data.columns:
                smiles_col = data[col]
                logging.info(
                    f"Found {len(smiles_col)} SMILES strings in column '{col}'"
                )
                break
        if smiles_col is None:
            raise ValueError(
                "No valid SMILES column found in the input data. Expected one of: "
                + ", ".join(self.smiles_column_names)
            )

        target_col = None
        for col in self.y_column_names:
            if col in data.columns:
                target_col = data[col]
                logging.info(
                    f"Note: Found a target column '{col}' in the input data. It will be used to compute metrics."
                )
                break

        # Clean the SMILES strings
        cleaner = SmilesCleaner()
        processed_smiles = smiles_col.apply(cleaner.clean)

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

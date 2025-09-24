import pandas as pd
import gin
import logging
import pickle
from src.utils import clean_smiles, get_nice_class_name
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
        self.data_path = data_path
        self.task_name = task_name
        self.smiles_column_names = ["smiles", "SMILES", "molecule"]

        # Load and prepare data
        self.data = self._prepare_data(data_path)

    def predict(self):

        # Inference
        y_pred = self.model.predict(self.data)
        logging.info(f"Predicted {len(y_pred)} values for task '{self.task_name}'")

        output_df = pd.DataFrame()
        output_df["smiles"] = self.data

        # Add predictions to the DataFrame
        if self.model.task_type == "classifier":
            output_df[f"{self.task_name}_class"] = [
                0 if pred < 0.5 else 1 for pred in y_pred
            ]
            output_df[f"{self.task_name}_class_probability"] = [
                float(pred) for pred in y_pred
            ]

        elif self.model.task_type == "regressor":
            output_df[f"{self.task_name}_pred"] = y_pred

        else:
            raise ValueError(
                f"Unknown task type: {self.model.task_type}. Expected 'classifier' or 'regressor'."
            )

        # Save the predictions to a new CSV file
        output_path = self.data_path.replace(".csv", f"_{self.task_name}.csv")
        output_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    def _load_model(self, model_path):
        """
        Loads the model from a pickle
        """
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        logging.info(
            f"Loaded {get_nice_class_name(self.model)} model from {model_path}"
        )

        return model

    def _prepare_data(self, data_path):
        """
        Loads the data and cleans it up in preparation for inference
        :return:
        """
        data = pd.read_csv(data_path)
        logging.info(f"Loaded data from {data_path}")

        # Parse the SMILES column
        smiles_list = []
        for col in self.smiles_column_names:
            if col in data.columns:
                smiles_list = data[col].tolist()
                logging.info(
                    f"Found {len(smiles_list)} SMILES strings in column '{col}'"
                )
                break
        if not smiles_list:
            raise ValueError(
                "No valid SMILES column found in the input data. Expected one of: "
                + ", ".join(self.smiles_column_names)
            )

        # Clean the SMILES strings
        processed_smiles = clean_smiles(smiles_list)
        if len(processed_smiles) != len(smiles_list):
            logging.warning(
                f"{len(smiles_list) - len(processed_smiles)} SMILES were dropped in preparation."
            )

        return processed_smiles

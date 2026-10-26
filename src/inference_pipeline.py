from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

import gin
import numpy as np
import pandas as pd
import yaml

from src.data.data_interface import DataInterface
from src.predictor.predictor_base import BinaryClassifierBase, PredictorBase
from src.utils import get_clean_smiles, log_markdown_table


@gin.configurable
class InferencePipeline:
    """Run a cached predictor on a dataset and save predictions and metrics to cache."""

    def __init__(
        self,
        data_interface: DataInterface,
        dataset_path: str | Path,
        model_cache_key: str,
        data_cache_key: str,
        task_setting: str = "regression",
        smiles_col: str = "smiles",
        source_col: str = "source",
        target_col: str = "y",
        use_refit_model: bool = True,
        do_evaluate: bool = True,
        do_append_to_input_file: bool = False,
        logfile: str | None = None,
        override_cache: bool = False,
    ):
        self.data_interface = data_interface
        self.dataset_path = Path(dataset_path)
        self.dataset_file_name = self.dataset_path.stem
        self.model_cache_key = model_cache_key
        self.data_cache_key = data_cache_key
        self.smiles_col = smiles_col
        self.source_col = source_col
        self.target_col = target_col
        self.clean_smiles_col = f"{smiles_col}_clean"
        # The refit model, with its feature scaler, was fit on train + test.
        self.use_refit_model = use_refit_model
        self.do_evaluate = do_evaluate
        self.do_append_to_input_file = do_append_to_input_file

        self.data_interface.set_task_setting(task_setting)
        if logfile is not None:
            self.data_interface.set_logfile(logfile)
        self.data_interface.set_override_cache(override_cache)

        self.out_dir = (
            self.data_interface.cache_dir
            / "predictions"
            / self.dataset_file_name
            / self.model_cache_key
            / self.data_cache_key
        )
        self.predictions_path = self.out_dir / "predictions.csv"
        self.metrics_path = self.out_dir / "metrics.yaml"
        self.label_name = None

        self.data: Optional[pd.DataFrame] = None
        self.predictor: Optional[PredictorBase] = None
        self.task_setting = task_setting

        self._validate_configuration()

    def run(self) -> Path:
        """Run inference and return the path of the saved predictions."""
        self._log_pipeline_start()
        self.data = self._load_dataset()
        self.predictor = self._load_predictor()
        self.task_setting = self._detect_task_setting(self.predictor)
        self.data = self._align_source_labels(self.data, self.predictor)

        predictions_df = self._predict_dataframe(self.data, self.predictor)
        self._save_predictions(predictions_df)

        if self.can_compute_metrics():
            metrics = self.evaluate()
            self._save_metrics(metrics)

        self.data_interface.dump_gin_config(self.out_dir)
        self.data_interface.dump_logs(self.out_dir)
        logging.info(
            f"Inference finished. Predictions saved to `{self.predictions_path}`"
        )
        return self.predictions_path

    def predict(self) -> Path:
        """Alias of `run`, kept for `predict.py`."""
        return self.run()

    def can_compute_metrics(self) -> bool:
        if not self.do_evaluate or self.data is None:
            return False
        return self.target_col in self.data.columns

    def evaluate(self) -> dict:
        """Evaluate the loaded predictor on the loaded dataset."""
        if self.data is None or self.predictor is None:
            raise RuntimeError("Call run() before evaluate().")
        if self.target_col not in self.data.columns:
            raise ValueError(
                f"Cannot evaluate without `{self.target_col}` in inference dataset."
            )
        eval_df = self._get_model_input(self.data).dropna(subset=[self.target_col])
        metrics = self.predictor.evaluate(eval_df)
        logging.info("Metrics (markdown):")
        log_markdown_table(metrics)
        return metrics

    def _load_dataset(self) -> pd.DataFrame:
        dataset = pd.read_csv(self.dataset_path)

        if self.smiles_col not in dataset.columns:
            raw_smiles_col = DataInterface.get_smiles_col_in_raw(dataset)
            dataset = dataset.rename(columns={raw_smiles_col: self.smiles_col})

        if self.target_col not in dataset.columns:
            try:
                raw_target_col = DataInterface.get_label_col_in_raw(dataset)
                dataset = dataset.rename(columns={raw_target_col: self.target_col})
            except ValueError:
                pass

        return self._clean_smiles(dataset)

    def _clean_smiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `<smiles_col>_clean` with SMILES cleaned as for training; invalid ones are NaN."""
        df[self.clean_smiles_col] = df[self.smiles_col].apply(get_clean_smiles)
        n_invalid = int(df[self.clean_smiles_col].isna().sum())
        if n_invalid:
            logging.warning(
                f"{n_invalid} input SMILES could not be cleaned; their predictions are left empty."
            )
        return df

    def _get_model_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the rows with valid SMILES, with the cleaned SMILES in `smiles_col`."""
        valid = df[df[self.clean_smiles_col].notna()].copy()
        valid[self.smiles_col] = valid[self.clean_smiles_col]
        return valid

    def _load_predictor(self) -> PredictorBase:
        predictor = self.data_interface.unpickle_model(
            model_cache_key=self.model_cache_key,
            data_cache_key=self.data_cache_key,
            refit=self.use_refit_model,
        )
        if not isinstance(predictor, PredictorBase):
            raise TypeError(
                f"Loaded object is not PredictorBase: {type(predictor).__name__}"
            )
        self.label_name = predictor.task
        predictor.set_column_ids(self.smiles_col, self.source_col, self.target_col)
        return predictor

    def _align_source_labels(
        self, df: pd.DataFrame, predictor: PredictorBase
    ) -> pd.DataFrame:
        """Match source labels to the endpoint map of attributed predictors."""
        endpoint_ohe_map = getattr(predictor, "endpoint_ohe_map", None)
        if not endpoint_ohe_map:
            return df

        expected_sources = set(endpoint_ohe_map.keys())
        observed_sources = set(df[self.source_col].astype(str).unique())
        if observed_sources.issubset(expected_sources):
            return df

        if len(expected_sources) == 1:
            inferred_source = next(iter(expected_sources))
            logging.warning(
                "Input source labels do not match model endpoint map; "
                f"using single available source `{inferred_source}` for all rows."
            )
            aligned = df.copy()
            aligned[self.source_col] = inferred_source
            return aligned

        raise ValueError(
            "Input source labels do not match model endpoint map. "
            f"Observed: {sorted(observed_sources)}; expected one of: {sorted(expected_sources)}"
        )

    def _predict_dataframe(
        self, df: pd.DataFrame, predictor: PredictorBase
    ) -> pd.DataFrame:
        model_input = self._get_model_input(df)
        preds = predictor.predict(model_input)
        if len(preds) != len(model_input):
            raise RuntimeError(
                f"Prediction length mismatch: {len(preds)} predictions for {len(model_input)} rows"
            )

        df[self.label_name] = np.nan
        df.loc[model_input.index, self.label_name] = np.asarray(preds)
        if self.task_setting in {"binary_classification", "multi_class_classification"}:
            if not isinstance(predictor, BinaryClassifierBase):
                raise TypeError(
                    "Classification task requires predictor to implement `classify`."
                )
            class_col = f"{self.label_name}_class"
            df[class_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
            df.loc[model_input.index, class_col] = np.asarray(predictor.classify(preds))
        return df

    def _save_predictions(self, results: pd.DataFrame) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.do_append_to_input_file:
            results.to_csv(self.dataset_path, index=False)
        results.to_csv(self.predictions_path, index=False)

    def _save_metrics(self, metrics: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w") as fh:
            # Metric values are numpy scalars, which `safe_dump` cannot represent.
            yaml.safe_dump({"metrics": {k: float(v) for k, v in metrics.items()}}, fh)

    def _detect_task_setting(self, predictor: PredictorBase) -> str:
        """Return the task setting implied by the loaded predictor."""
        if isinstance(predictor, BinaryClassifierBase):
            task_setting = "binary_classification"
        else:
            task_setting = "regression"
        if task_setting != self.task_setting:
            logging.info(f"* Task setting detected from predictor: {task_setting}")
        return task_setting

    def _validate_configuration(self) -> None:
        if not self.data_interface:
            raise ValueError("InferencePipeline requires a valid data_interface")
        if not self.dataset_path:
            raise ValueError("`dataset_path` must be provided")
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Inference dataset file does not exist: `{self.dataset_path}`"
            )
        if not self.model_cache_key:
            raise ValueError("`model_cache_key` must be provided")
        if not self.data_cache_key:
            raise ValueError("`data_cache_key` must be provided")

    def _log_pipeline_start(self) -> None:
        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info(
            "# ================================ ADMET-XSpec ================================ #"
        )
        logging.info("Inference configuration summary:")
        logging.info(f"* Dataset path: {self.dataset_path}")
        logging.info(f"* Model cache key: {self.model_cache_key}")
        logging.info(f"* Data cache key: {self.data_cache_key}")
        logging.info(f"* Use refit model: {self.use_refit_model}")
        logging.info(f"* Task setting: {self.task_setting}")
        logging.info(f"* Output dir: {self.out_dir}")

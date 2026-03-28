from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, cast
import logging

import gin
import pandas as pd
import yaml

from src.data.data_interface import DataInterface
from src.predictor.predictor_base import BinaryClassifierBase, PredictorBase
from src.utils import detect_csv_delimiter
from src.utils import log_markdown_table


@gin.configurable
class InferencePipeline:
    """Load dataset + cached predictor, run inference, and persist outputs to cache."""

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
        do_evaluate: bool = True,
        logfile: str | None = None,
        override_cache: bool = False,
    ):
        self.data_interface = data_interface
        self.dataset_path = Path(dataset_path)
        self.dataset_file_name = self.dataset_path.stem
        self.model_cache_key = model_cache_key
        self.data_cache_key = data_cache_key
        self.task_setting = task_setting
        self.smiles_col = smiles_col
        self.source_col = source_col
        self.target_col = target_col
        self.do_evaluate = do_evaluate

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

        self._validate_configuration()

    def run(self) -> Path:
        self._log_pipeline_start()
        self.data = self._load_dataset()
        self.predictor = self._load_predictor()
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

    # Backward-compatible entrypoint used by predict.py
    def predict(self) -> Path:
        return self.run()

    def can_compute_metrics(self) -> bool:
        if not self.do_evaluate or self.data is None:
            return False
        return self.target_col in self.data.columns

    def evaluate(self) -> dict:
        if self.data is None or self.predictor is None:
            raise RuntimeError("Call run() before evaluate().")
        if self.target_col not in self.data.columns:
            raise ValueError(
                f"Cannot evaluate without `{self.target_col}` in inference dataset."
            )
        metrics = self.predictor.evaluate(self.data)
        logging.info("Metrics (markdown):")
        log_markdown_table(metrics)
        return metrics

    def _load_dataset(self) -> pd.DataFrame:
        delimiter = detect_csv_delimiter(self.dataset_path)
        dataset = pd.read_csv(self.dataset_path, delimiter=delimiter).copy()

        if self.smiles_col not in dataset.columns:
            raw_smiles_col = DataInterface.get_smiles_col_in_raw(dataset)
            dataset = dataset.rename(columns={raw_smiles_col: self.smiles_col})

        if self.target_col not in dataset.columns:
            try:
                raw_target_col = DataInterface.get_label_col_in_raw(dataset)
                dataset = dataset.rename(columns={raw_target_col: self.target_col})
            except ValueError:
                pass

        if self.source_col not in dataset.columns:
            # Keep prediction input schema consistent with training/evaluation paths.
            dataset[self.source_col] = self.dataset_path.stem
        return dataset

    def _load_predictor(self) -> PredictorBase:
        predictor = self.data_interface.unpickle_model(
            model_cache_key=self.model_cache_key,
            data_cache_key=self.data_cache_key,
        )
        if not isinstance(predictor, PredictorBase):
            raise TypeError(
                f"Loaded object is not PredictorBase: {type(predictor).__name__}"
            )
        self.label_name = predictor.task
        predictor.inject_smiles_col_ID(self.smiles_col)
        predictor.inject_source_col_ID(self.source_col)
        predictor.inject_target_col_ID(self.target_col)
        return predictor

    def _align_source_labels(
        self, df: pd.DataFrame, predictor: PredictorBase
    ) -> pd.DataFrame:
        """Align source labels to match predictor endpoint map for multi-endpoint models."""
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
        preds = predictor.predict(df)
        if len(preds) != len(df):
            raise RuntimeError(
                f"Prediction length mismatch: {len(preds)} predictions for {len(df)} rows"
            )

        # Persist a narrow schema for inference artifacts.
        df[self.label_name] = preds
        if self.task_setting in {"binary_classification", "multi_class_classification"}:
            if not isinstance(predictor, BinaryClassifierBase):
                raise TypeError(
                    "Classification task requires predictor to implement `classify`."
                )
            classifier = cast(BinaryClassifierBase, predictor)
            df[f"{self.label_name}_class"] = classifier.classify(preds)
        return df

    def _save_predictions(self, results: pd.DataFrame) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.predictions_path, index=False)

    def _save_metrics(self, metrics: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w") as fh:
            yaml.safe_dump({"metrics": metrics}, fh)

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
        logging.info(f"* Task setting: {self.task_setting}")
        logging.info(f"* Output dir: {self.out_dir}")

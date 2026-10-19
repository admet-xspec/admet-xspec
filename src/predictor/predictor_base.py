import abc
import logging
from collections import defaultdict
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.data.featurizer import FeaturizerBase
from src.utils import get_metric_callable


HyperParams = dict[str, Any]
MetricsDict = dict[str, Any]
CvFolds = Iterator[tuple[np.ndarray, np.ndarray]]


class PredictorBase(abc.ABC):
    """Shared predictor API used by training and inference pipelines."""

    def __init__(
        self,
        random_state: int = 42,
        task: str | None = None,
        do_attributed_learning: bool = False,
    ) -> None:
        self.featurizer: FeaturizerBase | None = None
        self.random_state = random_state
        self.smiles_col = "smiles"
        self.source_col = "source"
        self.target_col = "y"
        self.task_name = task
        self.do_attributed_learning = do_attributed_learning
        self.endpoint_ohe_map: dict[Any, np.ndarray] | None = None
        # Number of native threads the model may use; None means the library default.
        self.n_threads: int | None = None
        # Set random seed for reproducibility
        np.random.seed(random_state)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the predictor."""
        raise NotImplementedError

    @property
    def task(self) -> str | None:
        """Semantic task/endpoint name associated with this model instance."""
        return self.task_name

    @property
    def is_attributed(self) -> bool:
        return self.do_attributed_learning

    @property
    def supports_threading(self) -> bool:
        """Whether `set_n_threads` actually controls the model's parallelism."""
        return False

    def set_n_threads(self, n_threads: int | None) -> None:
        """Limit the number of native threads used for training/prediction.

        No-op by default; predictors backed by multithreaded libraries override this.
        """
        self.n_threads = n_threads

    @abc.abstractmethod
    def get_hyperparameters(self) -> HyperParams:
        """Return current model hyperparameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        """Inject hyperparameters into the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict the target values for the given dataset.
        Returns a series of floats - either regression values or class probabilities.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train the model with set hyperparameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, df: pd.DataFrame) -> MetricsDict:
        """
        Evaluate the model and return task-appropriate metrics.

        Implementations may return plain floats or richer values that are serializable.
        """
        raise NotImplementedError

    def get_featurizer(self) -> FeaturizerBase | None:
        """Return the featurizer if set."""
        return self.featurizer if self.featurizer else None

    def set_featurizer(self, featurizer: FeaturizerBase) -> None:
        """Inject featurizer into the model."""
        if not isinstance(featurizer, FeaturizerBase):
            raise TypeError("Featurizer must be an instance of FeaturizerBase")
        self.featurizer = featurizer

    def set_task_name(self, name: str | None) -> None:
        """Set semantic task/endpoint name used in cache keys and logs."""
        self.task_name = name

    def get_cache_key(self) -> str:
        """Return a unique cache key for predictor + featurizer configuration.

        Hash is based on:
        - Predictor name
        - Featurizer name and its parameters (if any)
        Does not include model hyperparameters.
        """
        featurizer_key = (
            self.featurizer.get_cache_key() if self.featurizer else "nofeaturizer"
        )
        key = f"{self.task_name}_{self.name}_{featurizer_key}"
        if self.is_attributed:
            key = "attributed_" + key
        return key

    def get_cache_key_for_hyperparameter_parsing(self) -> str:
        # TODO: clean this mess
        if self.is_attributed:
            return self.get_cache_key().replace("attributed_", "")
        else:
            return self.get_cache_key()

    def cross_validate(self, df: pd.DataFrame, n_folds: int = 5) -> MetricsDict:
        """Run k-fold cross-validation and average the resulting metrics.

        Folds come from `_get_cv_folds`: plain shuffled KFold here, stratified for
        classifiers (see `BinaryClassifierBase`). Leaves the predictor trained on the
        last fold's training part.
        """
        if n_folds < 2:
            raise ValueError("`n_folds` must be at least 2 for cross-validation.")
        if len(df) < n_folds:
            raise ValueError(
                f"Not enough rows for {n_folds}-fold CV: got {len(df)} rows."
            )
        df = df.reset_index(drop=True)

        # Attributed models: build the endpoint map from every source up front. Otherwise
        # it would be built from the first fold's training part, and a source missing
        # there would make evaluation of later folds fail.
        if self.is_attributed and self.endpoint_ohe_map is None:
            self._require_columns(df, [self.source_col])
            self._create_endpoint_map(df[self.source_col])

        metrics_dicts: list[MetricsDict] = []
        for train_index, test_index in self._get_cv_folds(df, n_folds):
            self.train(df.iloc[train_index])
            metrics_dicts.append(self.evaluate(df.iloc[test_index]))
        return self._average_metrics(metrics_dicts)

    def _get_cv_folds(self, df: pd.DataFrame, n_folds: int) -> CvFolds:
        """Yield (train, test) positional indices for each CV fold."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        return kf.split(df)

    def set_column_ids(
        self,
        smiles_col: str | None = None,
        source_col: str | None = None,
        target_col: str | None = None,
    ) -> None:
        """Set dataframe column identifiers used by predictor methods."""
        if smiles_col is not None:
            self.smiles_col = smiles_col
        if source_col is not None:
            self.source_col = source_col
        if target_col is not None:
            self.target_col = target_col

    def get_endpoint_OHE_map(self) -> dict[Any, np.ndarray] | None:
        """Return endpoint one-hot map (legacy method name kept for compatibility)."""
        return self.endpoint_ohe_map if self.endpoint_ohe_map else None

    def _create_endpoint_map(self, endpoints: pd.Series) -> None:
        """Create endpoint -> one-hot mapping used for attributed predictors."""
        endpoint_map: dict[Any, np.ndarray] = {}
        unique_endpoints = sorted(endpoints.unique())
        for i, endpoint in enumerate(unique_endpoints):
            ohe = np.zeros(len(unique_endpoints), dtype=np.float32)
            ohe[i] = 1.0
            endpoint_map[endpoint] = ohe
        self.endpoint_ohe_map = endpoint_map

    @staticmethod
    def _average_metrics(metrics_dicts: list[MetricsDict]) -> MetricsDict:
        """Average each metric across CV folds."""
        grouped: dict[str, list[Any]] = defaultdict(list)
        for fold_metrics in metrics_dicts:
            for metric_name, metric_value in fold_metrics.items():
                grouped[metric_name].append(metric_value)
        return {metric_name: np.mean(values) for metric_name, values in grouped.items()}

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        """Fail fast with a clear message when input schema is incomplete."""
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class BinaryClassifierBase(PredictorBase, abc.ABC):
    """
    Base class for binary classification predictors. Implements common evaluation metrics
    and classification thresholding.
    """

    evaluation_metrics = ["accuracy", "roc_auc", "f1", "precision", "recall", "pr_auc"]

    # When the data mixes several sources (e.g. human + animal endpoints), stratify CV on
    # (source, class) pairs, so every fold keeps each source's share and class balance.
    stratify_cv_by_source: bool = True

    def evaluate(self, df: pd.DataFrame) -> MetricsDict:
        """Evaluate binary predictions using a fixed metric set."""
        self._require_columns(df, [self.target_col])
        y_true = self._as_binary_labels(df[self.target_col])
        preds = self.predict(df)
        binary_preds = self.classify(preds)
        metrics_dict: MetricsDict = {}
        for m in self.evaluation_metrics:
            if m in ["roc_auc", "pr_auc"]:
                # roc_auc needs class probabilities
                metrics_dict[m] = get_metric_callable(m)(y_true, preds)
            else:
                metrics_dict[m] = get_metric_callable(m)(y_true, binary_preds)
        return metrics_dict

    def _get_cv_folds(self, df: pd.DataFrame, n_folds: int) -> CvFolds:
        """Stratified folds, so every fold holds both classes in their overall ratio."""
        self._require_columns(df, [self.target_col])
        y = self._as_binary_labels(df[self.target_col])
        class_counts = np.bincount(y, minlength=2)
        if class_counts.min() < n_folds:
            # Some test fold would contain no sample of the minority class, which makes
            # roc_auc/pr_auc undefined and precision/recall meaningless.
            raise ValueError(
                f"Stratified {n_folds}-fold CV needs at least {n_folds} samples of each "
                f"class, got {{0: {class_counts[0]}, 1: {class_counts[1]}}}. "
                f"Lower `n_optim_cv_folds` or check the labels."
            )
        strata = self._get_cv_strata(df, y, n_folds)
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )
        return skf.split(np.zeros(len(df)), strata)

    def _get_cv_strata(
        self, df: pd.DataFrame, y: np.ndarray, n_folds: int
    ) -> np.ndarray:
        """Labels to stratify on: (source, class) pairs if possible, else the class."""
        if (
            not self.stratify_cv_by_source
            or self.source_col not in df.columns
            or df[self.source_col].nunique() < 2
        ):
            return y
        joint = (
            df[self.source_col].astype(str).to_numpy().astype(object)
            + "|"
            + y.astype(str).astype(object)
        )
        _, group_counts = np.unique(joint, return_counts=True)
        if group_counts.min() < n_folds:
            logging.warning(
                f"Some (source, class) groups have fewer than {n_folds} samples; "
                f"stratifying CV folds on the class only."
            )
            return y
        return joint

    def _as_binary_labels(self, y: pd.Series) -> np.ndarray:
        """Return labels as a 0/1 int array, or fail with a clear message.

        Also accepts 0/1 stored with object dtype (e.g. after concatenating with an
        empty, untyped DataFrame), which sklearn would reject as an "unknown" target.
        """
        try:
            y_num = pd.to_numeric(pd.Series(y), errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Target column `{self.target_col}` is not numeric: {exc}"
            ) from exc
        if y_num.isna().any():
            raise ValueError(
                f"Target column `{self.target_col}` contains {int(y_num.isna().sum())} NaN(s)."
            )
        unexpected = sorted(set(np.unique(y_num)) - {0, 1})
        if unexpected:
            raise ValueError(
                f"Expected binary 0/1 labels in `{self.target_col}`, "
                f"also found: {unexpected[:10]}"
            )
        return y_num.to_numpy().astype(int)

    def classify(self, preds: Any) -> np.ndarray:
        """Convert model scores/probabilities into hard labels using threshold."""
        return np.array(preds) > self.class_threshold

    @property
    def class_threshold(self) -> float:
        """Return the classification threshold value. Default is 0.5."""
        return 0.5


class RegressorBase(PredictorBase, abc.ABC):
    """
    Base class for regression predictors. Implements common evaluation metrics.
    """

    evaluation_metrics = ["mse", "rmse", "mae", "r2", "spearman"]

    def evaluate(self, df: pd.DataFrame) -> MetricsDict:
        """Evaluate regression predictions using a fixed metric set."""
        self._require_columns(df, [self.target_col])
        preds = self.predict(df)
        return {
            m: get_metric_callable(m)(df[self.target_col], preds)
            for m in self.evaluation_metrics
        }
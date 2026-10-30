"""Applicability domain: how close molecules are to the data a model was trained on.

A domain depends only on the training data, not on the model, so one fitted domain
is stored per split and shared by every model trained on it. Each model gets its own
calibration report: its test-set metrics inside and outside the domain.
"""

import abc
import copy
import hashlib
import logging
import pickle
import warnings
from typing import Hashable, List, Optional, Tuple

import gin
import numpy as np
import pandas as pd

from src.data.data_interface import DataInterface
from src.data.featurizer import FeaturizerBase
from src.predictor.predictor_base import PredictorBase

# Subsets smaller than this get no metrics; most metrics are undefined below it.
_MIN_SUBSET_SIZE = 2


class ApplicabilityDomainBase(abc.ABC):
    """Base class for applicability domains, fitted on a training set.

    Args:
        reference_sources: Sources (the `source` column) whose molecules form the
            reference set, e.g. only the human data of an augmented training set.
            None uses every training molecule.
    """

    score_col = "ad_similarity"
    in_domain_col = "ad_in_domain"
    nearest_col = "ad_nearest_smiles"

    def __init__(self, reference_sources: Optional[List[str]] = None) -> None:
        self.reference_sources = (
            sorted(reference_sources) if reference_sources is not None else None
        )
        self.threshold_: Optional[float] = None

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable domain name."""
        pass

    @abc.abstractmethod
    def get_hashable_params_values(self) -> List[Hashable]:
        """Return parameters for hashing/caching purposes."""
        pass

    def get_cache_key(self) -> str:
        """Return `<name>_<5-char hash of params>`."""
        params = str(self.get_hashable_params_values()).encode("utf-8")
        return f"{self.name}_{hashlib.md5(params).hexdigest()[:5]}"

    @property
    def is_fitted(self) -> bool:
        return self.threshold_ is not None

    @abc.abstractmethod
    def fit(self, smiles: List[str]) -> "ApplicabilityDomainBase":
        """Fit the domain on the reference SMILES and return self."""
        pass

    @abc.abstractmethod
    def score(self, smiles: List[str]) -> pd.DataFrame:
        """Return `score_col`, `in_domain_col` and `nearest_col`, one row per SMILES."""
        pass

    def get_reference_smiles(
        self, df: pd.DataFrame, smiles_col: str = "smiles", source_col: str = "source"
    ) -> List[str]:
        """Return the SMILES of `df` that form the reference set."""
        if self.reference_sources is not None:
            df = df[df[source_col].isin(self.reference_sources)]
            if df.empty:
                raise ValueError(
                    f"No training molecules from `reference_sources` {self.reference_sources}."
                )
        return df[smiles_col].tolist()


@gin.configurable
class TanimotoKnnDomain(ApplicabilityDomainBase):
    """Mean Tanimoto similarity to the `k` most similar reference molecules.

    A molecule is in the domain if its score is at least the `threshold_percentile`-th
    percentile of the leave-one-out scores of the reference molecules themselves,
    so by default 95% of the training set is in the domain.

    Args:
        featurizer: Binary fingerprint featurizer; its disk cache is not used.
        k: Number of nearest neighbours averaged.
        threshold_percentile: Percentile of the reference scores used as threshold.
        reference_sources: See `ApplicabilityDomainBase`.
        chunk_size: Rows per block of the similarity matrix; affects memory only.
    """

    def __init__(
        self,
        featurizer: FeaturizerBase,
        k: int = 5,
        threshold_percentile: float = 5.0,
        reference_sources: Optional[List[str]] = None,
        chunk_size: int = 2048,
    ) -> None:
        super().__init__(reference_sources)
        if k < 1:
            raise ValueError(f"`k` must be at least 1. Got: {k}")
        if not 0.0 <= threshold_percentile <= 100.0:
            raise ValueError(
                f"`threshold_percentile` must be in [0, 100]. Got: {threshold_percentile}"
            )
        # A pickle round trip drops the featurizer's cache, which may be shared with
        # the similarity filter. New molecules rarely repeat, so caching them would
        # only grow the cache file.
        self.featurizer = pickle.loads(pickle.dumps(featurizer))
        self.featurizer.cache_enabled = False
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return "tanimoto_knn_domain"

    def get_hashable_params_values(self) -> List[Hashable]:
        return [
            self.featurizer.get_cache_key(),
            self.k,
            self.threshold_percentile,
            self.reference_sources,
        ]

    def fit(self, smiles: List[str]) -> "TanimotoKnnDomain":
        """Store the reference fingerprints and set the threshold from their LOO scores."""
        unique_smiles = list(dict.fromkeys(smiles))
        if len(unique_smiles) < 2:
            raise ValueError(
                f"{self.name}: needs at least 2 unique reference molecules, got {len(unique_smiles)}."
            )
        bits = self._fingerprints(unique_smiles)
        self.n_bits_ = bits.shape[1]
        self.reference_smiles_ = np.array(unique_smiles, dtype=object)
        self.reference_bits_ = np.packbits(bits, axis=1)
        self.reference_counts_ = bits.sum(axis=1).astype(np.float32)
        # Leave-one-out leaves n - 1 neighbours per reference molecule.
        self.k_ = min(self.k, len(unique_smiles) - 1)
        if self.k_ < self.k:
            logging.warning(
                f"{self.name}: only {len(unique_smiles)} reference molecules; using k={self.k_}."
            )

        top_similarities, _ = self._knn(bits, exclude_self=True)
        loo_scores = top_similarities.mean(axis=1)
        self.threshold_ = float(np.percentile(loo_scores, self.threshold_percentile))
        self.reference_score_quantiles_ = {
            f"p{q}": float(np.percentile(loo_scores, q)) for q in (5, 25, 50, 75, 95)
        }
        logging.info(
            f"{self.name}: fitted on {len(unique_smiles)} molecules, "
            f"threshold {self.threshold_:.3f} (percentile {self.threshold_percentile:g})."
        )
        return self

    def score(self, smiles: List[str]) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before score().")
        if len(smiles) == 0:
            return pd.DataFrame(
                {
                    self.score_col: pd.Series(dtype=float),
                    self.in_domain_col: pd.Series(dtype=bool),
                    self.nearest_col: pd.Series(dtype=object),
                }
            )
        bits = self._fingerprints(list(smiles))
        if bits.shape[1] != self.n_bits_:
            raise ValueError(
                f"{self.name}: fingerprints have {bits.shape[1]} bits, the reference {self.n_bits_}."
            )
        top_similarities, top_indices = self._knn(bits, exclude_self=False)
        scores = top_similarities.mean(axis=1)
        return pd.DataFrame(
            {
                self.score_col: scores.astype(float),
                self.in_domain_col: scores >= self.threshold_,
                self.nearest_col: self.reference_smiles_[top_indices[:, 0]],
            }
        )

    def _fingerprints(self, smiles: List[str]) -> np.ndarray:
        """Return binary fingerprints as a boolean matrix."""
        return np.asarray(self.featurizer.featurize(smiles)) > 0

    def _knn(
        self, query_bits: np.ndarray, exclude_self: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the `k_` highest similarities to the reference and their indices.

        Both are sorted by decreasing similarity. With `exclude_self`, query `i` is
        reference molecule `i`, which is skipped.
        """
        k, step = self.k_, self.chunk_size
        n_reference = len(self.reference_bits_)
        all_similarities, all_indices = [], []
        for q_start in range(0, len(query_bits), step):
            query = query_bits[q_start : q_start + step].astype(np.float32)
            query_counts = query.sum(axis=1)
            best = np.full((len(query), 0), -np.inf, dtype=np.float32)
            best_indices = np.zeros((len(query), 0), dtype=np.int64)
            for r_start in range(0, n_reference, step):
                reference = np.unpackbits(
                    self.reference_bits_[r_start : r_start + step],
                    axis=1,
                    count=self.n_bits_,
                ).astype(np.float32)
                intersection = query @ reference.T
                union = (
                    query_counts[:, None]
                    + self.reference_counts_[r_start : r_start + step][None, :]
                    - intersection
                )
                similarities = np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 0,
                )
                if exclude_self:
                    rows = np.arange(len(query))
                    cols = rows + q_start - r_start
                    inside = (cols >= 0) & (cols < len(reference))
                    similarities[rows[inside], cols[inside]] = -np.inf

                candidates = np.hstack([best, similarities])
                candidate_indices = np.hstack(
                    [
                        best_indices,
                        np.broadcast_to(
                            np.arange(r_start, r_start + len(reference)),
                            similarities.shape,
                        ),
                    ]
                )
                n_keep = min(k, candidates.shape[1])
                keep = np.argpartition(-candidates, n_keep - 1, axis=1)[:, :n_keep]
                best = np.take_along_axis(candidates, keep, axis=1)
                best_indices = np.take_along_axis(candidate_indices, keep, axis=1)

            order = np.argsort(-best, axis=1, kind="stable")
            all_similarities.append(np.take_along_axis(best, order, axis=1))
            all_indices.append(np.take_along_axis(best_indices, order, axis=1))
        return np.vstack(all_similarities), np.vstack(all_indices)


def subset_metrics(
    predictor: PredictorBase, y_true: np.ndarray, preds: np.ndarray, mask: np.ndarray
) -> dict:
    """Return `{"n": ..., "metrics": ...}` of the predictions selected by `mask`.

    Undefined metrics (e.g. ROC-AUC of a subset with one class) are NaN. Metrics are
    None for subsets that are too small or where a metric raises; `note` says why.
    """
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n < _MIN_SUBSET_SIZE:
        return {"n": n, "metrics": None, "note": "too few molecules"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = predictor.evaluate_predictions(y_true[mask], preds[mask])
    except ValueError as exc:
        return {"n": n, "metrics": None, "note": str(exc)}
    return {"n": n, "metrics": {name: float(value) for name, value in metrics.items()}}


def calibrate_domain(
    domain: ApplicabilityDomainBase,
    predictor: PredictorBase,
    test_df: pd.DataFrame,
    n_bins: int = 4,
) -> dict:
    """Compare the test metrics of `predictor` inside and outside the fitted `domain`.

    Also reports metrics per quantile bin of the domain score, which shows whether
    errors grow as molecules get further from the training data.
    """
    if not hasattr(predictor, "evaluate_predictions"):
        raise TypeError(
            f"{type(predictor).__name__} does not support `evaluate_predictions`."
        )
    y_true = test_df[predictor.target_col].to_numpy()
    preds = np.asarray(predictor.predict(test_df))
    scores = domain.score(test_df[predictor.smiles_col].tolist())
    similarity = scores[domain.score_col].to_numpy()
    in_domain = scores[domain.in_domain_col].to_numpy()

    bins = []
    if len(test_df) >= n_bins:
        quantile_bins = pd.qcut(similarity, n_bins, duplicates="drop")
        for interval in quantile_bins.categories:
            mask = np.asarray(quantile_bins == interval)
            bins.append(
                {
                    "similarity_range": [
                        float(similarity[mask].min()),
                        float(similarity[mask].max()),
                    ],
                    **subset_metrics(predictor, y_true, preds, mask),
                }
            )

    return {
        "domain": domain.get_cache_key(),
        "threshold": float(domain.threshold_),
        "n_test": len(test_df),
        "fraction_in_domain": float(in_domain.mean()) if len(in_domain) else None,
        "in_domain": subset_metrics(predictor, y_true, preds, in_domain),
        "out_of_domain": subset_metrics(predictor, y_true, preds, ~in_domain),
        "similarity_bins": bins,
    }


def log_calibration(calibration: dict) -> None:
    """Log the in-domain vs out-of-domain test metrics of a calibration report."""
    logging.info(
        f"* Test set in domain: {calibration['in_domain']['n']} / {calibration['n_test']} "
        f"(threshold {calibration['threshold']:.3f})"
    )
    for part in ("in_domain", "out_of_domain"):
        report = calibration[part]
        if report["metrics"] is None:
            logging.info(f"* {part} (n={report['n']}): no metrics, {report['note']}")
        else:
            metrics = ", ".join(f"{k}={v:.3f}" for k, v in report["metrics"].items())
            logging.info(f"* {part} (n={report['n']}): {metrics}")


def get_or_fit_domain(
    domain: ApplicabilityDomainBase,
    data_interface: DataInterface,
    data_cache_key: str,
    refit: bool = False,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    smiles_col: str = "smiles",
    source_col: str = "source",
) -> ApplicabilityDomainBase:
    """Load the fitted domain of a split, fitting and saving it first if missing.

    The domain of `model.pkl` is fitted on the train split; with `refit`, the one of
    `model_final_refit.pkl` on train + test. The split is loaded from the cache unless
    `train_df` and `test_df` are given.
    """
    domain_key = domain.get_cache_key()
    with data_interface.domain_lock(data_cache_key, domain_key):
        exists = data_interface.check_domain_exists(data_cache_key, domain_key, refit)
        if exists and not data_interface.override_cache:
            return data_interface.load_domain(data_cache_key, domain_key, refit)

        if train_df is None or test_df is None:
            train_df, test_df = data_interface.load_train_test_split(data_cache_key)
        reference_df = (
            pd.concat([train_df, test_df], ignore_index=True) if refit else train_df
        )
        logging.info(
            f"Fitting {'refit ' if refit else ''}applicability domain `{domain_key}` "
            f"of split `{data_cache_key}`"
        )
        fitted = copy.deepcopy(domain).fit(
            domain.get_reference_smiles(reference_df, smiles_col, source_col)
        )
        data_interface.save_domain(fitted, data_cache_key, refit)
        return fitted


def fit_domain_for_model(
    domain: ApplicabilityDomainBase,
    data_interface: DataInterface,
    model_cache_key: str,
    data_cache_key: str,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    predictor: Optional[PredictorBase] = None,
) -> Optional[dict]:
    """Fit the domains of a model's split and calibrate them against the model.

    Works for models trained earlier: the split is loaded from the cache and the
    predictor from `model.pkl`. Returns the calibration report, or None if there is
    no `model.pkl` to calibrate against (the domains are fitted anyway).
    """
    if train_df is None or test_df is None:
        train_df, test_df = data_interface.load_train_test_split(data_cache_key)
    if predictor is None and data_interface.check_model_exists(
        model_cache_key, data_cache_key
    ):
        predictor = data_interface.unpickle_model(model_cache_key, data_cache_key)
    columns = {}
    if predictor is not None:
        columns = {
            "smiles_col": predictor.smiles_col,
            "source_col": predictor.source_col,
        }

    fitted = get_or_fit_domain(
        domain, data_interface, data_cache_key, False, train_df, test_df, **columns
    )
    get_or_fit_domain(
        domain, data_interface, data_cache_key, True, train_df, test_df, **columns
    )

    if predictor is None:
        logging.info(
            f"No `model.pkl` for `{model_cache_key}/{data_cache_key}`; "
            "applicability domain fitted but not calibrated."
        )
        return None
    calibration = calibrate_domain(fitted, predictor, test_df)
    data_interface.save_domain_calibration(
        calibration, model_cache_key, data_cache_key, fitted.get_cache_key()
    )
    log_calibration(calibration)
    return calibration

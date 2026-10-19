import abc
import hashlib
import logging
from typing import Hashable, Tuple

import gin
import pandas as pd

from src.data.featurizer import FeaturizerBase
from src.data.utils import TanimotoCalculator


class SimilarityFilterBase(abc.ABC):
    """Base class for similarity-based filtering of augmentation data.

    `against` selects the reference set: "test" (test split only) or
    "test_origin" (train + test).
    """

    def __init__(self, against: str) -> None:
        if against not in ["test", "test_origin"]:
            raise ValueError(
                f"Arg `against` must be 'test' or 'test_origin'. Got: {against}"
            )
        self.against = against

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""
        pass

    @abc.abstractmethod
    def get_hashable_params_values(self) -> list[str]:
        """Return parameters for hashing/caching purposes."""
        pass

    def get_cache_key(self) -> str:
        """Return `<name>_<5-char hash of params>`."""
        params = str(self.get_hashable_params_values()).encode("utf-8")
        return f"{self.name}_{hashlib.md5(params).hexdigest()[:5]}"

    @abc.abstractmethod
    def get_filtered_df(
        self, to_filter_df: pd.DataFrame, filter_against_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Return a copy of `to_filter_df` filtered by similarity to `filter_against_df`."""
        pass

    def get_filtered_train_test(
        self,
        augmenting_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter `augmenting_df` and return `(train + filtered augmentation, test)`."""
        if augmenting_df is None or augmenting_df.empty:
            return train_df, test_df

        if self.against == "test":
            filter_against = test_df
        else:
            filter_against = pd.concat([train_df, test_df], ignore_index=True)

        logging.info(
            f"Filtering {augmenting_df.shape[0]} molecules based on similarity to {filter_against.shape[0]} molecules."
        )
        filtered_aug = self.get_filtered_df(augmenting_df, filter_against)
        combined_train = pd.concat([train_df, filtered_aug], ignore_index=True)
        return combined_train, test_df.copy()


@gin.configurable
class TanimotoFilter(SimilarityFilterBase):
    """Drop molecules too close (in Tanimoto distance) to the reference set.

    Args:
        featurizer: Fingerprint featurizer.
        min_distance_to_test_post_filtering: Minimum Tanimoto distance to keep a
            molecule, in [0, 1].
        against: "test" or "test_origin".
    """

    def __init__(
        self,
        featurizer: FeaturizerBase,
        min_distance_to_test_post_filtering: float,
        against: str = "test",
    ) -> None:
        super().__init__(against)
        if not 0.0 <= min_distance_to_test_post_filtering <= 1.0:
            raise ValueError(
                "`min_distance_to_test_post_filtering` must be between 0.0 and 1.0"
            )
        self.featurizer = featurizer
        self.min_distance_to_test_post_filtering = min_distance_to_test_post_filtering

    def _format_distance_threshold(self) -> str:
        """Format the threshold as a percentage string (e.g. 0.1 -> '10p')."""
        percent = int(round(100.0 * self.min_distance_to_test_post_filtering))
        return f"{percent}p"

    @property
    def name(self) -> str:
        return f"tanimoto_{self._format_distance_threshold()}_filter"

    def get_hashable_params_values(self) -> list[Hashable]:
        return [
            self.featurizer.get_cache_key(),
            self.min_distance_to_test_post_filtering,
            self.against,
        ]

    def get_filtered_df(
        self, to_filter_df: pd.DataFrame, filter_against_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Keep molecules whose min distance to the reference set exceeds the threshold."""
        if to_filter_df is None or to_filter_df.empty:
            return pd.DataFrame(
                columns=to_filter_df.columns if to_filter_df is not None else ["smiles"]
            )
        if filter_against_df is None or filter_against_df.empty:
            return to_filter_df.copy().reset_index(drop=True)

        for df, name in [
            (to_filter_df, "to_filter_df"),
            (filter_against_df, "filter_against_df"),
        ]:
            if "smiles" not in df.columns:
                raise KeyError(f"`{name}` must contain a 'smiles' column")

        calculator = TanimotoCalculator(
            smiles_list=filter_against_df["smiles"].tolist(), featurizer=self.featurizer
        )
        results = calculator.run_batch(queries=to_filter_df["smiles"].tolist())
        min_distances = results.get("min_distance")
        if min_distances is None:
            raise RuntimeError(
                "TanimotoCalculator.run_batch did not return 'min_distance'"
            )

        return (
            to_filter_df.copy()
            .assign(min_distance=pd.Series(min_distances, index=to_filter_df.index))
            .query("min_distance > @self.min_distance_to_test_post_filtering")
            .drop(columns=["min_distance"])
            .reset_index(drop=True)
        )

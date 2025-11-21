import abc
from typing import Hashable, Tuple

import gin
import pandas as pd

from src.data.utils import TanimotoCalculator
from src.data.featurizer import FeaturizerBase


class SimilarityFilterBase(abc.ABC):
    """
    Base class for similarity-based filtering of molecular data.

    Implements two strategies controlled by the 'against' parameter:
      - "test": filters augmentation molecules against the test set
      - "test_origin": filters augmentation molecules against test origin (train + test)
    """

    def __init__(self, against: str) -> None:
        if against not in ["test", "test_origin"]:
            raise ValueError(f"Arg `against` must be 'test' or 'test_origin'. Got: {against}")
        self.against = against

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""
        pass

    @abc.abstractmethod
    def get_hashable_params_values(self) -> list[str]:
        pass

    def get_cache_key(self):
        """
        Generate a 5-character cache key based on the splitter's parameters.
        """
        return f"{self.name}_{abs(hash(frozenset(self.get_hashable_params_values()))) % (10 ** 5):05d}"

    @abc.abstractmethod
    def get_filtered_df(
        self, to_filter_df: pd.DataFrame, filter_against_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter `to_filter_df` based on similarity to `filter_against_df`.
        Returns a filtered copy.
        """
        pass

    def get_filtered_train_test(
        self,
        augmenting_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply the configured filtering strategy and return (filtered_train, test).

        Args:
            train_df: Training split
            test_df: Test split
            nosplit_df: Augmentation data (no split applied)

        Returns:
            Tuple of (combined training data with filtered augmentation, test data)
        """
        # Early exit if no augmentation data
        if augmenting_df is None or augmenting_df.empty:
            return train_df.copy(), test_df.copy()

        # Select reference set based on strategy
        if self.against == "test":
            filter_against = test_df
        else:
            filter_against = pd.concat([train_df, test_df], ignore_index=True)

        # Filter augmentation data
        filtered_aug = self.get_filtered_df(augmenting_df.copy(), filter_against.copy())

        # Combine train with filtered augmentation
        combined_train = pd.concat([train_df.copy(), filtered_aug], ignore_index=True)

        return combined_train, test_df.copy()


@gin.configurable
class TanimotoFilter(SimilarityFilterBase):
    """
    Filters molecules using Tanimoto similarity distance.

    Args:
        featurizer: Molecular featurizer for fingerprint generation
        min_distance_to_test_post_filtering: Minimum Tanimoto distance threshold (0.0-1.0)
        against: Filtering strategy - "test" or "whole"
    """

    def __init__(
            self,
            featurizer: FeaturizerBase,
            min_distance_to_test_post_filtering: float,
            against: str = "test",
    ) -> None:
        """Initialize Tanimoto filter with validation."""
        super().__init__(against)

        if not 0.0 <= min_distance_to_test_post_filtering <= 1.0:
            raise ValueError("`min_distance_to_test_post_filtering` must be between 0.0 and 1.0")

        self.featurizer = featurizer
        self.min_distance_to_test_post_filtering = min_distance_to_test_post_filtering

    def _format_distance_threshold(self) -> str:
        """Convert distance threshold to percentage string (e.g., 0.1 -> '10p')."""
        percent = int(round(100.0 * self.min_distance_to_test_post_filtering))
        return f"{percent}p"

    @property
    def name(self) -> str:
        """Generate descriptive filter name with threshold."""
        return f"tanimoto_{self._format_distance_threshold()}_filter"

    def get_hashable_params_values(self) -> list[Hashable]:
        """Return hashable parameter values for this filter."""
        return [
            self.featurizer.name,
            self.min_distance_to_test_post_filtering,
            self.against,
        ]

    def get_filtered_df(
            self, to_filter_df: pd.DataFrame, filter_against_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter molecules based on minimum Tanimoto distance to reference set.

        Keeps only molecules where the minimum distance to any reference molecule
        exceeds the configured threshold.
        """
        # Early exits
        if to_filter_df is None or to_filter_df.empty:
            return pd.DataFrame(columns=to_filter_df.columns if to_filter_df is not None else ["smiles"])

        if filter_against_df is None or filter_against_df.empty:
            return to_filter_df.copy().reset_index(drop=True)

        # Validate required columns
        required_col = "smiles"
        for df, name in [(to_filter_df, "to_filter_df"), (filter_against_df, "filter_against_df")]:
            if required_col not in df.columns:
                raise KeyError(f"`{name}` must contain a '{required_col}' column")

        # Compute distances using batch calculator
        calculator = TanimotoCalculator(
            smiles_list=filter_against_df["smiles"].tolist(),
            featurizer=self.featurizer
        )
        results = calculator.run_batch(queries=to_filter_df["smiles"].tolist())

        # Apply threshold filter
        min_distances = results.get("min_distance")
        if min_distances is None:
            raise RuntimeError("TanimotoCalculator.run_batch did not return 'min_distance'")

        filtered_df = (
            to_filter_df.copy()
            .assign(min_distance=pd.Series(min_distances, index=to_filter_df.index))
            .query("min_distance > @self.min_distance_to_test_post_filtering")
            .drop(columns=["min_distance"])
            .reset_index(drop=True)
        )

        return filtered_df
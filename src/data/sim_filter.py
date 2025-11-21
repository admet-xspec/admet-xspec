import abc
import gin
import pandas as pd

from src.data.utils import TanimotoCalculator
from src.data.featurizer import FeaturizerBase


class SimilarityFilterBase(abc.ABC):
    """Filtering class."""

    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the reducer."""
        pass

    @abc.abstractmethod
    def get_filtered_df(
        self, to_filter_df: pd.DataFrame, filter_against_df: pd.DataFrame
    ) -> pd.DataFrame:
        pass


@gin.configurable
class TanimotoFilter(FilterBase):

    def __init__(
        self,
        featurizer: FeaturizerBase,
        min_distance_to_test_post_filtering: float,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.min_distance_to_test_post_filtering = min_distance_to_test_post_filtering

    def _distance_to_str(self):
        # eg. 0.1 => 10.0 => 10 + p => 10p
        return str(100 * self.min_distance_to_test_post_filtering).split(".")[0] + "p"

    @property
    def name(self) -> str:
        return f"tanimoto_{self._distance_to_str()}_filter"

    def get_filtered_df(
        self, to_filter_df: pd.DataFrame, filter_against_df: pd.DataFrame
    ) -> pd.DataFrame:

        df = to_filter_df.copy()
        tc = TanimotoCalculator(
            smiles_list=filter_against_df["smiles"].tolist(), featurizer=self.featurizer
        )
        results = tc.run_batch(queries=df["smiles"].tolist())
        df["min_distance"] = results["min_distance"]
        df = df[df["min_distance"] > self.min_distance_to_test_post_filtering]
        df.drop(columns=["min_distance"], inplace=True)

        return df

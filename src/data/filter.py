import abc
import gin
import pandas as pd
import logging

from src.data.tanimoto import TanimotoCalculator
from src.data.featurizer import FeaturizerBase


class FilterBase(abc.ABC):
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
        test_smiles = filter_against_df["smiles"].tolist()
        tc = TanimotoCalculator(smiles_list=test_smiles, featurizer=self.featurizer)
        results = tc.run_batch(queries=to_filter_df["smiles"].tolist())
        to_filter_df["min_tanimoto"] = results["min_tanimoto"]
        to_filter_df = to_filter_df[
            to_filter_df["min_tanimoto"] > self.min_distance_to_test_post_filtering
        ]
        logging.info(
            f"Dropped {len(results) - len(to_filter_df)} molecules after filtering by minimal \
                            Tanimoto distance of {self.min_distance_to_test_post_filtering} to a test set."
        )

        to_filter_df.drop(columns=["min_tanimoto"], inplace=True)
        return to_filter_df

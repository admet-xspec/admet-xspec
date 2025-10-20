import abc
import gin

import pandas as pd
from sklearn.decomposition import PCA


class ReducerBase(abc.ABC):
    def __init__(self):
        self.model = self._init_model()
        self.input_dir = None
        self.output_dir = None

    @abc.abstractmethod
    def _init_model(self):
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def get_reduced_df(self, df: pd.DataFrame):
        pass

@gin.configurable
class PcaReducer(ReducerBase):
    def __init__(self, n_dims: int = 2):
        self.n_dims = n_dims
        self.model = self._init_model()

    def _init_model(self):
        """Initialize the model."""
        pca = PCA(n_components=self.n_dims)
        return pca

    def get_reduced_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform PCA on a dataframe, return that dataframe with dim_1, ..., dim_n columns corr. to PCA"""
        reduced_data_ndarray = self.model.fit_transform(df)
        reduced_df = pd.DataFrame({
            f"dim_{i + 1}": reduced_data_ndarray[:, i]
            for i in range(reduced_data_ndarray.shape[1])
        })

        df.join(reduced_df)
        return df
import abc
import gin

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from src.data.visualizer import ProjectionVisualizer
from typing import Dict, Any


class ReducerBase(abc.ABC):
    """Dimensionality reduction class."""

    def __init__(self, params):
        self.model = self._init_model(params)
        self.input_dir = None
        self.output_dir = None

    @abc.abstractmethod
    def get_associated_visualizer(self):
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the reducer."""
        pass

    @abc.abstractmethod
    def _init_model(self, params):
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def get_reduced_df(self, df: pd.DataFrame):
        pass


class ScikitReducerBase(ReducerBase):
    """Base class for scikit-learn based reducers."""

    def __init__(
        self,
        n_dims: int = 2,
        params: Dict[str, Any] = None,
        plot_title: str | None = None,
    ):
        super().__init__(params)
        self.visualizer = ProjectionVisualizer(
            n_dims=n_dims, projection_type=self.name, plot_title=plot_title
        )

    def get_associated_visualizer(self):
        """Return the visualizer object for this reducer."""
        return self.visualizer

    def get_reduced_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform PCA on a dataframe, return that dataframe with dim_1, ..., dim_n columns corr. to PCA"""
        reduced_data_ndarray = self.model.fit_transform(df)
        reduced_df = pd.DataFrame(
            {
                f"dim_{i + 1}": reduced_data_ndarray[:, i]
                for i in range(reduced_data_ndarray.shape[1])
            }
        )

        return reduced_df


@gin.configurable
class PcaReducer(ScikitReducerBase):
    """PCA reducer class."""

    def __init__(
        self, n_dims: int = 2, random_state: int = 42, plot_title: str | None = None
    ):
        params = {
            "n_components": n_dims,
            "random_state": random_state,
        }
        super().__init__(n_dims=n_dims, params=params, plot_title=plot_title)

    def _init_model(self, params):
        """Initialize the model."""
        pca = PCA(**params)
        return pca

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "PCA"


@gin.configurable
class TsneReducer(ScikitReducerBase):
    """T-SNE reducer class."""

    def __init__(
        self,
        n_dims: int = 2,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        params = {
            "perplexity": perplexity,
            "max_iter": max_iter,
            "random_state": random_state,
        }
        super().__init__(n_dims=n_dims, params=params)

    def _init_model(self, params):
        """Initialize the model."""
        tsne_model = TSNE(**params)
        return tsne_model

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "t-SNE"


@gin.configurable
class UmapReducer(ScikitReducerBase):
    """UMAP reducer class."""

    def __init__(
        self,
        n_dims: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        }
        super().__init__(n_dims=n_dims, params=params)

    def _init_model(self, params):
        """Initialize the model."""
        umap_model = UMAP(**params)
        return umap_model

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "UMAP"

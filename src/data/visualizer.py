import abc
import io
import gin
import matplotlib.pyplot as plt

from PIL import Image
import pandas as pd
import numpy as np


class VisualizerBase(abc.ABC):
    def __init__(self):
        self.model = self._init_model()
        self.input_dir = None
        self.output_dir = None

    @abc.abstractmethod
    def _init_model(self):
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def _get_visualizable_form(self, dataset_dict: dict):
        pass

    @abc.abstractmethod
    def get_visualization(self, dataset_dict: dict):
        pass


@gin.configurable
class PcaVisualizer(VisualizerBase):
    def __init__(self, n_dims: int = 2):
        self.n_dims = n_dims

    def _get_visualizable_form(self, dataset_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        pass

    def _visualize_2d(
            self,
            ndarray: np.ndarray,
            class_series: pd.Series,
            class_names: list[str]
    ) -> Image.Image:

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            ndarray[:, 0],
            ndarray[:, 1],
            c=class_series,
            s=40,
        )

        ax.set(
            title="First two PCA dimensions",
            xlabel="1st Eigenvector",
            ylabel="2nd Eigenvector",
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add a legend
        legend1 = ax.legend(
            scatter.legend_elements()[0],
            class_names,
            loc="upper right",
            title="Classes",
        )
        ax.add_artist(legend1)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)

        return img

    def _visualize_3d(
            self,
            ndarray: np.ndarray,
            class_series: pd.Series,
            class_names: list[str]
    ) -> Image.Image:

        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

        scatter = ax.scatter(
            ndarray[:, 0],
            ndarray[:, 1],
            ndarray[:, 2],
            c=class_series,
            s=40,
        )

        ax.set(
            title="First three PCA dimensions",
            xlabel="1st Eigenvector",
            ylabel="2nd Eigenvector",
            zlabel="3rd Eigenvector",
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        # Add a legend
        legend1 = ax.legend(
            scatter.legend_elements()[0],
            class_names,
            loc="upper right",
            title="Classes",
        )
        ax.add_artist(legend1)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)

        return img

    def get_visualization(self, reduced_df_dict: dict[str, pd.DataFrame]) -> Image.Image:
        class_series = pd.concat(
            [pd.Series([i] * len(df)) for i, df in enumerate(reduced_df_dict.values())],
            ignore_index=True
        )

        concatenated_df = pd.concat([df for df in reduced_df_dict.values()], ignore_index=True)
        concatenated_ndarray = concatenated_df.to_numpy()

        img = None
        if self.n_dims == 2:
            img = self._visualize_2d(
                concatenated_ndarray,
                class_series,
                list(reduced_df_dict.keys())
            )
        elif self.n_dims == 3:
            img = self._visualize_3d(
                concatenated_ndarray,
                class_series,
                list(reduced_df_dict.keys())
            )

        return img
import gin
import pandas as pd
from PIL.Image import Image
from pathlib import Path


@gin.configurable
class DataInterface:

    def __init__(
        self,
        dataset_dir: str,
        metrics_dir: str,
        visualizations_dir: str,
        data_config_filename: str,
        normalized_filename: str,
    ):
        self.dataset_dir: Path = Path(dataset_dir)
        self.metrics_dir: Path = Path(metrics_dir)
        self.visualizations_dir: Path = Path(visualizations_dir)
        self.data_config_filename: str = data_config_filename
        self.normalized_filename: str = normalized_filename

    def get_by_friendly_name(self, friendly_name: str) -> pd.DataFrame:
        dataset_dir_path: Path = self._find_dataset_dir(friendly_name)

        if not self._check_normalized_dataset_exists(friendly_name):
            self._generate_normalized_dataset(friendly_name)

        dataset_df = self._load_normalized_dataset(friendly_name)

        return dataset_df

    def save_metrics(
        self, friendly_name: str, metrics: pd.DataFrame | dict
    ) -> None: ...

    def save_visualization(self, friendly_name: str, visualization: Image) -> None: ...

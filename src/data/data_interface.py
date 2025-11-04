import gin
import pandas as pd
import yaml
from PIL.Image import Image
from pathlib import Path

from src.data.utils import load_multiple_datasets


@gin.configurable
class DataInterface:

    def __init__(
        self,
        dataset_dir: str,
        metrics_dir: str,
        visualizations_dir: str,
        data_config_filename: str,
        normalized_filename: str,
        handle_multiple_datasets_method: str = None,
    ):
        self.dataset_dir: Path = Path(dataset_dir)
        self.metrics_dir: Path = Path(metrics_dir)
        self.visualizations_dir: Path = Path(visualizations_dir)
        self.data_config_filename: str = data_config_filename
        self.normalized_filename: str = normalized_filename
        self.handle_multiple_datasets_method: str = handle_multiple_datasets_method

        self._init_create_dirs()

    def _init_create_dirs(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    def _find_dataset_dir(self, friendly_name: str) -> Path:
        dataset_dir = None
        for yaml_path in Path(self.dataset_dir).rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if data and data.get("friendly_name") == friendly_name:
                    dataset_dir = yaml_path.parent
                    break

        if dataset_dir:
            return dataset_dir
        raise FileNotFoundError(
            f"No dataset directory with yaml containing friendly_name: '{friendly_name}' found"
        )

    def _check_normalized_dataset_exists(self, dataset_dir_path: Path) -> bool:
        normalized_dataset_path = dataset_dir_path / self.normalized_filename
        return normalized_dataset_path.exists()

    def _generate_normalized_dataset(self, dataset_dir_path: Path) -> None:
        """
        Take whatever raw dataset in 'dataset_dir_path' and output
        normalized dataset under 'self.normalized_filename' in that dir
        """
        multiple_raw_datasets = False

        # note that check against existence of normalized dataset should have occured before this
        datasets_in_dir: list[Path] = [
            Path(globbed_ds) for globbed_ds in dataset_dir_path.rglob("*.csv")
        ]
        if len(datasets_in_dir) > 1:
            multiple_raw_datasets = True

        raw_datasets = load_multiple_datasets(datasets_in_dir)
        if multiple_raw_datasets:
            match self.handle_multiple_datasets_method:
                case "naive_aggregate":
                    aggregate_dataset = self._naive_aggregate_multiple_datasets(
                        raw_datasets
                    )
                    normalized_df = self.get_normalized_df(aggregate_dataset)
                case None:
                    raise ValueError(
                        f"Found multiple raw datasets to be processed in {dataset_dir_path}, "
                        "however, parameter 'self.handle_multiple_datasets_method' is not specified"
                    )
                case _:
                    raise NotImplementedError(
                        "Aggregation method 'self.handle_multiple_datasets_method' is not implemented"
                    )
        else:
            normalized_df = self.get_normalized_df(raw_datasets[0])

        if normalized_df:
            self._save_df(normalized_df)
        else:
            raise RuntimeError(
                f"Failed to generate a normalized dataset within dataset directory '{dataset_dir_path}'"
            )

    def get_by_friendly_name(self, friendly_name: str) -> pd.DataFrame:
        dataset_dir_path: Path = self._find_dataset_dir(friendly_name)

        if not self._check_normalized_dataset_exists(dataset_dir_path):
            self._generate_normalized_dataset(dataset_dir_path)

        dataset_df = self._load_normalized_dataset(friendly_name)

        return dataset_df

    def save_metrics(
        self, friendly_name: str, metrics: pd.DataFrame | dict
    ) -> None: ...

    def save_visualization(self, friendly_name: str, visualization: Image) -> None: ...

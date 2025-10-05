import abc
import pathlib

from PIL import Image
import pickle


class ExplorerBase(abc.ABC):
    def __init__(self):
        self.model = self._init_model()
        self.input_dir = None
        self.output_dir = None

    @abc.abstractmethod
    def _init_model(self):
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def visualize(self, df, method) -> list[tuple[str, Image.Image]]:
        """Generate data-exploratory visualizations for a dataframe"""
        pass

    def save(self, dataset_name, image, output_path):
        """Save the visualization under output_path"""
        path = self.working_dir / "model.pkl"
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Pickle the model
        with open(path, "wb") as f:
            pickle.dump(self, f)

class PcaExplorer(ExplorerBase):
    def __init__(self):
        self.model = self._init_model()
        self.input_dir = None
        self.output_dir = None

    def _init_model(self):
        """Initialize the model."""
        pass

    def visualize(self, df, method) -> list[tuple[str, Image.Image]]:
        """Generate data-exploratory visualizations for a dataframe"""
        pass

    def save(self, dataset_name, image, output_path):
        """Save the visualization under output_path"""
        path = self.working_dir / "model.pkl"
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Pickle the model
        with open(path, "wb") as f:
            pickle.dump(self, f)

class UmapExplorer(ExplorerBase):
    def __init__(self):
        self.model = self._init_model()
        self.input_dir = None
        self.output_dir = None

    def _init_model(self):
        """Initialize the model."""
        pass

    def visualize(self, df, method) -> list[tuple[str, Image.Image]]:
        """Generate data-exploratory visualizations for a dataframe"""
        pass

    def save(self, dataset_name, image, output_path):
        """Save the visualization under output_path"""
        path = self.working_dir / "model.pkl"
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Pickle the model
        with open(path, "wb") as f:
            pickle.dump(self, f)
import abc

class PredictorBase(abc.ABC):
    def __init__(self):
        model = self.init_model()

    @abc.abstractmethod
    def init_model(self):
        """Initialize the model"""
        pass

    @abc.abstractmethod
    def train(self, smiles_list, target_list):
        """Train the model with the given smiles and target list."""
        pass

    @abc.abstractmethod
    def name(self):
        """Return the name of the model"""
        pass

    @abc.abstractmethod
    def predict(self, smiles_list):
        """Predict the target value for the given list of smiles"""
        pass

    @abc.abstractmethod
    def save(self, path):
        """Save the model to the given path"""
        pass

    @abc.abstractmethod
    def load(self, path):
        """Load the model from the given path"""
        pass


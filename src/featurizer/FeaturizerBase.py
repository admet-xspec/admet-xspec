import abc


class FeaturizerBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def featurize(self, smiles):
        """
        Featurize the given SMILES string
        """
        pass

    @abc.abstractmethod
    def name(self):
        """
        Return the name of the featurizer
        """
        pass

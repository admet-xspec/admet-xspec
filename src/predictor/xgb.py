from PredictorBase import PredictorBase
import sklearn

class ScikitPredictor(PredictorBase):
    def __init__(self):
        super(ScikitPredictor, self).__init__()
        self.model = None

    def train(self, smiles_list, target_list):
        pass

    def name(self):
        return 'Scikit-Learn'

    def predict(self, smiles):
        pass

    def predict_batch(self, smiles_list):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


import pandas as pd
from src.utils import clean_smiles
import logging
from src.predictor.chemprop import ChempropBinaryClassifier

logging.basicConfig(level=logging.INFO)

df = pd.read_csv('data/permeability/bbbp_pampa.csv')

pre_cleaning_length = len(df)
df['SMILES'] = clean_smiles(df['SMILES'])
df = df.dropna(subset=['SMILES']).reset_index(drop=True)
new_length = len(df)
if pre_cleaning_length != new_length:
    logging.info(f"Dropped {pre_cleaning_length - new_length} invalid SMILES")
logging.info(f"Dataset size: {new_length}")

model = ChempropBinaryClassifier()

X, y = df['SMILES'], df['Y']
model.train(X, y)
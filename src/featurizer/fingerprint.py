from src.featurizer.FeaturizerBase import FeaturizerBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
from typing import List
import numpy as np
import gin

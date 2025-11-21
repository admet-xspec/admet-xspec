import logging
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from src.data.featurizer import FeaturizerBase, EcfpFeaturizer

def load_multiple_datasets(
    dataset_paths: list[Path],
) -> list[pd.DataFrame]:
    return [pd.read_csv(ds_path) for ds_path in dataset_paths]

def check_dataset_is_raw_chembl(dataset_path: Path) -> bool:
    with open(dataset_path, "r") as f:
        first_two_lines_str = "".join(f.readlines()[:2])
        if ";" in first_two_lines_str:
            return True
    return False

def get_label_counts(df: pd.DataFrame, column_name="source") -> dict:
    source_count = {}
    for name in df[column_name].unique():
        source_count[name] = len(df[df[column_name] == name])
    return source_count

class TanimotoCalculator:
    """Efficiently calculate Tanimoto distance between a query molecule and a set of molecules.
    Uses ECFP4 fingerprints by default, but can be configured to use any featurizer compatible with FeaturizerBase.
    """

    def __init__(
        self,
        smiles_list: List[str],
        featurizer: FeaturizerBase | None = None,
        return_closest_smiles: bool = False,
    ):
        """
        Initialize the calculator with a set of molecules.
        Args:
            featurizer: An instance of FeaturizerBase to compute fingerprints
            smiles_list: List of SMILES strings representing the molecules to compare against
        """
        if featurizer is None:
            logging.warning(
                "No featurizer object provided to the constructor of TanimotoCalculator, defaulting to 2048-bit ECFP4.")
            self.featurizer = EcfpFeaturizer(n_bits=2048, radius=2)
        else:
            self.featurizer = featurizer
        self.smiles_list = smiles_list
        self.fingerprints = self.precompute_fingerprints(smiles_list)
        self.return_closest_smiles = return_closest_smiles

    def run_single(self, query: str) -> dict:
        """
        Calculate Tanimoto distances for the query molecule and return summary statistics.
        Args:
            query: SMILES string of the query molecule
        Returns:
            Dictionary containing max, min, mean, and quartiles of Tanimoto distances
        """
        query_fp = self.featurizer.featurize([query])[0]
        similarities = self._calculate_similarities(query_fp)
        distances = 1 - similarities

        q1, q2, q3 = np.percentile(distances, [25, 50, 75])
        stat_dict = {
            "max_distance": float(np.max(distances)),
            "min_distance": float(np.min(distances)),
            "mean_distance": float(np.mean(distances)),
            "q1_distance": float(q1),
            "q2_distance": float(q2),
            "q3_distance": float(q3),
        }

        if self.return_closest_smiles:
            min_index = int(np.argmin(distances))
            stat_dict["query_smile"] = query
            stat_dict["closest_smile"] = self.smiles_list[min_index]

        return stat_dict

    def run_batch(self, queries: List[str]) -> dict:
        """
        Calculate Tanimoto distances for a batch of query molecules.
        Args:
            queries: List of SMILES strings of the query molecules
        Returns:
            Dictionary of lists containing summary statistics for each query
        """
        results = {
            "max_distance": [],
            "min_distance": [],
            "mean_distance": [],
            "q1_distance": [],
            "q2_distance": [],
            "q3_distance": [],
            "query_smile": [],
            "closest_smile": [],
        }

        if not self.return_closest_smiles:
            results.pop("closest_smile")
            results.pop("query_smile")

        for query in queries:
            stats = None
            try:
                stats = self.run_single(query)
            except Exception as e:
                logging.debug(
                    (
                        f"Failed to query Tanimoto for {query}, "
                        "Adding None for each key in 'results' and expecting caller to handle it"
                    )
                )

            for key in results.keys():
                if stats is not None:
                    results[key].append(stats[key])
                else:
                    results[key].append(None)

        return results

    def _calculate_similarities(self, query_fp: np.ndarray) -> np.ndarray:
        """
        Calculate Tanimoto similarities between a query molecule fingerprint and a set of precomputed fingerprints.
        Args:
            query: np.ndarray representing the fingerprint of the query molecule
        Returns:
            Array of Tanimoto similarities
        """
        query_fp_bit_vect = self.numpy_to_bitvect(query_fp)
        similarities = np.array(
            DataStructs.BulkTanimotoSimilarity(query_fp_bit_vect, self.fingerprints)
        )
        return similarities

    def precompute_fingerprints(self, smiles_list: List[str]) -> List[ExplicitBitVect]:
        """Precompute fingerprints for the given list of SMILES strings."""
        fps = self.featurizer.featurize(smiles_list)
        return [self.numpy_to_bitvect(fp) for fp in fps]

    def numpy_to_bitvect(self, np_array: np.ndarray) -> ExplicitBitVect:
        """
        Convert a numpy array fingerprint to an RDKit ExplicitBitVect object.
        Args:
            np_array: Numpy array containing binary fingerprint data
        Returns:
            ExplicitBitVect object
        """
        bitvect = DataStructs.ExplicitBitVect(len(np_array))
        on_bits = np.where(np_array > 0)[0].tolist()
        bitvect.SetBitsFromList(on_bits)
        return bitvect
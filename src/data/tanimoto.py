from typing import List
import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from src.data.featurizer import FeaturizerBase, EcfpFeaturizer


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
            self.featurizer = EcfpFeaturizer(n_bits=1024, radius=2)
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
            stats = self.run_single(query)
            for key in results.keys():
                results[key].append(stats[key])

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

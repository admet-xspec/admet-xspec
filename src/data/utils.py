import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from src.data.featurizer import EcfpFeaturizer, FeaturizerBase


def load_multiple_datasets(
    dataset_paths: list[Path],
) -> list[pd.DataFrame]:
    return [pd.read_csv(ds_path) for ds_path in dataset_paths]


def check_dataset_is_raw_chembl(dataset_path: Path) -> bool:
    """Return True if the file looks like a raw ChEMBL export (`;`-separated)."""
    with open(dataset_path, "r") as f:
        first_two_lines = "".join(f.readlines()[:2])
    return ";" in first_two_lines


def get_label_counts(df: pd.DataFrame, column_name="source") -> dict:
    """Return the number of rows per unique value of `column_name`."""
    return {
        name: len(df[df[column_name] == name]) for name in df[column_name].unique()
    }


class TanimotoCalculator:
    """Compute Tanimoto distances between query molecules and a reference set.

    Defaults to 2048-bit ECFP4; any `FeaturizerBase` producing binary vectors works.
    """

    def __init__(
        self,
        smiles_list: List[str],
        featurizer: FeaturizerBase | None = None,
        return_closest_smiles: bool = False,
    ):
        if featurizer is None:
            logging.warning(
                "No featurizer object provided to the constructor of TanimotoCalculator, defaulting to 2048-bit ECFP4."
            )
            featurizer = EcfpFeaturizer(n_bits=2048, radius=2)
        self.featurizer = featurizer
        self.smiles_list = smiles_list
        self.fingerprints = self.precompute_fingerprints(smiles_list)
        self.return_closest_smiles = return_closest_smiles

    def run_single(self, query: str) -> dict:
        """Return max/min/mean/quartile distances from `query` to the reference set.

        Adds `query_smile` and `closest_smile` if `return_closest_smiles` is set.
        """
        query_fp = self.featurizer.featurize([query])[0]
        distances = 1 - self._calculate_similarities(query_fp)

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
            stat_dict["query_smile"] = query
            stat_dict["closest_smile"] = self.smiles_list[int(np.argmin(distances))]

        return stat_dict

    def run_batch(self, queries: List[str]) -> dict:
        """Return `run_single` stats for each query, as a dict of lists.

        Failed queries get `None` for every key; callers must handle it.
        """
        keys = [
            "max_distance",
            "min_distance",
            "mean_distance",
            "q1_distance",
            "q2_distance",
            "q3_distance",
        ]
        if self.return_closest_smiles:
            keys += ["query_smile", "closest_smile"]
        results = {key: [] for key in keys}

        for query in queries:
            try:
                stats = self.run_single(query)
            except Exception:
                logging.debug(
                    f"Failed to query Tanimoto for {query}, "
                    "Adding None for each key in 'results' and expecting caller to handle it"
                )
                stats = None

            for key in keys:
                results[key].append(stats[key] if stats is not None else None)

        return results

    def _calculate_similarities(self, query_fp: np.ndarray) -> np.ndarray:
        """Return Tanimoto similarities of `query_fp` to all reference fingerprints."""
        query_bitvect = self.numpy_to_bitvect(query_fp)
        return np.array(
            DataStructs.BulkTanimotoSimilarity(query_bitvect, self.fingerprints)
        )

    def precompute_fingerprints(self, smiles_list: List[str]) -> List[ExplicitBitVect]:
        """Featurize `smiles_list` into RDKit bit vectors."""
        fps = self.featurizer.featurize(smiles_list)
        return [self.numpy_to_bitvect(fp) for fp in fps]

    def numpy_to_bitvect(self, np_array: np.ndarray) -> ExplicitBitVect:
        """Convert a numpy fingerprint to an `ExplicitBitVect` (nonzero -> on bit)."""
        bitvect = DataStructs.ExplicitBitVect(len(np_array))
        bitvect.SetBitsFromList(np.where(np_array > 0)[0].tolist())
        return bitvect

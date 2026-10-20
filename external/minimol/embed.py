"""Embed SMILES with MiniMol; run inside the isolated `external/minimol` venv.

Reads one SMILES per line from `--input` and writes a float32 `(N, 512)` array
to `--output` (`.npy`), in input order. Molecules that fail featurization get
NaN rows, so the caller decides how to treat them.
"""

import argparse
import logging
import sys

import numpy as np
from rdkit import Chem, RDLogger

EMBEDDING_DIM = 512


def embed(smiles: list[str], batch_size: int) -> np.ndarray:
    """Return MiniMol embeddings for `smiles`, with NaN rows for failures."""
    # Imported late, as loading graphium and the checkpoint takes a while.
    from minimol import Minimol

    out = np.full((len(smiles), EMBEDDING_DIM), np.nan, dtype=np.float32)
    # Unparseable SMILES would make MiniMol crash the whole batch, so skip them upfront.
    RDLogger.DisableLog("rdApp.*")
    valid = [i for i, smi in enumerate(smiles) if _is_valid(smi)]
    if not valid:
        return out

    # One instance per process: MiniMol initializes hydra globally.
    model = Minimol(batch_size=batch_size)
    for start in range(0, len(valid), batch_size):
        idx = valid[start : start + batch_size]
        try:
            out[idx] = _to_numpy(model([smiles[i] for i in idx]))
        except Exception:
            logging.warning(
                "Batch at %d failed, retrying one molecule at a time.", start
            )
            for i in idx:
                try:
                    out[i] = _to_numpy(model([smiles[i]]))[0]
                except Exception:
                    logging.warning("MiniMol failed to embed: %s", smiles[i])
    return out


def _is_valid(smi: str) -> bool:
    """Return whether RDKit parses `smi` into a non-empty molecule."""
    mol = Chem.MolFromSmiles(smi)
    return mol is not None and mol.GetNumAtoms() > 0


def _to_numpy(embeddings) -> np.ndarray:
    return np.stack([e.detach().cpu().numpy() for e in embeddings])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Text file, one SMILES per line."
    )
    parser.add_argument("--output", required=True, help="Destination `.npy` file.")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    with open(args.input) as f:
        smiles = f.read().splitlines()
    np.save(args.output, embed(smiles, args.batch_size))


if __name__ == "__main__":
    main()

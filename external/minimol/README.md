# MiniMol environment

Isolated environment for `MinimolFeaturizer` (`src/data/featurizer.py`), which
embeds molecules with the pre-trained [MiniMol](https://github.com/graphcore-research/minimol)
GNN (512-dimensional embeddings).

MiniMol depends on `graphium==2.4.7`, whose pins (`torch` 2.3, `numpy<2`,
`torchmetrics<0.11`, PyG extensions) clash with the main project. It therefore
lives in its own uv project here, and the featurizer calls `embed.py` in a
subprocess using `.venv/bin/python`. Nothing from it is imported by the main
project.

## Setup

```bash
external/minimol/setup.sh
```

This runs `uv sync --frozen` in this directory and embeds two molecules as a
smoke test. Then use `configs/featurizers/minimol.gin`.

## Notes

- **CPU-only wheels.** Embeddings are computed once per SMILES and cached on disk
  (see `fp_cache_dir` in `src/data/featurizer.py`, or set `$ADMET_FP_CACHE_DIR`),
  so the GPU isn't needed. For CUDA, switch the `pytorch-cpu` index and the PyG
  `find-links` URL in `pyproject.toml` to their `cu121` counterparts and re-lock.
- **Pins that matter:** `scipy<1.15` (graphium builds float16 sparse matrices),
  `torchvision` from the PyTorch CPU index (the PyPI build doesn't match CPU torch),
  `setuptools<81` (MiniMol imports `pkg_resources`). Don't upgrade casually:
  graphium 2.x is no longer maintained.
- **Speed:** model loading takes ~10 s per worker call; ~1000 molecules embed in
  ~10 s on CPU with the default batch size of 1000. Smaller batches are slower,
  since graphium starts a process pool per batch.
- **Failures:** SMILES that MiniMol cannot embed come back as NaN rows, and the
  featurizer raises a `ValueError` listing them.

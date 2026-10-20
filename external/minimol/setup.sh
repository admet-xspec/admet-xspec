#!/usr/bin/env bash
# Create the isolated MiniMol environment and check that it embeds molecules.
set -euo pipefail

cd "$(dirname "$0")"
# Ignore any activated venv (e.g. the main project's); this one lives in ./.venv.
unset VIRTUAL_ENV
uv sync --frozen

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
printf 'CCO\nc1ccccc1\n' > "$tmp_dir/smiles.txt"
.venv/bin/python -I embed.py --input "$tmp_dir/smiles.txt" --output "$tmp_dir/out.npy" 2> "$tmp_dir/log" \
    || { cat "$tmp_dir/log" >&2; exit 1; }
.venv/bin/python -I -c "
import numpy as np, sys
emb = np.load(sys.argv[1])
assert emb.shape == (2, 512) and np.isfinite(emb).all(), emb.shape
print('MiniMol environment OK:', emb.shape)
" "$tmp_dir/out.npy"

# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

# Make the `src` package importable for autodoc.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information

project = 'ADMET-XSpec'
copyright = '2026, Rybka, Masztalerz'
author = 'H. Rybka & K. Masztalerz'

release = '1.1.0'
version = '1.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Autodoc

# Heavy dependencies are mocked so the docs build without them. `gin` stays real
# (`@gin.configurable` must return the decorated class), and so do numpy, pandas
# and rdkit, whose types appear in `X | None` annotations evaluated at import.
autodoc_mock_imports = [
    'filelock',
    'joblib',
    'lightgbm',
    'optuna',
    'scipy',
    'sklearn',
    'threadpoolctl',
    'toml',
    'tqdm',
    'xgboost',
]
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

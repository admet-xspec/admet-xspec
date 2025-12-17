# ADMET-XSpec: A Tool for Systematic Cross-Species Data Integration in ADMET Prediction

___

<p align="center">
  <img src="imgs/logo_white_bg.png" alt="ADMET-Xspec logo" width="400"/>
</p>

___

## What is ADMET-XSpec?
ADMET-XSpec is an open-source tool that facilitates systematic cross-species data integration for training ADMET
(Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction models. **The tool is designed to help
researchers assess and understand the translatability of ADMET parameters measured in non-human assays in the context
of human drug development.**

ADMET-XSpec implements a comprehensive workflow that includes data preprocessing, feature engineering, model training,
and evaluation. ADMET-XSpec allows for the evaluation of predictive models trained on human data augmented by non-human
data points, leveraging a selection of machine learning algorithms, feature extraction techniques and additional
data preprocessing steps. The repository is currently under active development.

## Quick start

ADMET-XSpec can be set up either using plain `uv` or by creating a conda environment.
First, clone the repository:

```
git clone https://github.com/hubertrybka/admet-prediction.git
cd admet-prediction
```

### 1. UV setup
Follow the "Install uv" step at the [official docs](https://docs.astral.sh/uv/#__tabbed_1_1) to set up uv.
Then, have uv register a .venv within the current directory and install packages from the lockfile:
```bash
uv init .
uv sync
```
#### Run example with uv
```bash
uv run process.py --cfg configs/examples/train_optimize_rf_clf.gin
```

##### Optional, recommended: integrate uv with PyCharm
In your Status Bar (bottom right corner of IDE), click on the button which gives a "Current Interpreter" hover.
Select "Add New Interpreter" -> "Add Local Interpreter" -> "Select Existing".

Change the type to 'uv'. The uv path should be that of your system uv installation, the Environment path should be
that of the `.venv` that uv created inside of `./admet_prediction`.

### 2. Conda setup
```bash
conda create -n admet python=3.11.8
conda activate admet_xspec
conda install rdkit seaborn conda-forge::py-xgboost conda-forge::ray-all

pip install -r requirements.txt

# Dev dependencies
pre-commit install
```
#### Run example with conda
```bash
# if you haven't already:
conda activate admet_xspec

python -m process --cfg configs/examples/train_optimize_rf_clf.gin
```

##### Optional, recommended: integrate conda with PyCharm
Follow the above steps for uv, but select conda. The procedure should be pretty straightforward and analogous.
# TBA

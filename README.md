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

ADMET-XSpec can be set up either using plain `uv` (**can it???**) or by creating a conda environment.
First, clone the repository:

```
git clone https://github.com/hubertrybka/admet-prediction.git
cd admet-prediction
```

### 1. UV setup (in progress...)

### 2. Conda setup
```bash
conda create -n admet python=3.11.8
conda activate admet
conda install rdkit seaborn conda-forge::py-xgboost conda-forge::ray-all

pip install -r requirements.txt

# Dev dependencies
pre-commit install
```

# TBA
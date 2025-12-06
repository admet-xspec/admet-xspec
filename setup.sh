conda create -n admet python=3.11
conda activate admet

pip install -r requirements.txt
conda install rdkit seaborn conda-forge::py-xgboost conda-forge::ray-all

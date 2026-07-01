Quick Start
===========

Installation
------------

ADAMET-XSpec can be set up with both ``uv`` and ``conda``. Clone the repository and navigate to its directory.

.. code:: bash
   
   git clone https://github.com/admet-xspec/admet-xspec.git
   cd admet-xspec

UV setup
--------

.. _`official uv docs`: https://docs.astral.sh/uv/getting-started/

Follow the guide at the `official uv docs`_ to set up ``uv``.
Then, have ``uv`` register a .venv within the current directory and install packages from the lockfile:

.. code:: bash
   
   uv init .
   uv sync

Run a demo experiment with uv:

.. code:: bash
   
   uv run process.py --cfg configs/examples/train_lgbm.gin

Conda setup
-----------

.. _miniconda: https://www.anaconda.com/download/success?reg=skipped

Install miniconda_ following the instructions for your operating system.

.. code:: bash
   
   conda create -n admet python=3.11.8
   conda activate xspec
   conda install rdkit seaborn conda-forge::py-xgboost conda-forge::ray-all
   
   pip install -r requirements.txt
   
   # dev dependencies
   pre-commit install

Run a demo experiment with Conda:

.. code:: bash
   
   # if you haven't already:
   conda activate xspec
   
   python -m process --cfg configs/examples/train_lgbm.gin

3 Experiments you can run now!
------------------------------

1. **I'm training a classifier for the prediction of human acetylcholinesterase (AChE) inhibitory action of organic molecules. I want to test how the inclusion of datapoints from mouse and rat-based AChE inhibition assays into my exclusively-human dataset influences performance of the model**

   I want to input pre-optimized hyperparameters for the model, utilize scaffold-based train-test split strategy, represent the 
   molecules using ECFP4 fingerprints and employ a LightGBM classifier alogrithm.
 
   .. code:: bash
      
      uv run process.py --cfg configs/examples/train_lgbm.gin
      # or
      python -m process --cfg configs/examples/train_lgbm.gin

   Have a look at ``configs/examples/train_lgbm.gin``, as its structure corresponds with the exact experiment setup described above. 
   **This is an example of the general config file for an ADMET-Xspec experiment in which ML models would be trained and/or evaluated.** 
   The build process for a config file describing our desired experiment will be discussed in next chapters.

2. **I have a dataset of small molecules labeled with their continuous IC50 (inhibitory activity) values towards human AChE. 
   I want to identify a well-performing set of hyperparameter values for a chosen regressor, train, evaluate and save the 
   best-performing model.**

   I want my pipeline to include scaffold splitter, ECFP-count featurizer and RF regression model. I want to train on human-derived 
   data only.
 
   .. code:: bash
      
      uv run process.py --cfg configs/examples/train_rf_optimize.gin
      # or
      python -m process --cfg configs/examples/train_rf_optimize.gin

3. **I have a heterogenous dataset of IC50-labeled Monoamine oxidase A (MAO-A) inhibitors, obtained as a naive concatenation of the rat and the human-derived data.** I want to explore how **attributted learning** affects the predictive power of the trained regressor on human test data.

   I want to utilize scaffold split, KRFP (Klekota & Roth FP) featurizer, < 95% tanimoto similarity filter for the rat data (against 
   the whole human set) and an RF regressor in the attributed leatning mode.

   .. note::
      When working with heterogenous datasets (concatented from two or more data sources), ADMET-Xspec allows for training ML models in the **attributed learning** mode. According to this strategy, the :math:`n` unique **attributes**
      :math:`\mathbf{a}^{(k)} \to \mathcal{A}` (data source labels) found in the whole dataset are mapped to OHE vectors
      :math:`\phi: \mathcal{A} \to \{0,1\}^n`. For each data point, described by a feature vector
      :math:`\mathbf{x}^{(k)} \in \mathbb{R}^d`, we then construct an augmented representation
      :math:`\tilde{\mathbf{x}} \in \mathbb{R}^{d+n}` by concatenating the feature vector with the OHE attribute.


   .. code:: bash
      
      uv run process.py --cfg configs/examples/train_rf_attributed.gin
      # or
      python -m process --cfg configs/examples/train_rf_attributed.gin

Quick start
===========

.. admonition:: Goals
   :class: tip

   After reading this section, you will:

   #. Have ADMET-XSpec installed and ready to use.
   #. Be able to run three demo experiments with the provided config files.

ADMET-XSpec can be set up with both ``uv`` and ``conda``. Clone the repository and navigate to its directory. Then,
you can choose one of the two installation methods described below.

.. code:: bash

   git clone https://github.com/admet-xspec/admet-xspec.git
   cd admet-xspec

UV installation
---------------

.. _`official uv docs`: https://docs.astral.sh/uv/getting-started/

Follow the guide at the `official uv docs`_ to set up ``uv``.
Then, have ``uv`` create a ``.venv`` within the current directory and install packages from the lockfile
(``uv`` will download Python 3.11.8 if it is not available on your system):

.. code:: bash

   uv sync

Run a demo experiment with uv to test your installation:

.. code:: bash

   uv run process.py --cfg configs/examples/train_lgbm.gin

Conda installation
------------------

.. _miniconda: https://www.anaconda.com/download/success?reg=skipped

Install miniconda_ following the instructions for your operating system. The dependencies are declared in
``pyproject.toml``, which ``pip`` cannot install from directly in this repository, so we use ``uv pip`` inside
the conda environment:

.. code:: bash

   conda create -n xspec python=3.11.8
   conda activate xspec

   pip install uv
   uv pip install -r pyproject.toml

Run a demo experiment with Conda to test your installation:

.. code:: bash

   conda activate xspec # if you haven't already

   python -m process --cfg configs/examples/train_lgbm.gin

Development setup (optional)
----------------------------

Contributors should install the ``pre-commit`` hooks (``black`` and basic file checks):

.. code:: bash

   uvx pre-commit install
   # or, inside the conda environment
   pip install pre-commit && pre-commit install

Command-line options
--------------------

Both entry points, ``process.py`` (training, see :doc:`features`) and ``predict.py`` (inference, see
:doc:`inference`), accept the same two options:

- ``--cfg`` / ``-c`` -- path to the general ``.gin`` config file.
- ``--log-level`` -- one of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.

Three experiments to run right now
----------------------------------

As the setup of ADMET-XSpec relies on the use of ``gin`` config files, we provide three examples of experiments that you
should be able to run immediately after the installation. The config files for these experiments are located in the
``configs/examples/`` directory.

Below are three scenarios you can explore with the provided config files. Do not worry - the next chapters
will go into detail about how to build your own config files for your own experiments.

1. I'm training a classifier for the prediction of human acetylcholinesterase (AChE) inhibitory action on organic
   molecules. **I want to test how the inclusion of data points from mouse and rat-based AChE inhibition assays into my
   exclusively-human dataset influences performance of the model on human test data.**

   I want to use fixed hyperparameters for the model, utilize a scaffold-based train-test split strategy, represent the
   molecules using ECFP4 fingerprints and employ a LightGBM classifier algorithm.

   .. code:: bash

      uv run process.py --cfg configs/examples/train_lgbm.gin
      # or
      python -m process --cfg configs/examples/train_lgbm.gin

   Have a look at ``configs/examples/train_lgbm.gin``, as its structure corresponds with the exact experiment setup described above.
   **This is an example of the general config file for an ADMET-XSpec experiment in which ML models would be trained and/or evaluated.**
   The build process for a config file describing our desired experiment will be discussed in the next chapters.

2. **I have a dataset of small molecules labeled with their continuous IC50 (inhibitory activity) values towards human AChE.
   I want to identify a well-performing set of hyperparameter values for a chosen regressor, train, evaluate and save the
   best-performing model.**

   I want my pipeline to include a scaffold splitter, an ECFP-count featurizer and an RF regression model. I want to train on
   human-derived data only.

   .. code:: bash

      uv run process.py --cfg configs/examples/train_rf_optimize.gin
      # or
      python -m process --cfg configs/examples/train_rf_optimize.gin

3. **I have a heterogeneous dataset of IC50-labeled Monoamine oxidase A (MAO-A) inhibitors, obtained as a naive
   concatenation of the rat and the human-derived data.** I want to explore how **attributed learning** affects the
   predictive power of the trained regressor on human test data.

   I want to utilize a scaffold split, the KRFP (Klekota & Roth FP) featurizer, a similarity filter that drops rat molecules
   with a Tanimoto distance of at most 0.05 to any human molecule (``tanimoto_5p_to_whole``), and an RF regressor in the
   attributed learning mode.

   .. code:: bash

      uv run process.py --cfg configs/examples/train_rf_attributed.gin
      # or
      python -m process --cfg configs/examples/train_rf_attributed.gin

   .. note::
      When working with heterogeneous datasets (concatenated from two or more data sources), ADMET-XSpec allows for
      training ML models in the **attributed learning** mode. In this mode, the feature vector of each data point is
      concatenated with a vector representing the source of the particular data point.

      **This allows the model to learn the differences between the data sources and may improve its predictive
      performance on the target data source.**

      Each data point :math:`(\mathbf{x}, y, a)`, described by a feature vector :math:`\mathbf{x} \in \mathbb{R}^d`
      and labeled with :math:`y`, has an additional attribute :math:`a`, which identifies which of the :math:`m`
      source datasets (by friendly name, e.g. ``MAOA_human_pIC50`` or ``MAOA_rat_pIC50``) the data point comes from.
      A standard one-hot encoding map :math:`\phi: \{a_1, a_2, \dots, a_m\} \to \{0,1\}^m` encodes each attribute
      :math:`a` as a one-hot vector :math:`\mathbf{e}_{a}`. The ML models are trained on the concatenated feature
      vectors :math:`\hat{\mathbf{x}} = [\mathbf{x}, \mathbf{e}_{a}] \in \mathbb{R}^{d+m}` and the corresponding
      labels :math:`y`.

      Attributed learning is enabled with ``do_attributed_learning = True`` in the predictor config, as in the
      ``*_attributed.gin`` files under ``configs/predictors/``.

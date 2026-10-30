Configuring an experiment
=========================

ADMET-XSpec is configured exclusively through ``gin`` files. If you are unfamiliar with ``gin``,
we encourage you to take a brief look at the official `gin-config user guide`_ for a nice introduction.

.. _`gin-config user guide`: https://github.com/google/gin-config/blob/master/docs/index.md

General run config structure
----------------------------

In ADMET-XSpec, each experimental setup is to be defined using one single ``gin`` file, which we will refer to as the
**general config** file. The config file controls data flow through the pipeline, splitters, featurizers, filters, and ML predictors.

**The general config is composed of several smaller config files,** invoked using the ``include`` command, each defining
a single component of the pipeline.

.. note::
   ``include`` imports another gin file into the current config. This is how small reusable config
   fragments are combined into one experiment definition. The ``gin`` config files were meant to be composed,
   not copied.

**A template for a general config file** (``configs/examples/experiment_template.gin``) **looks like this:**

.. code-block:: text

   #=====================[ADMET-XSpec]=====================#
   #----------------------[pipeline]-----------------------#

   include 'configs/processing_plans/...'
   include 'configs/splitters/...'
   include 'configs/featurizers/...'
   include 'configs/sim_filters/...'
   include 'configs/predictors/...'

   #------------------------[data]-------------------------#

   ProcessingPipeline.datasets = [
      '..._friendly_name_1',
      '..._friendly_name_2'
   ]

   ProcessingPipeline.test_origin_dataset = '..._friendly_name_1'
   ProcessingPipeline.task_setting = '...'

   #========================================================#
   # Advanced gin config settings
   include 'configs/data_interface/data_interface.gin'
   ProcessingPipeline.hyperparams_source_sim_filter = None
   ProcessingPipeline.data_interface = %data_interface
   ProcessingPipeline.splitter = %splitter
   ProcessingPipeline.featurizer = %featurizer
   ProcessingPipeline.sim_filter = %sim_filter
   ProcessingPipeline.predictor = %predictor

To fill in the template:

1. Choose an appropriate processing plan for your workflow.

   - ``train.gin`` for a standard training run with fixed hyperparameters.
   - ``train_optimize.gin`` when hyperparameter optimization is part of the run.
   - ``train_load_hyperparams.gin`` when the model should load previously optimized hyperparameters.
   - ``split.gin`` when you only need the dataset splitting stage.
   - ``normalize.gin`` when you only need to prepare (clean, filter and label) the raw datasets.

   If you are unsure, consult the :doc:`features` section for a description of the available plans.
2. Include gin files representing the chosen splitter type, featurizer, similarity filter and predictor.
3. Set the dataset list using the datasets' friendly names. The list of all friendly names identified in the
   ``data/datasets/`` directory can be found in the ``data/datasets/registry.txt`` text file.
   A friendly name usually appears there twice, because the same name is shared by the ``regression`` and the
   ``binary_classification`` variant of a dataset (see :doc:`sourcing_data`).
4. Set ``ProcessingPipeline.test_origin_dataset`` to the friendly name of the dataset that should populate the test set.
   All other datasets in the list are treated as **augmenting** datasets and only ever enter the train set.
5. Set ``ProcessingPipeline.task_setting`` to match the task type (``"binary_classification"`` or ``"regression"``).
   Together with the friendly names, it selects which dataset directories are loaded.

Component config files
----------------------

Each component (splitter, featurizer, model etc.) of the ``ProcessingPipeline`` is represented by some gin config file
in ``configs/``. The directories containing user-editable gin configs, relevant to the assembly of an ADMET-XSpec
experiment, are outlined below:

.. code-block:: text

   configs
   ├── applicability_domains
   │   ├── none.gin                         # No applicability domain
   │   └── tanimoto_knn.gin                 # kNN Tanimoto similarity to the training set
   ├── data_interface
   │   └── data_interface.gin               # Data/cache directory locations
   ├── examples                             # Ready-to-run experiments (see Quick start)
   │   ├── experiment_template.gin
   │   ├── predict.gin
   │   ├── train_lgbm.gin
   │   ├── train_mlp_optimize.gin
   │   ├── train_rf_attributed.gin
   │   └── train_rf_optimize.gin
   ├── processing_plans
   │   ├── _possible_plans.gin              # List of all available flags
   │   ├── normalize.gin                    # Pre-process the data only
   │   ├── fit_applicability_domain.gin     # Fit the applicability domain of a trained model
   │   ├── split.gin                        # Perform train-test split
   │   ├── train.gin                        # Train on fixed hyperparams (parse from config)
   │   ├── train_load_hyperparams.gin       # Train on fixed hyperparams (parse from previous run)
   │   └── train_optimize.gin               # Perform hyperparam optimization and train
   ├── featurizers
   │   ├── ecfp_count.gin                   # ECFP4 count variant
   │   ├── ecfp.gin                         # ECFP4
   │   ├── krfp.gin                         # Klekota-Roth fingerprint
   │   ├── maccs.gin                        # MACCS keys fingerprint
   │   ├── maplight.gin                     # MapLight: Morgan + Avalon counts, ErG, RDKit descriptors
   │   ├── minimol.gin                      # MiniMol pre-trained GNN embeddings (see external/minimol)
   │   ├── property.gin                     # RDKit physicochemical descriptors
   │   ├── none.gin                         # No featurizer
   │   └── proprietary                      # Internal fingerprints of the similarity filters
   │       └── tanimoto_*.gin               #   (included by sim_filters/ and applicability_domains/; do not include directly)
   ├── predictors
   │   ├── classifiers                      # Classifiers hyperparams
   │   │   ├── lgbm_attributed.gin
   │   │   ├── lgbm.gin
   │   │   ├── mlp_attributed.gin
   │   │   ├── mlp.gin
   │   │   ├── rf_attributed.gin
   │   │   ├── rf.gin
   │   │   ├── svm_attributed.gin
   │   │   ├── svm.gin
   │   │   └── optimization                 # Optuna hyperparam distributions
   │   │       ├── lgbm_hyperparams.gin
   │   │       ├── mlp_hyperparams.gin
   │   │       ├── rf_hyperparams.gin
   │   │       └── svm_hyperparams.gin
   │   └── regressors                       # Regressors hyperparams
   │       ├── lgbm_attributed.gin
   │       ├── lgbm.gin
   │       ├── mlp_attributed.gin
   │       ├── mlp.gin
   │       ├── rf_attributed.gin
   │       ├── rf.gin
   │       ├── svm_attributed.gin
   │       ├── svm.gin
   │       └── optimization                 # Optuna hyperparam distributions
   │           ├── lgbm_hyperparams.gin
   │           ├── mlp_hyperparams.gin
   │           ├── rf_hyperparams.gin
   │           └── svm_hyperparams.gin
   ├── sim_filters
   │   ├── none.gin                         # No filtering of augmenting data
   │   ├── tanimoto_5p_to_test.gin          # Min. distance 0.05 to the test set
   │   ├── tanimoto_10p_to_test.gin         # Min. distance 0.10 to the test set
   │   ├── tanimoto_5p_to_whole.gin         # Min. distance 0.05 to the whole test-origin dataset
   │   ├── tanimoto_10p_to_whole.gin        # ...
   │   ├── ...                              # 20p, 30p, ..., 80p
   │   └── tanimoto_90p_to_whole.gin        # Min. distance 0.90 to the whole test-origin dataset
   └── splitters
       ├── random.gin                       # Random split (test_size = 0.2)
       └── scaffold.gin                     # Bemis-Murcko scaffold split (test_size = 0.2)

Each predictor config (e.g. ``configs/predictors/classifiers/lgbm.gin``) sets the fixed hyperparameters of the model
and includes the matching file from ``optimization/``, which defines the search space used by ``train_optimize.gin``
(see :doc:`models`). The ``*_attributed.gin`` variants additionally set ``do_attributed_learning = True``.

The available predictor configs are LightGBM (``lgbm``), multi-layer perceptron (``mlp``), random forest (``rf``) and
support vector machine (``svm``). The MLP has some specifics described in :ref:`mlp-predictors`.

.. note::

   The ``src/predictor`` package also contains XGBoost (``XGBoostClassifier``, ``XGBoostRegressor``) and Chemprop
   (``ChempropBinaryClassifier``, ``ChempropRegressor``) predictors. They are experimental and have no ready-made
   config files yet.

Similarity filters
------------------

A similarity filter removes augmenting (e.g. rat or mouse) training molecules that are too similar to the human data.
A molecule is kept only if its minimum Tanimoto distance to the reference set is **greater than**
``TanimotoFilter.min_distance_to_test_post_filtering``. The reference set is chosen with ``TanimotoFilter.against``:

- ``'test'`` (``*_to_test.gin``) -- the test set only;
- ``'test_origin'`` (``*_to_whole.gin``) -- the whole test-origin dataset (its train and test parts).

The ``5p``, ``10p``, ... in the file names stand for the minimum distance in percent, so ``tanimoto_5p_to_whole.gin``
drops every augmenting molecule with a Tanimoto similarity of 0.95 or more to any human molecule.

The Tanimoto distances are always computed on ECFP4 bit fingerprints (radius 2, 2048 bits), configured in
``configs/featurizers/proprietary/tanimoto_ecfp.gin``, regardless of the featurizer used by the model. The bindings in
that file are scoped to ``tanimoto_featurizer/``, so the filter and the model featurizer never share settings.

Advanced ``ProcessingPipeline`` settings
----------------------------------------

The following optional parameters can be added to the general config:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Parameter
     - Default
     - Description
   * - ``override_cache``
     - ``False``
     - Regenerate prepared datasets and train-test splits even if they exist in cache.
   * - ``manual_train_splits`` / ``manual_test_splits``
     - ``None``
     - Lists of friendly names of previously saved splits (see ``data/cache/splits/registry.txt``). Used instead of
       automatic splitting when ``datasets`` is empty; this requires a processing plan with
       ``do_load_datasets = False``.
   * - ``hyperparams_source_sim_filter``
     - ``None``
     - With ``train_load_hyperparams.gin``, the similarity filter of the run whose hyperparameters should be loaded
       (see :doc:`features`).
   * - ``ci_n_bootstraps``
     - ``1000``
     - Number of bootstrap resamples of the test set used for metric confidence intervals.
   * - ``ci_percentiles``
     - ``95.0``
     - Width of the confidence intervals, in percent.
   * - ``n_cpus``
     - all available
     - CPU budget of the run. If not set, the smallest of ``SLURM_CPUS_PER_TASK``, the CPU affinity mask and
       ``os.cpu_count()``.
   * - ``n_optim_jobs``
     - ``-1``
     - Number of concurrent Optuna trials; ``-1`` lets the pipeline split the CPU budget automatically.
   * - ``n_threads_per_trial``
     - ``None``
     - Threads per model during optimization; ``None`` lets the pipeline decide.
   * - ``smiles_col`` / ``source_col`` / ``target_col``
     - ``smiles`` / ``source`` / ``y``
     - Column names used internally.
   * - ``label_hist_bins`` / ``ascii_plots``
     - ``10`` / ``False``
     - Appearance of the label-distribution plots in the log.

Configuring an experiment
=========================

ADMET-XSpec is configured exclusively through ``gin`` files. If you are unfamiliar with ``gin``,
We encourage you to take a brief look into to the official `gin-config user guide`_ for a nice introduction.

.. _`gin-config user guide`: https://github.com/google/gin-config/blob/master/docs/index.md

General run config structure
----------------------------

In ADMET-XSpec, each experimental setup is to be defined using one single ``gin`` file, which we will refer to as the
**general config** file. The config file controls data flow through the pipeline, splitters, featurizers, filters, and ML predictors.

**The general config is composed of several smaller config files,** invoked using the ``include`` command, each defining
a single component of the pipeline.

.. info::
   ``include`` imports another gin file into the current config. This is how small reusable config
   fragments are combined into one experiment definition. The ``gin`` config files were meant to be composed,
   not copied.

**A template for a general config file may look like this:**

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

   ProcessingPipeline.test_origin_dataset = '..._friendly_name_1't
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

1. Choose an appropriate processing plan for your workflow.
   - ``train.gin`` for a standard training run with fixed hyperparameters.
   - ``train_optimize.gin`` when hyperparameter optimization is part of the run.
   - ``train_load_hyperparams.gin`` when the model should load previously optimized hyperparameters.
   - ``split.gin`` when you only need the dataset splitting stage.
   - ``normalize.gin`` for preprocessing/normalization-oriented workflows.
   If you are unsure, consult the :ref:`features` section for a description of the available plans.
2. Include gin files representing the chosen splitter type, featurizer, similarity filter and predictor.
3. Set the dataset list using the datasets' friendly names. The list of all friendly names identified in the
   ``data/datasets/`` directory can be found in ``data/datasets/registry.txt`` text file.
4. Set ``ProcessingPipeline.test_origin_dataset`` to the friendly name of a dataset that should populate the test set.
5. Set ``ProcessingPipeline.task_setting`` to match the task type ("binary_classification", "regression")

Component config files
----------------------

Each component (splitter, featurizer, model etc.) of the ``ProcessingPipeline`` is represented by some gin config file
in ``configs/``. The directories containing user-editable gin configs, relevant to the assembly of an ADMET-XSpec
experiment, are outlined below:

.. code-block:: text

   configs
   ├── processing_plans
   │   ├── normalize.gin                    # Pre-process the data only
   │   ├── split.gin                        # Perform train-test split
   │   ├── train.gin                        # Train on fixed hyperparams (parse from config)
   │   ├── train_load_hyperparams.gin       # Train on fixed hyperparams (parse from previous run)
   │   ├── train_optimize.gin               # Perform hyperparam optimization and train
   │   └── *.gin
   ├── featurizers
   │   ├── ecfp_count.gin                   # ECFP4 count variant
   │   ├── ecfp.gin                         # ECFP4
   │   ├── krfp.gin                         # Klekota-Roth fingerprint
   │   ├── maccs.gin                        # MACCS keys fingerprint
   │   ├── map4.gin                         # MAP4 fingerprint
   │   ├── property.gin                     # RDKit physiochemical descriptors
   │   └── *.gin
   ├── predictors
   │   ├── classifiers                      # Classifiers hyperparams
   │   │   ├── lgbm_attributed.gin
   │   │   ├── lgbm.gin
   │   │   ├── rf_attributed.gin
   │   │   ├── rf.gin
   │   │   ├── svm_attributed.gin
   │   │   ├── svm.gin
   │   │   └── optimization                 # Optuna hyperparam distributions
   │   │       ├── lgbm_hyperparams.gin
   │   │       ├── rf_hyperparams.gin
   │   │       └── svm_hyperparams.gin
   │   └── regressors                       # Regressors hyperparams
   │       ├── lgbm_attributed.gin
   │       ├── lgbm.gin
   │       ├── rf_attributed.gin
   │       ├── rf.gin
   │       ├── svm_attributed.gin
   │       ├── svm.gin
   │       └── optimization                 # Optuna hyperparam distributions
   │           ├── lgbm_hyperparams.gin
   │           └── svm_hyperparams.gin
   ├── sim_filters
   │   ├── none.gin
   │   ├── tanimoto_10p_to_test.gin
   │   ├── tanimoto_10p_to_whole.gin
   │   └── *.gin
   └── splitters
       ├── random.gin
       └── scaffold.gin
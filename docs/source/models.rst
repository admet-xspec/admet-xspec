========================================
Training and hyperparameter optimization
========================================

Introductory notes
------------------

ADMET-XSpec is a tool that allows researchers to investigate the
viability of integrating non-human ADMET data into the human drug development
process. It is worthwhile to keep this in mind when considering major elements of how the tool functions,
including:

- Aggregating splits during training
- Filtering using Tanimoto distance to the test set
- Using an "arbitrary" binary classification threshold, such as the median
  of human regression values

.. note::

   **The goal is to explore as many methods of enriching the training set**
   — normally composed of solely human data — **with data sourced from non-human assays or even simple in vitro studies**.
   If a performance loss is observed, we can modify the training set to include only those non-human data points that are
   sufficiently dissimilar to the human dataset (filtering by Tanimoto distance). If establishing a "proper" source of truth for
   the classification threshold is not necessary (a unanimously agreed-upon value for inactivity vs. activity), we can simply study
   if classification accuracy improves when adding new data using the median of the continuous label :math:`y` as the threshold.


How raw data is prepared
------------------------

.. admonition:: Goals
   :class: tip

   After reading this section, you will understand all of the
   transformations applied to molecular data before it ends up being used
   to train a ML model.

Let's discuss the flow of data through the training pipeline in terms of three major steps:

#. **Arriving at the prepared dataset.** This is handled by the
   :class:`~src.data.data_interface.DataInterface` class. The SMILES strings are canonicalized and neutralized, the
   counterions are stripped, and the labels are filtered and transformed according to the dataset's own
   ``params.yaml`` (see :doc:`sourcing_data`).
#. **Splitting the prepared human data** (and integrating augmenting data) to form
   the human test set and (augmented) train set. This is accomplished
   with :class:`~src.processing_pipeline.ProcessingPipeline`, relying on
   :class:`~src.data.data_interface.DataInterface` for data-loading operations and weaving in some of its own
   manipulations.
#. **Filtering the augmenting train samples by minimum Tanimoto distance** to
   molecules present in the whole human dataset / human test set. This is achieved through using
   :class:`~src.data.sim_filter.SimilarityFilterBase` in :class:`~src.processing_pipeline.ProcessingPipeline`.

.. _arriving-at-the-prepared-dataset:

Arriving at the prepared dataset
--------------------------------

Recall these parts of ``AChE/human/regression/params.yaml``:

.. code-block:: yaml
   :lineno-start: 6

   filter_criteria:
       Standard Units:
         - "nM"
       Standard Relation:
         - "'='"
       Standard Type:
         - "IC50"
   label_transformations:
     - "log10"
     - "negate"

and these parts in ``AChE/human/binary_classification/params.yaml``:

.. code-block:: yaml
   :lineno-start: 5

   task_setting: "binary_classification"
   ...
   threshold: -3

You can think of these parameters as "hard-coded" filters that you apply to
a raw dataset included in ``data/datasets``. By the time the data reaches
the boundary of **1: Arriving at the prepared dataset** and enters
**2: Splitting the prepared human data**, it is guaranteed to have the
following qualities:

1. Of the molecules in the raw ``.csv``, only those whose SMILES successfully passed canonicalization, neutralization
   and counterion-stripping remain; for the exact details of this step, see
   :func:`~src.utils.get_clean_smiles`.

2. Of those molecules, only those that met all ``filter_criteria`` (here: standard unit, standard type and, if
   applicable, standard relation) remain. In the case of transforming ChEMBL data to binary classification,
   only those that could be unambiguously placed in either the inactive or active category remain
   (e.g. a ``>`` value above the threshold or a ``<`` value below it).

3. For those molecules, the label transformations have been applied (``log10`` drops non-positive labels), and in the
   binary classification setting, the transformed labels have been converted to ``0`` (inactive) or ``1`` (active).
   The continuous label is kept in the ``y_cont`` column.

The result is saved as ``prepared_data.csv`` in the dataset's directory, together with a ``preparation_log``, and is
reused by later runs unless ``override_cache = True``.

.. important::

   Before proceeding to the next section, the SMILES undergo featurization. For
   the sake of example, we assume that the featurizer that was specified in the
   general config ``.gin`` file of this experiment is an ECFP featurizer.

Splitting and augmenting the human data
---------------------------------------

As the aim of ADMET-XSpec is exploring the usability of non-human ADMET data in the human drug development
process, **the test sets that the ML models are evaluated on are typically human-only data.** The train-test
split is therefore performed on the test-origin (human) dataset only. If any augmenting datasets (e.g. rat, mouse) are
to be included, only the human train set is augmented by concatenation with the new data. Every row keeps the friendly
name of its dataset in the ``source`` column.

The data splitting functionality is handled by the :class:`~src.data.split.DataSplitterBase` class
(:class:`~src.data.split.RandomSplitter`, :class:`~src.data.split.ScaffoldSplitter`).

Filtering the augmenting data by Tanimoto distance
--------------------------------------------------

The Tanimoto distance is a measure of dissimilarity between two molecular fingerprints.
In the context of ADMET-XSpec, it is used to filter the augmenting training samples (e.g. rat-based, mouse-based) based
on their similarity to molecules present in the full human dataset or just the human test set. **This filtering step can
be used to ensure that the label noise introduced by near-duplicate molecules from the rat or mouse datasets does not
throw off the model's performance on the human test set.** The available filters are described in :doc:`config`.

The filtering functionality is handled by the :class:`~src.data.sim_filter.SimilarityFilterBase` class.

Hyperparameter search space configs
-----------------------------------

The configs governing the hyperparameter search space can be found under:

- ``configs/predictors/classifiers/optimization/{model_name}_hyperparams.gin``
- ``configs/predictors/regressors/optimization/{model_name}_hyperparams.gin``

Here is the portion covering the distribution for the LightGBM classifier, as an example:

.. code-block:: text

   ProcessingPipeline.params_distribution = {
   'n_estimators': ('int_log', 100, 2000),
   'max_depth': ('int', 2, 50),
   'num_leaves': ('int', 20, 200),
   'min_child_samples': ('int', 10, 100),
   'learning_rate': ('categorical', [0.01, 0.05, 0.1])
   }

   ProcessingPipeline.n_optim_cv_folds = 5
   ProcessingPipeline.n_optim_iter = 100
   ProcessingPipeline.target_metric = 'roc_auc'

The ``params_distribution`` dictionary maps each hyperparameter to be optimized onto a tuple describing the
distribution to sample from. The optimization uses `Optuna <https://optuna.org/>`_ (TPE sampler), and each tuple is
translated into an Optuna ``suggest_*`` call:

.. list-table::
   :header-rows: 1

   * - Tuple
     - Optuna call
   * - ``('int', low, high)``
     - ``suggest_int(name, low, high)``
   * - ``('int_log', low, high)``
     - ``suggest_int(name, low, high, log=True)``
   * - ``('float', low, high)``
     - ``suggest_float(name, low, high)``
   * - ``('float_log', low, high)``
     - ``suggest_float(name, low, high, log=True)``
   * - ``('categorical', [v1, v2, ...])``
     - ``suggest_categorical(name, [v1, v2, ...])``

The ``n_optim_cv_folds`` and ``n_optim_iter`` parameters define the number of cross-validation folds and the number of trials
to run during the optimization process, respectively. The ``target_metric`` parameter defines the metric to optimize
for during hyperparameter tuning. Supported metrics are ``roc_auc``, ``pr_auc``, ``accuracy``, ``f1``, ``precision``,
``recall`` (classification) and ``r2``, ``spearman``, ``rmse``, ``mae``, ``mse`` (regression); the pipeline maximizes
or minimizes them as appropriate.

Optimization trials run in parallel. See ``n_optim_jobs``, ``n_threads_per_trial`` and ``n_cpus`` in :doc:`config`
to control how the CPU budget is shared between trials.

.. _mlp-predictors:

MLP predictors
--------------

:class:`~src.predictor.scikit_predictors.MlpRegressor` (``MLP_reg``) and
:class:`~src.predictor.scikit_predictors.MlpClassifier` (``MLP_clf``) wrap scikit-learn's
``MLPRegressor`` and ``MLPClassifier``. They are configured in ``configs/predictors/{classifiers,regressors}/mlp.gin``
(and ``mlp_attributed.gin``), and ``configs/examples/train_mlp_optimize.gin`` is a ready-to-run experiment that
optimizes an MLP regressor:

.. code-block:: text

   MlpRegressor.params = {
       'hidden_layer_sizes': [512, 256],
       'activation': 'relu',
       'alpha': 0.001,
       'learning_rate_init': 0.001,
       'batch_size': 64,
       }

Any keyword argument of the scikit-learn estimator can be set in ``params``. Unless overridden there, the predictor
sets the following defaults:

- ``random_state`` -- the predictor's ``random_state`` (default ``42``), so weight initialization, batch
  shuffling and the early-stopping split are reproducible;
- ``early_stopping = True`` -- training stops once the score on an internal validation split (10% of the train set,
  stratified for the classifier) stops improving for 10 epochs;
- ``max_iter = 500`` -- the maximum number of epochs.

**Network architecture.** ``hidden_layer_sizes`` lists the widths of the hidden layers. Optuna cannot sample lists, so
the search space describes the architecture with two scalar hyperparameters instead, which the predictor translates
into ``hidden_layer_sizes = [hidden_layer_width] * n_hidden_layers``:

.. code-block:: text

   ProcessingPipeline.params_distribution = {
       'n_hidden_layers': ('int', 1, 3),
       'hidden_layer_width': ('int_log', 64, 1024),
       'alpha': ('float_log', 1e-6, 1e-1),
       'learning_rate_init': ('float_log', 1e-4, 1e-2),
       'batch_size': ('categorical', [32, 64, 128, 256]),
       'activation': ('categorical', ['relu', 'tanh'])
       }

``n_hidden_layers`` and ``hidden_layer_width`` must be given together and cannot be combined with
``hidden_layer_sizes``. They can also be used in ``params``. The search space only covers networks whose hidden layers
all have the same width; tapered networks (e.g. ``[512, 256]``) can only be set as fixed ``hidden_layer_sizes``.
In ``hyperparams.yaml``, the architecture is always saved as a ``hidden_layer_sizes`` list, so
``train_load_hyperparams.gin`` can reuse it.

**Featurizers.** The MLP does not scale its input features. ``property.gin`` features are already standardized, and
bit fingerprints (``ecfp.gin``, ``maccs.gin``, ``krfp.gin``) are binary. ``ecfp_count.gin`` features are unscaled
counts, which can slow down or destabilize training. ``minimol.gin`` embeddings are unscaled too, but within a
narrow range (roughly 0 to 8).

**Convergence warnings.** scikit-learn emits a ``ConvergenceWarning`` when training reaches ``max_iter`` before
converging. These warnings are not suppressed, and during hyperparameter optimization they can be expected for some
trials (e.g. with a low ``learning_rate_init``). Frequent warnings with the final hyperparameters suggest raising
``max_iter`` in ``params``.

**Threading.** scikit-learn MLPs have no ``n_jobs`` parameter, so the pipeline treats them as single-threaded models:
during hyperparameter optimization, each trial gets one thread (its BLAS thread pool is limited accordingly) and the
CPUs are used by running more trials concurrently. Outside of optimization, BLAS uses its default number of threads.

Attributed learning (``mlp_attributed.gin``) works as for the other models: the endpoint one-hot vector is appended to
the features.

Where are the models saved?
---------------------------

.. admonition:: Goals
   :class: tip

   After reading this section, you will understand the contents of a
   successful training run outputted to ``data/cache``.

Models are outputted to ``data/cache/models``. Let's look at an example of
the result of a :class:`~src.processing_pipeline.ProcessingPipeline` run that successfully trained a
model:

.. code-block:: text

   AChE_human_pIC50_LightGBM_clf_ecfp_featurizer_4b52a
   └── scaffold_e4737_tanimoto_5p_filter_dfa1b_91da5
       ├── bootstrap_data.yaml
       ├── hyperparams.yaml
       ├── metrics.yaml
       ├── model_final_refit.pkl
       ├── model_metadata.yaml
       ├── model.pkl
       ├── operative_config.gin
       └── training_log
           └── console.log

Let us now discuss the contents of each of these files:

#. ``bootstrap_data.yaml`` contains the value of every metric for every bootstrap resample of the test set
   (only with ``do_get_metrics_confidence_interval = True``).

#. ``hyperparams.yaml`` contains the hyperparameters used to train the model. If
   hyperparameter optimization was performed, this file will contain the
   best hyperparameters found during that process.

#. ``metrics.yaml`` contains the evaluation metrics for the model on the test set and, if enabled,
   the bootstrap-derived confidence intervals for each metric.

#. ``model_final_refit.pkl`` is a binary file containing the model refit on the entire train+test dataset
   (only with ``do_refit_final_model = True`` and ``do_save_trained_model = True``).

#. ``model_metadata.yaml`` contains some useful metadata about the model, including the exact components of
   the :class:`~src.processing_pipeline.ProcessingPipeline` config used to train it.

#. ``model.pkl`` is a binary file containing the model trained on the train set only
   (only with ``do_save_trained_model = True``).

#. ``operative_config.gin`` is a copy of the full contents of the general ``gin`` config file used to train the model.

#. ``training_log/console.log`` contains the console output of the training process, including any warnings or errors.

Directory naming convention
---------------------------

The directory name (``AChE_human_pIC50_LightGBM_clf_ecfp_featurizer_4b52a``) for the model is composed of:

- The test-origin dataset's friendly name: ``AChE_human_pIC50``
- The model name: ``LightGBM_clf`` (a classifier, in this case)
- The featurizer name: ``ecfp_featurizer``
- The featurizer's "hash code", a result of MD5 hashing its parameters:
  ``4b52a``

Attributed models get an additional ``attributed_`` prefix. Model hyperparameters are **not** part of the name, so
retraining the same model with different hyperparameters on the same data overwrites the previous results.

The directory name (``scaffold_e4737_tanimoto_5p_filter_dfa1b_91da5``) for the data that resulted in this model being trained
(one model can have multiple such subdirectories) follows suit and is
composed of:

- The splitter key, containing the splitter type and its hashed parameters:
  ``scaffold_e4737``
- The filter key, again with a hash code of the filter's parameters (including its fingerprint):
  ``tanimoto_5p_filter_dfa1b`` (``nofilter`` if no filter was used)
- A hash code of the dataset list, the test-origin dataset and the task setting: ``91da5``

The same key is used for the train-test split saved in ``data/cache/splits``.

.. note::

   All of this hashing is done to aid tracking of models in a more
   organized way than simply appending the date to their name. As shown in
   the example above, there is plenty of metadata to go along with this
   technique, which was covered in :ref:`arriving-at-the-prepared-dataset`.

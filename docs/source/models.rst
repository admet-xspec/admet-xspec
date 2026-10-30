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
the search space describes the architecture with scalar hyperparameters instead, which the predictor translates into
``hidden_layer_sizes``:

.. code-block:: text

   ProcessingPipeline.params_distribution = {
       'n_hidden_layers': ('int', 1, 3),
       'hidden_layer_width': ('int_log', 64, 1024),
       'layer_shape': ('categorical', ['constant', 'funnel', 'free']),
       'alpha': ('float_log', 1e-6, 1e-1),
       'learning_rate_init': ('float_log', 1e-4, 1e-2),
       'batch_size': ('categorical', [32, 64, 128, 256]),
       'activation': ('categorical', ['relu', 'tanh'])
       }

``n_hidden_layers`` is the number of hidden layers and ``hidden_layer_width`` the width of the first one.
``layer_shape`` sets the widths of the following layers:

- ``constant`` -- every layer is ``hidden_layer_width`` wide, e.g. ``[512, 512, 512]``;
- ``funnel`` -- each layer is half as wide as the previous one, e.g. ``[512, 256, 128]``;
- ``free`` -- every layer gets its own width, e.g. ``[512, 96, 700]``. The widths of layers 2, 3, ... are sampled as
  ``hidden_layer_width_2``, ``hidden_layer_width_3``, ... from the ``hidden_layer_width`` distribution. These are
  conditional hyperparameters: a trial only samples the widths of the layers it has, and trials with another shape
  sample none.

Omitting ``layer_shape`` restricts the search to ``constant`` networks, and a shape can be excluded by leaving it out
of the list. ``layer_shape`` can only be sampled together with ``n_hidden_layers`` and ``hidden_layer_width``.

The same scalars can be used in ``params``, e.g. ``'n_hidden_layers': 3, 'hidden_layer_width': 512,
'layer_shape': 'funnel'``. ``n_hidden_layers`` and ``hidden_layer_width`` must be given together and cannot be combined
with ``hidden_layer_sizes``. ``layer_shape`` defaults to ``constant``, and the ``free`` shape needs exactly
``hidden_layer_width_2`` to ``hidden_layer_width_<n_hidden_layers>``. In ``hyperparams.yaml``, the architecture is
always saved as a ``hidden_layer_sizes`` list, so ``train_load_hyperparams.gin`` can reuse it.

**Featurizers.** The MLP does not scale its input features itself. Descriptors (``property.gin`` and the descriptor
block of ``maplight.gin``) are standardized by the predictor, with a scaler fitted on its training data, and bit
fingerprints (``ecfp.gin``, ``maccs.gin``, ``krfp.gin``) are binary. ``ecfp_count.gin`` features and the count
fingerprints of ``maplight.gin`` are unscaled counts, which can slow down or destabilize training. ``minimol.gin``
embeddings are unscaled too, but within a narrow range (roughly 0 to 8).

**Convergence warnings.** scikit-learn emits a ``ConvergenceWarning`` when training reaches ``max_iter`` before
converging. These warnings are not suppressed, and during hyperparameter optimization they can be expected for some
trials (e.g. with a low ``learning_rate_init``). Frequent warnings with the final hyperparameters suggest raising
``max_iter`` in ``params``.

**Threading.** scikit-learn MLPs have no ``n_jobs`` parameter, so the pipeline treats them as single-threaded models:
during hyperparameter optimization, each trial gets one thread (its BLAS thread pool is limited accordingly) and the
CPUs are used by running more trials concurrently. Outside of optimization, BLAS uses its default number of threads.

Attributed learning (``mlp_attributed.gin``) works as for the other models: the endpoint one-hot vector is appended to
the features.

.. _maplight-featurizer:

MapLight featurizer
-------------------

:class:`~src.data.featurizer.MapLightFeaturizer` (``configs/featurizers/maplight.gin``) reproduces the features of
MapLight (`Notwell & Wood, 2023 <https://arxiv.org/abs/2310.00174>`_), a combination of molecular fingerprints and
descriptors evaluated on the Therapeutics Data Commons ADMET benchmark (`reference implementation
<https://github.com/maplightrx/MapLight-TDC/blob/main/maplight.py>`_). Each molecule gets four blocks of features,
concatenated in this order:

.. list-table::
   :header-rows: 1

   * - Block
     - Columns
     - Values
   * - Morgan (ECFP4) counts
     - ``morgan_n_bits`` (1024)
     - integer counts
   * - Avalon counts
     - ``avalon_n_bits`` (1024)
     - integer counts
   * - ErG pharmacophore fingerprint
     - 315
     - non-negative floats
   * - RDKit descriptors
     - 200
     - raw descriptor values

With the defaults, that is 2563 columns. The parameters, set in ``maplight.gin``, are:

.. code-block:: text

   MapLightFeaturizer.morgan_radius = 2
   MapLightFeaturizer.morgan_n_bits = 1024
   MapLightFeaturizer.avalon_n_bits = 1024

The defaults match the original implementation, and the features are tested against it. The 200 descriptors are the
fixed list used by MapLight, not every descriptor of the installed RDKit, so new RDKit releases don't change the
columns. They differ from ``property.gin``, which uses all RDKit descriptors.

**Scaling.** Features are cached on disk unscaled. The predictor standardizes only the 200 descriptors, with a scaler
fitted on its training data (and refitted in every cross-validation fold), and passes the fingerprint blocks through
unchanged. This does not matter for tree models, but it does for the MLP and SVM (see :ref:`mlp-predictors`). The
original MapLight trains CatBoost on unscaled features.

**Invalid molecules.** SMILES that RDKit cannot parse raise an error naming them, instead of being featurized as
zeros. Descriptors that fail or overflow ``float32`` (e.g. ``Ipc`` of very large molecules) are replaced with zeros,
with a warning.

**Not included.** MapLight also standardizes the regression targets (optionally log-transforming them first) and
trains CatBoost models. Neither is part of the featurizer, and neither is done by this pipeline. The MapLight + GNN variant,
which adds pre-trained GIN embeddings, is not implemented either.

.. _applicability-domain:

Applicability domain
--------------------

A model is only reliable for molecules that resemble its training data. The **applicability domain** (AD) measures
that resemblance, so predictions for unfamiliar molecules can be flagged. It is configured in
``configs/applicability_domains/`` and implemented in :mod:`src.data.applicability_domain`.

**The measure.** :class:`~src.data.applicability_domain.TanimotoKnnDomain` (``tanimoto_knn.gin``) scores a molecule by
its mean Tanimoto similarity (2048-bit ECFP4) to the ``k`` most similar training molecules (5 by default), and also
reports the most similar training molecule. It does not depend on the model, so it works for every featurizer and
predictor.

**The threshold.** Each training molecule is scored against the rest of the training set (leave-one-out). A molecule is
in the domain if its score is at least the ``threshold_percentile``-th percentile of these scores (5 by default), so
95% of the training set is in the domain by construction. Duplicate training SMILES are counted once.

.. code-block:: text

   TanimotoKnnDomain.k = 5
   TanimotoKnnDomain.threshold_percentile = 5.0
   TanimotoKnnDomain.reference_sources = None   # e.g. ['AChE_human_pIC50'] to ignore augmenting data

**Calibration.** A threshold alone does not show that the domain matters for a given model. Each model therefore gets a
calibration report: its test-set metrics for molecules inside and outside the domain, and for each quartile of the
domain score. If out-of-domain errors are not clearly worse, the domain is not informative for that model. The report
is computed with ``model.pkl``, the model trained on the train split. Random splits leave few test molecules outside
the domain, so scaffold splits give more informative calibrations.

**Storage.** A fitted domain depends only on the training data, so it is saved once per split and shared by every model
trained on it. Each model keeps only its calibration:

.. code-block:: text

   data/cache/splits/<data_cache_key>/applicability_domain/<domain_key>/
       domain.pkl           # fitted on the train split; used with model.pkl
       domain_refit.pkl     # fitted on train + test; used with model_final_refit.pkl
   data/cache/models/<model_cache_key>/<data_cache_key>/applicability_domain/<domain_key>/
       calibration.yaml

**During training.** Set the flag and configure the domain in the experiment config:

.. code-block:: text

   include 'configs/applicability_domains/tanimoto_knn.gin'
   ProcessingPipeline.applicability_domain = %applicability_domain
   ProcessingPipeline.do_fit_applicability_domain = True

The domains are fitted and the model calibrated right after evaluation, before the refit. The domain is saved with the
split, so the split must be cached (``do_dump_train_test = True``).

**For models trained earlier.** Domains need only the cached split, and calibration also needs ``model.pkl``. There are
two ways to add them:

- ``applicability_domain.py`` processes every cached model, or those matching glob patterns of their cache keys:

  .. code-block:: bash

     python applicability_domain.py --models 'AChE_human_pIC50_*' --splits 'scaffold_*'

  Each split's domain is fitted once, then every model with a ``model.pkl`` is calibrated. ``--cfg`` selects another
  domain config, ``--no-calibrate`` only fits the domains, and ``--override-cache`` refits and recalibrates.
- To process a single experiment, swap its processing plan for ``configs/processing_plans/fit_applicability_domain.gin``
  and add the three lines above. The plan neither loads data nor trains; it uses the cache keys of the experiment.

Inference fits a missing domain on first use as well (see :doc:`inference`), but it cannot calibrate it.

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
       ├── applicability_domain
       │   └── tanimoto_knn_domain_1f43c
       │       └── calibration.yaml
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

#. ``applicability_domain/<domain_key>/calibration.yaml`` contains the test metrics of ``model.pkl`` inside and outside
   its applicability domain (only with ``do_fit_applicability_domain = True``; see :ref:`applicability-domain`).

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

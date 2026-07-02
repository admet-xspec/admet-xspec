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
- Using an "arbitrary" binary classification threshold of the mean or
  median of human regression values

.. note::

   **The goal is to explore as many methods of enriching the training set**
   — normally composed of solely human data — **with data sourced from non-human assays or even simple in vitro studies**.
   If a performance loss is observed, we can modify the training set to include only those non-human data points that are
   sufficiently dissimilar to the human dataset (filtering by Tanimoto distance). If establishing a "proper" source of truth for
   classification threshold is not necessary (a unanimously agreed-upon value for inactivity vs. activity), we can simply study
   if classification accuracy improves when adding new data using the median of the coninous label :math:`y` as the threshold.


How raw data is prepared
------------------------

.. admonition:: Goals
   :class: tip

   After reading this section, you will understand all of the
   transformations applied to molecular data before it ends up being used
   to train a ML model.

Let's discuss the flow of data through the training pipeline in terms of four major steps:

#. **Arriving at the prepared dataset.** This is handled by the
   :class:`DataInterface` class. The smiles strings are canonicalized, neutralized, the counterions
    are stripped, and the labels are transformed according to the specifications in ``params.yaml``.
#. **Splitting the prepared human data** (and integrating augmenting data) to form
   the human test set and (augmented) train set. This is accomplished
   with :class:`ProcessingPipeline`, relying on :class:`DataInterface` for
   data-loading operations and weaving in some of its own manipulations.
#. **Filtering the augmenting train samples by minimum Tanimoto distance** to
   molecules present in the whole human dataset / human test set. This is achieved through using
   :class:`SimilarityFilterBase` in :class:`ProcessingPipeline`.

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
   threshold: "median"

You can think of these parameters as "hard-coded" filters that you apply to
a raw dataset included in ``data/datasets``. By the time the data reaches
the boundary of **1: Arriving at the prepared dataset** and enters
**2: Splitting the prepared human data**, it is guaranteed to have the
following qualities:

1. Only those molecules in the ChEMBL ``.csv`` that met the standard unit,
   standard value, and (if applicable) standard relation criteria remain; in
   the case of transforming to binary classification, only those that could
   be unambiguously placed in either the inactive or active category
   (appropriate ``<`` or ``>`` values).

2. Of those molecules, only those whose SMILES successfully passed canonicalization, neutralization
   and counterion-stripping remain; for the exact details of this step, see
   :func:`~src.utils.get_clean_smiles`.

3. For those molecules, the label transformations have been applied in the
   regression setting, and in the binary classification setting, the labels
   have been converted to ``0`` (inactive) or ``1`` (active).

.. important::

   Before proceeding to the next section, the SMILES undergo featurization. For
   the sake of example, we assume that the featurizer that was specified in the
   general config `.gin` file of this experiment is an ECFP featurizer.

Splitting and augmenting the human data
---------------------------------------

As the aim of ADMET-XSpec is exploring the usability of non-human ADMET data in human drug development
process, **the test sets that the ML models are evaluated on are typically human-only data.*** The train-test
split is therefore performed first on human data only. If any augmenting datasets (ex. rat, mouse) are to be included,
only the human train is augmented by concatenation with the new data.

The data splitting functionality is handled by :class:`DataSplitterBase` class.

Filtering the augmenting data by Tanimoto distance
--------------------------------------------------

The Tanimoto distance is a measure of similarity between two molecular fingerprints.
In the context of ADMET-XSpec, it is used to filter the augmenting training samples (ex. rat-based, mouse-based) based
on their similarity to molecules present in the full human dataset or just the human test set. **This filtering step can
be used to ensure that the label noise introduced by near-duplicate molecules from the rat or mouse datasets does not
throw off the model's performance on the human test set.**

The filtering functionality is handled by :class:`SimilarityFilterBase` class.

Hyperparameter search space configs
-----------------------------------

The configs governing the hyperparameter search space can be found under:

- ``configs/predictors/classifiers/optimization/{model_name}_hyperparams.gin``
- ``configs/predictors/regressors/optimization/{model_name}_hyperparams.gin``

Here is the portion covering the distribution for LightGBM, as an example:

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

   The ``params_distribution`` dictionary defines the hyperparameters to be optimized,
   the type of distribution to sample from, and the range of values to sample. As our
   optimization method makes use of `Optuna <https://optuna.org/>`_, the types of distributions
   are those supported by Optuna.

   The ``n_optim_cv_folds`` and ``n_optim_iter`` parameters define the number of cross-validation folds and the number of trials
   to run during the optimization process, respectively. The ``target_metric`` parameter defines the metric to optimize for during hyperparameter tuning.

Where are the models saved?
---------------------------

.. admonition:: Goals
   :class: tip

   After reading this section, you will understand the contents of a
   successful training run outputted to ``data/cache``.

Models are outputted to ``data/cache/models``. Let's look at an example of
the result of a :class:`ProcessingPipeline` run that successfully trained a
model:

.. code-block:: text

   LightGBM_clf_ecfp_featurizer_4b52a
   └── scaffold_e4737_tanimoto_5p_filter_c2805_91da5
       ├── hyperparams.yaml
       ├── metrics.yaml
       ├── model_final_refit.pkl
       ├── model_metadata.yaml
       ├── model.pkl
       ├── operative_config.gin
       └── training_log
           └── console.log

Let us now discuss the contents of each of these files:

#. ``hyperparams.yaml`` contains the hyperparameters used to train the model. If
   hyperparameter optimization was performed, this file will contain the
   best hyperparameters found during that process.

#. ``metrics.yaml`` contains the evaluation metrics for the model on the test set,
   including bootstrap-derived confidence intervals for each metric.

#. ``model_final_refit.pkl`` is a binary file containing the model refit on the entire train+test dataset.

#. ``model_metadata.yaml`` contains some useful metadata about the model, including the exact components of
   :class:`ProcessingPipeline` config used to train it.

#. ``model.pkl`` is a binary file containing the model trained on the train set only.

#. ``operative_config.gin`` is a copy of the full contents of the general `gin` config file used to train the model.

#. ``training_log/console.log`` contains the console output of the training process, including any warnings or errors.

Directory naming convention
---------------------------

The directory name (``LightGBM_clf_ecfp_featurizer_4b52a``) for the model is composed of:

- The model name: ``LightGBM_clf`` (a classifier, in this case)
- The featurizer name: ``ecfp_featurizer``
- The featurizer's "hash code", a result of MD5 hashing its parameters:
  ``4b52a``

The directory name (``scaffold_e4737_tanimoto_5p_filter_c2805_91da5``) for the data that resulted in this model being trained
(one model can have multiple such subdirectories) follows suit and is
composed of:

- The splitter key, containing the splitter type and its hashed parameters:
  ``scaffold_e4737``
- The filter key, again with a hash code: ``tanimoto_5p_filter_c2805``
- The dataset's hash code: ``91da5``

.. note::

   All of this hashing is done to aid tracking of models in a more
   organized way than simply appending the date to their name. As shown in
   the example above, there is plenty of metadata to go along with this
   technique, which was covered in :ref:`arriving-at-the-prepared-dataset`.

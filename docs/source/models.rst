========================================
Training and hyperparameter optimization
========================================

Introductory notes
==================

ADMET-XSpec is a tool that seeks to allow researchers to investigate the
viability of integrating non-human assays into the human drug development
process, specifically with respect to ADMET properties. It is worthwhile to
keep this in mind when considering major elements of how the tool functions,
including:

- Aggregating splits during training
- Filtering using Tanimoto distance to the test set
- Using an "arbitrary" binary classification threshold of the mean or
  median of human regression values

.. note::

   In short, **we want to try out as many methods of enriching the training
   set** — normally composed of solely human data — **with mouse and rat
   data**. If performance worsens, we want to amend our approach
   (filtering). If establishing a "proper" source of truth for
   classification is not necessary (a unanimously agreed-upon value for
   inactivity vs. activity), we simply want to see if classification
   accuracy improves when adding new data (using the mean or median of
   regression values as the classification threshold).

.. contents:: Sections
   :local:
   :depth: 1

How the raw data is transformed
================================

.. admonition:: Goals
   :class: tip

   After reading this section, you will understand all of the
   transformations applied to the data before it ends up in a training loop
   for an ML method.

Let's base our discussion on a diagram that illustrates the flow of data:

.. figure:: train_optimize_diagram.png
   :alt: Data flow from raw ChEMBL data to model training
   :align: center

   Flow of raw ChEMBL human, mouse, and rat data into the model-training
   classes implementing :class:`ScikitPredictorBase`.

This diagram shows how we take raw ChEMBL human, mouse, and rat data and
provide it to model training classes implementing the
:class:`ScikitPredictorBase` interface.

Let's briefly note the three color-coded sections, which group together
concerns at the different stages of data transformation:

#. **Arriving at the prepared dataset.** This is mostly accomplished by the
   :class:`DataInterface` class.
#. **Splitting the prepared human data and integrating rodent data** to form
   the test set (final) and aggregate train set (raw). This is accomplished
   with :class:`ProcessingPipeline`, relying on :class:`DataInterface` for
   data-loading operations and weaving in some of its own manipulations.
#. **Filtering the aggregate train set by minimum Tanimoto distance** to
   molecules in the test set. This is best explained by the pseudocode
   ``for`` loop in the diagram and is achieved through using
   :class:`SimilarityFilterBase` in :class:`ProcessingPipeline`.

.. _arriving-at-the-prepared-dataset:

Arriving at the prepared dataset
---------------------------------

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

- Only those molecules in the ChEMBL ``.csv`` that met the standard unit,
  standard value, and (if applicable) standard relation criteria remain; in
  the case of transforming to binary classification, only those that could
  be unambiguously placed in either the inactive or active category
  (appropriate ``<`` or ``>`` values).
- Of those molecules, only those whose SMILES passed canonicalization
  remain; for the exact details of this step, see
  :func:`~src.utils.get_clean_smiles`.
- For those molecules, the label transformations have been applied in the
  regression setting, and in the binary classification setting, the labels
  have been converted to ``0`` (inactive) or ``1`` (active).

.. important::

   Before proceeding to the next section, the SMILES are featurized. For
   the sake of example, we assume that the ECFP4 fingerprint featurizer is
   employed.

Splitting the prepared human data, integrating rodent data
-------------------------------------------------------------

.. todo::

   Description of scaffold splitting and its motivation.

Filtering the aggregate train set by minimum Tanimoto distance
-------------------------------------------------------------------

.. todo::

   Description of filtering by Tanimoto distance and its motivation.

``ScikitPredictorBase`` as the model-training interface
=========================================================

.. admonition:: Goals
   :class: tip

   After reading this section, you will understand how we use
   scikit-learn's well-established interface to train models, as well as
   how we run optimization and save optimized hyperparameters, which can be
   used for future runs.

   You will also know where to find the configs responsible for the ranges
   and distributions from which we sample hyperparameters for the
   optimization search.

:class:`ScikitPredictorBase` exposes the following public methods:

.. code-block:: python
   :lineno-start: 1

   def train(self, smiles_list, target_list): ...
   def optimize(self, smiles_list, target_list): ...
   def predict(self, smiles_list): ...
   def get_hyperparameters(self): ...
   def set_hyperparameters(self, params): ...

.. py:method:: train(smiles_list, target_list)

   Trains a model based on a set of data and internal hyperparameters.
   These are provided through an ``experiment_config.gin`` (the filename is
   used as an example).

.. py:method:: optimize(smiles_list, target_list)

   Performs a `RandomizedSearchCV
   <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_
   to train a model (and discard it) and save the optimal hyperparameters
   internally. If :meth:`train` is run afterwards, the model is trained
   with those optimal hyperparameters.

.. py:method:: predict(smiles_list)

   Returns a list of predictions: probability of the positive class for
   classifiers, and predicted values for regressors. This can be used to
   collect metrics about the fit.

.. py:method:: get_hyperparameters()

   Retrieves the hyperparameters stored within the class.

.. py:method:: set_hyperparameters(params)

   Sets those hyperparameters.

Motivating use cases
---------------------

Here are three use cases of :class:`ProcessingPipeline` that motivate these
methods:

#. Simply training a model with predefined ``experiment_config.gin``
   parameters and outputting it to ``./data/cache/models``.
#. Finding optimal hyperparameters by having :meth:`optimize` find them, and
   then training a model on those parameters — in this case, *we provide
   ranges and distributions for sampling the hyperparameters* (see below).
#. Training a model on optimal hyperparameters without finding them via
   :meth:`optimize` — instead, loading optimal hyperparameters that were
   already saved to disk.

.. seealso::

   These various use cases are covered by different processing plans,
   which were introduced earlier in this documentation.

Hyperparameter search space configs
-------------------------------------

The configs governing the hyperparameter search space can be found under:

- ``./configs/predictors/classifiers/optimization/*_hyperparams.gin``
- ``./configs/predictors/regressors/optimization/*_hyperparams.gin``

Here is the portion covering the distribution for LightGBM, as an example:

.. code-block:: text

   LightGbmClassifier.params_distribution = {
       'n_estimators': @n_estimators/QLogUniform(),
       'max_depth': @max_depth/QUniform(),
       'num_leaves': @num_leaves/QUniform(),
       'min_child_samples': @min_child_samples/QUniform(),
       'learning_rate': [0.01, 0.05, 0.1],
   }

Where to find outputted models
================================

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

Directory naming convention
-----------------------------

The directory name for the model is composed of:

- The model name: ``LightGBM_clf`` (a classifier, in this case)
- The featurizer name: ``ecfp_featurizer``
- The featurizer's "hash code," a result of MD5 hashing its parameters:
  ``4b52a``

The directory name for the data that resulted in this model being trained
(one model can have multiple such subdirectories) follows suit and is
composed of:

- The splitter key, containing the splitter type and hash code:
  ``scaffold_e4737``
- The filter key, again with a hash code: ``tanimoto_5p_filter_c2805``
- The dataset's hash code: ``91da5``

.. note::

   All of this hashing is done to aid tracking of models in a more
   organized way than simply appending the date to their name. As shown in
   the example above, there is plenty of metadata to go along with this
   technique, which was covered in :ref:`arriving-at-the-prepared-dataset`.

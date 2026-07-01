=============================
Choosing your processing plan
=============================

**The main goal of** :class:`ProcessingPipeline` **is to provide a single interface for
training and evaluating models on heterogeneous datasets. This pipeline is
designed to be flexible and modular, allowing users to easily configure and
run experiments with different datasets, featurizers, models, and hyperparameter
optimization methods.** The pipeline is also designed to be efficient, with
caching and logging functionality built in to minimize redundant computations
and facilitate reproducibility.

The different dataset concatenations, train-test splits and trained model weights are
cached and easily retrieved, minimizing the need for the user to manually
copy, move, rename and organize files anywhere on disk.

To support this functionality, we use what we call **"processing plans"**. The
name reflects the fact that they control the flow of :class:`ProcessingPipeline`.

.. admonition:: Goals
   :class: tip

   After reading this section, you should understand:

   #. What processing plans represent and how they connect to the
      :class:`ProcessingPipeline` class.
   #. What processing plans are available out of the box and when to use them.
   #. How to create your own processing plan for a custom workflow.

The nine steps of ProcessingPipeline
------------------------------------

At a high level, a processing plan controls these nine steps in the
execution of :class:`ProcessingPipeline`:

.. code-block:: text

   # Step 1: Load datasets
   # Step 2: Create train/test splits (if they do not exist in cache)
   # Step 3: Save train/test splits to cache
   # Step 4: Load previously-optimized hyperparameters
   # Step 5: Optimize hyperparameters with Optuna
   # Step 7: Train the model
   # Step 8: Calculate confidence intervals for metrics
   # Step 9: Save the trained model to cache
   # Step 10: Refit on the entire train+test dataset and save to cache

These steps are all present in the :meth:`~ProcessingPipeline.run`
method itself. They are placed between control-flow blocks to disable
certain parts from running, depending on the chosen processing plan.

.. tip::

   The file ``configs/processing_plans/_possible_plans.gin`` serves as a
   reminder of what plans you can create whenever you find yourself outside
   of our docs.

Example plan - Train
--------------------

Let's look at ``configs/processing_plans/train.gin``, which is one of the simplest processing plans that yield
a trained model. The contents of this file are as follows:

.. code-block:: python
   ProcessingPipeline.do_load_datasets = True
   ProcessingPipeline.do_load_train_test = True
   ProcessingPipeline.do_dump_train_test = True
   ProcessingPipeline.do_load_optimized_hyperparams = False
   ProcessingPipeline.do_optimize_hyperparams = False
   ProcessingPipeline.do_train_model = True
   ProcessingPipeline.do_get_metrics_confidence_interval = True
   ProcessingPipeline.do_save_trained_model = True
   ProcessingPipeline.do_refit_final_model = False

According to this processing plan, the :class:`ProcessingPipeline` will:

#. Load your raw datasets and preprocess them.
#. Load your train-test splits from cache, if they exist. If they do not exist,
   the pipeline will perform the split on specified datasets.
#. Save the train-test splits to cache.
#. **Not** load hyperparameters found to be optimal in a previous run, since ``do_load_optimized_hyperparams`` it is set
   to ``False``.
#. **Not** Optimize hyperparameters of the model.
#. Train a model on the train-test split using set hyperparameter dictionary.
   An example of a config file defining a a fixed hyperparameter dict is ``configs/predictors/classifier/lgbm.gin``.
#. Estimate confidence intervals for the metrics with bootstrapping.
#. Save the trained predictor as a ``.pkl`` file to cache.
#. **Not** refit the model on the entire train+test dataset and save it to cache.

Example plan - Train and optimize
---------------------------------

Let's look at ``configs/processing_plans/train_optimize.gin``, the one you
are likely to be using rather often. The contents of this file are as follows:

.. code-block:: python
   ProcessingPipeline.do_load_datasets = True
   ProcessingPipeline.do_load_train_test = True
   ProcessingPipeline.do_dump_train_test = True
   ProcessingPipeline.do_load_optimized_hyperparams = False
   ProcessingPipeline.do_optimize_hyperparams = True
   ProcessingPipeline.do_train_model = True
   ProcessingPipeline.do_get_metrics_confidence_interval = True
   ProcessingPipeline.do_save_trained_model = True
   ProcessingPipeline.do_refit_final_model = True

According to this processing plan, the :class:`ProcessingPipeline` will:

#. Load your raw datasets and preprocess them.
#. Load your train-test splits from cache, if they exist. If they do not exist,
   the pipeline will perform the split on specified datasets.
#. Save the train-test splits to cache.
#. **Not** load hyperparameters found to be optimal in a previous run, since ``do_load_optimized_hyperparams`` it is set
   to ``False``.
#. Optimize hyperparameters using Optuna, retain the best hyperparameters found, and save them to cache.
#. Train a model on the train-test split using the best hyperparameters found in the previous step.
#. Estimate confidence intervals for the metrics with bootstrapping.
#. Save the trained predictor as a ``.pkl`` file to cache.
#. Refit the model on the entire train+test dataset and save it to cache.

Other processing plans
======================

The other processing plans are similar to the one above, but with different combinations of steps enabled or disabled.

#. ``normalize.gin`` will not train a model, but will instead perform the standard pre-processing steps on a raw dataset
and save the normalized dataset to cache.

#. ``split.gin`` will not train a model, but will instead perform the standard pre-processing steps on a raw dataset,
create train-test splits, handle non-human data augmentations / filtering and save the train-test splits to cache.

#. ``train_load_hyperparams.gin`` will search cache to find previously optimized hyperparameters for the exact splitter
/ featurizer / model combination, parse them and use them to train a model on some train-test split.
=============================
Choosing your processing plan
=============================

**The main goal of** :class:`~src.processing_pipeline.ProcessingPipeline` **is to provide a single interface for
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
name reflects the fact that they control the flow of :class:`~src.processing_pipeline.ProcessingPipeline`.

.. admonition:: Goals
   :class: tip

   After reading this section, you should understand:

   #. What processing plans represent and how they connect to the
      :class:`~src.processing_pipeline.ProcessingPipeline` class.
   #. What processing plans are available out of the box and when to use them.
   #. How to create your own processing plan for a custom workflow.

The nine steps of ProcessingPipeline
------------------------------------

At a high level, a processing plan controls these nine steps in the
execution of :class:`~src.processing_pipeline.ProcessingPipeline`. Each step is switched on or off by one
``do_*`` flag:

.. code-block:: text

   # Step 1: Load datasets                                   do_load_datasets
   # Step 2: Create train/test splits (or load from cache)   do_load_train_test
   # Step 3: Save train/test splits to cache                 do_dump_train_test
   # Step 4: Load previously-optimized hyperparameters       do_load_optimized_hyperparams
   # Step 5: Optimize hyperparameters with Optuna            do_optimize_hyperparams
   # Step 6: Train and evaluate the model                    do_train_model
   # Step 7: Calculate confidence intervals for metrics      do_get_metrics_confidence_interval
   # Step 8: Save the trained model to cache                 do_save_trained_model
   # Step 9: Refit on the entire train+test dataset          do_refit_final_model

These steps are all present in the :meth:`~src.processing_pipeline.ProcessingPipeline.run`
method itself. They are placed between control-flow blocks to disable
certain parts from running, depending on the chosen processing plan.
If ``do_train_model`` is ``False``, steps 7-9 are skipped as well. The refit model of step 9 is only saved
if ``do_save_trained_model`` is also ``True``.

.. tip::

   The file ``configs/processing_plans/_possible_plans.gin`` serves as a
   reminder of what plans you can create whenever you find yourself outside
   of our docs.

Example plan - Train
--------------------

Let's look at ``configs/processing_plans/train.gin``, which is one of the simplest processing plans that yield
a trained model. The contents of this file are as follows:

.. code-block:: text

   ProcessingPipeline.do_load_datasets = True
   ProcessingPipeline.do_load_train_test = True
   ProcessingPipeline.do_dump_train_test = True
   ProcessingPipeline.do_load_optimized_hyperparams = False
   ProcessingPipeline.do_optimize_hyperparams = False
   ProcessingPipeline.do_train_model = True
   ProcessingPipeline.do_get_metrics_confidence_interval = True
   ProcessingPipeline.do_save_trained_model = True
   ProcessingPipeline.do_refit_final_model = True

According to this processing plan, the :class:`~src.processing_pipeline.ProcessingPipeline` will:

#. Load your raw datasets and preprocess them.
#. Load your train-test splits from cache, if they exist. If they do not exist,
   the pipeline will perform the split on specified datasets.
#. Save the train-test splits to cache.
#. **Not** load hyperparameters found to be optimal in a previous run, since ``do_load_optimized_hyperparams`` is set
   to ``False``.
#. **Not** optimize hyperparameters of the model.
#. Train a model on the train set using the fixed hyperparameter dictionary from the predictor config, and evaluate
   it on the test set. An example of a config file defining a fixed hyperparameter dict is
   ``configs/predictors/classifiers/lgbm.gin``.
#. Estimate confidence intervals for the metrics with bootstrapping.
#. Save the trained predictor as a ``.pkl`` file to cache.
#. Refit the model on the entire train+test dataset and save it to cache.

Example plan - Train and optimize
---------------------------------

Let's look at ``configs/processing_plans/train_optimize.gin``, the one you
are likely to be using rather often. It differs from ``train.gin`` in a single flag:

.. code-block:: text

   ProcessingPipeline.do_optimize_hyperparams = True

According to this processing plan, the :class:`~src.processing_pipeline.ProcessingPipeline` will:

#. Load your raw datasets and preprocess them.
#. Load your train-test splits from cache, if they exist. If they do not exist,
   the pipeline will perform the split on specified datasets.
#. Save the train-test splits to cache.
#. **Not** load hyperparameters found to be optimal in a previous run, since ``do_load_optimized_hyperparams`` is set
   to ``False``.
#. Optimize hyperparameters with Optuna using cross-validation on the train set, and retain the best hyperparameters
   found.
#. Train a model on the train set using the best hyperparameters found in the previous step, evaluate it, and save
   the hyperparameters to ``hyperparams.yaml``.
#. Estimate confidence intervals for the metrics with bootstrapping.
#. Save the trained predictor as a ``.pkl`` file to cache.
#. Refit the model on the entire train+test dataset and save it to cache.

Other processing plans
----------------------

The other processing plans are similar to the ones above, but with different combinations of steps enabled or disabled.

#. ``normalize.gin`` will not split the data or train a model. It only runs the standard pre-processing steps on the
   raw datasets and writes the result as ``prepared_data.csv`` next to each raw dataset in ``data/datasets/``
   (see :doc:`models`).

#. ``split.gin`` will not train a model, but will instead perform the standard pre-processing steps on the raw datasets,
   create train-test splits, add (and optionally similarity-filter) the augmenting non-human data, and save the
   train-test splits to ``data/cache/splits``.

#. ``train_load_hyperparams.gin`` loads hyperparameters that were previously optimized on the
   **test-origin dataset alone**, for the same predictor / featurizer combination, and uses them to train a model on
   the current (possibly augmented) train set. The typical workflow is:

   #. run ``train_optimize.gin`` with ``ProcessingPipeline.datasets = ['AChE_human_pIC50']``;
   #. run ``train_load_hyperparams.gin`` with the human, rat and mouse datasets and the same
      ``test_origin_dataset``, splitter, featurizer and predictor.

   The hyperparameters are looked up in the cache directory of that earlier run. If the earlier run used a similarity
   filter, set ``ProcessingPipeline.hyperparams_source_sim_filter`` to the same filter. Attributed predictors load the
   hyperparameters of their non-attributed counterpart. Loaded hyperparameters override those in the predictor config.

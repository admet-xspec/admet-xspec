Inference with trained models
=============================

.. admonition:: Goals
   :class: tip

   After reading this section, you will be able to use a model trained by
   :class:`~src.processing_pipeline.ProcessingPipeline` to make predictions for new molecules.

Inference is run with ``predict.py``, configured, like training, by a single ``gin`` file. An example is provided in
``configs/examples/predict.gin``. It uses the model trained by ``configs/examples/train_rf_optimize.gin``
(see :doc:`quickstart`), so run that experiment first:

.. code-block:: text

   include 'configs/data_interface/data_interface.gin'

   InferencePipeline.data_interface = %data_interface

   # Inference input dataset path (CSV with a SMILES column; replace with your own molecules)
   InferencePipeline.dataset_path = 'data/datasets/AChE/human/regression/prepared_data.csv'

   # Cache keys identifying the trained model in data/cache/models/<model_cache_key>/<data_cache_key>/model.pkl
   InferencePipeline.model_cache_key = 'AChE_human_pIC50_RF_reg_ecfp_featurizer_4efda'
   InferencePipeline.data_cache_key = 'scaffold_e4737_nofilter_9a75c'

   # Task/column settings
   InferencePipeline.task_setting = 'regression'
   InferencePipeline.smiles_col = 'smiles'
   InferencePipeline.source_col = 'source'
   InferencePipeline.target_col = 'y'

   # Runtime behavior
   InferencePipeline.do_evaluate = True
   InferencePipeline.override_cache = False

Run it with:

.. code:: bash

   uv run process.py --cfg configs/examples/train_rf_optimize.gin   # trains the model, if not done yet
   uv run predict.py --cfg configs/examples/predict.gin
   # or
   python -m predict --cfg configs/examples/predict.gin

Choosing the model
------------------

A trained model is identified by the two directory names under ``data/cache/models`` described in :doc:`models`:

- ``model_cache_key`` -- the model directory, e.g. ``AChE_human_pIC50_RF_reg_ecfp_featurizer_4efda``;
- ``data_cache_key`` -- its data subdirectory, e.g. ``scaffold_e4737_nofilter_9a75c``.

By default (``use_refit_model = True``), ``model_final_refit.pkl``, refit on train + test, is loaded; with
``use_refit_model = False``, ``model.pkl``, trained on the train set only. The featurizer is stored inside the pickled
model, so it does not have to be configured again.

Input data
----------

``dataset_path`` points to a comma-separated ``.csv`` file with a SMILES column. If the column is not called
``smiles_col``, any of the SMILES column names accepted for datasets (see :doc:`sourcing_data`) is recognized.

The input SMILES are cleaned exactly as the training data is (canonicalization, neutralization and counterion
stripping with :func:`~src.utils.get_clean_smiles`), and the model predicts on the cleaned SMILES. The cleaned SMILES
are stored in an additional ``<smiles_col>_clean`` column; the original column is left untouched. Rows whose SMILES
cannot be cleaned are kept, with empty predictions, and their number is reported in the log.

For **attributed models**, every row needs a ``source`` column with one of the friendly names the model was trained on
(e.g. ``AChE_human_pIC50``), so that the right one-hot source vector is appended. If the model was trained on a single
source, that source is used for all rows.

Applicability domain
--------------------

With an applicability domain configured (see :ref:`applicability-domain`), every prediction is also scored for how
close the molecule is to the model's training data:

.. code-block:: text

   include 'configs/applicability_domains/tanimoto_knn.gin'
   InferencePipeline.applicability_domain = %applicability_domain

The fitted domain is loaded from the cached split named by ``data_cache_key``, matching the loaded model:
``domain_refit.pkl`` for the refit model, ``domain.pkl`` for ``model.pkl``. If it is missing, it is fitted on the cached
split and saved. The log reports how many molecules are in the domain and, if the model was calibrated, its test metrics
inside and outside the domain.

The domain is resolved before predicting, so a missing domain never costs the predictions:

- With ``fit_domain_if_missing = False``, a missing domain is treated as required: inference stops with an error before
  predicting, and nothing is written. The error names the ``applicability_domain.py`` command that fits it.
- Otherwise, a domain that cannot be loaded or fitted (e.g. because the model's split is no longer cached) is skipped
  with a warning. The predictions are saved as usual, without the domain columns.

Out-of-domain predictions are flagged, not dropped. Use the ``*_ad_similarity`` column to rank them, and the
``*_ad_nearest_smiles`` column to check what the model's judgement is based on.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``task_setting``
     - ``regression``
     - Overridden by the task setting of the loaded model.
   * - ``do_evaluate``
     - ``True``
     - If the input contains a label column (``target_col``, or one of the accepted label column names), compute
       metrics against it. Rows with invalid SMILES or missing labels are left out of the evaluation.
   * - ``do_append_to_input_file``
     - ``False``
     - Also write the predictions back into the input ``.csv`` file.
   * - ``override_cache``
     - ``False``
     - Passed on to :class:`~src.data.data_interface.DataInterface`. Also refits the applicability domain.
   * - ``applicability_domain``
     - ``None``
     - Domain config to assess the predictions with (see above).
   * - ``fit_domain_if_missing``
     - ``True``
     - Fit a missing domain on the cached split. If ``False``, a missing domain stops inference before predicting.

Output
------

The results are written to
``data/cache/predictions/<input file name>/<model_cache_key>/<data_cache_key>/``:

- ``predictions.csv`` -- the input data with the ``<smiles_col>_clean`` column and a prediction column named after
  the model's test-origin dataset (e.g. ``AChE_human_pIC50``). For classifiers, this column holds the predicted
  probability of class ``1``, and an additional ``<name>_class`` column holds the predicted class.
- With an applicability domain, three more columns per model: ``<name>_ad_similarity`` (the domain score),
  ``<name>_ad_in_domain`` (``True`` if the score reaches the domain's threshold) and ``<name>_ad_nearest_smiles`` (the
  most similar training molecule). They are empty for invalid SMILES.
- ``metrics.yaml`` -- only if the metrics were computed (see ``do_evaluate``). With an applicability domain, an
  ``applicability_domain`` section holds the metrics of the molecules inside and outside the domain.
- ``operative_config.gin`` and the console log of the run.

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

The ``model.pkl`` file (trained on the train set only) is loaded. The featurizer is stored inside the pickled model, so
it does not have to be configured again.

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
     - Passed on to :class:`~src.data.data_interface.DataInterface`.

Output
------

The results are written to
``data/cache/predictions/<input file name>/<model_cache_key>/<data_cache_key>/``:

- ``predictions.csv`` -- the input data with the ``<smiles_col>_clean`` column and a prediction column named after
  the model's test-origin dataset (e.g. ``AChE_human_pIC50``). For classifiers, this column holds the predicted
  probability of class ``1``, and an additional ``<name>_class`` column holds the predicted class.
- ``metrics.yaml`` -- only if the metrics were computed (see ``do_evaluate``).
- ``operative_config.gin`` and the console log of the run.

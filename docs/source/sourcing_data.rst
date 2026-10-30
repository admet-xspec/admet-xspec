Data sourcing & setup
=====================

.. note::

   ADMET-XSpec handles ChEMBL-sourced datasets automatically. All external datasets
   must be reformatted manually as ``.csv`` files containing at least the following two columns:

   - a **SMILES** column, named ``smiles``, ``SMILES``, ``Smiles`` or ``molecule``;
   - a **label** column with continuous or binary labels, named ``y``, ``Y``, ``label``, ``Label``, ``LABEL``
     or ``Standard Value``.

Introductory note
-----------------

**This page contains all the necessary information on how to use
ADMET-XSpec with your own datasets.**

The most important piece of information to keep in mind when using
ADMET-XSpec is that **each dataset must be placed in a separate
subdirectory somewhere in** ``data/datasets``. The exact organization of
the directories populating ``data/datasets`` is not critical and may be
modified according to personal preferences.

To facilitate experiment config creation, each dataset is assigned what
we will refer to as a **friendly name**. A friendly name for a new
dataset is established by the user. A good practice would be to keep it
short, yet sufficiently descriptive. The software regularly scans the
``data/datasets`` directory in search of previously unseen friendly
names — the current list of datasets (their friendly names) is stored in
``data/datasets/registry.txt``.

A dataset is identified by the pair **(friendly name, task setting)**. The regression and the binary classification
variant of the same data can therefore share a friendly name, while two directories with the same friendly name *and*
the same ``task_setting`` are an error.

.. admonition:: Goals
   :class: tip

   After reading this section, you should understand:

   #. Where to place raw datasets downloaded from ChEMBL
   #. How ``params.yaml`` defines how :class:`~src.processing_pipeline.ProcessingPipeline` will
      handle your dataset
   #. How to use regression datasets for binary classification

Suppose you have human- and mouse-sourced data for the prediction of
acetylcholinesterase (AChE) inhibition as continuous IC50 values. We
will discuss how to set those up to train both regression and binary
classification models within ADMET-XSpec.

Each dataset directory must contain the following two components:

- Exactly one tabular datafile containing the molecules (SMILES) and labels

  - Data exported as ``.csv`` from ChEMBL (semicolon-separated) can be placed directly, as it is
    handled by the software
  - External data must be hand-curated to make it into a ``.csv`` with a SMILES and a label column (see the note
    at the top of this page); the delimiter is detected automatically

- ``params.yaml`` file, containing data necessary for the dataset to be
  correctly interpreted by the software

After the first run, the directory will also contain the ``prepared_data.csv`` file and a ``preparation_log``
(see :doc:`models`).

Regression dataset setup
------------------------

Let's address the first point — we create a subdirectory in
``data/datasets`` and drop our ChEMBL-sourced ``.csv`` file there. As we
are preparing a dataset for training human AChE IC50 regressors, our
data file will land in the following path::

   data/datasets/AChE/human/regression/DOWNLOAD.csv

Now we should discuss what should end up in the contents of every
``params.yaml`` file.

``data/datasets/AChE/human/regression/params.yaml``:

.. code-block:: yaml
   :linenos:

   friendly_name: "AChE_human_pIC50"
   raw_or_derived: "raw"
   category: "AChE_IC50"
   is_chembl: true
   task_setting: "regression"
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

**Line 1** provides the dataset's ``friendly_name``, which you will use
to refer to it in your experiment's ``.gin`` configs.

**Line 2** states that the ``.csv`` accompanying this ``.yaml`` (the
downloaded dataset) is data in its ``raw``, original form, in contrast
to splits, which are output by the :class:`~src.processing_pipeline.ProcessingPipeline` when
integrating data and are always labelled as ``derived`` in their
accompanying ``.yaml`` (yes, they get a ``friendly_name`` too, albeit
theirs is less friendly in general).

**Line 3** is a legacy field no longer used by the current codebase and
can be skipped.

**Line 4** notes whether the dataset was downloaded from ChEMBL. ChEMBL exports are read as semicolon-separated files,
and their ``Standard Relation`` column is used when converting continuous labels to classes (see below). Set it to
``false`` for datasets from other sources.

**Line 5** states how the dataset should be treated: whether it is a
``regression`` or ``binary_classification`` dataset. Since we offer the
feature of running classification on datasets originally labelled as
regression, we wanted to be able to use the same ``friendly_names``,
with the :class:`~src.processing_pipeline.ProcessingPipeline` distinguishing between task
settings for us. This field accomplishes that.

**Lines 6–12** define the filter criteria. For a data point to remain in the set, it must have ``nM`` in
the ``Standard Units`` column, ``'='`` in the ``Standard Relation``
column, and ``IC50`` in the ``Standard Type`` column. The values of
units, relation, and type may change depending on what kind of
phenomenon you are trying to model and what kind of assay was used to
gather the data uploaded to ChEMBL.

.. note::

   Filter criteria are applied to any column that is present in the data file; criteria for missing
   columns (e.g. in non-ChEMBL datasets) are skipped with a warning.

**Lines 13–15** provide the label transformations to be used before
training, applied in the listed order. Two transformations are available: ``log10`` (rows with non-positive labels are
dropped) and ``negate``. In this example, we are transforming IC50 values into their
negated logarithms and using the transformed values as the new labels in training.

.. note::

   As ChEMBL reports IC50 in nM, ``-log10(IC50 [nM])`` equals pIC50 - 9. For example, an IC50 of 1 µM
   (1000 nM) becomes ``-3``.

Binary classification dataset setup
-----------------------------------

Now the continuous AChE IC50 dataset can be easily used to train binary
classifiers in ADMET-XSpec. All you have to do is decide on a threshold
value that will allow the software to automatically assign all
samples to one of two AChE activity classes (``0`` or ``1``).

Let's create a separate subdirectory for the binary classification AChE
dataset — for example, ``data/datasets/AChE/human/binary_classification``.
Next, we copy the whole ChEMBL-sourced ``DOWNLOAD.csv`` from the
regression directory into our new classification directory. As the raw
data file is the same, changes will have to be made in ``params.yaml``
to enable ADMET-XSpec to parse this dataset for training binary
classifiers.

``data/datasets/AChE/human/binary_classification/params.yaml``:

.. code-block:: yaml
   :linenos:

   friendly_name: "AChE_human_pIC50"
   raw_or_derived: "raw"
   category: "AChE_IC50"
   is_chembl: true
   task_setting: "binary_classification"
   filter_criteria:
       Standard Units:
         - "nM"
       Standard Type:
         - "IC50"
   threshold: -3
   label_transformations:
     - "log10"
     - "negate"

**Lines 1–4** are the same as in ``data/datasets/AChE/human/regression/params.yaml``. The friendly name can stay the
same, because **Line 5** sets a different task setting.

**Lines 6–10** keep the unit and type filters but drop the ``Standard Relation`` filter, so that censored
values (``<``, ``>``) can still be used where their class is unambiguous.

**Line 11** sets the threshold value used to map continuous labels onto binary classes.
**The threshold is applied after the label transformations** (lines 12–14), so it must be expressed in the
transformed units: ``-3`` corresponds to an IC50 of 1 µM. Values of :math:`y` greater than
or equal to the threshold are assigned class ``1`` (active), while all other data
points are assigned class ``0`` (inactive). For ChEMBL datasets, a data point whose ``Standard Relation`` does not
allow an unambiguous assignment is dropped.

.. tip::

   Apart from numbers, the ``threshold`` setting also accepts ``"median"``, which
   sets the threshold at the median of the (transformed) labels :math:`y` of this dataset and results in an
   approximately balanced dataset.

Sharing a threshold between datasets
------------------------------------

The rat and mouse AChE datasets should be split into classes in a way that is consistent with the human data.
This is done with ``threshold_source``, as in
``data/datasets/AChE/rat/binary_classification/params.yaml``:

.. code-block:: yaml
   :lineno-start: 11

   threshold: null
   threshold_source: "AChE_human_pIC50"

When ``threshold_source`` is set, it takes precedence over ``threshold``, and the dataset uses **the same threshold as
the** ``threshold_source`` **dataset**. Here, the rat data is classified with the human threshold of ``-3``
(IC50 = 1 µM). Note that:

- the ``threshold_source`` dataset is looked up with the same task setting, so it must be another
  ``binary_classification`` dataset;
- if the source's threshold is ``"median"``, it resolves to the median of the **source's** transformed labels (the value
  used to classify the source itself), not of the current dataset;
- the source may itself have a ``threshold_source``; such chains are followed, but a dataset must not refer back to
  itself, directly or through a cycle;
- the threshold is applied to the transformed labels of the current dataset, so both datasets should use the same
  ``label_transformations``.

.. important::

   Prepared datasets are cached. After changing ``threshold``, ``threshold_source`` or ``label_transformations`` of a
   dataset, delete its ``prepared_data.csv`` (and the train-test splits and models built from it in ``data/cache``) or
   run with ``ProcessingPipeline.override_cache = True``.

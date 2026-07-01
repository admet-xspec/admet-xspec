Data sourcing & setup
======================

.. warning::

   
   ADMET-XSpec handles ChEMBL-sourced datasets automatically. All external datasets
   are to be reformatted to `.csv` files with 'smiles' and 'y' columns as the features and labels!
   
   .. code-block:: text
      
      data/datasets/permeability/B3DB/binary_classification/b3db_classification.csv
      data/datasets/permeability/PAMPA/binary_classification/pampa_combined.csv

Introductory note
------------------

.. note::

   This page contains all the necessary information on how to use
   ADMET-XSpec with your own datasets.

The most important fact about uploading your own data to use with
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

Goals
-----

After reading this section, you should understand:

#. Where to place raw datasets downloaded from ChEMBL
#. How ``params.yaml`` defines how :class:`ProcessingPipeline` will
   handle your dataset
#. How to use regression datasets for binary classification

Suppose you have human- and mouse-sourced data for the prediction of
acetylcholinesterase (AChE) inhibition as continuous IC50 values. We
will discuss how to set those up to train both regression and binary
classification models within ADMET-XSpec.

Each dataset directory must contain the following two components:

- Tabular datafile containing the molecules (SMILES) and labels (``y``)

  - Data parsed as ``.csv`` from ChEMBL can be placed directly, as it is
    handled by the software
  - External data must be hand-curated to make it into a ``.csv`` that
    contains at least two columns: ``smiles`` and ``y``

- ``params.yaml`` file, containing data necessary for the dataset to be
  correctly interpreted by the software

Regression dataset setup
-------------------------

Let's address the first point — we create a subdirectory in
``data/datasets`` and drop our ChEMBL-sourced ``.csv`` file there. As we
are preparing a dataset for training human AChE IC50 regressors, our
data file will land in the following path::

   data/datasets/AChE/human/regression/DOWNLOAD(1).csv

Now we should discuss what should end up in the contents of every
``params.yaml`` file.

``data/datasets/AChE/human/regression/params.yaml``:

.. code-block:: yaml
   :linenos:

   friendly_name: "AChE_human_IC50"
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
to splits, which are output by the :class:`ProcessingPipeline` when
integrating data and are always labelled as ``derived`` in their
accompanying ``.yaml`` (yes, they get a ``friendly_name`` too, albeit
theirs is less friendly in general).

**Line 3** is a legacy field no longer used by the current codebase and
can be skipped.

**Line 4** notes whether the dataset was downloaded from ChEMBL. This
is added to accommodate datasets from other sources that have already
been preprocessed by the user. For now, you can safely assume
ADMET-XSpec can only accommodate ChEMBL datasets and proceed
accordingly.

**Line 5** states how the dataset should be treated: whether it is a
``regression`` or ``binary_classification`` dataset. Since we offer the
feature of running classification on datasets originally labelled as
regression, we wanted to be able to use the same ``friendly_names``,
with the :class:`ProcessingPipeline` distinguishing between task
settings for us. This field accomplishes that.

**Lines 6–12** define the filter criteria applied to ChEMBL-sourced
datasets. For a data point to remain in the set, it must have ``nM`` in
the ``Standard Units`` column, ``'='`` in the ``Standard Relation``
column, and ``IC50`` in the ``Standard Type`` column. The values of
units, relation, and type may change depending on what kind of
phenomenon you are trying to model and what kind of assay was used to
gather the data uploaded to ChEMBL.

.. warning::

   Filter criteria are not applied to non-ChEMBL datasets.

**Lines 13–15** provide the label transformations to be used before
training. In this example, we are transforming IC50 values into their
negated logarithms (pIC50) and using the transformed values as the new
labels in training.

Binary classification dataset setup
-------------------------------------

Now the continuous AChE IC50 dataset can be easily used to train binary
classifiers in ADMET-XSpec. All you have to do is decide on a threshold
value of IC50 that will allow the software to automatically assign all
samples to one of two AChE activity classes (``0`` or ``1``).

Let's create a separate subdirectory for the binary classification AChE
dataset — for example, ``data/datasets/AChE/human/binary_classification``.
Next, we copy the whole ChEMBL-sourced ``DOWNLOAD(1).csv`` from the
regression directory into our new classification directory. As the raw
data file is the same, changes will have to be made in ``params.yaml``
to enable ADMET-XSpec to parse this dataset for training binary
classifiers.

``data/datasets/AChE/human/binary_classification/params.yaml``:

.. code-block:: yaml
   :linenos:

   friendly_name: "AChE_human_IC50"
   raw_or_derived: "raw"
   category: "AChE_IC50"
   is_chembl: true
   task_setting: "binary_classification"
   filter_criteria:
       Standard Units:
         - "nM"
       Standard Type:
         - "IC50"
   threshold: null
   threshold_source: "AChE_human_IC50"

**Lines 1–10** follow the same configuration as for
``data/datasets/AChE/human/regression/params.yaml``.

**Line 11** allows the user to set the threshold value used to map
continuous labels onto binary classes. Values of :math:`y` greater than
or equal to the threshold are assigned class ``1``, while all other data
points are assigned class ``0``.

.. tip::

   Apart from floats, this setting also accepts ``"median"``, which
   results in a perfectly balanced dataset with the threshold set at the
   median value of all the labels :math:`y`.

**Line 12** specifies where to source the threshold value, if desired.
If we navigated to the dataset specified in ``threshold_source``, its
Line 11 would contain, for instance, ``"median"``. This means that to
obtain our ``0`` vs. ``1`` label threshold, we take the median of
regression values from ``threshold_source``. For this reason, Line 11 is
``null`` here, as we do not intend to use this dataset as the threshold
source for any other dataset.

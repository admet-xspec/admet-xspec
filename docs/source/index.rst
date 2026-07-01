Welcome to ADMET-XSpec's documentation
======================================

**ADMET-XSpec** is an open-source tool that facilitates systematic cross-species data integration for training ADMET
(Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction models.

**The tool is designed to help
researchers understand the translatability of ADMET parameters measured in non-human assays in the context
of human drug development.**

ADMET-XSpec implements:
   - Pre-processing and curation of molecular datasets for ADMET ML tasks.
   - A selection of train-test splitting strategies, molecular featurizers, ML algorithms, and
     hyperparameter optimization methods for building diverse data pipelines.
   - Both single-task and **attributed learning** modes for training on heterogeneous datasets.
   - Exclusively ``gin`` config-based interface for easy preparation of large experiments.
   - Model & data interface for facile inference on new molecules using previously trained models.

For the installation instructions refer to the :doc:`quickstart` section.

.. note::

   This project is under active development.

Table of contents
-----------------

.. toctree::

   quickstart
   features

   sourcing_data
   api

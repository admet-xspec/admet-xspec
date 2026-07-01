Welcome to ADMET-Xspec's documentation
======================================

**ADMET-XSpec** is an open-source tool that facilitates systematic cross-species data integration for training ADMET
(Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction models. **The tool is designed to help
researchers assess and understand the translatability of ADMET parameters measured in non-human assays in the context
of human drug development.**

ADMET-XSpec implements:
   - Pre-processing and curation of molecular datasets for ADMET tasks.
   - A selection of:
      - train-test splitting strategies
      - molecular representations
      - ML algorithms
   - Both single-task and multiend training
   - Exclusively ''gin'' config-based interface for easy preparation of large experiments
   - Predictions for new samples using previously trained models

Check out the :doc:`quickstart` section for installation instructions.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   quickstart
   usage
   sourcing_dara
   api

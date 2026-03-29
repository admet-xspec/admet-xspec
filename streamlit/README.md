# Streamlit utilities

## Experiment builder

This app helps you batch-create experiment `.gin` configs (similar to `configs/examples`) into a separate output directory.

### Run

```bash
uv run streamlit run streamlit/experiment_builder_app.py
```

If you are not using `uv`, install Streamlit in your active environment first:

```bash
pip install streamlit
python -m streamlit run streamlit/experiment_builder_app.py
```

### What it configures

- Processing plan
- Featurizer
- Splitter
- Similarity filter
- Predictor (filtered by task setting)
- Training datasets
- Test origin dataset
- Output directory for generated configs

Generated files include all selected combinations of:
`processing_plan x featurizer x splitter x sim_filter x predictor`


# Reproduction Instructions for Evidence Directory `evidence_rerun_20261022_050835`

To reproduce all experiments, figures, and CSV tables from this package:

```powershell
# Set Windows environment flags
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:PYTHONPATH="src;."

# 1. Run unit, integration, and regression test suite
pytest -v

# 2. Re-run complete experimental evidence pipeline
python scripts/run_reproducible_pipeline.py
```

Generated outputs will populate:
- `figures/`: Fig 1 to Fig 10 (PNG 300 DPI and vector PDF)
- `tables/`: Publication-ready CSV benchmark tables
- `predictions/`: Serialized prediction arrays and labels
- `run_config.json`: Configuration snapshot
- `run.log`: Full execution log

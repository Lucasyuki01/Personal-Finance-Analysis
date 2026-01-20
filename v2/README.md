# Personal Finance Analysis v2

This folder contains the v2 data pipeline for producing a canonical dataset. The
canonical dataset is the single source of truth for later Streamlit views.

## Structure
- `src/pfa`: Package with pipeline steps, IO helpers, schemas/validators, and utilities.
- `scripts/run_pipeline.py`: Runs the pipeline over sample inputs.
- `data/samples`: Place 1..N CSV/XLSX input files here.
- `data/outputs`: Pipeline outputs (canonical dataset + manifest).
- `tests`: Minimal tests for key pipeline behaviors.

## Running the pipeline
From the repo root:
```bash
python v2/scripts/run_pipeline.py
```

The script reads inputs from `v2/data/samples` and writes:
- `v2/data/outputs/canonical_base.csv`
- `v2/data/outputs/processing_manifest.json`

The canonical dataset includes `excluded_reason`. Downstream analysis should
filter rows where `excluded_reason` is empty or null.

## Running tests
From the repo root:
```bash
pytest v2/tests
```

## Notes
- Inputs are never modified; each file is copied in memory and tagged with
  `source_account`.
- `transaction_id` is a stable hash of source metadata, date, description,
  amount, and source row ID.
- A placeholder schema + IO utilities exist for `classification_rules` but
  no rules are applied yet.


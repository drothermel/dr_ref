# Agent Guide: dr_showntell

## Snapshot
- **Purpose**: exploratory lab for WandB run parsing, enrichment, and matched finetune + pretrain analyses.
- **Outputs**: pickled `processed_runs` bundles and combined plotting tables under `data/` (e.g., `combined_plotting_data_matched.pkl`).
- **Core Strength**: most complete understanding of run naming conventions and feature extraction for post-training evaluation.

## Reading Order
1. `src/dr_showntell/run_id_parsing.py` – regex patterns, mapping tables, and post-processing (`apply_processing`) that convert run IDs into structured fields (tokens, checkpoints, comparison models).
2. Legacy pretrain/finetune merge helpers have been removed; rely on the new DuckDB ingestion work in `datadec/notebooks/duck_wandb.py` for evaluation metric normalisation and future joins.
3. Remaining notebooks are historical debugging aids; expect them to shrink further as the DuckDB-centric pipeline replaces the old pickle workflow.

## Data Products
- None active. Historical pickles/parquet exports have been removed; future curated tables will come from the DuckDB ingestion work (`datadec/notebooks/duck_wandb.py`).

## Integration Points
- Consumes parquet outputs from `dr_wandb` and `datadec`.
- Feeds curated datasets to `by-tomorrow-app` (table/plot prototypes) and influences planned DuckDB schemas.
- Provides logic that should eventually be ported into DuckDB transformations or ingestion scripts (avoid re-implementing regex parsing elsewhere).

## Known Gaps / TODOs
- Paths to base data are absolute; convert to configurable inputs when integrating with DuckDB.
- Outputs are pickled Python objects; plan to replace with structured DuckDB tables.
- Regex catalogue needs periodic updates as new run naming conventions appear—capture new patterns before they fall into `"other"` bucket.
- Notebook code is not modularized; consider extracting reusable functions for ingestion scripts.

### Update Log
- _Add recent changes here._

Treat this repo as the reference for “how to interpret WandB experiments” until the logic migrates into the central ingestion pipeline.

# Agent Guide: dr_showntell

## Snapshot
- **Purpose**: exploratory lab for WandB run parsing, enrichment, and matched finetune + pretrain analyses.
- **Outputs**: pickled `processed_runs` bundles and combined plotting tables under `data/` (e.g., `combined_plotting_data_matched.pkl`).
- **Core Strength**: most complete understanding of run naming conventions and feature extraction for post-training evaluation.

## Reading Order
1. `src/dr_showntell/run_id_parsing.py` – regex patterns, mapping tables, and post-processing (`apply_processing`) that convert run IDs into structured fields (tokens, checkpoints, comparison models).
2. `src/dr_showntell/combine_pt_ft_utils.py` – merges processed runs with pretraining metrics, extracts evaluation metrics from WandB summaries, builds matched groups, and adds helper columns.
3. `src/dr_showntell/datadec_utils.py` – convenience loaders for DataDecide and exported WandB parquet files (paths currently hard-coded).
4. Notebooks:
   - `notebooks/quick_demo_modernized.py` – end-to-end demo (parse → validate → pickle export).
   - `notebooks/combine_processed_with_pretrain_data.py` – generates `combined_plotting_data_*` artifacts.

## Data Products
- `data/*_modernized_run_data.pkl`: dictionary with `processed_runs`, `processed_dataframes`, and metadata (used as source for further pipelines).
- `data/combined_plotting_data_matched.pkl`: enriched table with matched finetune + pretrain metrics (ready for dashboards and modelling).
- Notebook outputs printed to console (FancyTable) for QA during development.

## Integration Points
- Consumes parquet outputs from `dr_wandb` and `datadec`.
- Feeds curated datasets to `by-tomorrow-app` (table/plot prototypes) and influences planned DuckDB schemas.
- Provides logic that should eventually be ported into DuckDB transformations or ingestion scripts (avoid re-implementing regex parsing elsewhere).

## Known Gaps / TODOs
- Paths to base data are absolute; convert to configurable inputs when integrating with DuckDB.
- Outputs are pickled Python objects; plan to replace with structured DuckDB tables.
- Regex catalogue needs periodic updates as new run naming conventions appear—capture new patterns before they fall into `"other"` bucket.
- Notebook code is not modularized; consider extracting reusable functions for ingestion scripts.

Treat this repo as the reference for “how to interpret WandB experiments” until the logic migrates into the central ingestion pipeline.

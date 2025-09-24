# Agent Guide: dr_showntell

## Snapshot
- **Purpose**: exploratory lab for WandB run parsing, enrichment, and matched finetune + pretrain analyses.
- **Outputs**: pickled `processed_runs` bundles and combined plotting tables under `data/` (e.g., `combined_plotting_data_matched.pkl`).
- **Core Strength**: most complete understanding of run naming conventions and feature extraction for post-training evaluation.

## Reading Order
There is no parsing logic left here—only table display utilities:
1. `src/dr_showntell/fancy_table.py` – Rich table helper with multi-layer headers.
2. `src/dr_showntell/console_components.py` – Wrapper panels/blocks for rendering tables in Rich consoles.
3. `src/dr_showntell/table_formatter.py` – Configurable formatting/exports for tables (markdown/csv/etc.).
Parsing and data munging now live in `dr_ingest` and DuckDB workflows (`datadec/notebooks/duck_wandb.py`).

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

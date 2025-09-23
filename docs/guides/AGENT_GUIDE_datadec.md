# Agent Guide: datadec

## Snapshot
- **Purpose**: canonical ingestion pipeline for DataDecide pretraining/evaluation results (Hugging Face + AllenAI parquet releases).
- **Outputs**: typed parquet tables under `data/datadecide/` (`full_eval.parquet`, `mean_eval.parquet`, etc.) plus helper metadata (model/dataset details).
- **Entry Point**: instantiate `datadec.data.DataDecide` to trigger pipeline + load cached dataframes.

## Reading Order
1. `src/datadec/pipeline.py` – defines staged pipeline (`download`, `metrics_expand`, `parse`, `merge`, `enrich`, `aggregate`, `plotting_prep`). Shows how parquet files are produced.
2. `src/datadec/data.py` – `DataDecide` orchestrates pipeline execution and provides filtering utilities (`select_subset`, `prepare_plot_data`).
3. `src/datadec/parsing.py` – parsing/merging helpers; adds tokens/compute, learning rate columns, OLMES metric reshaping.
4. `src/datadec/constants.py` – definitive list of recipes, parameter sizes, metrics, column hierarchies.
5. `src/datadec/paths.py` + `loader.py` – path management and dataframe caching.

## Data Tables
- `full_eval.parquet`: merged perplexity + downstream metrics + metadata, enriched with LR columns.
- `mean_eval.parquet` / `std_eval.parquet`: seed-aggregated stats used for plotting.
- `full_eval_melted.parquet` / `mean_eval_melted.parquet`: precomputed tidy format for dashboards.
- `datasets/dataset_*.pkl`: legacy serialized subsets (may be deprecated).

## Integration Points
- `dr_showntell` consumes `full_eval.parquet` when enriching WandB runs with pretrain metrics.
- `ddpred` uses `DataDecide` APIs to fetch filtered windows for early-window features.
- Future DuckDB ingestion should reuse stage logic but output to shared schemas instead of per-stage parquet.

## Known Gaps / TODOs
- Pipeline still downloads directly from Hugging Face each run; plan is to refactor into DuckDB loaders while keeping current code as reference.
- `display.py` and some scripting helpers are stubs; safe to ignore unless reviving CLI plotting.
- Ensure new ingestion scripts update `paths.DataDecidePaths` or document alternative storage.

Keep this guide in sync when pipeline stages or outputs change.

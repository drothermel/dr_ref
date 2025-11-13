# Agent Guide: datadec

## Quick Navigation

- **Need reactivity / marimo context?** See `guides/marimo/agent_guide.md` and `guides/marimo/marimo-reactivity-general.md` for notebook instructions.
- **Need WandB ingestion pairing?** See `docs/guides/AGENT_GUIDE_dr_ingest.md` for how these tables are joined with run metadata.
- **Need DuckDB plans?** See `docs/DATA_ARTIFACTS.md` and `docs/ROADMAP.md` for schema targets and priorities.
- **Looking for scripts/notebooks?** See `datadec/notebooks/` (e.g., `duck_wandb.py`) and `dr_ref/guides/marimo` for Marimo-based prototypes.

## Table of Contents

1. [Snapshot](#snapshot)
2. [Current Status & High-Priority TODOs](#current-status--high-priority-todos)
3. [Reading Order](#reading-order)
4. [Data Tables](#data-tables)
5. [Integration Points](#integration-points)
6. [Known Gaps / Risks](#known-gaps--risks)
7. [Update Log](#update-log)

## Snapshot
- **Purpose**: canonical ingestion pipeline for DataDecide pretraining/evaluation results (Hugging Face + AllenAI parquet releases).
- **Outputs**: typed parquet tables under `data/datadecide/` (`full_eval.parquet`, `mean_eval.parquet`, etc.) plus helper metadata (model/dataset details).
- **Entry Point**: instantiate `datadec.data.DataDecide` to trigger pipeline + load cached dataframes.
- **Owner**: `datadec` repository under `repos/datadec` (mirrors from `github.com/DataDecide/datadec`).

## Current Status & High-Priority TODOs

- ✅ Pipeline stages (`download → metrics_expand → parse → merge → enrich → aggregate`) are stable for existing parquet sources.
- ⚠️ **DuckDB migration pending**: convert the stage outputs into DuckDB tables per `docs/DATA_ARTIFACTS.md` schemas (runs, history, finetune). Requires aligning `paths.DataDecidePaths` with new storage locations.
- ⚠️ **Download strategy**: still hits Hugging Face every run. Add caching or prebaked mirrors; see `ROADMAP.md` “DuckDB Ingestion MVP.”
- ⚠️ **Documentation drift**: keep README/guide updated when adding new stage outputs (e.g., `*_melted.parquet`, DuckDB tables).

## Reading Order
1. `src/datadec/pipeline.py` – defines staged pipeline (`download`, `metrics_expand`, `parse`, `merge`, `enrich`, `aggregate`, `plotting_prep`). Shows how parquet files are produced.
2. `src/datadec/data.py` – `DataDecide` orchestrates pipeline execution and provides filtering utilities (`select_subset`, `prepare_plot_data`).
3. `src/datadec/parsing.py` – parsing/merging helpers; adds tokens/compute, learning rate columns, OLMES metric reshaping.
4. `src/datadec/constants.py` – definitive list of recipes, parameter sizes, metrics, column hierarchies.
5. `src/datadec/paths.py` + `loader.py` – path management and dataframe caching; update these when migrating storage paths (e.g., DuckDB outputs).
6. `datadec/notebooks/` – ad-hoc DuckDB + Marimo experiments; use as reference for schema planning.

## Data Tables
- `full_eval.parquet`: merged perplexity + downstream metrics + metadata, enriched with LR columns.
- `mean_eval.parquet` / `std_eval.parquet`: seed-aggregated stats used for plotting.
- `full_eval_melted.parquet` / `mean_eval_melted.parquet`: precomputed tidy format for dashboards.
- `datasets/dataset_*.pkl`: legacy serialized subsets (may be deprecated).
- *(Planned)* DuckDB tables for runs, history, and finetune evaluation. See `docs/DATA_ARTIFACTS.md` for current schema drafts.

## Integration Points
- `dr_ingest` consumes `full_eval.parquet` when enriching WandB runs with pretrain metrics.
- `ddpred` uses `DataDecide` APIs to fetch filtered windows for early-window features.
- Future DuckDB ingestion should reuse stage logic but output to shared schemas instead of per-stage parquet.
- `by-tomorrow-app` and Marimo notebooks expect tidy tables (e.g., `full_eval_melted`), so document any breaking column changes in this guide.

## Known Gaps / Risks
- TODO: replace per-run downloads with cached/mirrored Hugging Face assets before running the pipeline.
- TODO: either delete or revive `display.py` + CLI helpers; document their status.
- TODO: document how DuckDB outputs will be wired into `paths.DataDecidePaths` before landing the new ingestion scripts.
- TODO: add schema/version metadata to parquet/tidy outputs so downstream code can detect drift.

### Update Log
- _Add recent changes here._

Keep this guide in sync when pipeline stages or outputs change.

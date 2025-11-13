# Agent Guide: ddpred

## Quick Navigation

- **Data sources**: see `docs/guides/AGENT_GUIDE_datadec.md` (pretraining tables) and future DuckDB schemas in `docs/DATA_ARTIFACTS.md`.
- **WandB/finetune enrichment**: see `docs/guides/AGENT_GUIDE_dr_ingest.md` for normalized tables this repo should ingest next.
- **Notebook references**: see `docs/guides/AGENT_GUIDE_datadec_notebooks.md` for exploratory feature work.
- **Process / roadmap context**: see `docs/ROADMAP.md` for modelling milestones.

## Table of Contents

1. [Snapshot](#snapshot)
2. [Current Status & High-Priority TODOs](#current-status--high-priority-todos)
3. [Reading Order](#reading-order)
4. [Key Concepts](#key-concepts)
5. [Data Flow](#data-flow)
6. [Current Outputs](#current-outputs)
7. [Integration Notes](#integration-notes)
8. [Known Gaps / TODOs](#known-gaps--todos)
9. [Update Log](#update-log)

## Snapshot
- **Purpose**: research-grade framework for predicting final LLM performance from early training signals.
- **Inputs**: DataDecide pretraining curves (via `DataDecide` API), optional matched finetune tables.
- **Outputs**: Cross-validation results, markdown reports, trained models saved under `results/` and `markdowns/`.

## Current Status & High-Priority TODOs

- ✅ ElasticNet + GP trainers work against pretraining-only data with nested CV.
- ⚠️ **Data abstraction**: TODO decouple from `DataDecide` parquet paths and support DuckDB queries (pretrain + finetune) via a datasource layer.
- ⚠️ **Feature catalog**: TODO document all feature generators and how they map to schema columns once DuckDB integration lands.
- ⚠️ **Reproducibility**: TODO standardize experiment config capture (seeds, parameters) in `results/` outputs.
## Reading Order
1. `src/ddpred/core/data_pipeline.py` – `DataPipelineFactory` entry points (`prepare_standard_data`, `prepare_progressive_data`, `prepare_sequence_data`); orchestrates filtering, feature extraction, and target construction.
2. `src/ddpred/core/data_preparation.py` – converts filtered dataframes into feature matrices / target arrays; handles progressive targets, sequence extraction, scaling.
3. `src/ddpred/core/feature_extraction.py` + `features/` package – windowing (`EarlyWindowData`), basic stats, regression features, fixed metadata.
4. `src/ddpred/cross_validation/engine.py` – nested parameter-size CV, hyperparameter search, metric collection.
5. Trainers: `trainers/elasticnet_trainer.py`, `trainers/gp_trainer.py`, etc.

## Key Concepts
- **Windowed Features**: specify steps or percentage windows; features include per-step values, slopes, regressions, and basic stats.
- **Targets**: best perplexity, final value, or progressive (percentage-based) metrics derived from pretraining curves.
- **Cross-Validation**: parameter-size group splits (outer + inner) to avoid leakage between model sizes.

## Data Flow
1. Load DataDecide parquet via `DataDecide` (currently pretraining-only).
2. Filter by metric/data/params; extract windowed subframes.
3. Featurize into numpy arrays; optional scaling/target transforms.
4. Run nested CV (ElasticNet or alternative models); evaluate metrics (RMSE, correlation, ranking).

## Current Outputs
- `experiment_results.csv` / `experiment_results_fixed.csv`: aggregated sweep results.
- `markdowns/` and `plots/`: rendered reports and visualizations for specific analysis runs.
- Model artefacts saved per experiment (check `results/` or `output/`).

## Integration Notes
- Data source currently assumes pretraining metrics only. Plan is to refactor to read from upcoming DuckDB tables (both pretrain and finetune).
- Feature topology may need expansion once matched finetune metrics are available (e.g., include group-level deltas or finetune tokens).

## Known Gaps / TODOs
- TODO: replace hard-coded `DataDecide` parquet reliance with an abstraction that can query DuckDB tables.
- TODO: consolidate feature metadata (currently scattered in `features/__init__.py`) and map each to schema columns.
- TODO: audit legacy scripts/notebooks (e.g., `temp_plotter`, `scripts/b*`) and archive or update once new pipeline stabilizes.
- TODO: capture random seeds and config snapshots in outputs for reproducibility.

### Update Log
- _Add recent changes here._

Consult this guide alongside `README.md` and analysis scripts when adjusting modelling experiments.

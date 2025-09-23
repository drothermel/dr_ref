# Agent Guide: ddpred

## Snapshot
- **Purpose**: research-grade framework for predicting final LLM performance from early training signals.
- **Inputs**: DataDecide pretraining curves (via `DataDecide` API), optional matched finetune tables.
- **Outputs**: Cross-validation results, markdown reports, trained models saved under `results/` and `markdowns/`.

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
- Hard-coded reliance on `DataDecide` and parquet paths; introduce abstraction layer for DuckDB queries.
- Feature metadata is scattered (see `features/__init__.py`); document which features map to which columns in new schema.
- Some scripts/notebooks are legacy (e.g., `temp_plotter`, `scripts/b*`). Clean up or archive once new pipeline stabilizes.
- Ensure reproducibility by capturing random seeds and config snapshots in outputs.

Consult this guide alongside `README.md` and analysis scripts when adjusting modelling experiments.

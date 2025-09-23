# Data Artifacts Inventory

Snapshot of the main datasets produced or consumed across repos. Update this as new DuckDB tables or parquet exports are added.

## Legacy Parquet Outputs

| Location | Description | Source | Notes |
|----------|-------------|--------|-------|
| `datadec/data/datadecide/full_eval.parquet` | Enriched merge of perplexity + downstream eval metrics with model/dataset metadata. | Hugging Face parquet downloads (`DataDecide` pipeline). | Includes LR curves (`lr_at_step`, `cumulative_lr`). |
| `datadec/data/datadecide/mean_eval.parquet` / `std_eval.parquet` | Seed-aggregated statistics. | Derived from `full_eval`. | Useful for plots; retains metric columns. |
| `dr_wandb/data/runs_metadata.parquet` | Run-level configs, summaries, system metadata. | WandB API via `dr_wandb`. | JSON-heavy columns; needs schema design before DuckDB ingest. |
| `dr_wandb/data/runs_history.parquet` | Step-by-step history entries with metrics dict. | WandB API via `dr_wandb`. | Flattening into metric columns pending. |
| `dr_showntell/data/*_modernized_run_data.pkl` | Pickled dict with parsed runs (`processed_runs`) and metadata. | `dr_showntell` parsing notebooks. | Intermediate format; will be replaced by structured DuckDB tables. |
| `dr_showntell/data/combined_plotting_data_matched.pkl` | Finetune + pretrain merged table (matched runs). | `combine_processed_with_pretrain_data.py`. | Candidate to become curated DuckDB view for analysis/dashboard. |

## In-Progress DuckDB / Parquet Experiments

| Artifact | Status | Notes |
|----------|--------|-------|
| `datadec/notebooks/wandb_runs.parquet` & `wandb_history.parquet` | ✅ JSONL → Parquet conversion using DuckDB (prototype). | Stored in `notebooks/`; size reduction observed (~60 MB → ~5 MB). |
| QA task tarball extracts (`instances_extracted/*`) | ⏳ Manual extraction of Hugging Face QA evaluation outputs. | `start.py` notebook shows structuring plan; need schema + ingestion script. |

## Planned DuckDB Schemas

1. **WandB Runs**: normalized tables for run metadata, config, summary metrics, evaluation results (flattened per-task metrics, matched group IDs). Likely stored as DuckDB tables with JSON extra columns for rarely-used blobs.
2. **WandB History**: step/time-series table with typed metric columns; consider LIST/STRUCT for dynamic metrics.
3. **Finetune Evaluations**: QA per-question outputs (`TaskOutputData` schema) and aggregated metrics per task.
4. **Pretraining Curves**: compressed typed equivalent of `full_eval.parquet`; may live as base tables in DuckDB to enable direct joins with WandB data.

Document concrete schemas here once defined (column name, type, description) to keep ingestion scripts and consumers aligned.

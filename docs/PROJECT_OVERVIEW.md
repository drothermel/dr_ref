# Project Overview

## Mission
Develop a unified, queryable data platform for the DataDecide experiments that:
- Preserves raw pretraining/finetuning/evaluation artifacts in compressed, schema-aware storage (primarily DuckDB + Parquet).
- Produces curated, analysis-ready tables for interactive exploration (Marimo + Altair) and dashboard-style reporting.
- Supports predictive modelling workflows that relate early training signals to final model performance.

## Current Architecture (High-Level)
```mermaid
flowchart TD
    HF[HuggingFace raw datasets]
    WANDB[Weights & Biases API]
    RAW[Raw JSON/CSV dumps]
    DUCK_PIPE[DuckDB ingestion]
    PARQUET[Structured Parquet tables]
    ANALYSIS[Interactive analysis (Marimo/Altair)]
    MODELLING[Prediction pipelines (ddpred)]

    HF -->|download + convert| RAW
    WANDB -->|download + clean| RAW
    RAW -->|ingest scripts| DUCK_PIPE
    DUCK_PIPE --> PARQUET
    PARQUET --> ANALYSIS
    PARQUET --> MODELLING
```

- **Raw Data Acquisition**: legacy scripts and notebooks pull pretraining metrics from Hugging Face and run logs from WandB. Current experiments rely on Python downloaders (`dr_wandb`, `datadec/notebooks/duck_wandb.py`) and tarball unpackers (`datadec/notebooks/start.py`).
- **Processing & Enrichment**: `dr_showntell` contains the most complete WandB parsing (regex classification, config normalization) and merges with DataDecide pretraining metrics to produce matched finetune+pretrain tables.
- **Prediction Stack**: `ddpred` extracts windowed features from pretraining curves and runs cross-validated ElasticNet/GP models to forecast final metrics.
- **Visualization / Frontend**: `by-tomorrow-app` hosts the Svelte UI; future interactive analysis will lean on Marimo notebooks and DuckDB-backed queries.

## Status Snapshot
- ✅ Pretraining parquet pipeline (DataDecide) is stable for Hugging Face datasets.
- ✅ Robust WandB parsing lives in `dr_showntell`; new DuckDB ingestion notebooks are prototyping direct JSONL → Parquet conversion.
- ✅ Predictive modelling (`ddpred`) works on pretraining-only data with mature CV infrastructure.
- 🚧 Need to migrate ingestion + cleaning to DuckDB so all repos consume the same structured tables.
- 🚧 Post-training datasets (finetune evaluation, early-window signals) need schemas + storage strategy.
- 🚧 Frontend/dashboard work currently depends on raw APIs; future plan is to query DuckDB/DuckLake directly.

## Key Questions / Next Moves
1. Finalize DuckDB schemas for:
   - WandB run metadata/history (normalized config, eval metrics, matched groups).
   - Hugging Face pretraining datasets (compressed, typed variants of existing parquet).
   - Finetune evaluation outputs (QA task logs, aggregated metrics).
2. Decide migration path for existing parsing code (e.g., port `dr_showntell` logic into DuckDB transforms or Python loaders that emit structured tables).
3. Define the interface between interactive notebooks and the new tables (e.g., canonical views for Altair/Marimo vs. modelling features for `ddpred`).
4. Establish automation for the documentation index (see `scripts/render_agent_index.py`).

For per-repo details and onboarding instructions, see [`docs/REPO_MAP.md`](REPO_MAP.md) and the guides in `docs/guides/`.

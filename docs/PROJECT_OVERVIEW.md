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

## Documentation Index
<!-- START INDEX -->
- **Core Docs**
  - [Project Overview](docs/PROJECT_OVERVIEW.md) — Mission, architecture snapshot.
  - [Repo Map](docs/REPO_MAP.md) — Purpose + entry points for each repo.
  - [Data Artifacts](docs/DATA_ARTIFACTS.md) — Existing outputs and planned schemas.
  - [Roadmap](docs/ROADMAP.md) — Near/long-term priorities.

- **Per-Repo Guides**
  - [by-tomorrow-app](docs/guides/AGENT_GUIDE_by-tomorrow-app.md) — Per-repo onboarding guide.
  - [datadec](docs/guides/AGENT_GUIDE_datadec.md) — Per-repo onboarding guide.
  - [datadec notebooks](docs/guides/AGENT_GUIDE_datadec_notebooks.md) — Per-repo onboarding guide.
  - [ddpred](docs/guides/AGENT_GUIDE_ddpred.md) — Per-repo onboarding guide.
  - [dr showntell](docs/guides/AGENT_GUIDE_dr_showntell.md) — Per-repo onboarding guide.
  - [dr wandb](docs/guides/AGENT_GUIDE_dr_wandb.md) — Per-repo onboarding guide.

- **Process References**
  - [Coding Principles - Learnings from DataDec Development](docs/processes/CODING_PRINCIPLES.md) — This document captures key coding principles extracted from refactoring experiences, particularly from the `supervised_f…
  - [Audit Synthesis Pipeline](docs/processes/audit_synthesis_pipeline.md) — Transform multiple independent audit reports into validated, evidence-based implementation recommendations. This pipelin…
  - [Design Philosophy](docs/processes/design_philosophy.md) — The library's primary purpose is to accelerate the research process by making it easy to create a wide range of publicat…
  - [Documentation Organizer Guide](docs/processes/documentation_organizer_guide.md) — You are the **organizational memory curator** for this multi-agent collaboration system. Your mission is to ensure that …
  - [Fresh Eyes Review Guide](docs/processes/fresh_eyes_review_guide.md) — You are the **external auditor** coming to this codebase with fresh perspective. Your job is to evaluate the current sta…
  - [General Project Information Extraction](docs/processes/general_project_extraction_prompt.md) — You are tasked with conducting comprehensive information extraction for a completed project in the dr_plotter repository…
  - [Project Consolidation Methodology](docs/processes/project_consolidation_methodology.md) — **Purpose**: Framework for consolidating completed project documentation into central repository summaries that preserve…
  - [Project Reporting Guide](docs/processes/reporting_guide.md) — We use two complementary report types optimized for different purposes:
  - [Strategic Collaboration Guide](docs/processes/strategic_collaboration_guide.md) — **You are the conductor, not the performer** - provide strategic frameworks and decision criteria, let executing agents …
  - [Tactical Execution Guide](docs/processes/tactical_execution_guide.md) — You are the **skilled performer** in the conductor-performer paradigm. Your job is to take strategic guidance and framew…
<!-- END INDEX -->


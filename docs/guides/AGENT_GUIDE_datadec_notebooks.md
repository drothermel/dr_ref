# Agent Guide: datadec/notebooks

## Quick Navigation

- **Canonical ingestion pipeline**: see `docs/guides/AGENT_GUIDE_datadec.md` for production parquet outputs.
- **WandB normalization**: see `docs/guides/AGENT_GUIDE_dr_ingest.md` for classifier/post-process steps referenced by these notebooks.
- **QA schemas + helpers**: see `src/dr_ingest/qa/` and `docs/processes/` for struct definitions used here.
- **Marimo/reactivity guidance**: see `guides/marimo/` for building/maintaining the notebooks themselves.

## Table of Contents

1. [Snapshot](#snapshot)
2. [Current Status & High-Priority TODOs](#current-status--high-priority-todos)
3. [Notebook Inventory](#notebook-inventory)
4. [Workflow Recommendations](#workflow-recommendations)
5. [TODOs / Next Steps](#todos--next-steps)
6. [Update Log](#update-log)

## Snapshot
- **Purpose**: sandbox for the next-generation ingestion stack (DuckDB, Marimo, Altair). Contains prototypes for converting WandB exports to parquet and unpacking QA evaluation tarballs using shared helpers.
- **Key Files**: `duck_wandb.py`, `qa_instances_ingest.py`, plus ad-hoc exploration notebooks (`explore_data.py`, etc.).
- **Location**: `repos/datadec/notebooks/`.

## Current Status & High-Priority TODOs

- ✅ DuckDB JSONL → Parquet experiment demonstrates ~10× size reduction and initial schema ideas.
- ✅ QA tarball unpacking notebook bridges Hugging Face downloads with `dr_ingest/qa` helpers.
- ⚠️ **Automation gap**: TODO extract reusable modules from notebooks before wiring into CI/CLI.
- ⚠️ **Schema tracking**: TODO reflect any new column layouts here and in `docs/DATA_ARTIFACTS.md` to prevent drift.

## Notebook Inventory
1. `duck_wandb.py`
   - Downloads WandB runs/history (if JSONL not present), converts to Parquet using DuckDB (`read_json` + `COPY`).
   - Normalizes select summary fields (`oe_eval_metrics`), drops duplicates/constants, creates `runs_clean` table inside DuckDB, and previews data via Quak widget.
   - Demonstrates size savings (~60 MB JSONL → ~5 MB Parquet) and sets the stage for schema design.

2. `qa_instances_ingest.py`
   - Marimo notebook for unpacking Hugging Face QA tarballs.
   - Delegates tarball discovery/extraction and JSONL reshaping to `dr_ingest/qa` helpers (schemas + transforms) before previewing structured payloads.
   - Demonstrates attrs/cattrs structuring for typed question/answer records without relying on Postgres.

3. `explore_data.py`, `explore_wandb_data.py`
   - Legacy exploratory notebooks; safe to skim for context but focus primarily on the two notebooks above.

## Workflow Recommendations
- Treat these notebooks as reference implementations. When formalizing ingestion scripts, extract reusable functions into dedicated modules (e.g., under `datadec/src`).
- Replace hard-coded paths with configurable inputs before automation.
- Document any new DuckDB tables created here in `docs/DATA_ARTIFACTS.md`.

## TODOs / Next Steps
- TODO: generalize JSONL → Parquet conversion to handle large batches (streaming, chunking).
- TODO: implement schema definitions for WandB runs/history inside DuckDB (struct/list columns, flattening strategies).
- TODO: build a pipeline for bulk downloading QA tarballs, converting to DuckDB tables, and disposing of temporary files.
- TODO: capture notebook outputs (metrics, sizes) in text form for agents who cannot run Marimo interactively.

Use this guide when replicating notebook logic or porting it into production scripts.

### Update Log
- _Add recent changes here._

# Agent Guide: datadec/notebooks

## Snapshot
- **Purpose**: sandbox for the next-generation ingestion stack (DuckDB, Marimo, Altair). Contains prototypes for converting WandB exports to parquet and unpacking QA evaluation tarballs.
- **Key Files**: `duck_wandb.py`, `start.py`, plus ad-hoc exploration notebooks (`explore_data.py`, etc.).

## Notebooks
1. `duck_wandb.py`
   - Downloads WandB runs/history (if JSONL not present), converts to Parquet using DuckDB (`read_json` + `COPY`).
   - Normalizes select summary fields (`oe_eval_metrics`), drops duplicates/constants, creates `runs_clean` table inside DuckDB, and previews data via Quak widget.
   - Demonstrates size savings (~60 MB JSONL → ~5 MB Parquet) and sets the stage for schema design.

2. `start.py`
   - Marimo notebook for unpacking Hugging Face QA tarballs.
   - Uses `attrs` + `cattrs` to map nested JSON to typed dataclasses (`ModelAnswerOutput`, `QuestionOutputData`, `TaskOutputData`).
   - Experiments with Postgres composite types as an interim schema; lessons carry over to planned DuckDB struct definitions.

3. `explore_data.py`, `explore_wandb_data.py`
   - Legacy exploratory notebooks; safe to skim for context but focus primarily on the two notebooks above.

## Workflow Recommendations
- Treat these notebooks as reference implementations. When formalizing ingestion scripts, extract reusable functions into dedicated modules (e.g., under `datadec/src`).
- Replace hard-coded paths with configurable inputs before automation.
- Document any new DuckDB tables created here in `docs/DATA_ARTIFACTS.md`.

## TODOs / Next Steps
- Generalize JSONL → Parquet conversion to handle large batches (streaming, chunking).
- Implement schema definitions for WandB runs/history inside DuckDB (struct/list columns, flattening strategies).
- Build a pipeline for bulk downloading QA tarballs, converting to DuckDB tables, and disposing of temporary files.
- Capture notebook outputs (metrics, sizes) in text form for agents who cannot run Marimo interactively.

Use this guide when replicating notebook logic or porting it into production scripts.

### Update Log
- _Add recent changes here._

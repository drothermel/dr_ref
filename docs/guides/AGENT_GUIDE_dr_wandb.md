# Agent Guide: dr_wandb

## Quick Navigation

- **Downstream parsing**: see `docs/guides/AGENT_GUIDE_dr_ingest.md` for how these exports are normalized.
- **DuckDB/schema targets**: see `docs/DATA_ARTIFACTS.md` for the destination tables this repo should eventually populate.
- **Notebook experiments**: see `docs/guides/AGENT_GUIDE_datadec_notebooks.md` (`duck_wandb.py`) for alternative ingestion flows.
- **Process context / roadmap**: see `docs/ROADMAP.md` for migration milestones.

## Table of Contents

1. [Snapshot](#snapshot)
2. [Current Status & High-Priority TODOs](#current-status--high-priority-todos)
3. [Reading Order](#reading-order)
4. [Data Tables](#data-tables)
5. [Integration Points](#integration-points)
6. [Known Gaps / TODOs](#known-gaps--todos)
7. [Update Log](#update-log)

## Snapshot
- **Purpose**: isolate WandB data acquisition. Downloads runs/history, stores in PostgreSQL via SQLAlchemy ORM, and exports parquet files for offline analysis.
- **Outputs**: `data/runs_metadata.parquet`, `data/runs_history.parquet`, plus component-specific parquet slices and optional Postgres database.
- **Entry Points**: `src/dr_wandb/downloader.py` (API fetch + orchestration) and `src/dr_wandb/store.py` (database persistence/export).

## Current Status & High-Priority TODOs

- ✅ Downloader + ORM flow works for targeted projects, exporting parquet for downstream use.
- ⚠️ **Postgres dependency**: TODO evaluate replacing the intermediate database with DuckDB or direct parquet writes.
- ⚠️ **Schema clarity**: TODO document which JSONB fields need flattening so dr_ingest can rely on stable columns.
- ⚠️ **History volume**: TODO design chunked export strategy for large histories (avoid memory spikes).

## Reading Order
1. `src/dr_wandb/downloader.py` – `Downloader` class and CLI hook; handles incremental updates, force-refresh, history streaming.
2. `src/dr_wandb/store.py` – ORM models (`RunRecord`, `HistoryEntryRecord`), Postgres bootstrap, export logic (`export_to_parquet`).
3. `src/dr_wandb/run_record.py` / `history_entry_record.py` – JSONB columns, config/sweep info, helper queries.
4. `src/dr_wandb/utils.py` – large-int safe conversions, progress callbacks, run filtering.
5. `src/dr_wandb/cli/download.py` – `wandb-download` command (env-configurable) used in automation.

## Data Tables
- `runs_metadata.parquet`: per-run fields (config, summary, metadata, system info). JSON-heavy; needs normalization before loading into DuckDB.
- `runs_metadata_{component}.parquet`: same data split by config/summary/etc. (useful when working with specific payloads).
- `runs_history.parquet`: flattened history rows with `_step`, `_timestamp`, `_wandb` fields and metrics map.

## Integration Points
- `dr_ingest` reads the exported parquet files for regex parsing and matching to pretrain metrics.
- New DuckDB ingestion notebooks (`datadec/notebooks/duck_wandb.py`) reuse the JSON exports to test schema ideas.
- Future plan is to replace Postgres dependency with direct DuckDB writes; keep existing ORM as reference until parity is achieved.

## Known Gaps / TODOs
- TODO: finalize schema design for JSONB fields; today many columns remain serialized strings after export.
- TODO: flatten history metrics or adopt DuckDB `struct` columns instead of nested dict blobs.
- TODO: document or implement multi-project download workflows so CLI users can batch repositories.

### Update Log
- _Add recent changes here._

Update this guide when the export format changes or when DuckDB ingestion supersedes the Postgres pipeline.

# Agent Guide: by-tomorrow-app

## Quick Navigation

- **Data sources**: see `docs/guides/AGENT_GUIDE_datadec.md` (pretraining tables) and `docs/guides/AGENT_GUIDE_dr_ingest.md` (WandB normalization).
- **Notebook prototypes**: see `docs/guides/AGENT_GUIDE_datadec_notebooks.md` for DuckDB + Marimo experiments that feed this UI.
- **Reactivity/UI details**: see `guides/marimo/` for building interactive notebooks that can complement the app.
- **API dumps**: see `docs/DATA_ARTIFACTS.md` for current parquet exports powering the backend.

## Table of Contents

1. [Snapshot](#snapshot)
2. [Current Status & High-Priority TODOs](#current-status--high-priority-todos)
3. [Reading Order (Frontend)](#reading-order-frontend)
4. [Backend (FastAPI prototype)](#backend-fastapi-prototype)
5. [Data Flow Today](#data-flow-today)
6. [Transition Plan](#transition-plan)
7. [Known Gaps / Risks](#known-gaps--risks)
8. [Update Log](#update-log)

## Snapshot
- **Purpose**: SvelteKit frontend + FastAPI backend prototype for interactive exploration (tables + charts) of DataDecide/WandB datasets.
- **Current State**: Hosted on Vercel (frontend) and Railway (backend). Backend serves static parquet slices; plan is to remove it in favor of direct DuckDB queries from the frontend.
- **Owner**: `repos/by-tomorrow-app`.

## Current Status & High-Priority TODOs

- ✅ Frontend renders sample charts and tables using precomputed parquet slices.
- ⚠️ **Backend deprecation**: TODO confirm if Railway backend is still required or if DuckDB-in-browser is feasible; document cut-over plan.
- ⚠️ **Auth/layout context**: TODO capture how `(protected)` routes enforce auth so agents editing data viz don’t break gating.
- ⚠️ **Data contracts**: TODO sync column naming assumptions with upcoming DuckDB schemas before replacing the backend.

## Reading Order (Frontend)
1. `src/routes/(protected)/data_viz/+page.svelte` – landing page linking to demo chart + table views.
2. `src/routes/(protected)/data_viz/chart/+page.svelte` – loads small sample data via `fetchDataFrame` and renders `LineChartComponent`.
3. `src/routes/(protected)/data_viz/ft-pt-table/+page.svelte` – fetches finetune/pretrain merged table and renders `DataGridComponent`.
4. Components:
   - `src/lib/components/dataviz/LineChartComponent.svelte` – LayerChart wrapper with param-size sorting.
   - `src/lib/components/dataviz/DataGridComponent.svelte` – SVAR grid with grouped headers and dynamic column discovery.
5. Data service: `src/lib/services/dataService.ts` – simple fetch helper pointing to FastAPI endpoint.

## Backend (FastAPI prototype)
- `python-backend/main.py`: loads pre-generated parquet/pickle files (`initial_data/`), filters them, and serves JSON responses at `/api/visualize/data*`.
- No database; plan is to delete backend once frontend can query DuckDB/ducklake directly.

## Data Flow Today
1. Precomputed pickle/parquet (generated offline from ingestion notebooks) are copied into `python-backend/initial_data/`.
2. FastAPI endpoints read entire dataset on each request, perform simple filtering, return JSON.
3. Frontend fetches these endpoints and renders charts/tables.

## Transition Plan
- Replace FastAPI with DuckDB queries executed client-side (e.g., via WASM) or through a lightweight DuckLake service.
- Move data fetching logic to point at curated DuckDB tables once schemas are finalized.
- Use Marimo notebooks for exploratory plots; keep frontend for canonical dashboards.

## Known Gaps / Risks
- TODO: parameterize the data service base URL (`dataService.ts`) so local dev isn’t tied to `https://api.bytomorrow.app`.
- TODO: document column contracts for LayerChart/DataGrid before DuckDB schemas land.
- TODO: summarize authentication/layout approach from `(protected)` routes for future contributors.

Use this guide to orient yourself before modifying frontend components or replacing the backend.

### Update Log
- _Add recent changes here._

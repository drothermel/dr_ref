# Agent Guide: by-tomorrow-app

## Snapshot
- **Purpose**: SvelteKit frontend + FastAPI backend prototype for interactive exploration (tables + charts) of DataDecide/WandB datasets.
- **Current State**: Hosted on Vercel (frontend) and Railway (backend). Backend serves static parquet slices; plan is to remove it in favor of direct DuckDB queries from the frontend.

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

## Known Issues / TODOs
- Data service currently hard-coded to `https://api.bytomorrow.app`; parameterize or expose environment variables for local development.
- LayerChart + DataGrid components assume specific column naming; update when DuckDB schemas change.
- Authentication/layout code omitted here—consult routes under `(protected)` and `(public)` if working on broader app features.

Use this guide to orient yourself before modifying frontend components or replacing the backend.

### Update Log
- _Add recent changes here._

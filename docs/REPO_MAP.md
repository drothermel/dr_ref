# Repository Map

Quick reference to the major codebases that make up the DataDecide tooling. Each entry lists its purpose, where to start reading, and key data products. Detailed onboarding lives in the per-repo guides under [`docs/guides/`](guides/).

| Repo | Purpose | Start Here | Outputs / Data Products |
|------|---------|------------|-------------------------|
| `datadec` | Source of truth for pretraining datasets and pipelines. Downloads Hugging Face results, parses perplexity + downstream metrics, and produces parquet tables. | `src/datadec/pipeline.py` (stage definitions), `src/datadec/data.py` (`DataDecide` API). | `data/datadecide/*.parquet` (full_eval, mean_eval, etc.). |
| `dr_wandb` | Standalone WandB downloader. Syncs runs/history into Postgres-backed store and emits parquet exports. | `src/dr_wandb/downloader.py`, `src/dr_wandb/store.py`. | `data/runs_metadata*.parquet`, `data/runs_history.parquet`. |
| `dr_ingest` | Shared ingestion utilities that classify/parses WandB runs and apply normalization via declarative configs. | `src/dr_ingest/wandb/classifier.py`, `src/dr_ingest/wandb/postprocess.py`, `configs/wandb/*.cfg`. | Cleaned run metadata DataFrames; planned DuckDB tables. |
| `ddpred` | Early-window prediction experiments. Feature extraction, cross-validation, and modelling to predict final metrics. | `src/ddpred/core/data_pipeline.py`, `src/ddpred/core/data_preparation.py`. | Model weights, CV reports (`results/`, `markdowns/`), evaluation artefacts. |
| `by-tomorrow-app` | SvelteKit frontend experiment. Displays tables (SVAR grid) and charts (LayerChart) driven by FastAPI backend (to be replaced with DuckDB queries). | `src/routes/(protected)/data_viz/`, `src/lib/components/dataviz/*`. | Deployed UI (Vercel), API endpoints serving parquet subsets. |
| `datadec/notebooks` | Prototyping area for new ingestion workflows (DuckDB, Marimo). | `duck_wandb.py`, `start.py`. | Ad-hoc parquet conversions, schema experiments. |

Other supporting folders in `dr_ref`:
- `docs/processes/`: collaboration, reporting, and methodology guides.
- `projects/`: project-specific references and notes.
- `repos/`: small convenience scripts for cloning and syncing external repos.

Use this map together with the per-repo guides to choose which code to open first.

# Roadmap (Working Draft)

## Immediate Priorities (Next 2–4 Weeks)
1. **DuckDB Ingestion MVP**
   - Finalize JSONL → Parquet → DuckDB flow for WandB runs/history (build on `duck_wandb.py`).
   - Define initial DuckDB schemas for runs, history, and matched finetune data; document in `docs/DATA_ARTIFACTS.md`.
   - Automate conversion scripts so new dumps can be processed reproducibly.
2. **Finetune Evaluation Imports**
   - Implement chunked download/unzip of Hugging Face QA tarballs (temp storage → DuckDB → Parquet) per plan in `start.py`.
   - Capture raw data with minimal loss; design typed struct columns for answers/metrics.
3. **Documentation & Guides**
   - Complete per-repo `AGENT_GUIDE` files.
   - Hook up the agent index script to regenerate doc summaries as repos change.

## Near Term (1–2 Months)
- **Shared Data Views**: create curated DuckDB views for common analyses (e.g., matched finetune vs pretrain, early-window feature windows).
- **Modelling Integration**: refactor `ddpred` pipelines to read from DuckDB tables instead of legacy parquet paths.
- **Interactive Analysis**: build exemplar Marimo notebooks querying the new tables and rendering Altair charts; evaluate integration with `by-tomorrow-app`.

## Longer Term
- **Storage Optimization**: evaluate hosting options (DuckLake, MotherDuck, R2) once local pipeline stabilizes.
- **Automation / CI**: schedule periodic refreshes (download → ingest → publish docs) and validation checks.
- **Predictive Experiments**: extend early-window modelling to post-training metrics, incorporate matched group features, and benchmark against baselines.

> Update this roadmap as priorities shift. Each change should reference commits or issues where the decision was made.

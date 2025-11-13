# dr_ref: Reference Hub

This repository now serves as the shared knowledge base for the DataDecide ecosystem. It contains:

- **Project overview & roadmap**: see [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md), [`docs/ROADMAP.md`](docs/ROADMAP.md).
- **Repository map**: quick summary of each codebase in [`docs/REPO_MAP.md`](docs/REPO_MAP.md).
- **Per-repo agent guides**: onboarding instructions in [`docs/guides/`](docs/guides/) (datadec, dr_wandb, dr_showntell, ddpred, by-tomorrow-app, notebooks, etc.).
- **Data artifacts inventory**: current parquet/pickle outputs and planned DuckDB tables in [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md).
- **Process references**: existing collaboration and methodology guides under [`docs/processes/`](docs/processes/).

## Getting Started (Agent Checklist)
1. Read [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) for mission, architecture, and current status.
2. Consult [`docs/REPO_MAP.md`](docs/REPO_MAP.md) to see which repository/module to inspect.
3. Open the relevant `AGENT_GUIDE_<repo>.md` under `docs/guides/` for detailed entry points, key files, and current TODOs.
4. Update documentation when you make changes: add new data artifacts, adjust guides, or revise the roadmap.

## Future Automation
- `scripts/render_agent_index.py` (to be added) will generate an index of guides/sections for quick navigation. Until then, maintain the docs manually.
- Consider linking each external repo’s README to the corresponding guide here (symlink or relative link).

Keep `dr_ref` in sync with active development so onboarding remains fast and reliable.

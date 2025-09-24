# Tooling Stack Notes

Overview of lightweight utilities adopted for data ingestion and analysis workflows. These libraries complement `clumper` and support the push toward shared parsing/ingestion modules.

## memo
- **What**: decorators for logging function calls (`memfunc`, `memlist`, `memfile`, `memweb`).
- **Usage**: wrap critical parsing/transformation functions to record inputs/outputs during development or debugging.
- **Example**: decorate `classify_run_id_type_and_extract` to append structured logs to a list or file, helping identify unmatched run IDs.

## catalogue
- **What**: lightweight function registry.
- **Usage**: register regex patterns, parsers, or ingestion steps under namespaced registries (e.g., `catalogue.create("dr", "wandb", "patterns")`).
- **Benefits**: modular addition/removal of handlers without editing central dispatch tables; downstream tools can query registries dynamically.

## confection
- **What**: configuration loader/validator (supports nested configs, integrates with Pydantic when available).
- **Usage**: define schema-driven configs for parsing rules, default mappings, ingestion parameters. Load once and pass to registry functions.
- **Benefits**: central, version-controlled config for both legacy pipelines and new DuckDB scripts.

## Typer
- **What**: CLI framework on top of Click with modern ergonomics.
- **Usage**: expose ingestion/refactor utilities (e.g., `parse-wandb`, `ingest-qa-tarballs`) as command-line subcommands; easy integration with uv.
- **Benefits**: consistent interface for running parsing tasks locally or in automation.

## Integration Plan (Example: Shared WandB Parsing Module)
1. Extract parsing functions into a new module (planned: `dr_ingest/wandb.py`).
2. Load pattern/config metadata via Confection; register parsing handlers with Catalogue.
3. Decorate key entry points with memo to capture per-run diagnostics during rollout.
4. Provide a Typer CLI (`python -m dr_ingest.cli parse-wandb …`) for scripted execution.

Keep this note updated as additional utilities are adopted or deprecated.

# Agent Guide: dr_ingest

## Snapshot
- **Purpose**: shared ingestion toolkit for the DataDecide ecosystem. Hosts canonical WandB parsing, normalization, and ingest-ready transforms that downstream repos can reuse.
- **Status**: classification + post-processing pipelines now consolidated behind declarative configs. DuckDB export helpers and CLIs are in progress.
- **Outputs**: cleaned per-run DataFrames (grouped by run type), token-normalized fields, and fixtures for regression tests; future goal is emitting DuckDB tables directly.

## Reading Order
1. `src/dr_ingest/wandb/config.py` & `configs/wandb/*.cfg` — layered Confection configs that register patterns, defaults, converters, and hooks.
2. `src/dr_ingest/wandb/classifier.py` — run-ID classification driven by catalogue-registered regex specs.
3. `src/dr_ingest/wandb/postprocess.py` & `processing_context.py` — apply renames, defaults, config merges, and token normalization.
4. `src/dr_ingest/wandb/metrics.py` & `tokens.py` — canonicalize metric labels and harmonize finetune token counts.
5. Support utilities: `src/dr_ingest/df_ops.py`, `json_utils.py`, `normalization.py`, `wandb/utils.py`.
6. Tests + fixtures under `tests/` and `tests/fixtures/` — sample parquet slices and expectations for classifier/post-processing behavior.

## Key Modules & Concepts
- **Pattern Registry**: `pattern_builders.py` composes regex fragments from `constants.py`; specs live in config files and are resolved via Confection + catalogue.
- **Processing Context**: `ProcessingContext.from_config` loads defaults, column renames, recipe mappings, converter hooks, and per-run-type hooks for a single pass pipeline.
- **Value Converters**: `wandb/utils.py` registers `timestamp` and `tokens_to_number` converters so configs stay declarative.
- **Classification Log**: memo-backed log in `classifier.py` captures in-flight matches for debugging and telemetry.
- **Fixtures**: `tests/fixtures/build_wandb_samples.py` regenerates small parquet subsets from upstream datasets for reproducible testing.

## Common Workflows
- **Classify run IDs**: load a runs parquet, call `parse_and_group_run_ids`, then `convert_groups_to_dataframes` for per-run-type tables.
- **Apply post-processing**: feed grouped frames into `apply_processing(...)`, optionally supplying full runs/history DataFrames to hydrate config/summary fields.
- **Extend patterns**: add new entries to `configs/wandb/patterns.cfg`; tests ensure regex coverage before deploying.
- **Run tests**: `uv run pytest` (or equivalent) to validate classifiers, metric canonicalization, and token filling logic against fixtures.

## Integration Points
- Consumes exports from `dr_wandb` (runs/history) and pretraining parquet from `datadec` for recipe normalization.
- Supplies cleaned tables to modelling stacks like `ddpred` and any DuckDB ingestion notebooks.
- Future CLIs will provide Typer commands for batch parsing and DuckDB materialization used by frontend experiments (`by-tomorrow-app`).

## Known Gaps / TODOs
- Emit standardized DuckDB schemas/tables (runs, history, matched finetune) instead of pandas-only outputs.
- Broaden run-type hooks beyond `matched` to cover new pipelines (e.g., DPO variants, reduce-loss experiments).
- Fold history-aware enrichment (learning rate curves, metrics) into `apply_processing` once schema settled.
- Package Typer CLI entry points for reproducible ingestion jobs and documentation index automation.

### Update Log
- _Add recent changes here._

# Agent Guide: dr_ingest

## Quick Navigation

- **WandB acquisition**: see `docs/guides/AGENT_GUIDE_dr_wandb.md` (JSON exports feeding this toolkit).
- **Pretraining metrics**: see `docs/guides/AGENT_GUIDE_datadec.md` for `full_eval.parquet` consumed during enrichment.
- **Notebook sandboxes**: see `docs/guides/AGENT_GUIDE_datadec_notebooks.md` for DuckDB prototype notebooks that import these helpers.
- **Schema plans**: see `docs/DATA_ARTIFACTS.md` for the target DuckDB tables this repo should emit.

## Table of Contents

1. [Snapshot](#snapshot)
2. [Current Status & High-Priority TODOs](#current-status--high-priority-todos)
3. [Reading Order](#reading-order)
4. [Key Modules & Concepts](#key-modules--concepts)
5. [Common Workflows](#common-workflows)
6. [Integration Points](#integration-points)
7. [Known Gaps / TODOs](#known-gaps--todos)
8. [Update Log](#update-log)

## Snapshot
- **Purpose**: shared ingestion toolkit for the DataDecide ecosystem. Hosts canonical WandB parsing, normalization, and ingest-ready transforms that downstream repos can reuse.
- **Outputs**: cleaned per-run DataFrames (grouped by run type), token-normalized fields, and fixtures for regression tests; future goal is emitting DuckDB tables directly.
- **Status**: classification + post-processing pipelines now consolidated behind declarative configs. DuckDB export helpers and CLIs are in progress.

## Current Status & High-Priority TODOs

- ✅ Pattern registry + processing context support the main WandB run types (`matched`, `pretrain`, etc.).
- ✅ QA ingestion helpers (`dr_ingest/qa`) power tarball structuring without Postgres.
- ⚠️ **DuckDB export**: TODO design & implement canonical DuckDB tables produced directly from `apply_processing`.
- ⚠️ **CLI tooling**: TODO expose Typer/uv commands for batch ingestion + documentation updates.
- ⚠️ **History-aware enrichment**: TODO integrate history metrics (LR curves, evals) once schema is available.

## Reading Order
1. `src/dr_ingest/wandb/config.py` & `configs/wandb/*.cfg` — layered Confection configs that register patterns, defaults, converters, and hooks.
2. `src/dr_ingest/wandb/classifier.py` — run-ID classification driven by catalogue-registered regex specs.
3. `src/dr_ingest/wandb/postprocess.py` & `processing_context.py` — apply renames, defaults, config merges, and token normalization.
4. `src/dr_ingest/wandb/metrics.py` & `tokens.py` — canonicalize metric labels and harmonize finetune token counts.
5. QA ingestion helpers: `src/dr_ingest/qa/extraction.py`, `qa/schemas.py`, `qa/transform.py` — shared tarball extraction, attrs/cattrs schemas, and Clumper-based JSONL reshaping for evaluation data.
6. Support utilities: `src/dr_ingest/df_ops.py`, `json_utils.py`, `normalization.py`, `wandb/utils.py`.
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
- TODO: emit standardized DuckDB schemas/tables (runs, history, matched finetune) instead of pandas-only outputs.
- TODO: broaden run-type hooks beyond `matched` to cover new pipelines (e.g., DPO variants, reduce-loss experiments).
- TODO: fold history-aware enrichment (learning rate curves, metrics) into `apply_processing` once schema settles.
- TODO: package Typer CLI entry points for reproducible ingestion jobs and documentation index automation.

### Update Log
- _Add recent changes here._

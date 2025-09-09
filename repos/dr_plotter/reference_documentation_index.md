# Reference Documentation Index

## Active Documentation Locations

### Verification System Architecture
- Location: Current dr_plotter verification system (8 core files)
- Description: Clean fail-fast verification architecture with unified engine
- Components: `plot_data_extractor.py`, `unified_verification_engine.py`, `verif_decorators.py`, `verification_formatter.py`, `comparison_utils.py`
- Last Updated: August 2025 (post-consolidation)

### Verification User Interface  
- Location: `verif_decorators.py` - Simplified decorator interface
- Description: 2-3 clear decorator patterns replacing complex overlapping system
- Usage: Assertion-based validation with consistent formatting
- Last Updated: August 2025

### Verification Output Formatting
- Location: `verification_formatter.py` - VerificationFormatter class
- Description: Consistent symbols (✅🔴⚠️🔍💥🎉), 4-space indentation, standardized templates
- Integration: Centralized convenience functions for verification logic files
- Last Updated: August 2025

### Architectural Reference Documentation
- Location: `docs/reference/architecture.md`
- Description: Comprehensive architectural reference covering foundation improvements, style systems, configuration management
- Coverage: Type systems, configuration objects, StyleApplicator patterns, legend coordination strategies
- Last Updated: August 2025 (post-architectural enhancement)

### Faceted Plotting API Documentation
- Location: `docs/reference/api/faceting_api_reference.md`
- Description: Complete API reference for native faceting functionality
- Coverage: `fm.plot_faceted()` interface, parameter specifications, usage patterns
- Last Updated: August 2025

### Faceting Migration Guide
- Location: `docs/reference/api/faceting_migration_guide.md`  
- Description: Migration guidance from manual subplot management to faceting API
- Coverage: Before/after examples, common patterns, troubleshooting
- Last Updated: August 2025

### Configuration System Architecture
- Location: Current dr_plotter unified configuration system (all 8 plotters)
- Description: Unified parameter resolution with clear precedence hierarchy (Context → User → Theme)
- Components: `_resolve_phase_config()` method, component schema definitions, artist utilities
- Last Updated: September 2025 (post-configuration system fixes)

### Artist Property Extraction Utilities
- Location: `src/dr_plotter/artist_utils.py`
- Description: Atomic, reusable functions for matplotlib artist property extraction
- Coverage: Color extraction, alpha extraction, fail-fast validation patterns
- Last Updated: September 2025

### CLI Interface Documentation
- Location: `src/dr_plotter/scripting/cli.py`
- Description: Generic command-line interface for data visualization without external dependencies
- Coverage: Parquet/CSV file processing, flexible dimensional plotting options, generic data source support
- Last Updated: September 2024 (post-DataDecide dependency removal)

### Implementation Methodology Guides  
- Location: `docs/guides/audit_methodology.md`, `docs/guides/implementation_patterns.md`
- Description: Evidence-based assessment processes and validated development strategies
- Coverage: 5-stage audit pipeline, multi-agent collaboration patterns, architectural transformation approaches, unified configuration patterns, retrospective forensic analysis
- Last Updated: September 2025 (updated with retrospective analysis patterns)

## Archive References

### Verification System Consolidation (2025-08-29)
- Archive: `docs/projects/archive/verification_system_consolidation_2024/`
- Summary: 14 implementation documents + git commit history
- Key Artifacts: Multi-phase consolidation prompts, implementation logs, completion documentation
- Historical Value: Methodology validation, strategic/tactical collaboration patterns

### Architectural Enhancement Project (2025-08-28)
- Archive: `docs/archive/2025/`
- Summary: 130+ analysis files, 37 audit reports across 5 architectural categories
- Key Artifacts: Evidence-based audit reports, multi-agent collaboration documentation, systematic improvement tracking
- Historical Value: Large-scale architectural transformation methodology, cross-category integration patterns

### Faceted Plotting Implementation (2025-08-28)
- Archive: `docs/archive/2025/faceting_project_artifacts/`
- Summary: Complete implementation documentation including requirements, design specifications, implementation plans
- Key Artifacts: 6-phase implementation strategy, comprehensive testing approach, API design rationale
- Historical Value: Systematic feature development methodology, infrastructure integration patterns

### Configuration System and Parameter Flow Fixes (2025-08-30 to 2025-09-03)
- Archive: `docs/projects/active/manual_cleanup/`
- Summary: Comprehensive parameter flow resolution with unified configuration system implementation
- Key Artifacts: `done__configuration_system_and_parameter_flow.md`, `todos.md` with systematic cleanup tracking
- Historical Value: Unified configuration resolution methodology, defensive programming elimination patterns, architectural compliance improvement strategies

### DataDecide Dependency Removal (2024-08-28 to 2024-09-09) 
- Archive: `docs/projects/active/extract_datadec/` (planning documentation), Git commits (implementation evidence)
- Summary: Complete scope reversal from planned integration to dependency elimination with CLI pivot
- Key Artifacts: Comprehensive planning documents (1,515 lines), retrospective forensic analysis, architectural reversal evidence
- Historical Value: Strategic scope reversal methodology, comprehensive planning processes, CLI as integration alternative patterns, forensic project reconstruction techniques
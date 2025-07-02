# Implementation Plan: Tier 4 Analysis Suite (deconCNN)

## 📋 Context from Tier 3 Implementation

**What Tier 3 Accomplished:**
- ✅ Created `src/deconcnn/analysis/` directory structure (existing from prior work)
- ✅ Implemented LossSlopeLogger callback with slope calculation (alpha_5_15, alpha_full)
- ✅ Configured CurvatureMonitor (500-step frequency) and NoiseMonitor (2-epoch frequency)
- ✅ Created monitoring script `scripts/monitor_experiment.py` for job tracking
- ✅ Set up basic failure recovery in `scripts/recover_failed.py`
- ✅ Created validation notebook `notebooks/validate_metrics_basic.ipynb` with H1/H2 checks
- ✅ Created operational documentation `docs/operational_runbook_basic.md`
- ✅ Added dependencies: pandas>=2.3.0, matplotlib>=3.10.3, statsmodels>=0.14.4, rich>=14.0.0
- ✅ Established dr_exp metrics.jsonl logging format
- ✅ Created data pipeline scripts: `collect_and_archive.sh`, `verify_completeness.py`, `prepare_dataset.py`
- ✅ Created test scripts: `validate_local.py`, `test_harness.py` 
- ✅ Created submission wrapper `scripts/submit_all_loss_slope.sh` for 216-job sweep
- ✅ Created utils module and extract EWMA
- ✅ Created fitting module and extract power law
- ✅ Created spectral module and extract PSD
- ✅ Extracted slope calculation to utils

**What Tier 4 Must Complete:**
- Analysis library functions (knee detection, power law fitting, metric validation)
- Extract and modularize existing analysis code from callbacks
- Advanced SLURM job submission and monitoring infrastructure
- Intelligent failure recovery with pattern analysis
- Efficient result collection and archival systems
- Comprehensive documentation and tutorials
- Data format conversion utilities (JSON to CSV/Parquet)
- Missing visualization dependencies (seaborn, plotly, jupyter)

## Agent Instructions
**IMPORTANT: Read these before starting implementation**

1. **Quality Gates**: Before EVERY commit:
   - Run `lint_fix` and resolve all issues
   - Run tests and ensure they pass
   - Fix any issues found, even if pre-existing

2. **Progress Tracking**: 
   - Mark each step with ✅ when complete
   - Add notes in [brackets] for any deviations
   - Update the "Current Status" section after each work session

3. **Adaptation Protocol**:
   - Follow the plan but use judgment
   - Document any necessary modifications in [brackets]
   - Flag significant changes for retrospective

4. **Commit Message Protocol**:
   - **CRITICAL**: Use the EXACT commit message provided in each step
   - Messages are marked with "Use exact message: `<message>`"
   - Do NOT modify, add to, or abbreviate the provided commit messages
   - Each commit should be atomic and focused on a single logical change
   - If a step has no commit message, no commit is needed for that step

5. **Implementation Philosophy**:
   - Build on existing foundations from Tier 3
   - Extract and consolidate existing code to avoid duplication
   - Maintain backward compatibility with existing callbacks
   - Cross-validate extensively against current behavior

## Current Status
- Last updated: 2025-07-02 - Merged advanced SLURM integration from phase_4_slurm_update.md
- Last completed step: Plan restructured with advanced SLURM features
- Active agent: ready for implementation
- Blocked by: none

### [x] Phase 1: Enhanced Mathematical Functions (4 commits)
- [x] **Commit 1**: Add AIC computation to utils ✅ COMPLETED
- [x] **Commit 2**: Implement exponential fitting
- [x] **Commit 3**: Implement Welch's method for spectral analysis ✅ COMPLETED
- [x] **Commit 4**: Add noise floor detection ✅ COMPLETED

### [x] Phase 2: Detection & Analysis Functions (6 commits)
- [x] **Commit 5**: Implement knee detection
- [x] **Commit 6**: Implement windowed slope analysis
- [x] **Commit 7**: Implement segmented power law fitting
- [x] **Commit 8**: Implement Hutch++ algorithm - IN PROGRESS
- [x] **Commit 9**: Implement plateau detection
- [x] **Commit 10**: Add statistical utilities

### [x] Phase 3: Primary Analysis Scripts (6 commits)
- [x] **Commit 11**: Implement loss mosaic visualization
- [x] **Commit 12**: Create knee detection script
- [x] **Commit 13**: Create slope analysis script
- [x] **Commit 14**: Create model comparison script
- [x] **Commit 15**: Test primary analysis scripts
- [x] **Commit 16**: Add progress tracking to scripts

### [x] Phase 4: Supporting Analysis Scripts (5 commits)
- [x] **Commit 17**: Create architecture ablation script
- [x] **Commit 18**: Create hyperparameter heatmap script
- [x] **Commit 19**: Create noise correlation script
- [x] **Commit 20**: Create curvature timing script
- [x] **Commit 21**: Integration test supporting scripts

### [x] Phase 5: Slide Generation Notebooks (13 commits)
- [x] **Commit 22**: Create H1 hypothesis notebook
- [x] **Commit 23**: Create H2 hypothesis notebook
- [x] **Commit 24**: Add interactivity to primary notebooks
- [x] **Commit 25**: Create model comparison notebook
- [x] **Commit 26**: Create window robustness notebook
- [x] **Commit 27**: Create architecture ablation notebook
- [x] **Commit 28**: Create curvature exploration notebook
- [x] **Commit 29**: Add export functionality to notebooks
- [x] **Commit 30**: Create notebook utilities
- [x] **Commit 31**: Create integration test suite
- [x] **Commit 32**: Expand operational documentation
- [x] **Commit 33**: Create analysis library README

# After Runs Complete (Manual) 

- [ ] **Commit 1**: Test all notebooks
  - [ ] Run all notebooks end-to-end
  - [ ] Verify outputs and exports
  - [ ] Check narrative flow
  - **Commit**: `test: verify all analysis notebooks`

## Technical Guidelines

### Mathematical Approach
- Use AIC = 2k - 2ln(L) for model comparison (k = parameters, L = likelihood)
- Implement Hutch++ for O(1/ε) complexity trace estimation
- Enhance power law fitting with robust outlier detection
- Use Welch's method for spectral analysis with proper windowing

### Integration Philosophy  
- Maintain native deconCNN integration within project structure
- Ensure extracted functions match existing callback behavior exactly
- Leverage Lightning's metric collection and checkpoint systems
- Use Hydra configuration for all analysis script parameters

### Performance Requirements
- Process 216 experimental runs without memory issues
- Maintain <15% overhead for callback operations
- Enable parallel processing for large-scale analysis
- Implement graceful error handling for incomplete data

### Validation Approach
- Cross-validate all extracted functions against originals
- Use synthetic data for testing new mathematical functions
- Maintain comprehensive regression test suite
- Perform end-to-end testing with real experimental data

## Risk Mitigation Updates

### **Critical Implementation Risks**
1. **Callback Compatibility Risk**: Create comprehensive regression tests before extraction
   - Mitigation: Baseline test all callback outputs before making changes
   - Verification: Side-by-side comparison of outputs before/after refactoring

2. **Memory Issues During Hutch++ Upgrade**: Monitor memory usage during curvature analysis
   - Mitigation: Test on smaller models first, implement memory monitoring
   - Verification: Ensure <15% total runtime overhead as specified in plan

3. **Directory Access Limitations**: Verify session working directory includes deconCNN
   - Mitigation: Use absolute paths, verify repository access before starting
   - Verification: Test write/read permissions in target directories

4. **Dependency Conflicts**: Test new visualization packages with existing PyTorch stack
   - Mitigation: Install incrementally, test imports after each addition
   - Verification: Ensure existing functionality continues to work

### **Quality Assurance Protocol**
- **Before each commit**: Run `lint_fix` and resolve all issues
- **After each phase**: Run `pt` to ensure tests pass
- **Before extraction**: Create baseline comparison data
- **After extraction**: Verify identical behavior with integration tests

## Handoff Instructions
To continue this implementation:
1. Read the full plan and agent instructions
2. Check "Current Status" to see where we are  
3. Use TodoWrite to plan your work session
4. Find the first unchecked [ ] commit
5. Implement exactly what that commit describes
6. Run `lint_fix` before committing
7. Move to the next commit (often a test commit)
8. Update "Current Status" when stopping work

**Critical**: Follow the atomic commit pattern - each commit should do exactly ONE thing. Don't bundle changes together.

**Key Principles**:
1. Every implementation commit is followed by a test commit
2. Each commit does exactly ONE thing
3. Functions are extracted and tested before enhancement
4. Scripts are created only after their dependencies are tested
5. Integration testing validates the complete system

**Testing Strategy**:
- Unit tests for every new function
- Integration tests for extracted functions
- End-to-end tests for analysis scripts
- Notebook verification for all visualizations
- Final integration test of complete pipeline

## Retrospective Notes
[Track significant decisions and deviations here]

---
Remember: Build on existing foundations. Cross-validate extensively. Maintain compatibility.



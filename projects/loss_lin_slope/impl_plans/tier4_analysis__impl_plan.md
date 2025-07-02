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
- Total commits: 74 atomic commits across 6 phases

## Pre-implementation Checklist

**Environment Verification:**
- [ ] Confirm deconCNN repository accessibility from implementation session
- [ ] Verify Python 3.12+ environment active in deconCNN directory  
- [ ] Verify `uv` package manager availability
- [ ] Run `uv sync --group all` successfully
- [ ] Run `pt` and confirm all current tests pass
- [ ] Check available disk space for notebooks and analysis outputs

**Dependency Verification:**
- [ ] Add visualization dependencies to pyproject.toml
- [ ] Verify all new imports work correctly
- [ ] Test matplotlib backend for headless operation (if needed for cluster)

**Baseline Testing:**
- [ ] Run existing callbacks on sample data to establish baseline
- [ ] Document current NoiseMonitor and CurvatureMonitor output formats
- [ ] Verify existing checkpoints/logs are readable and contain expected metrics

**Integration Validation:**
- [ ] Check that `dr_exp` integration still works after changes
- [ ] Verify Hydra configuration system compatibility
- [ ] Test callback execution overhead is acceptable (<15% as per plan)

## Implementation Steps

**CRITICAL**: Follow Tier 3's atomic commit pattern:
- Each commit does ONE thing
- Implementation → Unit Test → Integration Test
- Run `lint_fix` before EVERY commit

### Phase 0: Environment Setup & Verification
- [ ] **Precommit**: Verify deconCNN environment
  - [ ] Ensure deconCNN directory is accessible
  - [ ] Verify Python 3.12+ with: `uvrp -c "import sys; print(sys.version)"`
  - [ ] Run `us` to sync dependencies
  - [ ] Run `pt` to verify existing tests pass
  - [ ] Run `lint` to check code quality baseline
  - [ ] Test import: `uvrp -c "import seaborn; print('seaborn OK')"`
  - [ ] Test imports: `uvrp -c "import plotly, jupyter; print('All imports OK')"`
  - [ ] Verify `src/deconcnn/analysis/__init__.py`, `scripts/analysis/` and `notebooks/` exist


### Phase 1: Enhanced Mathematical Functions (4 commits)
- [x] **Commit 1**: Add AIC computation to utils ✅ COMPLETED
- [x] **Commit 2**: Implement exponential fitting
- [x] **Commit 3**: Implement Welch's method for spectral analysis ✅ COMPLETED
- [x] **Commit 4**: Add noise floor detection ✅ COMPLETED

### Phase 2: Detection & Analysis Functions (6 commits)

- TODO: verify that the training code outputs CSV

- [ ] **Commit 5**: Implement knee detection
  - [ ] Create `src/deconcnn/analysis/detection.py`
  - [ ] Add imports and module structure
  - [ ] Add `knee_epoch()` function
  - [ ] Use AIC to compare exponential vs power law
  - [ ] Return burn-in epoch estimate
  - [ ] Create `tests/test_detection.py`
  - [ ] Test with synthetic curves
  - [ ] Verify burn-in identification
  - **Commit**: `feat and test: implement AIC-based knee detection`

- [ ] **Commit 6**: Implement windowed slope analysis
  - [ ] Add `alpha_window()` function
  - [ ] Support flexible window parameters
  - [ ] Return slope time series
  - [ ] Unit test window sliding behavior
  - [ ] Verify slope accuracy per window
  - **Commit**: `feat and test: implement windowed slope computation`

- [ ] **Commit 7**: Implement segmented power law fitting
  - [ ] Add `fit_two_power()` to `fitting.py`
  - [ ] Use dynamic programming for changepoint
  - [ ] Return both segments and changepoint
  - [ ] unit test with known two-regime data
  - [ ] Verify changepoint detection
  - **Commit**: `feat and test: implement segmented power law fitting`

- [ ] **Commit 8**: Implement Hutch++ algorithm
  - [ ] Create `src/deconcnn/analysis/curvature.py`
  - [ ] Add module structure
  - [ ] Add `hutchpp_trace()` function
  - [ ] Improve on Hutchinson estimator
  - [ ] Include variance reduction
  - [ ] Create `tests/test_curvature.py`
  - [ ] Compare with Hutchinson baseline
  - [ ] Verify variance reduction
  - **Commit**: `feat and test: implement Hutch++ trace estimator`

- [ ] **Commit 9**: Implement plateau detection
  - [ ] Add `lambda_plateau_epoch()` function
  - [ ] Use statistical change detection
  - [ ] Return plateau start epoch
  - [ ] unit Test with synthetic eigenvalue trajectories
  - [ ] Verify detection accuracy
  - **Commit**: `feat and test: implement eigenvalue plateau detection`
  
- [ ] **Commit 10**: Add statistical utilities
  - [ ] Add correlation functions to `utils.py`
  - [ ] Include Pearson, Spearman, partial
  - [ ] Add bootstrap confidence intervals
  - [ ] Unit Test correlation functions
  - [ ] Unit Test confidence intervals
  - [ ] Unit Test edge cases
  - **Commit**: `feat: add statistical correlation utilities`

### Phase 3: Primary Analysis Scripts (6 commits)

- [ ] **Commit 11**: Implement loss mosaic visualization
  - [ ] Create `src/deconcnn/analysis/visualization.py`
  - [ ] Add basic imports and module structure
  - [ ] Add `plot_loss_mosaic()` function
  - [ ] Support 6-panel layout (train/val × 3 scales)
  - [ ] Add automatic scaling and labeling
  - **Commit**: `feat: implement loss mosaic visualization`

- [ ] **Commit 12**: Create knee detection script
  - [ ] Create `scripts/analysis/A-knee.py`
  - [ ] Use click CLI interface
  - [ ] Process all runs and detect burn-in
  - **Commit**: `feat: create knee detection analysis script`

- [ ] **Commit 13**: Create slope analysis script
  - [ ] Create `scripts/analysis/A-slope.py`
  - [ ] Compute α_early (5-15) and α_full
  - [ ] Export results to CSV
  - **Commit**: `feat: create slope analysis script`

- [ ] **Commit 14**: Create model comparison script
  - [ ] Create `scripts/analysis/A-fits.py`
  - [ ] Compare exponential/power/two-power models
  - [ ] Generate AIC comparison table
  - **Commit**: `feat: create model comparison script`

- [ ] **Commit 15**: Test primary analysis scripts
  - [ ] Create test data subset
  - [ ] Run all three scripts
  - [ ] Verify output formats
  - **Commit**: `test: verify primary analysis scripts`

- [ ] **Commit 16**: Add progress tracking to scripts
  - [ ] Add rich progress bars
  - [ ] Add error recovery on failure
  - [ ] Add logging for debugging
  - **Commit**: `feat: add progress tracking to analysis scripts`

### Phase 4: Supporting Analysis Scripts (5 commits)

- [ ] **Commit 17**: Create architecture ablation script
  - [ ] Create `scripts/analysis/A-optim_arch.py`
  - [ ] Generate ablation comparison table
  - [ ] Include effect sizes
  - **Commit**: `feat: create architecture ablation analysis`

- [ ] **Commit 18**: Create hyperparameter heatmap script
  - [ ] Create `scripts/analysis/A-grid_heat.py`
  - [ ] Generate LR×WD performance heatmap
  - [ ] Add statistical significance
  - **Commit**: `feat: create hyperparameter heatmap analysis`

- [ ] **Commit 19**: Create noise correlation script
  - [ ] Create `scripts/analysis/A-noise_corr.py`
  - [ ] Compute noise proxy correlations
  - [ ] Add confidence intervals
  - **Commit**: `feat: create noise correlation analysis`

- [ ] **Commit 20**: Create curvature timing script
  - [ ] Create `scripts/analysis/A-curvature_timing.py`
  - [ ] Analyze λ₁ plateau vs α knee timing
  - [ ] Include statistical tests
  - **Commit**: `feat: create curvature timing analysis`

- [ ] **Commit 21**: Integration test supporting scripts
  - [ ] Run all supporting scripts
  - [ ] Verify outputs compatible
  - [ ] Check performance
  - **Commit**: `test: verify supporting analysis scripts`

### Phase 5: Slide Generation Notebooks (13 commits)

- [ ] **Commit 22**: Create H1 hypothesis notebook
  - [ ] Create `notebooks/h1_primary.ipynb`
  - [ ] Add to `.gitignore`: `notebooks/*.ipynb_checkpoints`, `*.png`, `*.pdf`, `*.svg`
  - [ ] Add R² histogram visualization
  - [ ] Add loss mosaic for best/worst fits
  - **Commit**: `feat: create H1 hypothesis notebook`

- [ ] **Commit 23**: Create H2 hypothesis notebook
  - [ ] Create `notebooks/h2_primary.ipynb`
  - [ ] Add α_early vs best_val_CE scatter
  - [ ] Include correlation statistics
  - **Commit**: `feat: create H2 hypothesis notebook`

- [ ] **Commit 24**: Add interactivity to primary notebooks
  - [ ] Add plotly interactive plots
  - [ ] Add parameter selection widgets
  - [ ] Test interactivity
  - **Commit**: `feat: add interactivity to hypothesis notebooks`

- [ ] **Commit 25**: Create model comparison notebook
  - [ ] Create `notebooks/backup_b_fam.ipynb`
  - [ ] Add AIC comparison visualizations
  - [ ] Include model selection analysis
  - **Commit**: `feat: create model comparison notebook`

- [ ] **Commit 26**: Create window robustness notebook
  - [ ] Create `notebooks/backup_c_window.ipynb`
  - [ ] Add robustness heatmaps
  - [ ] Include sensitivity analysis
  - **Commit**: `feat: create window robustness notebook`

- [ ] **Commit 27**: Create architecture ablation notebook
  - [ ] Create `notebooks/backup_h_arch.ipynb`
  - [ ] Add ablation result tables
  - [ ] Include statistical comparisons
  - **Commit**: `feat: create architecture ablation notebook`

- [ ] **Commit 28**: Create curvature exploration notebook
  - [ ] Create `notebooks/curvature_explore.ipynb`
  - [ ] Add eigenvalue trajectory plots
  - [ ] Include plateau analysis
  - **Commit**: `feat: create curvature exploration notebook`

- [ ] **Commit 29**: Add export functionality to notebooks
  - [ ] Add PNG/PDF export for all plots
  - [ ] Add LaTeX table export
  - [ ] Test export quality
  - **Commit**: `feat: add export functionality to notebooks`

- [ ] **Commit 30**: Create notebook utilities
  - [ ] Create `notebooks/utils.py`
  - [ ] Add common plotting functions
  - [ ] Add data loading helpers
  - **Commit**: `feat: create notebook utility functions`

- [ ] **Commit 31**: Test all notebooks
  - [ ] Run all notebooks end-to-end
  - [ ] Verify outputs and exports
  - [ ] Check narrative flow
  - **Commit**: `test: verify all analysis notebooks`

- [ ] **Commit 32**: Create integration test suite
  - [ ] Create `tests/test_integration.py`
  - [ ] Add end-to-end pipeline tests
  - [ ] Include synthetic data generation
  - **Commit**: `test: create integration test suite`

- [ ] **Commit 33**: Expand operational documentation
  - [ ] Expand `docs/operational_runbook_basic.md`
  - [ ] Add troubleshooting scenarios
  - [ ] Include performance tips
  - **Commit**: `docs: expand operational runbook`

- [ ] **Commit 34**: Create analysis library README
  - [ ] Create `src/deconcnn/analysis/README.md`
  - [ ] Add API documentation
  - [ ] Include usage examples
  - **Commit**: `docs: create analysis library documentation`

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



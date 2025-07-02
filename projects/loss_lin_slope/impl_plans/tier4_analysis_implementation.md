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
- Total commits: 82 atomic commits across 7 phases

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

### Phase 0: Environment Setup & Verification (3 commits)

- [ ] **Commit 0.1**: Verify deconCNN environment
  - [ ] Ensure deconCNN directory is accessible
  - [ ] Verify Python 3.12+ with: `uvrp -c "import sys; print(sys.version)"`
  - [ ] Run `us` to sync dependencies
  - [ ] Run `pt` to verify existing tests pass
  - [ ] Run `lint` to check code quality baseline
  - **Commit**: `chore: verify deconCNN environment`

- [ ] **Commit 0.2**: Add visualization dependencies
  - [ ] Add to pyproject.toml: `seaborn>=0.12.0`
  - [ ] Run `us` to install
  - [ ] Test import: `uvrp -c "import seaborn; print('seaborn OK')"`
  - **Commit**: `feat: add seaborn visualization dependency`

- [ ] **Commit 0.3**: Add remaining analysis dependencies
  - [ ] Add to pyproject.toml: `plotly>=5.17.0`, `jupyter>=1.0.0`
  - [ ] Run `us` to install
  - [ ] Test imports: `uvrp -c "import plotly, jupyter; print('All imports OK')"`
  - **Commit**: `feat: add plotly and jupyter dependencies`

### Phase 1: Library Structure & Function Extraction (12 commits)

- [ ] **Commit 1.1**: Create analysis library structure
  - [ ] Verify `src/deconcnn/analysis/` exists, add `__init__.py` if missing
  - [ ] Create `scripts/analysis/` directory
  - [ ] Verify/create `notebooks/` directory
  - [ ] Test: `uvrp -c "from deconcnn.analysis import *; print('Import OK')"`
  - **Commit**: `feat: complete analysis library structure`

- [ ] **Commit 1.2**: Update gitignore for notebooks
  - [ ] Add to `.gitignore`: `notebooks/*.ipynb_checkpoints`, `*.png`, `*.pdf`, `*.svg`
  - [ ] Test by creating dummy file and verifying it's ignored
  - **Commit**: `chore: update gitignore for notebook outputs`

- [ ] **Commit 1.3**: Create utils module and extract EWMA
  - [ ] Create `src/deconcnn/analysis/utils.py`
  - [ ] Extract `ewma()` from NoiseMonitor (lines 67-72)
  - [ ] Add proper docstring and type hints
  - **Commit**: `refactor: extract EWMA computation to utils`

- [ ] **Commit 1.4**: Add unit tests for EWMA
  - [ ] Create `tests/test_analysis_utils.py`
  - [ ] Test EWMA with known inputs/outputs
  - [ ] Test edge cases (empty list, single value)
  - **Commit**: `test: add unit tests for EWMA function`

- [ ] **Commit 1.5**: Create fitting module and extract power law
  - [ ] Create `src/deconcnn/analysis/fitting.py`
  - [ ] Extract power law fitting from NoiseMonitor (lines 102-107)
  - [ ] Add docstring explaining log-log regression approach
  - **Commit**: `refactor: extract power law fitting to fitting module`

- [ ] **Commit 1.6**: Add unit tests for power law fitting
  - [ ] Create `tests/test_fitting.py`
  - [ ] Test with synthetic power law data
  - [ ] Test R² calculation accuracy
  - **Commit**: `test: add unit tests for power law fitting`

- [ ] **Commit 1.7**: Create spectral module and extract PSD
  - [ ] Create `src/deconcnn/analysis/spectral.py`
  - [ ] Extract PSD analysis from NoiseMonitor (lines 114-131)
  - [ ] Document FFT approach and frequency bands
  - **Commit**: `refactor: extract PSD analysis to spectral module`

- [ ] **Commit 1.8**: Add unit tests for PSD analysis
  - [ ] Create `tests/test_spectral.py`
  - [ ] Test with synthetic signals
  - [ ] Verify frequency band extraction
  - **Commit**: `test: add unit tests for PSD analysis`

- [ ] **Commit 1.9**: Extract slope calculation to utils
  - [ ] Extract slope logic from LossSlopeLogger to `utils.py`
  - [ ] Add `calculate_slope()` function with window parameters
  - [ ] Include least squares implementation
  - **Commit**: `refactor: extract slope calculation to utils`

- [ ] **Commit 1.10**: Add unit tests for slope calculation
  - [ ] Add tests to `test_analysis_utils.py`
  - [ ] Test against numpy.polyfit reference
  - [ ] Test windowing behavior
  - **Commit**: `test: add unit tests for slope calculation`

- [ ] **Commit 1.11**: Update callbacks to use extracted functions
  - [ ] Update NoiseMonitor imports
  - [ ] Update LossSlopeLogger imports
  - [ ] Ensure backward compatibility
  - **Commit**: `refactor: update callbacks to use analysis library`

- [ ] **Commit 1.12**: Integration test extracted functions
  - [ ] Run training with updated callbacks
  - [ ] Compare outputs before/after extraction
  - [ ] Verify identical behavior
  - **Commit**: `test: verify callback compatibility after extraction`

### Phase 2: Enhanced Mathematical Functions (10 commits)

- [ ] **Commit 2.1**: Add AIC computation to utils
  - [ ] Add `compute_AIC()` function to `utils.py`
  - [ ] Implement: AIC = 2k - 2ln(L)
  - [ ] Add docstring explaining parameters
  - **Commit**: `feat: add AIC computation to utils`

- [ ] **Commit 2.2**: Add unit tests for AIC
  - [ ] Test AIC calculation with known values
  - [ ] Test model comparison scenarios
  - **Commit**: `test: add unit tests for AIC computation`

- [ ] **Commit 2.3**: Enhance power law fitting with outlier detection
  - [ ] Add outlier detection to `fitting.py`
  - [ ] Use IQR or MAD method for robustness
  - [ ] Return outlier mask with fit results
  - **Commit**: `feat: add outlier detection to power law fitting`

- [ ] **Commit 2.4**: Test enhanced power law fitting
  - [ ] Test outlier detection accuracy
  - [ ] Test fit improvement with outliers removed
  - **Commit**: `test: verify outlier detection in fitting`

- [ ] **Commit 2.5**: Implement exponential fitting
  - [ ] Add `fit_exponential()` to `fitting.py`
  - [ ] Implement log-linear regression
  - [ ] Return parameters and R²
  - **Commit**: `feat: implement exponential fitting function`

- [ ] **Commit 2.6**: Test exponential fitting
  - [ ] Test with synthetic exponential data
  - [ ] Compare with scipy.optimize results
  - **Commit**: `test: add unit tests for exponential fitting`

- [ ] **Commit 2.7**: Implement Welch's method for spectral analysis
  - [ ] Add `welch_psd()` to `spectral.py`
  - [ ] Implement windowing and overlap
  - [ ] Add frequency resolution control
  - **Commit**: `feat: implement Welch's method for PSD`

- [ ] **Commit 2.8**: Test Welch's method
  - [ ] Test against scipy.signal.welch
  - [ ] Verify frequency resolution
  - **Commit**: `test: verify Welch's method implementation`

- [ ] **Commit 2.9**: Add noise floor detection
  - [ ] Add `detect_noise_floor()` to `spectral.py`
  - [ ] Implement statistical noise floor estimation
  - **Commit**: `feat: add noise floor detection`

- [ ] **Commit 2.10**: Test noise floor detection
  - [ ] Test with synthetic signal + noise
  - [ ] Verify detection accuracy
  - **Commit**: `test: verify noise floor detection`

### Phase 3: Detection & Analysis Functions (16 commits)

- [ ] **Commit 3.1**: Create data utils module
  - [ ] Create `src/deconcnn/analysis/data_utils.py`
  - [ ] Add module docstring and imports
  - **Commit**: `feat: create data utils module`

- [ ] **Commit 3.2**: Implement JSONL to CSV conversion
  - [ ] Add `convert_metrics_jsonl_to_csv()` function
  - [ ] Handle dr_exp metrics format
  - [ ] Include data validation
  - **Commit**: `feat: implement JSONL to CSV conversion`

- [ ] **Commit 3.3**: Test JSONL conversion
  - [ ] Create `tests/test_data_utils.py`
  - [ ] Test with sample metrics.jsonl
  - [ ] Verify CSV output format
  - **Commit**: `test: add tests for JSONL conversion`

- [ ] **Commit 3.4**: Add batch processing for conversions
  - [ ] Add `batch_convert_experiments()` function
  - [ ] Support directory traversal
  - [ ] Add progress tracking
  - **Commit**: `feat: add batch processing for data conversion`

- [ ] **Commit 3.5**: Create detection module
  - [ ] Create `src/deconcnn/analysis/detection.py`
  - [ ] Add imports and module structure
  - **Commit**: `feat: create detection analysis module`

- [ ] **Commit 3.6**: Implement knee detection
  - [ ] Add `knee_epoch()` function
  - [ ] Use AIC to compare exponential vs power law
  - [ ] Return burn-in epoch estimate
  - **Commit**: `feat: implement AIC-based knee detection`

- [ ] **Commit 3.7**: Test knee detection
  - [ ] Create `tests/test_detection.py`
  - [ ] Test with synthetic curves
  - [ ] Verify burn-in identification
  - **Commit**: `test: add tests for knee detection`

- [ ] **Commit 3.8**: Implement windowed slope analysis
  - [ ] Add `alpha_window()` function
  - [ ] Support flexible window parameters
  - [ ] Return slope time series
  - **Commit**: `feat: implement windowed slope computation`

- [ ] **Commit 3.9**: Test windowed slope
  - [ ] Test window sliding behavior
  - [ ] Verify slope accuracy per window
  - **Commit**: `test: verify windowed slope analysis`

- [ ] **Commit 3.10**: Implement segmented power law fitting
  - [ ] Add `fit_two_power()` to `fitting.py`
  - [ ] Use dynamic programming for changepoint
  - [ ] Return both segments and changepoint
  - **Commit**: `feat: implement segmented power law fitting`

- [ ] **Commit 3.11**: Test segmented fitting
  - [ ] Test with known two-regime data
  - [ ] Verify changepoint detection
  - **Commit**: `test: verify segmented power law fitting`

- [ ] **Commit 3.12**: Create curvature analysis module
  - [ ] Create `src/deconcnn/analysis/curvature.py`
  - [ ] Add module structure
  - **Commit**: `feat: create curvature analysis module`

- [ ] **Commit 3.13**: Implement Hutch++ algorithm
  - [ ] Add `hutchpp_trace()` function
  - [ ] Improve on Hutchinson estimator
  - [ ] Include variance reduction
  - **Commit**: `feat: implement Hutch++ trace estimator`

- [ ] **Commit 3.14**: Test Hutch++ implementation
  - [ ] Create `tests/test_curvature.py`
  - [ ] Compare with Hutchinson baseline
  - [ ] Verify variance reduction
  - **Commit**: `test: verify Hutch++ implementation`

- [ ] **Commit 3.15**: Implement plateau detection
  - [ ] Add `lambda_plateau_epoch()` function
  - [ ] Use statistical change detection
  - [ ] Return plateau start epoch
  - **Commit**: `feat: implement eigenvalue plateau detection`

- [ ] **Commit 3.16**: Test plateau detection
  - [ ] Test with synthetic eigenvalue trajectories
  - [ ] Verify detection accuracy
  - **Commit**: `test: verify plateau detection`

### Phase 4: Primary Analysis Scripts (8 commits)

- [ ] **Commit 4.1**: Create visualization module
  - [ ] Create `src/deconcnn/analysis/visualization.py`
  - [ ] Add basic imports and module structure
  - **Commit**: `feat: create visualization module`

- [ ] **Commit 4.2**: Implement loss mosaic visualization
  - [ ] Add `plot_loss_mosaic()` function
  - [ ] Support 6-panel layout (train/val × 3 scales)
  - [ ] Add automatic scaling and labeling
  - **Commit**: `feat: implement loss mosaic visualization`

- [ ] **Commit 4.3**: Test loss mosaic
  - [ ] Create `tests/test_visualization.py`
  - [ ] Test layout generation
  - [ ] Test export formats (PNG/PDF/SVG)
  - **Commit**: `test: add tests for loss mosaic`

- [ ] **Commit 4.4**: Create knee detection script
  - [ ] Create `scripts/analysis/A-knee.py`
  - [ ] Use click CLI interface
  - [ ] Process all runs and detect burn-in
  - **Commit**: `feat: create knee detection analysis script`

- [ ] **Commit 4.5**: Create slope analysis script
  - [ ] Create `scripts/analysis/A-slope.py`
  - [ ] Compute α_early (5-15) and α_full
  - [ ] Export results to CSV
  - **Commit**: `feat: create slope analysis script`

- [ ] **Commit 4.6**: Create model comparison script
  - [ ] Create `scripts/analysis/A-fits.py`
  - [ ] Compare exponential/power/two-power models
  - [ ] Generate AIC comparison table
  - **Commit**: `feat: create model comparison script`

- [ ] **Commit 4.7**: Test primary analysis scripts
  - [ ] Create test data subset
  - [ ] Run all three scripts
  - [ ] Verify output formats
  - **Commit**: `test: verify primary analysis scripts`

- [ ] **Commit 4.8**: Add progress tracking to scripts
  - [ ] Add rich progress bars
  - [ ] Add error recovery on failure
  - [ ] Add logging for debugging
  - **Commit**: `feat: add progress tracking to analysis scripts`

### Phase 5: Supporting Analysis Scripts (12 commits)

- [ ] **Commit 5.1**: Create metric validator module
  - [ ] Create `src/deconcnn/analysis/metric_validator.py`
  - [ ] Add basic structure and imports
  - **Commit**: `feat: create metric validator module`

- [ ] **Commit 5.2**: Implement metric validation
  - [ ] Add outlier detection methods
  - [ ] Add consistency checks
  - [ ] Add cross-experiment validation
  - **Commit**: `feat: implement metric validation functions`

- [ ] **Commit 5.3**: Test metric validator
  - [ ] Create `tests/test_metric_validator.py`
  - [ ] Test outlier detection
  - [ ] Test validation rules
  - **Commit**: `test: add tests for metric validator`

- [ ] **Commit 5.4**: Create window robustness script
  - [ ] Create `scripts/analysis/A-window_scan.py`
  - [ ] Scan different window sizes for slope
  - [ ] Generate robustness heatmap
  - **Commit**: `feat: create window robustness analysis`

- [ ] **Commit 5.5**: Create unit invariance script
  - [ ] Create `scripts/analysis/A-unit_inv.py`
  - [ ] Verify nats/bits conversion invariance
  - [ ] Check numerical stability
  - **Commit**: `feat: create unit invariance verification`

- [ ] **Commit 5.6**: Create architecture ablation script
  - [ ] Create `scripts/analysis/A-optim_arch.py`
  - [ ] Generate ablation comparison table
  - [ ] Include effect sizes
  - **Commit**: `feat: create architecture ablation analysis`

- [ ] **Commit 5.7**: Create hyperparameter heatmap script
  - [ ] Create `scripts/analysis/A-grid_heat.py`
  - [ ] Generate LR×WD performance heatmap
  - [ ] Add statistical significance
  - **Commit**: `feat: create hyperparameter heatmap analysis`

- [ ] **Commit 5.8**: Create noise correlation script
  - [ ] Create `scripts/analysis/A-noise_corr.py`
  - [ ] Compute noise proxy correlations
  - [ ] Add confidence intervals
  - **Commit**: `feat: create noise correlation analysis`

- [ ] **Commit 5.9**: Create curvature timing script
  - [ ] Create `scripts/analysis/A-curvature_timing.py`
  - [ ] Analyze λ₁ plateau vs α knee timing
  - [ ] Include statistical tests
  - **Commit**: `feat: create curvature timing analysis`

- [ ] **Commit 5.10**: Add statistical utilities
  - [ ] Add correlation functions to `utils.py`
  - [ ] Include Pearson, Spearman, partial
  - [ ] Add bootstrap confidence intervals
  - **Commit**: `feat: add statistical correlation utilities`

- [ ] **Commit 5.11**: Test statistical utilities
  - [ ] Test correlation functions
  - [ ] Test confidence intervals
  - [ ] Test edge cases
  - **Commit**: `test: verify statistical utilities`

- [ ] **Commit 5.12**: Integration test supporting scripts
  - [ ] Run all supporting scripts
  - [ ] Verify outputs compatible
  - [ ] Check performance
  - **Commit**: `test: verify supporting analysis scripts`

### Phase 6: Slide Generation Notebooks (10 commits)

- [ ] **Commit 6.1**: Create H1 hypothesis notebook
  - [ ] Create `notebooks/h1_primary.ipynb`
  - [ ] Add R² histogram visualization
  - [ ] Add loss mosaic for best/worst fits
  - **Commit**: `feat: create H1 hypothesis notebook`

- [ ] **Commit 6.2**: Create H2 hypothesis notebook
  - [ ] Create `notebooks/h2_primary.ipynb`
  - [ ] Add α_early vs best_val_CE scatter
  - [ ] Include correlation statistics
  - **Commit**: `feat: create H2 hypothesis notebook`

- [ ] **Commit 6.3**: Add interactivity to primary notebooks
  - [ ] Add plotly interactive plots
  - [ ] Add parameter selection widgets
  - [ ] Test interactivity
  - **Commit**: `feat: add interactivity to hypothesis notebooks`

- [ ] **Commit 6.4**: Create model comparison notebook
  - [ ] Create `notebooks/backup_b_fam.ipynb`
  - [ ] Add AIC comparison visualizations
  - [ ] Include model selection analysis
  - **Commit**: `feat: create model comparison notebook`

- [ ] **Commit 6.5**: Create window robustness notebook
  - [ ] Create `notebooks/backup_c_window.ipynb`
  - [ ] Add robustness heatmaps
  - [ ] Include sensitivity analysis
  - **Commit**: `feat: create window robustness notebook`

- [ ] **Commit 6.6**: Create architecture ablation notebook
  - [ ] Create `notebooks/backup_h_arch.ipynb`
  - [ ] Add ablation result tables
  - [ ] Include statistical comparisons
  - **Commit**: `feat: create architecture ablation notebook`

- [ ] **Commit 6.7**: Create curvature exploration notebook
  - [ ] Create `notebooks/curvature_explore.ipynb`
  - [ ] Add eigenvalue trajectory plots
  - [ ] Include plateau analysis
  - **Commit**: `feat: create curvature exploration notebook`

- [ ] **Commit 6.8**: Add export functionality to notebooks
  - [ ] Add PNG/PDF export for all plots
  - [ ] Add LaTeX table export
  - [ ] Test export quality
  - **Commit**: `feat: add export functionality to notebooks`

- [ ] **Commit 6.9**: Create notebook utilities
  - [ ] Create `notebooks/utils.py`
  - [ ] Add common plotting functions
  - [ ] Add data loading helpers
  - **Commit**: `feat: create notebook utility functions`

- [ ] **Commit 6.10**: Test all notebooks
  - [ ] Run all notebooks end-to-end
  - [ ] Verify outputs and exports
  - [ ] Check narrative flow
  - **Commit**: `test: verify all analysis notebooks`

### Phase 7: Advanced SLURM Integration & Documentation (11 commits)

- [ ] **Commit 7.1**: Create advanced job submission script
  - [ ] Create `scripts/submit_loss_lin_slope.py` with priority optimization
  - [ ] Implement parameter grid generation with intelligent prioritization
  - [ ] Add dry-run and force-resubmit options
  - [ ] Include test-one mode for validation
  - [ ] Use dr_exp JobSubmitter API for batch submission
  - **Commit**: `feat: add loss_lin_slope experiment submission script`

- [ ] **Commit 7.2**: Create SLURM worker launcher with embedded parameters
  - [ ] Create `scripts/launch_workers_embedded.sh`
  - [ ] Use embedded parameter method to avoid environment variable issues
  - [ ] Configure CUDA MPS for efficient GPU sharing
  - [ ] Set up singularity container with dr_exp overlay
  - [ ] Include automatic cleanup on exit
  - **Commit**: `feat: add SLURM launcher for loss_lin_slope workers`

- [ ] **Commit 7.3**: Create real-time monitoring dashboard
  - [ ] Replace basic `scripts/monitor_experiment.py` with advanced version
  - [ ] Add variant-specific tracking and progress visualization
  - [ ] Implement completion percentage and ETA calculation
  - [ ] Add failure warnings and alerts
  - [ ] Use dr_exp JSON output for detailed status
  - **Commit**: `feat: add real-time experiment monitoring`

- [ ] **Commit 7.4**: Create automatic failure recovery script
  - [ ] Replace basic `scripts/recover_failed.py` with intelligent version
  - [ ] Add failure pattern analysis (OOM, gradient explosion, etc.)
  - [ ] Implement adaptive parameter adjustment for retries
  - [ ] Include priority boost for failed jobs
  - [ ] Add interactive resubmission interface
  - **Commit**: `feat: add automatic failure detection and recovery`

- [ ] **Commit 7.5**: Create efficient result collection script
  - [ ] Create `scripts/collect_and_archive.sh`
  - [ ] Implement selective collection of logs and checkpoints
  - [ ] Add checkpoint compression with gzip
  - [ ] Create consolidated metrics parquet file
  - [ ] Generate timestamped archives
  - **Commit**: `feat: add efficient result collection and archival`

- [ ] **Commit 7.6**: Create job monitoring script
  - [ ] Create `scripts/monitor_jobs.py` for SLURM job analytics
  - [ ] Add GPU utilization tracking
  - [ ] Include worker health monitoring
  - [ ] Integrate with dr_exp status
  - **Commit**: `feat: create job monitoring script`

- [ ] **Commit 7.7**: Create integration test suite
  - [ ] Create `tests/test_integration.py`
  - [ ] Add end-to-end pipeline tests
  - [ ] Include synthetic data generation
  - **Commit**: `test: create integration test suite`

- [ ] **Commit 7.8**: Expand operational documentation
  - [ ] Expand `docs/operational_runbook_basic.md`
  - [ ] Add troubleshooting scenarios
  - [ ] Include performance tips
  - **Commit**: `docs: expand operational runbook`

- [ ] **Commit 7.9**: Create analysis library README
  - [ ] Create `src/deconcnn/analysis/README.md`
  - [ ] Add API documentation
  - [ ] Include usage examples
  - **Commit**: `docs: create analysis library documentation`

- [ ] **Commit 7.10**: Enhance validation notebook
  - [ ] Expand `notebooks/validate_metrics_basic.ipynb`
  - [ ] Add advanced validation checks
  - [ ] Include data quality reports
  - **Commit**: `feat: enhance metric validation notebook`

- [ ] **Commit 7.11**: Final integration verification
  - [ ] Run full analysis pipeline
  - [ ] Verify all components work together
  - [ ] Document any issues found
  - **Commit**: `test: final integration verification`

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

## Retrospective Notes
[Track significant decisions and deviations here]

---
Remember: Build on existing foundations. Cross-validate extensively. Maintain compatibility.

## Implementation Summary

**Total Commits**: 82 atomic commits organized in 7 phases

**Phase Breakdown**:
- Phase 0: Environment Setup (3 commits)
- Phase 1: Library Structure & Function Extraction (12 commits)
- Phase 2: Enhanced Mathematical Functions (10 commits)
- Phase 3: Detection & Analysis Functions (16 commits)
- Phase 4: Primary Analysis Scripts (8 commits)
- Phase 5: Supporting Analysis Scripts (12 commits)
- Phase 6: Slide Generation Notebooks (10 commits)
- Phase 7: Advanced SLURM Integration & Documentation (11 commits)

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

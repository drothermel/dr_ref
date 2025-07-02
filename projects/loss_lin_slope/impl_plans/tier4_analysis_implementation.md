# Implementation Plan: Tier 4 Analysis Suite (deconCNN)

## 🔄 DEFERRED ELEMENTS FROM TIER 3 OPTIMIZATION (Added 2025-07-01)

**NOTE**: The following elements were deferred from Tier 3 to focus on the critical path. They should be implemented in Tier 4 for complete functionality.

### Analysis Library Implementation (Deferred from Tier 3)

**Location**: `src/deconcnn/analysis/` (directory created in Tier 3)

1. **Knee Detection** (`knee_detector.py`)
   - AIC-based knee detection for burn-in analysis
   - Required for identifying training phase transitions
   - Should include docstrings and type hints
   - Test with synthetic data for validation

2. **Power Law Fitting** (`power_law_fitter.py`)
   - Extract and consolidate existing implementation from `src/deconcnn/callbacks/noise_monitor.py:102-107`
   - Enhanced functionality for robust fitting
   - Handle numerical stability issues
   - Update NoiseMonitor to use shared utility
   - **AVOIDS DUPLICATION**: Existing implementation needs consolidation

3. **Enhanced Metric Validation** (`metric_validator.py` - extends basic version from Tier 3)
   - **Advanced data quality validation** (extends real-time monitoring from Tier 3 Commit 9)
   - Statistical validation: outlier detection, consistency checks, temporal validation
   - Cross-experiment validation: variant comparison, parameter sensitivity, reproducibility
   - Comprehensive completeness checking beyond basic range validation
   - **NOTE**: Basic NaN/inf detection and range validation covered in Tier 3 Commit 9

### Advanced Monitoring (Deferred from Tier 3)

1. **Enhanced Failure Recovery** (extends basic version from Tier 3)
   - Expand `scripts/recover_failed.py` with advanced retry strategies
   - Add machine learning-based failure prediction
   - Advanced parameter adjustment algorithms
   - **NOTE**: Basic failure recovery covered in Tier 3 Commit 9

2. **Real-time Monitoring Dashboard** (`scripts/monitor_experiment.py` enhancement)
   - Advanced progress tracking by variant and parameter
   - Failure warnings and completion rate analytics
   - Integration with existing basic monitoring from Tier 3

3. **Advanced Job Monitoring** (`scripts/monitor_jobs.py` enhancement)
   - Enhanced job status tracking and completion analytics
   - Failure detection and automated reporting
   - Integration with dr_exp job management

### Comprehensive Documentation (Deferred from Tier 3)

1. **Enhanced Metric Validation** (extends basic version from Tier 3)
   - Expand `notebooks/validate_metrics_basic.ipynb` into comprehensive analysis
   - Add advanced metric quality checks and statistical validation
   - Deep-dive analysis of logging patterns and data quality
   - **NOTE**: Basic validation covered in Tier 3 Commit 7

2. **Enhanced Operational Documentation** (extends basic version from Tier 3)
   - Expand `docs/operational_runbook_basic.md` into comprehensive guide
   - Add advanced troubleshooting scenarios and solutions
   - Detailed performance optimization procedures
   - **NOTE**: Basic operational readiness covered in Tier 3 Commit 9

3. **Advanced Examples and Tutorials**
   - Integration examples for analysis utilities
   - Performance optimization guides
   - Research insights and optimizations

---

## ⚠️ ADDITIONAL CONSIDERATIONS FROM TIER 3 REVIEW (Added 2025-01-01)

**NOTE**: The following items may become relevant during implementation. They were identified during Tier 3 review but are not critical for the core logging infrastructure.

### Data Format Considerations

1. **JSON to CSV Conversion**
   - Tier 3 will log to dr_exp's metrics.jsonl format
   - Tier 4 needs to implement conversion utilities for pandas analysis
   - Example utility needed:
     ```python
     def convert_metrics_jsonl_to_csv(experiment_dir):
         """Convert dr_exp metrics.jsonl to CSV for analysis."""
     ```

2. **Analysis Functions Status Update**
   - ✅ **Deferred to Tier 4**: `knee_detector.py`, `power_law_fitter.py`, `metric_validator.py`
   - ✅ **Directory Created**: `src/deconcnn/analysis/` in Tier 3 Commit 1
   - ⚠️ **Power Law Consolidation**: Extract from `noise_monitor.py:102-107` to avoid duplication
   - 📋 **Implementation Priority**: Phase 1 of Tier 4 implementation

3. **Library Strategy Updates**
   - BackPACK is being used for trace estimation (more efficient)
   - PyHessian reserved for spectral analysis only
   - No `hutch_nn` library exists - use PyHessian or BackPACK features

4. **Dependencies Already Added**
   - All visualization libraries (pandas, matplotlib, etc.) already in deconCNN
   - BackPACK already added for trace estimation
   - No need to add these again

5. **Potential Resource Optimizations**
   - Workers per GPU will be determined by Tier 3 testing
   - May vary by experiment variant (BN-off may need fewer workers)
   - Check Tier 3 results before assuming 3 workers per GPU

---

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

## Current Status
- Last updated: 2025-07-01 comprehensive plan review completed
- Last completed step: Plan review and refinement with environment verification
- Active agent: ready for Phase 0 implementation
- Blocked by: none
- Review findings: Added Phase 0 verification, atomic commit messages, risk mitigation

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

### Phase 0: Environment Setup & Verification (Day 0)

- [ ] **Step 0.1**: Verify deconCNN environment access and setup
  - [ ] Ensure deconCNN directory is accessible for implementation
  - [ ] Verify `uv` and Python 3.12+ environment in deconCNN
  - [ ] Run `us` (uv sync --group all) to ensure current dependencies work
  - [ ] Run `pt` to verify existing tests pass
  - [ ] Run `lint` to check current code quality
  - **Testing**: All existing functionality works before any changes
  - **Commit**: None needed - verification only

### Phase 1: Dependencies & Analysis Library Setup (Days 1-2)

- [ ] **Step 1.1**: Add analysis dependencies with specific versions
  - [ ] Add to `pyproject.toml` dependencies: `pandas>=2.0.0`, `seaborn>=0.12.0`, `plotly>=5.17.0`, `jupyter>=1.0.0`, `matplotlib>=3.7.0`
  - [ ] Note: `click>=8.2.1` already added, `scipy>=1.11.0` already available
  - [ ] Run `us` to install new dependencies
  - [ ] Create simple import test: `uvrp -c "import pandas, seaborn, plotly, matplotlib; print('All imports successful')"`
  - [ ] Verify all imports work correctly
  - **Testing**: Test import of all new packages
  - **Commit**: Use exact message: `feat: add visualization and analysis dependencies`

- [ ] **Step 1.2**: Create analysis library structure
  - [ ] Create `src/deconcnn/analysis/` directory with `__init__.py`
  - [ ] Create `scripts/analysis/` directory for analysis scripts
  - [ ] Create `notebooks/` directory at project root for slide generation
  - [ ] Update `.gitignore` to exclude: `notebooks/*.ipynb_checkpoints`, `*.png`, `*.pdf`, `*.svg` in notebooks/
  - **Testing**: Verify directory structure and `from deconcnn.analysis import` works
  - **Commit**: Use exact message: `feat: create analysis library structure`

- [ ] **Step 1.3**: Extract and modularize existing analysis functions with careful compatibility
  - [ ] Create analysis module files: `fitting.py`, `spectral.py`, `utils.py` in `src/deconcnn/analysis/`
  - [ ] Extract `ewma()` computation from NoiseMonitor (lines 67-72) to `utils.py`
  - [ ] Extract power law fitting from NoiseMonitor (lines 102-107) to `fitting.py`
  - [ ] Extract PSD analysis from NoiseMonitor (lines 114-131) to `spectral.py`
  - [ ] Add `compute_AIC()` function to `utils.py`
  - [ ] Update NoiseMonitor to import and use extracted functions (maintain identical behavior)
  - [ ] Create integration test comparing old vs new function outputs
  - **Testing**: Run existing callbacks and verify identical output before/after extraction
  - **Commit**: Use exact message: `refactor: extract analysis functions to reusable library`

### Phase 2: Enhanced Mathematical Functions (Days 2-3)

- [ ] **Step 2.1**: Enhance fitting functions in analysis library
  - [ ] Improve extracted power law fitting in `fitting.py` with better error handling
  - [ ] Add `fit_exponential()` - log-linear space fitting
  - [ ] Enhance R² computation and residual analysis
  - [ ] Add robust outlier detection to existing scipy.optimize usage
  - **Testing**: Cross-validate with original NoiseMonitor results
  - **Commit**: Use exact message: `feat: enhance power law and exponential fitting functions`

- [ ] **Step 2.2**: Implement advanced segmented fitting
  - [ ] Add `fit_two_power()` to `fitting.py` - segmented power-law with dynamic programming
  - [ ] Add changepoint detection using AIC comparison
  - [ ] Include confidence intervals for parameters using bootstrap methods
  - [ ] Handle edge cases (insufficient data, poor fits)
  - **Testing**: Test with synthetic two-regime power laws, verify changepoint detection
  - **Commit**: Use exact message: `feat: implement segmented power law fitting`

- [ ] **Step 2.3**: Enhance spectral analysis functions
  - [ ] Enhance `spectral.py` with Welch's method for robust spectral estimation
  - [ ] Add frequency band filtering based on existing mid-frequency analysis
  - [ ] Add noise floor detection capabilities
  - [ ] Maintain backward compatibility with NoiseMonitor
  - **Testing**: Cross-validate with NoiseMonitor results, test synthetic signals
  - **Commit**: Use exact message: `feat: enhance spectral analysis with Welch method`

### Phase 3: Detection & Analysis Functions (Days 3-4)

- [ ] **Step 3.1**: Implement burn-in detection
  - [ ] Create `src/deconcnn/analysis/detection.py` with `knee_epoch()` function
  - [ ] Implement AIC-based burn-in detection using power law vs exponential models
  - [ ] Build on Phase 2 fitting functions for model comparison
  - [ ] Add visualization helpers for detection results
  - **Testing**: Test with synthetic learning curves and real training logs
  - **Commit**: Use exact message: `feat: implement AIC-based burn-in detection`

- [ ] **Step 3.2**: Implement slope analysis
  - [ ] Add `alpha_window()` to `detection.py` - windowed slope computation
  - [ ] Build on existing power law slope computation patterns
  - [ ] Add confidence intervals for slope estimates using bootstrap
  - [ ] Handle edge cases at curve boundaries
  - **Testing**: Validate against existing NoiseMonitor slope computation
  - **Commit**: Use exact message: `feat: implement windowed slope computation`

- [ ] **Step 3.3**: Upgrade curvature analysis
  - [ ] Create `src/deconcnn/analysis/curvature.py` for advanced curvature analysis
  - [ ] Upgrade CurvatureMonitor's Hutchinson estimator to Hutch++ algorithm
  - [ ] Implement `lambda_plateau_epoch()` using existing lambda_max patterns
  - [ ] Add statistical tests for plateau detection
  - **Testing**: Compare Hutch++ vs Hutchinson performance on real models
  - **Commit**: Use exact message: `feat: upgrade to Hutch++ curvature estimation`

### Phase 4: Primary Analysis Scripts (Days 4-5)

- [ ] **Step 4.1**: Create core analysis scripts
  - [ ] Create `scripts/analysis/A-knee.py` - detect power-law onset across all runs
  - [ ] Create `scripts/analysis/A-slope.py` - compute α_early (5-15) and α_full (knee-50)
  - [ ] Create `scripts/analysis/A-fits.py` - compare exponential/power/two-power models
  - [ ] Use click for CLI interfaces and hydra integration for configs
  - [ ] Include progress tracking and error recovery
  - **Testing**: Run on existing deconCNN experimental data
  - **Commit**: Use exact message: `feat: implement primary analysis scripts`

- [ ] **Step 4.2**: Implement visualization core
  - [ ] Create `src/deconcnn/analysis/visualization.py` with `plot_loss_mosaic()`
  - [ ] Add customizable layout and styling using matplotlib/seaborn
  - [ ] Include automatic subplot scaling and labeling
  - [ ] Add export functionality for slides (PNG/PDF/SVG)
  - **Testing**: Generate test mosaics with real training data
  - **Commit**: Use exact message: `feat: implement loss mosaic visualization system`

### Phase 5: Supporting Analysis Scripts (Days 5-6)

- [ ] **Step 5.1**: Implement robustness analysis
  - [ ] Create `scripts/analysis/A-window_scan.py` - window robustness heatmap
  - [ ] Create `scripts/analysis/A-unit_inv.py` - verify nats/bits invariance
  - [ ] Add comprehensive parameter sensitivity analysis
  - [ ] Include statistical significance testing using existing patterns
  - **Testing**: Verify robustness metrics with real experimental data
  - **Commit**: Use exact message: `feat: implement robustness analysis scripts`

- [ ] **Step 5.2**: Implement ablation analysis
  - [ ] Create `scripts/analysis/A-optim_arch.py` - ablation comparison table
  - [ ] Create `scripts/analysis/A-grid_heat.py` - LR×WD performance heatmap
  - [ ] Add statistical comparison methods using scipy.stats
  - [ ] Include effect size computations (Cohen's d, r²)
  - **Testing**: Generate comparison tables with deconCNN experimental grid
  - **Commit**: Use exact message: `feat: implement ablation comparison scripts`

- [ ] **Step 5.3**: Implement correlation analysis
  - [ ] Create `scripts/analysis/A-noise_corr.py` - noise proxy correlations
  - [ ] Create `scripts/analysis/A-curvature_timing.py` - λ₁ plateau vs α knee timing
  - [ ] Add Pearson and Spearman correlation with confidence intervals
  - [ ] Include partial correlation analysis using pandas
  - **Testing**: Test correlation computations with real training metrics
  - **Commit**: Use exact message: `feat: implement correlation analysis scripts`

### Phase 6: Slide Generation Notebooks (Days 6-7)

- [ ] **Step 6.1**: Create primary hypothesis notebooks
  - [ ] Create `h1_primary.ipynb` - R² histogram + loss mosaic for H1
  - [ ] Create `h2_primary.ipynb` - α vs CE scatter plot for H2
  - [ ] Add interactive plotting with plotly/bokeh
  - [ ] Include statistical annotations
  - **Testing**: Generate primary slides, verify narrative flow
  - **Commit**: Use exact message: `feat: create primary hypothesis notebooks`

- [ ] **Step 6.2**: Create backup analysis notebooks
  - [ ] Create `backup_b_fam.ipynb` - AIC comparison bars
  - [ ] Create `backup_c_window.ipynb` - window robustness heatmap
  - [ ] Create `backup_h_arch.ipynb` - architecture ablation results
  - [ ] Add comprehensive error handling
  - **Testing**: Generate backup slides, verify completeness
  - **Commit**: Use exact message: `feat: create backup analysis notebooks`

- [ ] **Step 6.3**: Create exploration notebook
  - [ ] Create `curvature_explore.ipynb` - future track exploration
  - [ ] Add hypothesis generation tools
  - [ ] Include interactive parameter exploration
  - [ ] Add export functionality for follow-up studies
  - **Testing**: Test interactive features, verify exploration workflows
  - **Commit**: Use exact message: `feat: create curvature exploration notebook`

### Phase 7: Statistical Validation & Integration (Days 7-8)

- [ ] **Step 7.1**: Implement statistical testing suite
  - [ ] Add Pearson and Spearman correlation tests with p-values
  - [ ] Implement bootstrap confidence intervals
  - [ ] Add multiple comparison corrections (Bonferroni, FDR)
  - [ ] Include effect size measures (Cohen's d, r²)
  - **Testing**: Verify statistical test implementations against known results
  - **Commit**: Use exact message: `feat: implement statistical testing suite`

- [ ] **Step 7.2**: Create integration testing
  - [ ] Create end-to-end validation pipeline
  - [ ] Add synthetic data generation for testing
  - [ ] Implement regression tests for key analyses
  - [ ] Add performance benchmarking
  - **Testing**: Run full pipeline on synthetic data, verify performance
  - **Commit**: Use exact message: `feat: add integration tests and validation pipeline`

- [ ] **Step 7.3**: Add documentation and final polish
  - [ ] Add detailed README with usage examples
  - [ ] Create API documentation
  - [ ] Add troubleshooting guide
  - [ ] Include performance optimization tips
  - **Testing**: Review documentation completeness, test examples
  - **Commit**: Use exact message: `docs: add comprehensive documentation and examples`

## Key Implementation Details

### Mathematical Requirements
- **AIC Computation**: AIC = 2k - 2ln(L) where k = parameters, L = likelihood (already used in NoiseMonitor)
- **Hutch++ Upgrade**: Improve existing Hutchinson trace estimator to O(1/ε) complexity
- **Power Law Fitting**: Enhance existing log-log regression from NoiseMonitor with outlier detection
- **FFT Analysis**: Refactor existing Welch's method and mid-frequency band filtering from NoiseMonitor
- **Statistical Testing**: Add proper multiple comparison corrections and effect sizes

### Integration Requirements
- **Native deconCNN Integration**: Analysis suite lives within deconCNN project structure
- **Callback Compatibility**: Extracted functions must match existing callback behavior exactly
- **Direct Data Access**: Use Lightning's built-in metric collection and existing checkpoint data
- **Hydra Configuration**: Leverage existing Hydra config system for analysis script parameters

### Performance Targets
- **Analysis Efficiency**: Process 216 experimental runs without memory issues
- **Backward Compatibility**: Maintain existing callback performance in deconCNN
- **Computational Efficiency**: Parallel processing for large run sets
- **Error Handling**: Graceful degradation for missing data or corrupted logs

### Validation Strategy
- **Cross-Validation**: Compare extracted functions with original callback results
- **Synthetic Data**: Generate known ground truth for new fitting functions
- **Regression Tests**: Ensure compatibility with existing deconCNN training workflows
- **Integration Tests**: End-to-end analysis pipeline with real deconCNN experimental data

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
3. Find the first unchecked [ ] item
4. Begin implementation from that point
5. Update status when stopping work

## Retrospective Notes
[Track significant decisions and deviations here]

---
Remember: Build on existing foundations. Cross-validate extensively. Maintain compatibility.
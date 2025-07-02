# H1/H2 Hypothesis Testing - Implementation Breakdown

## Overview
This document breaks down the implementation tasks into four tiers for testing hypotheses H1 (power-law linearity) and H2 (early slope prediction) for the advisor meeting.

## Tier 1: Training Code Changes (deconCNN modifications)

### 1.1 Architecture Ablations
- [ ] Add width multiplier parameter (1×, 0.5×) to ResNet config

### 1.2 Optimizer Configuration
- [ ] Verify AdamW default betas match requirements (β₁=0.9, β₂=0.999)

## Tier 2: Logging Infrastructure & Validation

### 2.1 Basic Training Metrics
- [ ] Change logging to per-batch frequency for training loss
- [ ] Add bits conversion (loss_train_bits = loss_train_nats / ln(2))
- [ ] Add explicit lr logging per batch
- [ ] Add wd (weight decay) logging per batch
- [ ] Add step/timestamp tracking

### 2.2 Validation Metrics
- [ ] Change validation logging to ¼ epoch frequency
- [ ] Track and log best validation CE explicitly

### 2.3 Gradient & Weight Norms
- [ ] Implement `grad_norm` logging in `on_after_backward`
- [ ] Implement `weight_norm` logging
- [ ] Set up ¼ epoch logging frequency

### 2.4 Curvature Metrics
- [ ] Create new `CurvatureCallback` with Hutch++ trace estimation
- [ ] Add power iteration for `lambda_max` computation
- [ ] Configure to log every 500 optimizer steps
- [ ] Validate <10% runtime overhead

### 2.5 Noise & Residual Metrics
- [ ] Create new `NoiseMetricsCallback` with 512-sample buffer
- [ ] Implement `psd_tail` via FFT on residuals
- [ ] Implement `grad_var` with running statistics
- [ ] Implement `resid_power` in mid-frequency band
- [ ] Configure to log every 2 epochs

### 2.6 Checkpoint Management
- [ ] Configure to save only best checkpoint based on val_loss
- [ ] Add checkpoint size logging
- [ ] Create compression script for archival

### 2.7 Validation Scripts
- [ ] Create `verify_callbacks.ipynb` to test each metric
- [ ] Verify nats→bits conversion
- [ ] Validate ¼ epoch logging frequency
- [ ] Test curvature estimation convergence
- [ ] Test FFT buffer and residual computation

## Tier 3: Hydra Configuration & Job Setup

See /Users/daniellerothermel/drotherm/repos/dr_ref/projects/loss_lin_slope/top_level_planning/2025.07.03_Meeting_Prep_Plan.md to identify the set of runs that need to be executed.  This step will require planning both in terms of what runs to execute, how to locally on my mac test and verify all the necessary functionality in a way that I, the user, can see how it works and personally verify it is what I expect, and how to use the combination of /Users/daniellerothermel/drotherm/repos/deconCNN and /Users/daniellerothermel/drotherm/repos/dr_exp to execute the runs on the slurm cluster.

## Tier 4: Analysis Implementation

### 4.1 Core Analysis Library (`analysis_lib/`)
- [ ] `knee_epoch()` - AIC-based burn-in detection
- [ ] `fit_exponential()` - log-linear space fitting
- [ ] `fit_power()` - log-log space fitting
- [ ] `fit_two_power()` - segmented power-law fitting
- [ ] `compute_AIC()` - model comparison metric
- [ ] `alpha_window()` - windowed slope computation
- [ ] `ewma()` - smoothing for noisy signals
- [ ] `lambda_plateau_epoch()` - curvature stabilization detector
- [ ] `psd_tail()` - FFT-based noise metric
- [ ] `plot_loss_mosaic()` - 6-panel visualization

### 4.2 Primary Analysis Scripts
- [ ] `A-knee.py` - Detect power-law onset across runs
- [ ] `A-slope.py` - Compute α_early (5-15) and α_full (knee-50)
- [ ] `A-fits.py` - Compare exp/power/two-power models

### 4.3 Supporting Analysis Scripts
- [ ] `A-window_scan.py` - Window robustness heatmap
- [ ] `A-unit_inv.py` - Verify nats/bits invariance
- [ ] `A-optim_arch.py` - Ablation comparison table
- [ ] `A-grid_heat.py` - LR×WD performance heatmap
- [ ] `A-noise_corr.py` - Noise proxy correlations
- [ ] `A-curvature_timing.py` - λ₁ plateau vs α knee

### 4.4 Slide Generation Notebooks
- [ ] `h1_primary.ipynb` - R² histogram + loss mosaic
- [ ] `h2_primary.ipynb` - α vs CE scatter plot
- [ ] `backup_b_fam.ipynb` - AIC comparison bars
- [ ] `backup_c_window.ipynb` - Window robustness heatmap
- [ ] `backup_h_arch.ipynb` - Architecture ablation results
- [ ] `curvature_explore.ipynb` - Future track exploration

### 4.5 Statistical Validation
- [ ] Implement Pearson and Spearman correlation tests
- [ ] Add significance testing (p-values)
- [ ] Create correlation confidence intervals
- [ ] Handle NaN/divergent runs in statistics

## Implementation Order & Dependencies

### Phase 1: Core Infrastructure (Day 0)
1. Training code changes (Tier 1)
2. Basic logging (Tier 2.1-2.3)
3. Validation scripts (Tier 2.7)

### Phase 2: Configuration & Launch (Day 1)
1. Create Hydra configs (Tier 3)
2. Launch SLURM array
3. Monitor for crashes, especially BN-off runs
4. Quick validation of logged metrics

### Phase 3: Advanced Metrics (Day 1-2)
1. Curvature callbacks (Tier 2.4)
2. Noise metrics (Tier 2.5)
3. Verify <15% total overhead

### Phase 4: Analysis (Day 2-4)
1. Core analysis functions (Tier 4.1)
2. Run primary analyses as jobs complete
3. Generate backup analyses
4. Create slide visualizations

### Phase 5: Presentation Prep (Day 4-5)
1. Finalize slide deck
2. Rehearse narrative
3. Prepare for advisor questions

## Success Criteria

### Tier 1
- All 216 runs complete without crashes
- BN-off runs remain stable with gradient clipping
- Configs correctly specify all ablations

### Tier 2  
- All metrics logged at specified frequencies
- Total runtime overhead <15%
- Validation scripts confirm metric correctness

### Tier 3
- R² histogram shows >80% of runs with R²≥0.98
- α_early vs CE correlation ρ≥0.7
- All backup analyses support main narrative

## Risk Mitigation

1. **BN-off instability**: Pre-test gradient clipping values
2. **Adam sawtooth**: Add schedule-aware knee detection
3. **Weak correlations**: Have λ₁ plateau analysis ready as backup
4. **Compute delays**: Prioritize core grid (S-core) for primary slides
5. **Logging overhead**: Test curvature/noise callbacks on single GPU first

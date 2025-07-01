# Tier 3 Implementation Review Findings (Temporary)

## Review Date: 2025-01-01

This document captures findings from the comprehensive review of the Tier 3 Job Setup and Verification implementation plan against the actual codebase and ground truth requirements.

## 1. Critical Findings

### 1.1 Code Alignment Issues

**✅ What's Already in Place:**
- deconCNN has robust Hydra configuration infrastructure with proper defaults composition
- Existing callback system (gradient_monitor, curvature_monitor, noise_monitor) provides foundation
- dr_exp integration via `dr_exp_metrics` callback is functional and tested
- SLURM submission patterns well-documented in dr_exp with embedded parameter method

**❌ Critical Gaps:**
- No `/configs/experiment/` directory structure exists (must be created)
- Loss conversion to bits not implemented (only nats currently logged)
- Knee detection and power-law fitting utilities completely missing
- No batch-level CSV/metrics logging (current system only logs epoch-level to dr_exp)
- Parquet conversion utilities not found in either repository
- Path discrepancy: plan references `scripts/train_cnn.py` but actual is `train_cnn.py` at root

### 1.2 Resource Configuration Conflicts

**Workers per GPU:**
- Plan specifies: 2 workers per GPU
- dr_exp documentation recommends: 3 workers for RTX8000
- **Action needed**: Test resource-intensive configurations before finalizing

**Checkpoint Strategy:**
- Current plan: save_top_k=1 (only best model)
- Risk: May miss early good models before validation improves
- **Recommendation accepted**: Update to save_top_k=3

### 1.3 Logging Frequency Mismatches

Ground truth requirements vs current capabilities:
- **Batch-level logging needed for**: loss_train_nats, loss_train_bits, acc_train, lr, wd
- **Quarter-epoch logging needed for**: validation metrics, grad_norm
- **Current system**: Only logs at epoch boundaries to dr_exp

## 2. Pre-implementation Requirements

### 2.1 Core Analysis Functions (Missing)
Must implement before Phase 1:
```python
# Required in dr_ref/projects/loss_lin_slope/analysis_lib/
- knee_detector.py  # AIC-based knee detection
- power_law_fitter.py  # Single/two-power law fitting  
- metric_validator.py  # Completeness and range checking
- parquet_converter.py  # CSV to Parquet conversion
```

### 2.2 Logging Infrastructure (Partial)
Need to create:
- Batch-level logging callback for loss metrics
- Metric aggregation for quarter-epoch intervals
- CSV export functionality alongside dr_exp metrics

### 2.3 Testing Infrastructure (New Finding)
**Critical addition**: Need cluster testing phase for resource optimization
- Test with 1, 2, 3, 4 workers per GPU
- Monitor GPU memory usage and throughput
- Identify optimal configuration for each experiment variant
- Special attention to BN-off runs (may need more memory)

## 3. Implementation Order Adjustments

### Recommended Sequence:
1. **Pre-Phase 0**: Implement core analysis functions with unit tests
2. **Pre-Phase 0**: Create cluster resource testing script
3. **Phase 0.5**: Run resource optimization tests on cluster
4. **Phase 1**: Create Hydra configs (with findings from resource tests)
5. **Continue as planned...**

## 4. Specific Corrections Needed

### 4.1 Configuration Updates
```yaml
# loss_lin_slope_base.yaml corrections:
checkpoint:
  save_top_k: 3  # Updated from 1
  monitor: loss_val_nats
  mode: min
  save_last: true  # Add for safety

# Add logging configuration:
logging:
  log_every_n_steps: 1  # For batch-level metrics
  val_check_interval: 0.25  # Quarter-epoch validation
```

### 4.2 Worker Configuration
```bash
# Make configurable based on testing:
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}  # Default 3, but allow override
```

### 4.3 Analysis Scripts Location
- Current plan puts scripts in various locations
- Standardize under `dr_ref/projects/loss_lin_slope/scripts/`
- Create clear separation: `/submission/`, `/monitoring/`, `/analysis/`

## 5. Risk Mitigation Strategies

### 5.1 Resource Constraints
- **Risk**: OOM with multiple workers on resource-intensive configs
- **Mitigation**: 
  - Implement dynamic batch size adjustment
  - Create variant-specific worker configurations
  - Add memory monitoring to submission script

### 5.2 Logging Completeness
- **Risk**: Missing metrics for analysis
- **Mitigation**:
  - Create comprehensive validation script
  - Test all metrics locally before cluster submission
  - Implement backup CSV logging alongside dr_exp

### 5.3 Cluster Reliability
- **Risk**: Job failures due to cluster issues
- **Mitigation**:
  - Use embedded parameter method (already in plan)
  - Add automatic retry logic with exponential backoff
  - Implement checkpoint recovery

## 6. Positive Findings

### 6.1 Strong Foundation
- NoiseMonitor already implements power law fitting (can extract/reuse)
- dr_exp provides robust job management and monitoring
- Existing callbacks can be extended rather than replaced

### 6.2 Good Practices in Plan
- Embedded parameter method for SLURM
- Phased validation approach
- Comprehensive monitoring utilities

## 7. Next Steps

1. **Immediate**: Implement missing analysis functions
2. **Before Phase 1**: Run cluster resource optimization tests
3. **Update plan**: Incorporate all findings and corrections
4. **Verify against**: 
   - H1/H2 implementation breakdown
   - Tier 4 analysis requirements
   - Any other relevant documentation

## 8. Questions for Clarification

1. Should we implement a separate CSV logger or extend dr_exp metrics callback?
2. What's the preferred method for quarter-epoch validation scheduling?
3. Should analysis functions live in dr_ref or deconCNN?
4. Do we need to preserve raw checkpoint files or just metrics?

## 9. Testing Phase Design (New)

### Cluster Resource Optimization Test
```python
# scripts/test_resource_configs.py
test_configs = [
    {"variant": "S-core", "workers": [1, 2, 3, 4]},
    {"variant": "S-bn-off", "workers": [1, 2, 3]},  # May need fewer
    {"variant": "S-narrow", "workers": [2, 3, 4, 5]},  # Can handle more
]

for config in test_configs:
    # Submit 10-epoch test runs
    # Monitor: GPU utilization, memory usage, throughput
    # Record: time per epoch, OOM errors, optimal workers
```

This testing phase should take ~4 hours and will inform final submission parameters.

---

**Note**: This is a temporary document for review purposes. Once additional sources are verified, these findings will be incorporated into the official implementation plan update.
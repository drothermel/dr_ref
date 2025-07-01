# Gap Analysis: Tier 3 Plan vs Tier 4 Requirements

## Executive Summary

The Tier 4 analysis implementation requires specific logged metrics and analysis functions that are not fully addressed in the current Tier 3 plan or existing codebase. This document identifies critical gaps that must be resolved before Tier 4 can succeed.

## 1. Critical Logging Gaps

### 1.1 Required by Tier 4 but Missing from Tier 3

| Metric | Tier 4 Requirement | Current State | Gap |
|--------|-------------------|---------------|-----|
| **loss_train_bits** | Every batch | Not implemented | Need conversion from nats |
| **loss_train_nats** | Every batch | Only epoch-level | Need batch-level logging |
| **acc_train** | Every batch | Only epoch-level | Need batch-level logging |
| **lr, wd** | Every batch | Not explicitly logged | Need per-batch tracking |
| **grad_norm, weight_norm** | ¼ epoch | Not implemented | New callback needed |
| **lambda_max, hutch_trace** | Every 500 iters | Exists but needs Hutch++ upgrade | Enhancement needed |
| **psd_tail, grad_var, resid_power** | Every 2 epochs | Partially exists in NoiseMonitor | Need extraction and scheduling |
| **checkpoint_size_mb** | Every epoch | Not implemented | New logging needed |

### 1.2 Data Format Issues

**Tier 4 expects:**
- CSV files with all metrics for easy pandas loading
- Metrics at precise intervals (¼ epoch, every 500 steps, etc.)
- Both training and validation metrics in same file

**Current system provides:**
- JSON lines format from dr_exp (`metrics.jsonl`)
- Only epoch-level granularity
- Separate training/validation logging

## 2. Missing Analysis Functions

### 2.1 Core Mathematical Functions Not Found

| Function | Tier 4 Usage | Current State | Implementation Needed |
|----------|-------------|---------------|----------------------|
| **knee_epoch()** | Detect burn-in via AIC | Not exists | Full implementation |
| **fit_exponential()** | Model comparison | Not exists | scipy.optimize based |
| **fit_two_power()** | Segmented fitting | Not exists | Dynamic programming |
| **compute_AIC()** | Model selection | Not exists | Standard formula |
| **lambda_plateau_epoch()** | Curvature analysis | Not exists | EWMA + detection |

### 2.2 Existing Functions Need Enhancement

**NoiseMonitor has:**
- Basic power law fitting (scipy.stats.linregress)
- PSD computation
- EWMA calculation

**Tier 4 needs:**
- Robust power law fitting with outlier detection
- Welch's method for spectral estimation
- Bootstrap confidence intervals

## 3. Library Compatibility Issues

### 3.1 Python Libraries Mismatch

**Tier 4 assumes:**
- `statsmodels.api.OLS` for regression (not in dependencies)
- `hessian_eigenthings` or `pyhessian` for curvature (not available)
- `hutch_nn` for Hutchinson trace (not found)

**Current available:**
- Basic scipy.stats functions
- PyTorch built-in functions
- NumPy/Pandas basics

### 3.2 Missing Utilities

**Tier 4 scripts expect:**
- Click CLI framework (added but not used in Tier 3)
- Hydra integration for analysis scripts
- Parquet conversion utilities
- Progress tracking with rich/tqdm

## 4. Data Pipeline Gaps

### 4.1 Collection → Analysis Flow

**Tier 4 expects:**
1. Raw CSV logs from training
2. Scripts process CSVs directly
3. Generate plots/tables

**Current flow:**
1. dr_exp writes metrics.jsonl
2. No CSV export
3. No direct pandas compatibility

### 4.2 Missing Intermediate Steps

- No batch-level metric aggregation
- No CSV export from dr_exp metrics
- No validation of metric completeness
- No data cleaning/preprocessing utilities

## 5. Implementation Order Conflicts

### 5.1 Circular Dependencies

**Problem:** Tier 4 Phase 1 extracts functions from NoiseMonitor, but Tier 3 hasn't implemented the enhanced metrics logging these functions need.

**Resolution needed:**
1. Implement basic logging first (Tier 3)
2. Then extract/enhance functions (Tier 4)

### 5.2 Testing Infrastructure

**Tier 4 assumes:**
- Existing experimental data to test against
- Baseline metrics for regression testing

**Reality:**
- No runs completed yet
- Need synthetic data generation first

## 6. Specific Implementation Recommendations

### 6.1 Tier 3 Additions Needed

1. **Create batch-level logging callback:**
   ```python
   class LossSlopeLogger(Callback):
       def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
           # Log loss_train_nats, loss_train_bits, acc_train, lr, wd
           
       def on_validation_epoch_end(self, trainer, pl_module):
           # Log at ¼ epoch intervals
   ```

2. **Add CSV export alongside JSON:**
   ```python
   # In dr_exp_metrics callback
   self.csv_logger = CSVLogger(path / "metrics.csv")
   ```

3. **Implement missing metrics:**
   - grad_norm and weight_norm computation
   - checkpoint size tracking
   - proper scheduling for different intervals

### 6.2 Tier 4 Adjustments Needed

1. **Add missing dependencies first:**
   ```toml
   [project.dependencies]
   statsmodels = ">=0.14.0"
   click = ">=8.1.0"  # Already there
   rich = ">=13.0.0"
   ```

2. **Create synthetic data utilities:**
   ```python
   # Before Phase 1
   def generate_synthetic_training_logs():
       # Create test data for analysis development
   ```

3. **Build JSON→CSV converter:**
   ```python
   # Bridge existing dr_exp format
   def convert_metrics_jsonl_to_csv(jsonl_path, csv_path):
       # Parse and flatten JSON metrics
   ```

## 7. Critical Path Forward

### 7.1 Immediate Actions for Tier 3

1. **Implement LossSlopeLogger callback** with all batch-level metrics
2. **Add CSV export** to dr_exp_metrics callback
3. **Create metric validation script** to ensure completeness
4. **Test logging at all required intervals**

### 7.2 Pre-Tier 4 Requirements

1. **Generate synthetic test data** if real runs not available
2. **Add all missing dependencies** to pyproject.toml
3. **Create JSON→CSV conversion utilities**
4. **Document expected data formats**

### 7.3 Risk Mitigation

1. **Fallback for missing libraries:**
   - Implement Hutch++ manually if library not available
   - Use scipy.optimize instead of statsmodels where possible

2. **Data format flexibility:**
   - Support both JSON and CSV inputs in analysis scripts
   - Add format detection and conversion

3. **Testing strategy:**
   - Create minimal synthetic dataset first
   - Validate each analysis function independently
   - Integration test with subset before full sweep

## 8. Conclusion

The gap between Tier 3 and Tier 4 is significant but bridgeable. The main issues are:

1. **Logging granularity** - Need batch-level and sub-epoch metrics
2. **Data format** - Need CSV export for pandas compatibility  
3. **Missing functions** - Need to implement core analysis utilities
4. **Dependencies** - Several key libraries not available

With the additions outlined above, Tier 4 can proceed successfully. The critical path is implementing proper logging first, then building the analysis functions on that foundation.
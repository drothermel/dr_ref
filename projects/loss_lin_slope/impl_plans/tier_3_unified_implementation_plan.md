# Tier 3 Unified Implementation Plan: Job Setup and Critical Logging Infrastructure

## Date: 2025-07-01 - Updated 2025-07-02
## Purpose: Streamlined Tier 3 implementation focused on critical path to get experiments running

This document provides an atomic implementation plan that gets from 0 to running 216-job sweep in 23 focused commits, ensuring each change is testable and reversible.

**Atomic commits - each commit does ONE thing well, with explicit testing phases.**

## 📁 Repository Architecture

**IMPORTANT**: This implementation follows a deliberate multi-repository architecture:

### Primary Repositories and Their Roles:

1. **deconCNN** (`/Users/daniellerothermel/drotherm/repos/deconCNN/`)
   - **Role**: Primary implementation repository
   - **Contains**: All code, callbacks, configurations, and scripts
   - **Why**: Has existing ML infrastructure, dependencies, and callback framework
   - **Work here**: 95% of implementation happens here

2. **dr_ref** (`/Users/daniellerothermel/drotherm/repos/dr_ref/`)
   - **Role**: Documentation and planning repository
   - **Contains**: Implementation plans, architecture docs, retrospectives
   - **Why**: Centralized knowledge base, separate from code churn
   - **Work here**: Planning documents and cross-project documentation only

3. **dr_exp** (`/Users/daniellerothermel/drotherm/repos/dr_exp/`)
   - **Role**: Experiment tracking infrastructure
   - **Contains**: Job submission utilities, logging infrastructure
   - **Why**: Reusable experiment management across projects
   - **Used via**: Import and integration, not direct modification

### Working Directory Commands:

```bash
# For implementation work (most commands):
cd /Users/daniellerothermel/drotherm/repos/deconCNN

# For documentation updates:
cd /Users/daniellerothermel/drotherm/repos/dr_ref

# Commands in this plan assume you're in deconCNN unless specified
```

## 🎯 CRITICAL SUCCESS FACTORS

1. **Atomic Commits** - Each commit does exactly one thing
2. **Explicit Testing** - Separate commits for implementation and testing
3. **No Bundling** - Scripts, configs, and tests in separate commits
4. **Quality gates maintained** - Run `lint_fix` before EVERY commit

## Agent Instructions
**IMPORTANT: Read these before starting implementation**

1. **Quality Gates**: Before EVERY commit:
   - Run `lint_fix` and resolve all issues
   - Run tests and ensure they pass
   - Fix any issues found, even if pre-existing

2. **Progress Tracking**: 
   - Use TodoWrite/TodoRead tools proactively for tasks with 3+ steps
   - Mark each step when complete
   - Add notes in [brackets] for any deviations
   - Update the "Current Status" section after each work session

3. **Adaptation Protocol**:
   - Follow the plan but use judgment
   - Document any necessary modifications in [brackets]
   - Flag significant changes for retrospective

4. **Commit Strategy**:
   - Functional commits that complete full features
   - Use git shortcuts: `gst`, `gd`, `ga .`, `gc -m "msg"`, `glo`
   - Commit messages: imperative mood, describe purpose not implementation

## Current Status
- Last updated: 2025-07-02
- Last completed step: [Comprehensive review completed - findings documented below]
- Active agent: [none]
- Blocked by: [none]
- Plan status: **Atomic structure - 23 focused commits organized in 6 phases**

## 🔍 Comprehensive Review Findings (2025-07-02)

### ✅ Repository Structure Verified
- All three repositories exist exactly as documented
- deconCNN has expected callback infrastructure with BaseMonitor pattern
- configs/experiment/ directory empty and ready for new configs
- Only missing: `notebooks/` directory (easy creation in Phase 0)

### ⚠️ Critical Implementation Updates Needed

1. **Submission Script Syntax** 
   - Plan shows: `--sweep optim.lr=0.05,0.1,0.2`
   - **Actual syntax**: `--param lr=0.05,0.1,0.2`
   - All submission commands need updating

2. **CurvatureMonitor Implementation**
   - Current code does NOT use PyHessian (uses custom implementation)
   - Already has `compute_every_n_steps=500` parameter
   - **Decision**: Keep existing implementation, skip PyHessian refactor
   - Memory optimization can be added to existing code

3. **Existing Metric Logging**
   - Lightning module ALREADY logs: `train_loss`, `train_loss_bits`, `train_acc`, `lr`, `wd`
   - LossSlopeLogger should coordinate to avoid duplication
   - Focus on adding slope calculations and periodic gradient/weight norms

4. **Dr_exp Integration**
   - Integration exists via dr_exp_callback.py and dr_exp_utils.py
   - Need to verify exact API before implementing job generation
   - Consider using existing submit_experiments.py for all submissions

### ✅ Dependencies Confirmed
- All required packages installed: pandas, matplotlib, statsmodels, rich
- PyHessian installed but not needed
- Training infrastructure fully operational
- Hydra configuration working as expected

### 📋 Implementation Recommendations

1. **Simplify CurvatureMonitor updates** - Add memory params to existing implementation
2. **Reuse existing metrics** - Don't duplicate train_loss_bits logging
3. **Use submit_experiments.py** - May not need custom job generation
4. **Test metric conflicts early** - Run all callbacks together in Commit 6

### ⚡ Updated Commands

```bash
# Submission command (corrected syntax)
python scripts/submit_experiments.py \
  --config configs/experiment/loss_lin_slope_base.yaml \
  --param lr=0.05,0.1,0.2 \
  --param weight_decay=1e-4,1e-3,1e-2 \
  --param seed=0,1,2,3,4,5

# Test with fast_dev_run (confirmed working)
uv run python scripts/train_cnn.py trainer.fast_dev_run=true
```

## 📋 Pre-Implementation Checklist

**Complete these steps BEFORE starting Commit 1:**

- [ ] Switch to deconCNN repository: `cd /Users/daniellerothermel/drotherm/repos/deconCNN`
- [ ] Create notebooks directory: `mkdir -p notebooks`
- [ ] Verify dependencies: `uv run python -c "import pandas, matplotlib, statsmodels, rich; print('Dependencies OK')"`
- [ ] Test training script: `uv run python scripts/train_cnn.py trainer.fast_dev_run=true`
- [ ] Review BaseMonitor pattern: `cat src/deconcnn/callbacks/base_monitor.py`
- [ ] Check existing metrics: `grep -n "self.log" src/deconcnn/models/cifar_resnet.py`
- [ ] Verify submit script syntax: `python scripts/submit_experiments.py --help`

## Phase 0: Pre-Implementation Setup

### Critical Setup Requirements

- [ ] **Pre-Setup**: Verify environment and create necessary directories
  ```bash
  # Working directory: deconCNN
  cd /Users/daniellerothermel/drotherm/repos/deconCNN
  
  # Verify dependencies
  uv run python -c "import pandas, matplotlib, statsmodels, rich; print('Dependencies OK')"
  
  # Verify training script location
  ls scripts/train_cnn.py  # Should exist
  
  # Test basic training with fast_dev_run
  uv run python scripts/train_cnn.py trainer.fast_dev_run=true
  
  # Create notebooks directory
  mkdir -p notebooks
  ```
  - Dependencies present: pandas>=2.3.0, matplotlib>=3.10.3, statsmodels>=0.14.4, rich>=14.0.0
  - Verify training script runs without errors
  - Create notebooks directory for validation scripts
  - **MUST COMPLETE**: Before any other commits
  - Commit: `chore: verify environment and create notebooks directory`

## Phase 1: Core Callbacks (6 commits)

```bash
# Working directory for all commits
cd /Users/daniellerothermel/drotherm/repos/deconCNN
```

- [ ] **Commit 1**: Implement LossSlopeLogger callback
  - Create `src/deconcnn/callbacks/loss_slope_logger.py`
  - Inherit from BaseMonitor (see existing callbacks for pattern)
  - Key implementation structure:
    ```python
    class LossSlopeLogger(BaseMonitor):
        def __init__(self, 
                     log_every_n_steps: int = 1,
                     log_epoch_freq: int = 1,
                     burn_in_epochs: int = 5,
                     slope_window_start: int = 5,
                     slope_window_end: int = 15,
                     debug_mode: bool = True):
            super().__init__(log_every_n_steps, log_epoch_freq, debug_mode)
    ```
  - **[UPDATED]** Focus on NEW metrics only (existing metrics already logged by Lightning module):
    - Skip: `train_loss`, `train_loss_bits`, `train_acc`, `lr`, `wd` (already logged)
    - Add gradient and weight norm tracking every 0.25 epochs:
      - `grad_norm_l2`: L2 norm of all gradients
      - `weight_norm_l2`: L2 norm of all weights
    - Implement slope calculation:
      - `loss_slope_5_15`: Slope computed over epochs 5-15 using least squares
      - `loss_slope_full`: Slope from burn-in to current epoch
      - Store loss history internally for slope computation
  - Use existing loss values from `trainer.callback_metrics['train_loss']`
  - Use `self.pl_module.log_dict()` for logging new metrics
  - Add comprehensive docstrings explaining slope methodology
  - Commit: `feat: implement LossSlopeLogger callback`

- [ ] **Commit 2**: Test LossSlopeLogger functionality
  - Test with: `python scripts/train_cnn.py epochs=2 batch_size=32`
  - Verify batch-level metrics are logged correctly
  - Verify gradient norms logged every 0.25 epochs
  - Commit: `test: verify LossSlopeLogger batch and epoch logging`

- [ ] **Commit 3**: Configure CurvatureMonitor settings
  - **[UPDATED]** Keep existing implementation (no PyHessian refactor needed)
  - Verify `compute_every_n_steps: int = 500` is already set (confirmed)
  - Add memory optimization parameters to existing implementation:
    ```python
    def __init__(self,
                 compute_every_n_steps: int = 500,
                 subsample_ratio: float = 0.1,  # Sample 10% of data for Hessian
                 max_batch_size: int = 32,      # Limit batch size for memory
                 ...):
    ```
  - Update compute method to use subsampling:
    ```python
    # Subsample data for memory efficiency
    n_samples = int(len(dataloader.dataset) * self.subsample_ratio)
    indices = torch.randperm(len(dataloader.dataset))[:n_samples]
    ```
  - Add docstring with memory usage tips and expected compute times
  - Test memory usage with different subsample ratios
  - Commit: `feat: add memory optimization to CurvatureMonitor`

- [ ] **Commit 4**: Test CurvatureMonitor configuration
  - Test with: `python scripts/train_cnn.py epochs=1 batch_size=128`
  - Verify metrics computed exactly every 500 steps
  - Monitor memory usage during computation
  - Commit: `test: verify CurvatureMonitor 500-step frequency`

- [ ] **Commit 5**: Configure NoiseMonitor settings
  - Update NoiseMonitor configuration in callback YAML:
    ```yaml
    noise_monitor:
      _target_: deconcnn.callbacks.noise_monitor.NoiseMonitor
      compute_every_epochs: 2
      log_epoch_freq: 1
    ```
  - Set checkpoint save_top_k=3 in trainer defaults
  - Ensure PSD tail, gradient variance use correct frequencies
  - Commit: `feat: configure NoiseMonitor and checkpoint settings`

- [ ] **Commit 6**: Integration test all callbacks
  - Test all callbacks together: LossSlopeLogger + DrExpMetricsCallback + CurvatureMonitor + NoiseMonitor
  - Verify no metric name conflicts
  - Check total memory usage
  - Test with: `python scripts/train_cnn.py epochs=4 batch_size=32 enable_checkpointing=true`
  - Commit: `test: validate callback integration`

## Phase 2: Experiment Configurations (5 commits)

- [ ] **Commit 7**: Create base experiment config
  - Create `configs/experiment/loss_lin_slope_base.yaml`:
    ```yaml
    defaults:
      - _self_
      - /model: resnet18_cifar
      - /callbacks: [
          gradient_monitor,
          noise_monitor,
          curvature_monitor,
          loss_slope_logger,  # New callback
          dr_exp_metrics
        ]
      - /machine: cluster
    
    # Experimental design parameters
    seed: ???  # Required override
    optim:
      lr: ???  # Will be {0.05, 0.1, 0.2}
      weight_decay: ???  # Will be {1e-4, 1e-3, 1e-2}
      momentum: 0.9
    
    trainer:
      max_epochs: 50
      val_check_interval: 0.25  # Critical for H2 hypothesis
      gradient_clip_val: null
    
    data:
      batch_size: 128
      dataset: cifar10
      augmentation:
        - RandomResizedCrop
        - RandomHorizontalFlip
    
    # Cosine schedule with 5-epoch warmup
    scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 50
      eta_min: 0
    
    warmup:
      epochs: 5
    ```
  - Commit: `feat: create base experiment configuration`

- [ ] **Commit 8**: Test base configuration
  - Load and validate config structure
  - Run 1 epoch test: `python scripts/train_cnn.py +experiment=loss_lin_slope_base epochs=1`
  - Verify validation runs 4 times per epoch
  - Commit: `test: verify base experiment configuration`

- [ ] **Commit 9**: Create BN-off variant
  - Create `configs/experiment/loss_lin_slope_bn_off.yaml`:
    ```yaml
    defaults:
      - loss_lin_slope_base
    
    model:
      norm_type: none  # Disable BN
    
    trainer:
      gradient_clip_val: 1.0  # Critical for stability without BN
    ```
  - Commit: `feat: create BN-off experiment variant`

- [ ] **Commit 10**: Create narrow and AdamW variants
  - Create `configs/experiment/loss_lin_slope_narrow.yaml`:
    ```yaml
    defaults:
      - loss_lin_slope_base
    
    model:
      width_mult: 0.5  # Half width for all layers
    ```
  - Create `configs/experiment/loss_lin_slope_adamw.yaml`:
    ```yaml
    defaults:
      - loss_lin_slope_base
    
    optim:
      _target_: torch.optim.AdamW
      lr: ???
      weight_decay: ???
      betas: [0.9, 0.999]
    ```
  - Commit: `feat: create narrow and AdamW variants`

- [ ] **Commit 11**: Create unified callback config
  - Create `configs/callbacks/loss_lin_slope_metrics.yaml`
  - Consolidate all callback settings in one place
  - Update experiment configs to reference it
  - Commit: `feat: create unified callback configuration`

## Phase 3: Local Validation (3 commits)

- [ ] **Commit 12**: Create validation script
  - Create `scripts/validate_local.py`
  - Quick 3-epoch test for single configuration
  - Check output formats and logging
  - Commit: `feat: create local validation script`

- [ ] **Commit 13**: Create test harness
  - Create `scripts/test_harness.py`
  - Test all 4 variants sequentially
  - Report pass/fail for each
  - Commit: `feat: create test harness for all variants`

- [ ] **Commit 14**: Create validation notebook and unit tests
  - Create `notebooks/validate_metrics_basic.ipynb`
  - Add unit tests for LossSlopeLogger in `tests/test_loss_slope_logger.py`:
    ```python
    def test_slope_calculation_accuracy():
        """Test that slope calculation matches numpy least squares"""
        # Generate synthetic loss curve
        # Calculate slope using LossSlopeLogger method
        # Compare with numpy.polyfit reference
        assert np.abs(calculated_slope - reference_slope) < 1e-6
    ```
  - Implement validation for H1 (Power-law prevalence):
    - Load sample run outputs
    - Compute R² for power-law fits after burn-in (CE ≤ 1.0)
    - Verify >90% of runs have R² ≥ 0.98
    - Create histogram of R² values
  - Implement validation for H2 (Early slope prediction):
    - Extract α_early (slope epochs 5-15) using least squares
    - Extract best validation CE within 50 epochs
    - Compute correlation ρ(α_early, best_val_CE)
    - Verify ρ ≥ 0.7 for predictive power
  - Data quality checks:
    - Verify logging frequencies match spec:
      - Batch-level: every batch
      - Validation: 4x per epoch (0.25 interval)
      - Curvature: every 500 steps
      - Noise: every 2 epochs
    - Validate nats→bits conversion: bits = nats / ln(2)
    - Check for NaN/inf values in metrics
  - Edge case handling tests:
    - Early epochs (< slope_window_start)
    - Insufficient data points
    - NaN/inf loss values
  - GO/NO-GO decision framework:
    - GO if: All frequencies correct AND R² > 0.95 for sample AND no data issues
    - NO-GO if: Missing metrics OR wrong frequencies OR poor fits OR data corruption
  - Commit: `feat: create metric validation notebook and unit tests`

## Phase 4: Job Management (4 commits)

- [ ] **Commit 15**: Create job generation script
  - **[UPDATED]** Consider if needed - submit_experiments.py may handle everything
  - If needed, create `scripts/generate_jobs.py`
  - Use existing dr_exp infrastructure:
    ```python
    from dr_exp_utils import submit_job_to_dr_exp
    
    GRID = {
        'variants': ['base', 'bn_off', 'narrow', 'adamw'],
        'learning_rates': [0.05, 0.1, 0.2],
        'weight_decays': [1e-4, 1e-3, 1e-2],
        'seeds': [0, 1, 2, 3, 4, 5]
    }
    # Total: 4 × 3 × 3 × 6 = 216 jobs
    ```
  - Implement priority assignment:
    ```python
    def get_priority(variant, lr, wd, seed):
        base_priority = 100
        if lr == 0.1:  # Default LR
            base_priority += 50
        if wd == 1e-4:  # Default WD
            base_priority += 25
        if seed == 0:  # First seed
            base_priority += 10
        return base_priority
    ```
  - Generate and submit jobs using dr_exp format:
    ```python
    for variant in GRID['variants']:
        for lr in GRID['learning_rates']:
            for wd in GRID['weight_decays']:
                for seed in GRID['seeds']:
                    job_config = {
                        "experiment": f"loss_lin_slope_{variant}",
                        "config_overrides": {
                            "seed": seed,
                            "optim.lr": lr,
                            "optim.weight_decay": wd
                        },
                        "priority": get_priority(variant, lr, wd, seed),
                        "tags": [f"lr{lr}", f"wd{wd}", variant]
                    }
                    submit_job_to_dr_exp(job_config)
    ```
  - Commit: `feat: create job generation script`

- [ ] **Commit 16**: Create submission wrapper
  - **[UPDATED]** Use existing `scripts/submit_experiments.py` with correct syntax:
    ```bash
    # Submit all jobs for base variant
    python scripts/submit_experiments.py \
      --config configs/experiment/loss_lin_slope_base.yaml \
      --param lr=0.05,0.1,0.2 \
      --param weight_decay=1e-4,1e-3,1e-2 \
      --param seed=0,1,2,3,4,5
    
    # Repeat for other variants
    python scripts/submit_experiments.py \
      --config configs/experiment/loss_lin_slope_bn_off.yaml \
      --param lr=0.05,0.1,0.2 \
      --param weight_decay=1e-4,1e-3,1e-2 \
      --param seed=0,1,2,3,4,5
    ```
  - Alternative: Create wrapper script `scripts/submit_all_loss_slope.sh`:
    ```bash
    #!/bin/bash
    # Submit all 216 jobs using existing infrastructure
    
    VARIANTS="base bn_off narrow adamw"
    LRS="0.05,0.1,0.2"
    WDS="1e-4,1e-3,1e-2"
    SEEDS="0,1,2,3,4,5"
    
    for variant in $VARIANTS; do
        echo "Submitting $variant variant..."
        python scripts/submit_experiments.py \
          --config configs/experiment/loss_lin_slope_${variant}.yaml \
          --param lr=$LRS \
          --param weight_decay=$WDS \
          --param seed=$SEEDS \
          --tags loss_lin_slope,$variant
    done
    ```
  - Commit: `feat: create submission wrapper using existing infrastructure`

- [ ] **Commit 17**: Create monitoring script
  - Create `scripts/monitor_experiment.py`
  - Implement job status tracking using dr_exp API:
    ```python
    # Track: jobs queued/running/completed/failed
    status = dr_exp.get_experiment_status("loss_lin_slope")
    ```
  - Add data quality checks:
    ```python
    QUALITY_CHECKS = {
        'loss_range': (0, 10),  # CE loss reasonable bounds
        'accuracy_range': (0, 1.0),
        'lr_positive': lambda x: x > 0,
        'no_nans': lambda x: not np.isnan(x).any(),
        'knee_epoch_range': (1, 20)  # Burn-in should end early
    }
    ```
  - Implement alert conditions:
    ```python
    ALERT_CONDITIONS = {
        'high_failure_rate': lambda s: s['failed'] / s['total'] > 0.1,
        'stuck_jobs': lambda j: j['runtime'] > 3600 and j['epoch'] < 5,
        'data_corruption': lambda m: any(np.isnan(m.values()))
    }
    ```
  - Progress reporting with ETA based on average job runtime
  - Commit: `feat: create experiment monitoring script`

- [ ] **Commit 18**: Create failure recovery script
  - Create `scripts/recover_failed.py`
  - Detect OOM, timeout, gradient explosion
  - Implement retry logic with parameter adjustment
  - Create `docs/operational_runbook_basic.md`
  - Commit: `feat: create failure recovery system`

## Phase 5: Data Pipeline (3 commits)

- [ ] **Commit 19**: Create collection script
  - Create `scripts/collect_and_archive.sh`
  - Gather results from scratch directory
  - Organize by experiment variant
  - Commit: `feat: create result collection script`

- [ ] **Commit 20**: Create verification script
  - Create `scripts/verify_completeness.py`
  - Check all 216 runs completed
  - Report missing or failed runs
  - Commit: `feat: create completeness verification script`

- [ ] **Commit 21**: Create export script
  - Create `scripts/prepare_dataset.py`
  - Define output columns for Tier 4 analysis:
    ```python
    OUTPUT_COLUMNS = [
        # Identifiers
        'run_id', 'variant', 'seed', 'lr', 'weight_decay',
        
        # Metrics for H1 (power-law prevalence)
        'r_squared', 'knee_epoch', 'burn_in_loss',
        
        # Metrics for H2 (early slope prediction)
        'alpha_early',  # Slope epochs 5-15
        'alpha_full',   # Slope from burn-in to end
        'best_val_ce',  # Best validation CE in 50 epochs
        'best_val_epoch',
        
        # Time series (store as separate files)
        'loss_trajectory',  # Every 1/4 epoch
        'val_trajectory',   # Every 1/4 epoch
        
        # Future track metrics
        'lambda_max_trajectory',  # Every 500 steps
        'hutch_trace_trajectory',  # Every 500 steps
    ]
    ```
  - Create Parquet schema with metadata:
    ```python
    import pyarrow as pa
    schema = pa.schema([
        ('run_id', pa.string()),
        ('variant', pa.string()),
        ('seed', pa.int32()),
        ('lr', pa.float32()),
        ('weight_decay', pa.float32()),
        ('r_squared', pa.float32()),
        ('alpha_early', pa.float32()),
        ('best_val_ce', pa.float32()),
        # ... etc
    ])
    ```
  - Include data dictionary documenting each column
  - Commit: `feat: create dataset export script`

## Phase 6: Cluster Execution (2 commits - MANUAL EXECUTION)

**Note: These commits require cluster access and will be executed manually**

- [ ] **Commit 22**: Resource testing (MANUAL)
  - Create `scripts/test_resource_configs.py`
  - Define expected resource profiles:
    ```python
    RESOURCE_PROFILES = {
        'base': {'gpu_memory_gb': 4, 'optimal_workers': 3},
        'bn_off': {'gpu_memory_gb': 5, 'optimal_workers': 2},  # More memory, less stable
        'narrow': {'gpu_memory_gb': 2, 'optimal_workers': 4},  # 0.5x width = less memory
        'adamw': {'gpu_memory_gb': 6, 'optimal_workers': 2},   # Optimizer states need memory
    }
    ```
  - Test protocol:
    ```python
    TEST_CONFIGS = [
        {'workers': 2, 'variants': ['base', 'bn_off', 'adamw']},
        {'workers': 3, 'variants': ['base', 'narrow']},
        {'workers': 4, 'variants': ['narrow']},
    ]
    ```
  - Success criteria:
    - No OOM errors
    - GPU utilization > 80%
    - Job completion < 30 minutes for 50 epochs
    - Memory usage within RTX8000 48GB limit
  - Document optimal workers per variant in commit message
  - Commit: `test: determine resource allocation on cluster`

- [ ] **Commit 23**: Execute experiment sweep (MANUAL)
  - Run submission with determined worker counts:
    ```bash
    # Submit jobs by variant with optimal workers
    python scripts/submit_loss_lin_slope.py --variant base --workers 3
    python scripts/submit_loss_lin_slope.py --variant bn_off --workers 2
    python scripts/submit_loss_lin_slope.py --variant narrow --workers 4
    python scripts/submit_loss_lin_slope.py --variant adamw --workers 2
    ```
  - Monitor execution:
    ```bash
    # Real-time monitoring
    watch -n 30 "python scripts/monitor_experiment.py"
    
    # Check for failures
    dr_exp --experiment loss_lin_slope list --status failed
    ```
  - Expected timeline: ~12 hours on 24 GPUs
  - Collect results using `scripts/collect_and_archive.sh`
  - Verify completeness: All 216 runs should have R² values and slope metrics
  - Document in commit: total runtime, failure rate, resource usage
  - Commit: `chore: execute 216-job loss slope sweep`

## Additional Implementation Recommendations

### Testing Strategy
1. **Unit Tests First**: Create unit tests for LossSlopeLogger before integration
   - Test slope calculation accuracy against numpy reference
   - Verify edge case handling (early epochs, missing data)
   - Ensure robust handling of NaN/inf values
   
2. **Integration Testing**: Test callbacks together before full experiment
   - Verify no metric name conflicts
   - Check memory usage with all callbacks active
   - Ensure logging frequencies are correct

### Documentation Standards
1. **Comprehensive Docstrings**: 
   - Explain slope calculation methodology in detail
   - Document expected memory usage and compute times
   - Include examples of metric interpretation
   
2. **Inline Comments**: Add comments for complex calculations
   - Slope fitting algorithm
   - Memory management techniques
   - Error handling logic

### Error Handling
1. **Graceful Degradation**: 
   - Continue training if slope calculation fails
   - Log warnings for data quality issues
   - Provide fallback values for missing metrics
   
2. **Early Detection**: 
   - Validate configuration on startup
   - Check for required dependencies
   - Verify GPU memory availability

### Performance Considerations
1. **Batch Processing**: 
   - Accumulate metrics before logging to reduce overhead
   - Use vectorized operations for slope calculations
   - Cache frequently accessed values
   
2. **Memory Management**:
   - Clear intermediate tensors after use
   - Use gradient checkpointing if needed
   - Monitor GPU memory usage during development

## Critical Validation Checkpoints

### **REQUIRED BEFORE STARTING:**
- [ ] Complete Pre-Implementation Checklist above
- [ ] Complete Phase 0 pre-setup 
- [ ] Understand key findings from Comprehensive Review section
- [ ] Note: configs/experiment directory exists but is empty (ready for use)
- [ ] Note: BaseMonitor pattern is straightforward (log_every_n_steps, should_log methods)
- [ ] Note: ResNet width_mult confirmed working (scales channels, min 8)

### Validation Checkpoints:
- [ ] **Commit 6**: All callbacks work together without conflicts
- [ ] **Commit 8**: Base experiment config loads without errors
- [ ] **Commit 11**: All 4 variants configured correctly
- [ ] **Commit 14**: Metric validation notebook confirms proper logging formats
- [ ] **Commit 21**: All scripts created and tested locally
- [ ] **Commit 22**: Resource testing prevents OOMs on cluster
- [ ] **Commit 23**: 216-job sweep successfully submitted and completed

### Pre-submission Checklist:
- [ ] LossSlopeLogger logs every batch with nats and bits
- [ ] Validation runs at 0.25 epoch intervals
- [ ] CurvatureMonitor computes every 500 steps
- [ ] NoiseMonitor logs every 2 epochs
- [ ] All 4 variants tested locally
- [ ] Resource testing completed
- [ ] Checkpoint strategy set to save_top_k=3

### Post-execution Checklist:
- [ ] All 216 runs completed or accounted for
- [ ] No missing metrics or frequencies
- [ ] CSV export successful for Tier 4
- [ ] Checkpoints ~44MB each
- [ ] Dataset in Parquet format
- [ ] Handoff documentation complete

## Key Technical Details

### Logging Frequencies (CRITICAL):
- **Batch-level**: loss_train_nats/bits, acc_train, lr, wd - EVERY batch
- **Quarter-epoch (0.25)**: loss_val_nats, acc_val, grad_norm, weight_norm
- **500 steps**: lambda_max, hutchinson_trace (for Track A curvature metrics)
- **2 epochs**: psd_tail, gradient_variance, residual_power (for Track C noise)
- **Slope windows**: 
  - alpha_early: epochs 5-15 (matching warmup length)
  - alpha_full: from burn-in (CE ≤ 1.0) to current

### Resource Configuration:
- **Base variant**: 3 workers per GPU (~4GB each)
- **BN-off variant**: 2 workers per GPU (~5GB each, gradient clipping overhead)
- **Narrow variant**: 4 workers per GPU (~2GB each, 0.5x width)
- **AdamW variant**: 2 workers per GPU (~6GB each, optimizer states)

### Critical Paths:
- **Working directory**: `/Users/daniellerothermel/drotherm/repos/deconCNN`
- **Training script**: `scripts/train_cnn.py` (verified location)
- **Callback configs**: `configs/callbacks/`
- **Experiment configs**: `configs/experiment/` (already created)
- **SLURM account**: CDS
- **Scratch path**: `/scratch/ddr8143/`
- **Container**: Singularity with overlay at `/scratch/work/public/singularity/`
- **dr_exp integration**: Via imports from dr_exp package

### Experimental Design:
- **Hyperparameter Grid**: 
  - LR: {0.05, 0.1, 0.2}
  - WD: {1e-4, 1e-3, 1e-2}
  - Seeds: {0, 1, 2, 3, 4, 5}
  - Variants: {base, bn_off, narrow, adamw}
- **Schedule**: Cosine annealing with 5-epoch warmup
- **Augmentation**: RandomResizedCrop + RandomHorizontalFlip
- **Success Metrics**:
  - H1: >90% runs with R² ≥ 0.98 for power-law fit
  - H2: ρ(alpha_early, best_val_CE) ≥ 0.7

### Infrastructure Status:
- **✅ Working**: dr_exp integration, callback system, batch logging, ML dependencies
- **✅ Created**: configs/experiment directory, src/deconcnn/analysis directory
- **✅ Dependencies**: pandas, matplotlib, statsmodels, rich installed
- **✅ Callbacks**: CurvatureMonitor has compute_every_n_steps=500
- **❌ Missing**: LossSlopeLogger, notebooks directory
- **🔄 Updated**: All paths reference deconCNN repository

## Risk Mitigations

1. **Memory Issues**: Variant-specific worker counts from testing
2. **Gradient Explosions**: grad_clip_norm=1.0 for BN-off
3. **Data Loss**: Checkpoint frequently, dual format logging
4. **Incomplete Runs**: Recovery script with adaptive strategies
5. **Analysis Gaps**: Ensure proper data formatting for Tier 4
6. **Callback Conflicts**: Test all callbacks together before cluster deployment
7. **PyHessian Memory**: Use mini-batch settings to control memory usage
8. **Experiment Failure Risk**: Automated failure recovery prevents job loss during 216-job execution
9. **Data Corruption Risk**: Real-time data quality monitoring prevents wasted compute on corrupted experiments

## Handoff Instructions

To continue this implementation:
1. **FIRST**: Read the "Comprehensive Review Findings" section for critical updates
2. `cd /Users/daniellerothermel/drotherm/repos/deconCNN`
3. Complete the "Pre-Implementation Checklist" section
4. Read the full plan including repository architecture
5. Check "Current Status" section for latest progress
6. Use TodoWrite to plan your session
7. Find first unchecked [ ] item (start with Commit 1)
8. Implement with quality gates (lint_fix before every commit)
9. Update "Current Status" section when stopping

### **CRITICAL DEPENDENCIES:**
- Phase 0 → Commit 1 (environment must be verified first)
- Commit 1 → Commit 2 (must test LossSlopeLogger before proceeding)
- Commit 6 → Commit 7 (callbacks must work before creating configs)
- Commit 14 → Commit 15 (validation must pass before job generation)
- Commit 21 → Commit 22 (all scripts ready before cluster testing)

### **Repository Context for Each Commit:**
- Commits 1-21: Execute in deconCNN repository (local development)
- Commits 22-23: Execute from deconCNN on cluster (GPU required, MANUAL)

## Success Metrics

**Phase 1 (Callbacks):**
- Each callback tested individually
- Integration test passes without conflicts

**Phase 2 (Configs):**
- All 4 variants load and run
- Validation occurs 4x per epoch

**Phase 3 (Validation):**
- Test harness confirms all variants work
- Notebook validates data formats

**Phase 4 (Job Management):**
- 216 job configurations generated
- Monitoring and recovery tested

**Phase 5 (Data Pipeline):**
- Collection and export scripts functional
- Data format ready for Tier 4

**Phase 6 (Execution):**
- Resource testing complete
- 216 jobs successfully executed
- All data collected and verified

---

Remember: Quality over speed. Test thoroughly. Fix all linting issues. Document changes.
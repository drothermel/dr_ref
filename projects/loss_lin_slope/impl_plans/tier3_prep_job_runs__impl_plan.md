# Tier 3 Implementation Plan: Job Setup and Critical Logging Infrastructure
## Date: 2025-07-02

This document provides an atomic implementation plan that gets from 0 to running 216-job sweep in 31 focused commits, ensuring each change is testable and reversible with comprehensive unit testing.

**Atomic commits - each commit does ONE thing well, with explicit testing phases.**

## 📁 Repository Architecture

**IMPORTANT**: This implementation follows a deliberate multi-repository architecture:

### Primary Repositories and Their Roles:

1. **deconCNN** (`/Users/daniellerothermel/drotherm/repos/deconCNN/`)
   - **Role**: Primary implementation repository
   - **Contains**: All code, callbacks, configurations, and scripts
   - **Why**: Has existing ML infrastructure, dependencies, and callback framework
   - **Work here**: 95% of implementation happens here

2. **dr_exp** (`/Users/daniellerothermel/drotherm/repos/dr_exp/`)
   - **Role**: Experiment tracking infrastructure
   - **Contains**: Job submission utilities, logging infrastructure
   - **Why**: Reusable experiment management across projects
   - **Used via**: Already integrated in deconCNN via dr_exp_callback.py

## 🎯 CRITICAL SUCCESS FACTORS

1. **Atomic Commits** - Each commit does exactly one thing
2. **Explicit Testing** - Separate commits for implementation and testing
3. **No Bundling** - Scripts, configs, and tests in separate commits
4. **Quality gates maintained** - Run `lint_fix` before EVERY commit

## Agent Instructions
**IMPORTANT: Read these before starting implementation**

1. **Quality Gates**: Before EVERY commit:
   - Run `lint_fix` and fix ALL errors, even if you didn't create them
   - Run `pt` (pytest) and ensure ALL tests pass - do NOT commit if any test fails
   - Fix any issues found, even if pre-existing - the codebase must be clean
   - NEVER skip these steps or commit with failing tests/lints

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
- Last completed step: Plan fully reviewed
- Active agent: [none]
- Blocked by: [none]
- Plan status: **Ready for implementation - 31 focused commits organized in 6 phases with unit tests**

## Phase 0: Pre-Implementation Setup

### ⚠️ IMPORTANT: Machine Configuration for Local Testing
When running on Mac, you MUST specify `machine=mac` for all training commands to avoid cluster-specific paths and CUDA errors. The default configuration points to cluster paths that don't exist on Mac.

### Critical Setup Requirements

- [ ] **Pre-Setup**: Verify environment and create necessary directories
  ```bash
  # CRITICAL: Navigate to deconCNN repository first
  cd /Users/daniellerothermel/drotherm/repos/deconCNN
  
  # Verify you're in the correct directory
  pwd  # Should show: /Users/daniellerothermel/drotherm/repos/deconCNN
  
  # Verify dependencies
  uv run python -c "import pandas, matplotlib, statsmodels, rich; print('Dependencies OK')"
  
  # Verify dr_exp_utils module exists and imports correctly
  uv run python -c "from deconcnn.dr_exp_utils import list_decon_jobs, submit_training_job; print('dr_exp_utils OK')"
  
  # Verify training script location
  ls scripts/train_cnn.py  # Should exist
  
  # Test basic training with fast_dev_run
  uv run python scripts/train_cnn.py machine=mac trainer.fast_dev_run=true
  ```
  - Dependencies present: pandas>=2.3.0, matplotlib>=3.10.3, statsmodels>=0.14.4, rich>=14.0.0
  - Verify training script runs without errors: `uv run python scripts/train_cnn.py machine=mac trainer.fast_dev_run=true`
  - Review BaseMonitor pattern: `cat src/deconcnn/callbacks/base_monitor.py`
  - Check existing metrics: `grep -n "self.log" src/deconcnn/training/lightning_module.py`
  - Verify submit script syntax: `uv run python scripts/submit_experiments.py sweep --help`
  - Run `lint_fix` then commit: `chore: verify environment and dependencies`

## Phase 1: Core Callbacks (9 commits)

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
  - Focus on NEW metrics only with unique names:
    - Use `loss_slope/` prefix for all metrics to avoid conflicts
    - Periodic gradient and weight norm tracking (every 0.25 epochs):
      - `loss_slope/grad_norm_periodic`: L2 norm of all gradients
      - `loss_slope/weight_norm_periodic`: L2 norm of all weights
    - Implement slope calculation:
      - `loss_slope/alpha_5_15`: Slope computed over epochs 5-15 using least squares
      - `loss_slope/alpha_full`: Slope from burn-in to current epoch
      - Store loss history internally for slope computation
  - Use existing loss values from `trainer.callback_metrics['train_loss']`
  - Use `self.pl_module.log_dict()` for logging new metrics
  - Add comprehensive docstrings explaining slope methodology
  - Run `lint_fix` then commit: `feat: implement LossSlopeLogger callback`

- [ ] **Commit 2**: Add unit tests for LossSlopeLogger
  - Create `tests/test_loss_slope_logger.py`
  - Test slope calculation accuracy:
    ```python
    def test_slope_calculation_accuracy():
        """Test that slope calculation matches numpy least squares"""
        logger = LossSlopeLogger()
        # Generate synthetic loss curve: y = 2.0 - 0.1*x + noise
        losses = [2.0 - 0.1*x + 0.01*np.random.randn() for x in range(20)]
        calculated_slope = logger._calculate_slope(losses, start=5, end=15)
        reference_slope = np.polyfit(range(5, 15), losses[5:15], 1)[0]
        assert np.abs(calculated_slope - reference_slope) < 1e-6
    ```
  - Test metric naming and prefixing:
    ```python
    def test_metric_prefixing():
        """Ensure all metrics use loss_slope/ prefix"""
        logger = LossSlopeLogger()
        metrics = logger._prepare_metrics(...)
        assert all(k.startswith('loss_slope/') for k in metrics.keys())
    ```
  - Test edge cases:
    ```python
    def test_early_epoch_handling():
        """Test behavior before slope_window_start"""
        logger = LossSlopeLogger(slope_window_start=5)
        # Should not compute slope before epoch 5
        assert logger._should_compute_slope(epoch=3) is False
    
    def test_nan_handling():
        """Test graceful handling of NaN values"""
        logger = LossSlopeLogger()
        losses_with_nan = [1.0, np.nan, 0.8, 0.7]
        # Should filter NaN and still compute or return None
        slope = logger._calculate_slope(losses_with_nan)
        assert slope is None or not np.isnan(slope)
    ```
  - Test gradient norm computation:
    ```python
    def test_gradient_norm_computation():
        """Test gradient norm calculation"""
        # Mock model with known gradients
        mock_model = create_mock_model_with_gradients()
        logger = LossSlopeLogger()
        grad_norm = logger._compute_gradient_norm(mock_model)
        expected_norm = np.sqrt(sum(g.norm()**2 for g in known_gradients))
        assert np.abs(grad_norm - expected_norm) < 1e-6
    ```
  - Run `lint_fix` then commit: `test: add unit tests for LossSlopeLogger`

- [ ] **Commit 3**: Integration test LossSlopeLogger
  - Test with: `uv run python scripts/train_cnn.py machine=mac epochs=2 batch_size=32`
  - Verify batch-level metrics are logged correctly
  - Verify gradient norms logged every 0.25 epochs
  - Run `lint_fix` then commit: `test: verify LossSlopeLogger batch and epoch logging`

- [ ] **Commit 4**: Configure CurvatureMonitor settings
  - Edit `src/deconcnn/callbacks/curvature_monitor.py`
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
  - Run `lint_fix` then commit: `feat: add memory optimization to CurvatureMonitor`

- [ ] **Commit 5**: Add unit tests for CurvatureMonitor changes
  - Create `tests/test_curvature_monitor.py` (or update existing)
  - Test subsampling functionality:
    ```python
    def test_subsampling():
        """Test that subsampling reduces dataset size correctly"""
        monitor = CurvatureMonitor(subsample_ratio=0.1)
        dataset_size = 1000
        subsampled_size = monitor._get_subsample_size(dataset_size)
        assert subsampled_size == 100
    ```
  - Test memory bounds:
    ```python
    def test_max_batch_size_limit():
        """Test that batch size is limited correctly"""
        monitor = CurvatureMonitor(max_batch_size=32)
        original_batch_size = 128
        limited_size = monitor._limit_batch_size(original_batch_size)
        assert limited_size == 32
    ```
  - Test compute frequency:
    ```python
    def test_compute_frequency():
        """Test that computation happens every N steps"""
        monitor = CurvatureMonitor(compute_every_n_steps=500)
        assert monitor.should_compute(global_step=499) is False
        assert monitor.should_compute(global_step=500) is True
        assert monitor.should_compute(global_step=501) is False
    ```
  - Run `lint_fix` then commit: `test: add unit tests for CurvatureMonitor memory optimization`

- [ ] **Commit 6**: Integration test CurvatureMonitor
  - Test with: `uv run python scripts/train_cnn.py machine=mac epochs=1 batch_size=128`
  - Verify metrics computed exactly every 500 steps
  - Monitor memory usage during computation
  - Run `lint_fix` then commit: `test: verify CurvatureMonitor 500-step frequency`

- [ ] **Commit 7**: Configure NoiseMonitor settings
  - Update NoiseMonitor configuration in callback YAML:
    ```yaml
    noise_monitor:
      _target_: deconcnn.callbacks.noise_monitor.NoiseMonitor
      compute_every_epochs: 2
      log_epoch_freq: 1
    ```
  - Set checkpoint save_top_k=3 in trainer defaults
  - Ensure PSD tail, gradient variance use correct frequencies
  - Run `lint_fix` then commit: `feat: configure NoiseMonitor and checkpoint settings`

- [ ] **Commit 8**: Add unit tests for NoiseMonitor configuration
  - Create `tests/test_noise_monitor_config.py`
  - Test configuration loading:
    ```python
    def test_noise_monitor_frequency():
        """Test NoiseMonitor computes every 2 epochs"""
        config = load_callback_config('noise_monitor')
        assert config['compute_every_epochs'] == 2
        assert config['log_epoch_freq'] == 1
    ```
  - Test checkpoint configuration:
    ```python
    def test_checkpoint_save_top_k():
        """Test checkpoint saves only top 3"""
        trainer_config = load_trainer_defaults()
        assert trainer_config['checkpoint']['save_top_k'] == 3
    ```
  - Run `lint_fix` then commit: `test: add unit tests for NoiseMonitor configuration`

- [ ] **Commit 9**: Integration test all callbacks
  - Test all callbacks together: LossSlopeLogger + DrExpMetricsCallback + CurvatureMonitor + NoiseMonitor
  - Verify no metric name conflicts
  - Check total memory usage
  - Test with: `uv run python scripts/train_cnn.py machine=mac epochs=4 batch_size=32 enable_checkpointing=true`
  - Run `lint_fix` then commit: `test: validate callback integration`

## Phase 2: Experiment Configurations (5 commits)

- [ ] **Commit 10**: Create base experiment config
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
      - /machine: cluster  # Override with machine=mac for local testing
    
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
  - Run `lint_fix` then commit: `feat: create base experiment configuration`

- [ ] **Commit 11**: Test base configuration
  - Load and validate config structure
  - Run 1 epoch test: `uv run python scripts/train_cnn.py machine=mac +experiment=loss_lin_slope_base epochs=1 seed=0 optim.lr=0.1 optim.weight_decay=1e-4`
  - Verify validation runs 4 times per epoch
  - Run `lint_fix` then commit: `test: verify base experiment configuration`

- [ ] **Commit 12**: Create BN-off variant
  - Create `configs/experiment/loss_lin_slope_bn_off.yaml`:
    ```yaml
    defaults:
      - loss_lin_slope_base
    
    model:
      norm_type: none  # Disable BN
    
    trainer:
      gradient_clip_val: 1.0  # Critical for stability without BN
    ```
  - Run `lint_fix` then commit: `feat: create BN-off experiment variant`

- [ ] **Commit 13**: Create narrow and AdamW variants
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
  - Run `lint_fix` then commit: `feat: create narrow and AdamW variants`

- [ ] **Commit 14**: Create unified callback config
  - Create `configs/callbacks/loss_lin_slope_metrics.yaml`
  - Consolidate all callback settings in one place
  - Update experiment configs to reference it
  - Run `lint_fix` then commit: `feat: create unified callback configuration`

## Phase 3: Local Validation (5 commits)

- [ ] **Commit 15**: Create validation script
  - Create `scripts/validate_local.py`
  - Quick 3-epoch test for single configuration
  - Include machine=mac parameter for local testing
  - Check output formats and logging
  - Example command in script:
    ```python
    cmd = [
        "uv", "run", "python", "scripts/train_cnn.py",
        "machine=mac",  # Critical for Mac testing
        "+experiment=loss_lin_slope_base",
        "epochs=3",
        "seed=0",
        "optim.lr=0.1",
        "optim.weight_decay=1e-4"
    ]
    ```
  - Run `lint_fix` then commit: `feat: create local validation script`

- [ ] **Commit 16**: Add unit tests for validation script
  - Create `tests/test_validate_local.py`
  - Test configuration validation:
    ```python
    def test_validate_config_structure():
        """Test that config validation catches missing keys"""
        config = {"epochs": 3}  # Missing required keys
        validator = LocalValidator()
        with pytest.raises(ValueError, match="Missing required"):
            validator.validate_config(config)
    ```
  - Test output format checking:
    ```python
    def test_validate_output_format():
        """Test output format validation"""
        outputs = {"loss": [1.0, 0.9], "accuracy": [0.5, 0.6]}
        validator = LocalValidator()
        assert validator.check_output_format(outputs) is True
    ```
  - Test metric presence validation:
    ```python
    def test_required_metrics_present():
        """Test that all required metrics are logged"""
        required = ["loss_slope/alpha_5_15", "loss_slope/grad_norm_periodic"]
        logged = {"loss_slope/alpha_5_15": 0.1, "other": 0.2}
        validator = LocalValidator()
        missing = validator.find_missing_metrics(required, logged)
        assert "loss_slope/grad_norm_periodic" in missing
    ```
  - Run `lint_fix` then commit: `test: add unit tests for validation script`

- [ ] **Commit 17**: Create test harness
  - Create `scripts/test_harness.py`
  - Test all 4 variants sequentially with machine=mac
  - Report pass/fail for each
  - Run `lint_fix` then commit: `feat: create test harness for all variants`

- [ ] **Commit 18**: Add unit tests for test harness
  - Create `tests/test_test_harness.py`
  - Test variant loading:
    ```python
    def test_load_all_variants():
        """Test that all 4 variants can be loaded"""
        harness = TestHarness()
        variants = harness.get_variants()
        assert len(variants) == 4
        assert set(variants) == {"base", "bn_off", "narrow", "adamw"}
    ```
  - Test failure detection:
    ```python
    def test_detect_training_failure():
        """Test that training failures are detected"""
        harness = TestHarness()
        result = {"status": "failed", "error": "OOM"}
        assert harness.is_failure(result) is True
    ```
  - Test report generation:
    ```python
    def test_generate_report():
        """Test report generation with mixed results"""
        results = {
            "base": {"status": "success"},
            "bn_off": {"status": "failed", "error": "gradient explosion"}
        }
        harness = TestHarness()
        report = harness.generate_report(results)
        assert "1/2 variants failed" in report
    ```
  - Run `lint_fix` then commit: `test: add unit tests for test harness`

- [ ] **Commit 19**: Create validation notebook and comprehensive tests
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
  - Run `lint_fix` then commit: `feat: create metric validation notebook and unit tests`

## Phase 4: Job Management (5 commits)

- [ ] **Commit 20**: Create submission wrapper script
  - Create `scripts/submit_all_loss_slope.sh`:
    ```bash
    #!/bin/bash
    # Submit all 216 jobs using existing infrastructure
    
    # Configuration
    VARIANTS="base bn_off narrow adamw"
    LRS="0.05,0.1,0.2"
    WDS="1e-4,1e-3,1e-2"
    SEEDS="0,1,2,3,4,5"
    
    # Submit jobs for each variant
    for variant in $VARIANTS; do
        echo "Submitting $variant variant (54 jobs)..."
        uv run python scripts/submit_experiments.py sweep \
          configs/experiment/loss_lin_slope_${variant}.yaml \
          --param lr=$LRS \
          --param weight_decay=$WDS \
          --param seed=$SEEDS \
          --experiment loss_lin_slope \
          --priority 100
        
        # Add small delay to avoid overwhelming the system
        sleep 2
    done
    
    echo "All 216 jobs submitted!"
    ```
  - Make script executable: `chmod +x scripts/submit_all_loss_slope.sh`
  - Test with dry run first:
    ```bash
    # Modify script temporarily to add --dry-run flag
    uv run python scripts/submit_experiments.py sweep \
      configs/experiment/loss_lin_slope_base.yaml \
      --param lr=0.05,0.1,0.2 \
      --param weight_decay=1e-4,1e-3,1e-2 \
      --param seed=0,1,2,3,4,5 \
      --dry-run
    ```
  - Run `lint_fix` then commit: `feat: create submission wrapper for 216-job sweep`

- [ ] **Commit 21**: Create monitoring script
  - Create `scripts/monitor_experiment.py`
  - Implement job status tracking using dr_exp API:
    ```python
    #!/usr/bin/env python
    """Monitor deconCNN experiment progress."""
    
    import numpy as np
    from datetime import datetime
    
    from deconcnn.dr_exp_utils import list_decon_jobs
    
    def monitor_experiment(experiment_name: str = "loss_lin_slope"):
        """Monitor job status for loss slope experiment."""
        # Track: jobs queued/running/completed/failed
        jobs = list_decon_jobs(experiment=experiment_name)
        
        status = {
            'queued': sum(1 for j in jobs if j['status'] == 'queued'),
            'running': sum(1 for j in jobs if j['status'] == 'running'),
            'completed': sum(1 for j in jobs if j['status'] == 'completed'),
            'failed': sum(1 for j in jobs if j['status'] == 'failed')
        }
        
        print(f"Experiment: {experiment_name}")
        print(f"Total jobs: {len(jobs)}")
        print(f"Status breakdown: {status}")
        return status
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
  - Run `lint_fix` then commit: `feat: create experiment monitoring script`

### Python Script Template for All Scripts
For consistency, all Python scripts in `scripts/` should follow this pattern:
```python
#!/usr/bin/env python
"""Script description."""

from deconcnn.dr_exp_utils import list_decon_jobs, submit_training_job
# ... other imports as needed
```

- [ ] **Commit 22**: Add unit tests for monitoring script
  - Create `tests/test_monitor_experiment.py`
  - Test job status tracking:
    ```python
    def test_job_status_counting():
        """Test correct counting of job statuses"""
        mock_jobs = [
            {"status": "completed"}, {"status": "running"},
            {"status": "completed"}, {"status": "failed"}
        ]
        monitor = ExperimentMonitor()
        status = monitor.count_statuses(mock_jobs)
        assert status["completed"] == 2
        assert status["running"] == 1
        assert status["failed"] == 1
    ```
  - Test data quality checks:
    ```python
    def test_quality_check_loss_range():
        """Test loss range validation"""
        monitor = ExperimentMonitor()
        assert monitor.check_loss_range(2.5) is True  # Valid
        assert monitor.check_loss_range(15.0) is False  # Too high
        assert monitor.check_loss_range(-1.0) is False  # Negative
    ```
  - Test alert conditions:
    ```python
    def test_high_failure_rate_alert():
        """Test failure rate alert trigger"""
        monitor = ExperimentMonitor()
        status = {"total": 100, "failed": 15}
        assert monitor.should_alert_failure_rate(status) is True
        
        status = {"total": 100, "failed": 5}
        assert monitor.should_alert_failure_rate(status) is False
    ```
  - Test ETA calculation:
    ```python
    def test_eta_calculation():
        """Test ETA calculation based on completed jobs"""
        monitor = ExperimentMonitor()
        completed_times = [300, 320, 310]  # seconds
        remaining_jobs = 10
        eta = monitor.calculate_eta(completed_times, remaining_jobs)
        assert 3000 < eta < 3300  # ~10 jobs * ~310 seconds
    ```
  - Run `lint_fix` then commit: `test: add unit tests for experiment monitoring`

- [ ] **Commit 23**: Create failure recovery script
  - Create `scripts/recover_failed.py`
  - Detect OOM, timeout, gradient explosion
  - Implement retry logic with parameter adjustment
  - Create `docs/operational_runbook_basic.md`
  - Run `lint_fix` then commit: `feat: create failure recovery system`

- [ ] **Commit 24**: Add unit tests for failure recovery
  - Create `tests/test_recover_failed.py`
  - Test failure detection:
    ```python
    def test_detect_oom_failure():
        """Test OOM detection from error logs"""
        recovery = FailureRecovery()
        error_log = "RuntimeError: CUDA out of memory"
        failure_type = recovery.detect_failure_type(error_log)
        assert failure_type == "OOM"
    ```
  - Test parameter adjustment logic:
    ```python
    def test_adjust_params_for_oom():
        """Test parameter adjustment for OOM failures"""
        recovery = FailureRecovery()
        original_params = {"batch_size": 128, "workers": 4}
        adjusted = recovery.adjust_for_oom(original_params)
        assert adjusted["batch_size"] == 64  # Halved
        assert adjusted["workers"] == 2  # Reduced
    ```
  - Test retry decision logic:
    ```python
    def test_should_retry_decision():
        """Test retry decision based on failure type"""
        recovery = FailureRecovery()
        assert recovery.should_retry("OOM", attempts=1) is True
        assert recovery.should_retry("OOM", attempts=4) is False  # Max 3
        assert recovery.should_retry("unknown", attempts=1) is False
    ```
  - Test runbook generation:
    ```python
    def test_generate_runbook_entry():
        """Test operational runbook entry generation"""
        recovery = FailureRecovery()
        failure = {
            "job_id": "123", 
            "error": "OOM",
            "action": "reduce batch size"
        }
        entry = recovery.generate_runbook_entry(failure)
        assert "job_id: 123" in entry
        assert "Action taken: reduce batch size" in entry
    ```
  - Run `lint_fix` then commit: `test: add unit tests for failure recovery`

## Phase 5: Data Pipeline (5 commits)

- [ ] **Commit 25**: Create collection script
  - Create `scripts/collect_and_archive.sh`
  - Gather results from scratch directory
  - Organize by experiment variant
  - Run `lint_fix` then commit: `feat: create result collection script`

- [ ] **Commit 26**: Create verification script
  - Create `scripts/verify_completeness.py`
  - Check all 216 runs completed
  - Report missing or failed runs
  - Run `lint_fix` then commit: `feat: create completeness verification script`

- [ ] **Commit 27**: Add unit tests for verification script
  - Create `tests/test_verify_completeness.py`
  - Test completeness checking:
    ```python
    def test_find_missing_runs():
        """Test detection of missing runs"""
        verifier = CompletenessVerifier()
        expected = [(lr, wd, seed) for lr in [0.05, 0.1] 
                    for wd in [1e-4] for seed in [0, 1]]
        actual = [(0.05, 1e-4, 0), (0.1, 1e-4, 1)]  # Missing one
        missing = verifier.find_missing(expected, actual)
        assert len(missing) == 1
        assert (0.05, 1e-4, 1) in missing
    ```
  - Test report generation:
    ```python
    def test_completeness_report():
        """Test completeness report generation"""
        verifier = CompletenessVerifier()
        results = {"completed": 214, "failed": 2, "missing": 0}
        report = verifier.generate_report(results)
        assert "214/216 runs completed" in report
        assert "2 runs failed" in report
    ```
  - Run `lint_fix` then commit: `test: add unit tests for completeness verification`

- [ ] **Commit 28**: Create export script
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
  - Create CSV export with proper data types:
    ```python
    def export_results(data, output_path):
        """Export to CSV with proper data types"""
        df = pd.DataFrame(data)
        
        # Ensure proper types
        df['seed'] = df['seed'].astype(int)
        df['lr'] = df['lr'].astype(float)
        df['weight_decay'] = df['weight_decay'].astype(float)
        df['r_squared'] = df['r_squared'].astype(float)
        
        # Save with explicit float format to avoid scientific notation
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        # Also save metadata
        metadata = {
            'total_runs': len(df),
            'export_date': datetime.now().isoformat(),
            'columns': list(df.columns),
            'column_types': {col: str(df[col].dtype) for col in df.columns}
        }
        with open(output_path.replace('.csv', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    ```
  - Include data dictionary documenting each column in README
  - Save time series data as separate CSV files per run
  - Run `lint_fix` then commit: `feat: create dataset export script`

- [ ] **Commit 29**: Add unit tests for export script
  - Create `tests/test_prepare_dataset.py`
  - Test schema validation:
    ```python
    def test_validate_output_schema():
        """Test that output schema matches expected columns"""
        exporter = DatasetExporter()
        data = {"run_id": "123", "lr": 0.1}  # Missing required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            exporter.validate_schema(data)
    ```
  - Test data type conversion:
    ```python
    def test_convert_data_types():
        """Test proper data type conversion"""
        exporter = DatasetExporter()
        raw_data = {"lr": "0.1", "seed": "42", "r_squared": "0.98"}
        converted = exporter.convert_types(raw_data)
        assert isinstance(converted["lr"], float)
        assert isinstance(converted["seed"], int)
        assert converted["r_squared"] == 0.98
    ```
  - Test CSV export:
    ```python
    def test_export_to_csv():
        """Test CSV file generation"""
        exporter = DatasetExporter()
        test_data = [{"run_id": "1", "lr": 0.1, "seed": 0}]
        output_path = Path("test_results.csv")
        exporter.export_results(test_data, output_path)
        
        # Verify CSV exists and can be read
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]['lr'] == 0.1
        
        # Verify metadata file exists
        metadata_path = output_path.with_suffix('').with_suffix('_metadata.json')
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata['total_runs'] == 1
    ```
  - Test float formatting:
    ```python
    def test_float_formatting():
        """Test that floats avoid scientific notation"""
        exporter = DatasetExporter()
        test_data = [{"lr": 0.00001, "weight_decay": 1e-6}]
        output_path = Path("test_floats.csv")
        exporter.export_results(test_data, output_path)
        
        # Read as text to check formatting
        with open(output_path) as f:
            content = f.read()
        assert "0.000010" in content  # Not 1e-05
        assert "0.000001" in content  # Not 1e-06
    ```
  - Run `lint_fix` then commit: `test: add unit tests for dataset export`

## Phase 6: Cluster Execution (2 commits - MANUAL EXECUTION)

**Note: These commits require cluster access and will be executed manually**

- [ ] **Commit 30**: Resource verification (MANUAL)
  - Create `scripts/verify_resource_config.py`
  - Define resource profiles (fixed at 2 workers per GPU):
    ```python
    RESOURCE_PROFILES = {
        'base': {'gpu_memory_gb': 4, 'workers': 2},
        'bn_off': {'gpu_memory_gb': 5, 'workers': 2},  # More memory, less stable
        'narrow': {'gpu_memory_gb': 2, 'workers': 2},  # 0.5x width = less memory
        'adamw': {'gpu_memory_gb': 6, 'workers': 2},   # Optimizer states need memory
    }
    ```
  - Verification protocol:
    ```python
    VERIFY_CONFIG = {
        'workers': 2,  # Fixed for all variants
        'variants': ['base', 'bn_off', 'narrow', 'adamw']
    }
    ```
  - Success criteria:
    - No OOM errors with 2 workers per GPU
    - GPU utilization > 80%
    - Job completion < 30 minutes for 50 epochs
    - Memory usage within RTX8000 48GB limit
  - Verify all variants run successfully with 2 workers per GPU
  - Run `lint_fix` then commit: `test: verify resource allocation on cluster`

- [ ] **Commit 31**: Execute experiment sweep (MANUAL)
  - Step 1: Submit jobs to dr_exp queue using wrapper script:
    ```bash
    # Submit all 216 jobs to dr_exp queue
    ./scripts/submit_all_loss_slope.sh
    
    # Verify jobs are queued
    uv run python scripts/submit_experiments.py list --status queued | grep -c "queued"
    # Should show 216 queued jobs
    ```
  - Step 2: Launch SLURM workers (CRITICAL - uses embedded parameters):
    ```bash
    # Create temporary SLURM script with embedded values
    cat > /tmp/loss_lin_slope_workers_$$.sbatch << 'EOF'
    #!/bin/bash
    #SBATCH --job-name=loss_lin_slope_workers
    #SBATCH --output=/scratch/ddr8143/logs/slurm_logs/%x_%j.out
    #SBATCH --error=/scratch/ddr8143/logs/slurm_logs/%x_%j.err
    #SBATCH --time=12:00:00
    #SBATCH --gres=gpu:rtx8000:8
    #SBATCH --mem=160G
    #SBATCH --cpus-per-task=48
    #SBATCH --account=cds
    
    # Setup CUDA MPS for GPU sharing
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    
    # Launch dr_exp workers in container
    singularity exec --nv --overlay $SCRATCH/drexp.ext3:ro \
      /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
      /bin/bash -c "
        source /scratch/ddr8143/repos/dr_exp/.venv/bin/activate
        cd /scratch/ddr8143/repos/dr_exp
        source .env
        
        # Launch workers (2 per GPU = 16 total workers)
        uv run dr_exp --base-path /scratch/ddr8143/repos/deconCNN \
          --experiment loss_lin_slope \
          system launcher --workers-per-gpu 2 --max-hours 11
      "
    EOF
    
    # Submit SLURM job
    sbatch /tmp/loss_lin_slope_workers_$$.sbatch
    rm /tmp/loss_lin_slope_workers_$$.sbatch
    ```
  - Monitor execution:
    ```bash
    # Check SLURM job status
    squeue -u $USER
    
    # Monitor dr_exp queue progress
    watch -n 30 "uv run python scripts/submit_experiments.py list --status all | grep -c completed"
    
    # Check for failures
    uv run python scripts/monitor_experiment.py --show-failed
    ```
  - Expected timeline: ~12 hours with 16 workers (2 per GPU × 8 GPUs)
  - Collect results using `scripts/collect_and_archive.sh`
  - Verify completeness: All 216 runs should have R² values and slope metrics
  - Document in commit: total runtime, failure rate, resource usage
  - Run `lint_fix` then commit: `chore: execute 216-job loss slope sweep`

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
- [ ] **Commit 9**: All callbacks work together without conflicts
- [ ] **Commit 11**: Base experiment config loads without errors
- [ ] **Commit 14**: All 4 variants configured correctly
- [ ] **Commit 19**: Metric validation notebook confirms proper logging formats
- [ ] **Commit 29**: All scripts created and tested locally
- [ ] **Commit 30**: Resource testing prevents OOMs on cluster
- [ ] **Commit 31**: 216-job sweep successfully submitted and completed

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
- [ ] Dataset in CSV format with metadata JSON
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
- **All variants**: 2 workers per GPU (fixed configuration)
- **Base variant**: ~4GB per worker
- **BN-off variant**: ~5GB per worker (gradient clipping overhead)
- **Narrow variant**: ~2GB per worker (0.5x width)
- **AdamW variant**: ~6GB per worker (optimizer states)

### Critical Paths:
- **Working directory**: `/Users/daniellerothermel/drotherm/repos/deconCNN`
- **Training script**: `scripts/train_cnn.py`
- **Submission script**: `scripts/submit_experiments.py sweep`
- **Callback configs**: `configs/callbacks/`
- **Experiment configs**: `configs/experiment/` (empty, ready for new configs)
- **SLURM account**: CDS
- **Scratch path**: `/scratch/ddr8143/`
- **Container**: Singularity with overlay at `/scratch/work/public/singularity/`
- **dr_exp integration**: Already complete via dr_exp_callback.py

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
- **❌ Missing**: LossSlopeLogger
- **🔄 Updated**: All paths reference deconCNN repository

## Risk Mitigations

1. **Memory Issues**: Fixed 2 workers per GPU for all variants
2. **Gradient Explosions**: grad_clip_norm=1.0 for BN-off
3. **Data Loss**: Checkpoint frequently, dual format logging
4. **Incomplete Runs**: Recovery script with adaptive strategies
5. **Analysis Gaps**: Ensure proper data formatting for Tier 4
6. **Callback Conflicts**: Test all callbacks together before cluster deployment
7. **CurvatureMonitor Memory**: Use subsampling to control memory usage
8. **Experiment Failure Risk**: Automated failure recovery prevents job loss during 216-job execution
9. **Data Corruption Risk**: Real-time data quality monitoring prevents wasted compute on corrupted experiments

## Handoff Instructions

To continue this implementation:
1. Complete the "Pre-Implementation Checklist" section
2. Check "Current Status" section for latest progress
3. Use TodoWrite to plan your session
4. Find first unchecked [ ] item (start with Phase 0 pre-setup)
5. Implement with quality gates (lint_fix before every commit)
6. Update "Current Status" section when stopping

### **CRITICAL DEPENDENCIES:**
- Phase 0 → Commit 1 (environment must be verified first)
- Commit 1 → Commit 2 (must test LossSlopeLogger before proceeding)
- Commit 6 → Commit 7 (callbacks must work before creating configs)
- Commit 14 → Commit 15 (validation must pass before job generation)
- Commit 21 → Commit 22 (all scripts ready before cluster testing)

### **Repository Context for Each Commit:**
- Commits 1-21: Execute in deconCNN repository (local development)
- Commits 22-23: Execute from deconCNN on cluster (GPU required, MANUAL)

### **Script Execution Pattern:**
All Python scripts should be executed with `uv run` from repository root:
- `uv run python scripts/monitor_experiment.py`
- `uv run python scripts/recover_failed.py`
- `uv run python scripts/verify_completeness.py`
- `uv run python scripts/prepare_dataset.py`
- `uv run python scripts/validate_local.py`
- `uv run python scripts/test_harness.py`

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

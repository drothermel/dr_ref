# Tier 3 Implementation Plan: Job Setup and Critical Logging Infrastructure
## Date: 2025-07-02

This document provides an atomic implementation plan that gets from 0 to running 216-job sweep in 29 focused commits, ensuring each change is testable and reversible with comprehensive unit testing.

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

- [ ] **Pre-Setup**: Verify environment and create validation script
  - First, create `scripts/validate_tier3_setup.py`:
    ```python
    """Pre-flight validation for Tier 3 implementation."""
    
    import sys
    from pathlib import Path
    
    from omegaconf import OmegaConf
    
    
    def check_dependencies() -> tuple[bool, list[str]]:
        """Check if required dependencies are installed."""
        required = [
            'pandas', 'matplotlib', 'statsmodels', 'rich',
            'pyhessian', 'scipy', 'hessian_eigenthings'
        ]
        
        missing = []
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        return len(missing) == 0, missing
    
    
    def validate_basemonitor() -> tuple[bool, str]:
        """Validate BaseMonitor exists with expected interface."""
        try:
            from deconcnn.callbacks.base_monitor import BaseMonitor
            
            # Verify it's a Lightning callback
            import lightning.pytorch as pl
            if not issubclass(BaseMonitor, pl.Callback):
                return False, "BaseMonitor not a Lightning Callback"
                
            return True, "BaseMonitor validated"
            
        except ImportError as e:
            return False, f"Cannot import BaseMonitor: {e}"
    
    
    def check_dr_exp() -> tuple[bool, str]:
        """Check dr_exp integration."""
        try:
            from deconcnn.dr_exp_utils import list_decon_jobs, submit_training_job
            # The dr_exp_utils module already handles ImportError gracefully
            return True, "dr_exp_utils available"
        except ImportError:
            return False, "Cannot import dr_exp_utils"
    
    
    def validate_configs() -> tuple[bool, list[str]]:
        """Check if required config directories exist."""
        config_root = Path("configs")
        required_dirs = ["callbacks", "experiment", "model", "optim", "lrsched"]
        
        missing = []
        for dir_name in required_dirs:
            if not (config_root / dir_name).exists():
                missing.append(f"configs/{dir_name}")
        
        return len(missing) == 0, missing
    
    
    def main():
        """Run validation checks."""
        print(f"Working directory: {Path.cwd()}")
        print("Running pre-flight validation...\n")
        
        all_ok = True
        
        # Check dependencies
        deps_ok, missing = check_dependencies()
        print(f"✓ Dependencies: {'PASS' if deps_ok else 'FAIL'}")
        if not deps_ok:
            print(f"  Missing: {', '.join(missing)}")
            print(f"  Run: uv add {' '.join(missing)}")
            all_ok = False
        
        # Check BaseMonitor
        base_ok, msg = validate_basemonitor()
        print(f"✓ BaseMonitor: {'PASS' if base_ok else 'FAIL'} - {msg}")
        all_ok &= base_ok
        
        # Check dr_exp
        dr_ok, msg = check_dr_exp()
        print(f"✓ dr_exp: {'PASS' if dr_ok else 'FAIL'} - {msg}")
        all_ok &= dr_ok
        
        # Check configs
        config_ok, missing = validate_configs()
        print(f"✓ Config dirs: {'PASS' if config_ok else 'FAIL'}")
        if not config_ok:
            print(f"  Missing: {', '.join(missing)}")
            all_ok = False
        
        # Test basic training if all checks pass
        if all_ok:
            print("\nTesting basic training...")
            import subprocess
            result = subprocess.run(
                ["uv", "run", "python", "scripts/train_cnn.py", 
                 "machine=mac", "trainer.fast_dev_run=true"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✓ Training test: PASS")
            else:
                print("✓ Training test: FAIL")
                print(f"  Error: {result.stderr}")
                all_ok = False
        
        print(f"\n{'✅ Ready to proceed!' if all_ok else '❌ Fix issues before proceeding'}")
        return 0 if all_ok else 1
    
    
    if __name__ == "__main__":
        sys.exit(main())
    ```
  - Run the validation script: `uv run python scripts/validate_tier3_setup.py`
  - Fix any issues identified by the script
  - Run `lint_fix` then commit: `chore: add pre-flight validation script`

## Phase 1: Library Extraction & Refactoring (5 commits)

**Strategy**: Extract existing implementations from NoiseMonitor and CurvatureMonitor into library modules first, then create new callbacks using the library. This avoids duplicate implementations and ensures consistency from the start.

- [ ] **Commit 1**: Extract analysis functions from NoiseMonitor and create library
  - Create `src/deconcnn/analysis/__init__.py` (empty with docstring)
  - Create `src/deconcnn/analysis/utils.py` and extract EWMA from NoiseMonitor:
    ```python
    import numpy as np
    from typing import Tuple
    
    def ewma_update(current_value: float, 
                    running_mean: float, 
                    running_var: float, 
                    alpha: float = 0.1) -> Tuple[float, float]:
        """Update exponential weighted moving average.
        
        Extracted from NoiseMonitor lines 67-72.
        """
        new_mean = alpha * current_value + (1 - alpha) * running_mean
        diff = current_value - running_mean
        new_var = alpha * (diff ** 2) + (1 - alpha) * running_var
        return float(new_mean), float(new_var)
    ```
  - Create `src/deconcnn/analysis/fitting.py` and extract power law fitting:
    ```python
    import numpy as np
    from typing import Tuple
    
    def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Fit power law y = a * x^b using log-log linear regression.
        
        Extracted from NoiseMonitor lines 102-107.
        """
        mask = (x > 0) & (y > 0)
        if np.sum(mask) < 2:
            return np.nan, np.nan, 0.0
            
        log_x = np.log(x[mask])
        log_y = np.log(y[mask])
        
        coeffs = np.polyfit(log_x, log_y, deg=1)
        slope, log_intercept = coeffs
        
        # Calculate R²
        y_pred = np.polyval(coeffs, log_x)
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return float(slope), float(log_intercept), float(r_squared)
    ```
  - Create `src/deconcnn/analysis/spectral.py` and extract PSD analysis:
    ```python
    import numpy as np
    from typing import Dict, Tuple
    
    def compute_psd_metrics(signal: np.ndarray, 
                          low_freq_range: Tuple[float, float] = (0.1, 0.3),
                          high_freq_range: Tuple[float, float] = (0.3, 0.5)) -> Dict[str, float]:
        """Compute power spectral density metrics.
        
        Extracted from NoiseMonitor lines 114-131.
        """
        if len(signal) < 4:
            return {'psd_tail': np.nan, 'total_power': np.nan}
            
        signal = signal - np.mean(signal)
        window = np.hanning(len(signal))
        signal = signal * window
        
        fft = np.fft.rfft(signal)
        psd = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(signal))
        
        low_mask = (freqs >= low_freq_range[0]) & (freqs <= low_freq_range[1])
        high_mask = (freqs >= high_freq_range[0]) & (freqs <= high_freq_range[1])
        
        low_power = np.sum(psd[low_mask])
        high_power = np.sum(psd[high_mask])
        
        psd_tail = high_power / (low_power + 1e-10)
        total_power = np.sum(psd)
        
        return {
            'psd_tail': float(psd_tail),
            'total_power': float(total_power)
        }
    ```
  - Immediately refactor NoiseMonitor to use these library functions:
    - Replace EWMA implementation with `from deconcnn.analysis.utils import ewma_update`
    - Replace power law fitting with `from deconcnn.analysis.fitting import fit_power_law`
    - Replace PSD computation with `from deconcnn.analysis.spectral import compute_psd_metrics`
  - Create comprehensive tests in `tests/test_analysis_utils.py`, `tests/test_analysis_fitting.py`, and `tests/test_analysis_spectral.py`
  - Run `lint_fix` then commit: `refactor: extract NoiseMonitor functions to analysis library`

- [ ] **Commit 2**: Extract Hutchinson trace from CurvatureMonitor
  - Create `src/deconcnn/analysis/curvature.py` and extract Hutchinson trace:
    ```python
    import torch
    from typing import Tuple, List
    import numpy as np
    
    def hutchinson_trace(model: torch.nn.Module, 
                        loss_fn: callable,
                        data_batch: torch.Tensor,
                        target_batch: torch.Tensor,
                        num_samples: int = 50) -> float:
        """Estimate Hessian trace using Hutchinson's method.
        
        Extracted from CurvatureMonitor lines 63-121.
        """
        trace_estimates = []
        
        for _ in range(num_samples):
            # Rademacher random vector
            z = torch.randint_like(
                torch.randn(sum(p.numel() for p in model.parameters())), 
                low=0, high=2, dtype=torch.float32
            ) * 2 - 1
            
            # Compute loss and gradients
            loss = loss_fn(model(data_batch), target_batch)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            
            # Flatten and compute Hessian-vector product
            grad_vec = torch.cat([g.flatten() for g in grads])
            hz = torch.autograd.grad(grad_vec, model.parameters(), z, retain_graph=True)
            hz_flat = torch.cat([h.flatten() for h in hz])
            
            # Compute z^T H z
            trace_estimates.append(torch.dot(z, hz_flat).item())
            
        return float(np.mean(trace_estimates))
    ```
  - Immediately refactor CurvatureMonitor to use library function:
    - Replace Hutchinson implementation with `from deconcnn.analysis.curvature import hutchinson_trace`
    - Update method calls to use the library function
  - Create tests in `tests/test_analysis_curvature.py`:
    ```python
    def test_hutchinson_trace_estimation():
        """Test Hutchinson trace against known matrices."""
        # Create simple quadratic loss with known Hessian
        model = SimpleQuadraticModel()
        # Test that trace estimation is accurate
        estimated_trace = hutchinson_trace(model, loss_fn, data, targets)
        assert abs(estimated_trace - true_trace) < 0.1 * true_trace
    ```
  - Run `lint_fix` then commit: `refactor: extract CurvatureMonitor functions to analysis library`

- [ ] **Commit 3**: Create slope calculation in library
  - Add to `src/deconcnn/analysis/utils.py`:
    ```python
    def calculate_slope(losses: Dict[int, List[float]], 
                       start: int = 5, 
                       end: int = 15) -> Optional[float]:
        """Calculate slope of loss curve using least squares.
        
        Args:
            losses: Dictionary mapping epoch to list of batch losses
            start: Start epoch for slope calculation (inclusive)
            end: End epoch for slope calculation (exclusive)
            
        Returns:
            Slope value or None if insufficient data
        """
        epochs = []
        mean_losses = []
        
        for epoch in range(start, min(end, max(losses.keys()) + 1)):
            if epoch in losses and len(losses[epoch]) > 0:
                epochs.append(epoch)
                mean_losses.append(np.mean(losses[epoch]))
        
        if len(epochs) < 2:
            return None
            
        coeffs = np.polyfit(epochs, mean_losses, deg=1)
        return float(coeffs[0])
    ```
  - Add comprehensive tests to `tests/test_analysis_utils.py`:
    ```python
    def test_slope_calculation_synthetic():
        """Test slope calculation with known linear function."""
        # Generate y = 2.0 - 0.1*x
        losses = {i: [2.0 - 0.1*i + 0.01*np.random.randn() for _ in range(10)] 
                  for i in range(20)}
        
        slope = calculate_slope(losses, start=5, end=15)
        assert slope is not None
        assert abs(slope - (-0.1)) < 0.01
    
    def test_slope_insufficient_data():
        """Test handling of insufficient data."""
        losses = {0: [1.0], 1: [0.9]}
        slope = calculate_slope(losses, start=5, end=15)
        assert slope is None
    ```
  - Run `lint_fix` then commit: `feat: add slope calculation to analysis library`

- [ ] **Commit 4**: Integration test extracted functions
  - Run training with original callbacks to capture baseline metrics
  - Run training with refactored callbacks using library functions
  - Compare outputs to ensure identical behavior:
    ```python
    def test_noise_monitor_refactoring():
        """Verify NoiseMonitor produces same results after refactoring."""
        # Run with original implementation
        original_metrics = run_training_with_original_callbacks()
        
        # Run with refactored implementation
        refactored_metrics = run_training_with_refactored_callbacks()
        
        # Compare key metrics
        for key in ['psd_tail', 'power_law_slope', 'grad_var']:
            assert np.allclose(original_metrics[key], refactored_metrics[key])
    ```
  - Verify CurvatureMonitor still computes every 500 steps
  - Verify NoiseMonitor still computes every 2 epochs
  - Run `lint_fix` then commit: `test: verify callback refactoring maintains behavior`

## Phase 2: Create New Callback (2 commits)

- [ ] **Commit 5**: Create LossSlopeLogger using library
  - Create `src/deconcnn/callbacks/loss_slope_logger.py`:
    ```python
    import torch
    from typing import Dict, List
    from .base_monitor import BaseMonitor
    from deconcnn.analysis.utils import calculate_slope
    
    class LossSlopeLogger(BaseMonitor):
        """Log loss curve slopes and periodic gradient/weight norms."""
        
        def __init__(self, 
                     log_every_n_steps: int = 1,
                     log_epoch_freq: int = 1,
                     slope_window_start: int = 5,
                     slope_window_end: int = 15,
                     debug_mode: bool = True):
            super().__init__(log_every_n_steps, log_epoch_freq, debug_mode)
            self.slope_window_start = slope_window_start
            self.slope_window_end = slope_window_end
            self.epoch_losses: Dict[int, List[float]] = {}
            
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            """Log metrics every batch and quarter epoch."""
            train_loss = trainer.callback_metrics.get("train_loss", None)
            if train_loss is None:
                return
                
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
                
            # Store for slope calculation
            current_epoch = trainer.current_epoch
            if current_epoch not in self.epoch_losses:
                self.epoch_losses[current_epoch] = []
            self.epoch_losses[current_epoch].append(train_loss)
            
            # Quarter-epoch periodic logging
            total_batches = len(trainer.train_dataloader)
            quarter_epoch_interval = max(1, total_batches // 4)
            if batch_idx % quarter_epoch_interval == 0:
                grad_norm = self._compute_gradient_norm(pl_module)
                weight_norm = self._compute_weight_norm(pl_module)
                
                pl_module.log_dict({
                    "loss_slope/grad_norm_periodic": grad_norm,
                    "loss_slope/weight_norm_periodic": weight_norm
                }, on_step=True)
                
        def on_train_epoch_end(self, trainer, pl_module):
            """Calculate and log slope metrics at epoch end."""
            if trainer.current_epoch >= self.slope_window_start:
                # Use library function for slope calculation
                alpha_early = calculate_slope(
                    self.epoch_losses, 
                    start=self.slope_window_start, 
                    end=min(self.slope_window_end, trainer.current_epoch + 1)
                )
                if alpha_early is not None:
                    pl_module.log("loss_slope/alpha_5_15", alpha_early)
    ```
  - Create comprehensive unit tests in same commit:
    ```python
    # tests/test_loss_slope_logger.py
    def test_slope_logger_uses_library():
        """Test that LossSlopeLogger correctly uses library functions."""
        logger = LossSlopeLogger()
        # Test that it calls calculate_slope from library
        # Test metric naming and prefixing
        # Test edge cases
    ```
  - Run `lint_fix` then commit: `feat: create LossSlopeLogger using analysis library`

- [ ] **Commit 6**: Integration test all callbacks
  - Test all callbacks together: LossSlopeLogger + DrExpMetricsCallback + CurvatureMonitor + NoiseMonitor
  - Verify no metric name conflicts
  - Verify each callback's specific behavior:
    - LossSlopeLogger: logs every batch, computes slopes after epoch 5
    - CurvatureMonitor: computes every 500 steps
    - NoiseMonitor: computes every 2 epochs
    - DrExpMetricsCallback: logs to dr_exp format
  - Check total memory usage and performance overhead (<15%)
  - Test with: `uv run python scripts/train_cnn.py machine=mac epochs=4 batch_size=32 enable_checkpointing=true`
  - Run `lint_fix` then commit: `test: validate all callbacks integration`

## Phase 3: Experiment Configurations (5 commits)

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
  - **NOTE on callback registration**: Callbacks listed in the YAML are automatically instantiated by Hydra and registered with the PyTorch Lightning Trainer. No manual registration is needed - the trainer receives all callbacks via the `callbacks` parameter when instantiated.
  - Ensure `configs/callbacks/loss_slope_logger.yaml` exists with proper `_target_` path
  - Run `lint_fix` then commit: `feat: create base experiment configuration`

- [ ] **Commit 8**: Test base configuration
  - Load and validate config structure
  - Run 1 epoch test: `uv run python scripts/train_cnn.py machine=mac +experiment=loss_lin_slope_base epochs=1 seed=0 optim.lr=0.1 optim.weight_decay=1e-4`
  - Verify validation runs 4 times per epoch
  - Run `lint_fix` then commit: `test: verify base experiment configuration`

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
  - Run `lint_fix` then commit: `feat: create BN-off experiment variant`

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
  - Run `lint_fix` then commit: `feat: create narrow and AdamW variants`

- [ ] **Commit 11**: Create unified callback config
  - Create `configs/callbacks/loss_slope_logger.yaml`:
    ```yaml
    _target_: deconcnn.callbacks.loss_slope_logger.LossSlopeLogger
    log_every_n_steps: 1
    log_epoch_freq: 1
    burn_in_epochs: 5
    slope_window_start: 5
    slope_window_end: 15
    debug_mode: ${debug_mode:false}  # Can be overridden
    ```
  - Create `configs/callbacks/loss_lin_slope_metrics.yaml` that includes all callbacks:
    ```yaml
    - gradient_monitor
    - noise_monitor
    - curvature_monitor
    - loss_slope_logger
    - dr_exp_metrics
    ```
  - Update experiment configs to reference the unified list
  - Run `lint_fix` then commit: `feat: create unified callback configuration`

## Phase 4: Local Validation (5 commits)

- [ ] **Commit 13**: Create validation script
  - Create `scripts/validate_local.py`:
    ```python
    """Validate loss slope experiment configuration locally."""
    
    import subprocess
    import sys
    from pathlib import Path
    
    
    def validate_single_config():
        """Run quick validation of single configuration."""
        cmd = [
            "uv", "run", "python", "scripts/train_cnn.py",
            "machine=mac",  # Critical for Mac testing
            "+experiment=loss_lin_slope_base",
            "epochs=3",
            "seed=0",
            "optim.lr=0.1",
            "optim.weight_decay=1e-4"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
            
        # Check for expected output patterns
        output = result.stdout
        checks = {
            "loss_slope/alpha_5_15": "loss_slope/alpha_5_15" in output,
            "grad_norm_periodic": "loss_slope/grad_norm_periodic" in output,
            "validation frequency": output.count("Validation") >= 12  # 4x per epoch * 3 epochs
        }
        
        print("\nValidation results:")
        for check, passed in checks.items():
            print(f"  {check}: {'✓' if passed else '✗'}")
            
        return all(checks.values())
    
    
    def main():
        """Main validation entry point."""
        print(f"Working directory: {Path.cwd()}")
        
        if validate_single_config():
            print("\n✅ Local validation passed!")
            return 0
        else:
            print("\n❌ Local validation failed!")
            return 1
    
    
    if __name__ == "__main__":
        sys.exit(main())
    ```
  - Run `lint_fix` then commit: `feat: create local validation script`

- [ ] **Commit 14**: Add unit tests for validation script
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

- [ ] **Commit 15**: Create test harness
  - Create `scripts/test_harness.py`
  - Test all 4 variants sequentially with machine=mac
  - Report pass/fail for each
  - Run `lint_fix` then commit: `feat: create test harness for all variants`

- [ ] **Commit 16**: Add unit tests for test harness
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

- [ ] **Commit 17**: Create validation notebook and comprehensive tests
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

## Phase 5: Job Management (5 commits)

- [ ] **Commit 18**: Create submission wrapper script
  - Create `scripts/submit_all_loss_slope.sh`:
    ```bash
    #!/bin/bash
    # Submit all 216 jobs for loss slope experiment
    
    # Check we're in the right place
    if [[ ! -f "scripts/submit_experiments.py" ]]; then
        echo "Error: Please run this script from the project root directory"
        echo "Current directory: $(pwd)"
        exit 1
    fi
    
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
        
        # Check submission status
        if [[ $? -ne 0 ]]; then
            echo "Error submitting $variant jobs"
            exit 1
        fi
        
        sleep 2  # Brief pause between variants
    done
    
    echo "All 216 jobs submitted successfully!"
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

- [ ] **Commit 19**: Create monitoring script
  - Create `scripts/monitor_experiment.py`:
    ```python
    """Monitor deconCNN experiment progress."""
    
    from datetime import datetime
    
    import numpy as np
    from rich.console import Console
    from rich.table import Table
    
    from deconcnn.dr_exp_utils import list_decon_jobs
    
    
    def monitor_experiment(experiment_name: str = "loss_lin_slope"):
        """Monitor job status for loss slope experiment."""
        console = Console()
        
        # Get jobs using existing utility
        jobs = list_decon_jobs(experiment=experiment_name)
        
        # Count statuses
        status_counts = {
            'queued': 0,
            'running': 0,
            'completed': 0,
            'failed': 0
        }
        
        for job in jobs:
            status = job.get('status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1
        
        # Display results
        table = Table(title=f"Experiment: {experiment_name}")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta")
        
        for status, count in status_counts.items():
            table.add_row(status.capitalize(), str(count))
        
        console.print(table)
        return status_counts
    
    
    def main():
        """Main entry point."""
        status = monitor_experiment()
        
        # Return non-zero if failures
        return 1 if status.get('failed', 0) > 0 else 0
    
    
    if __name__ == "__main__":
        import sys
        sys.exit(main())
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

### Python Script Import Pattern
For consistency, all Python scripts in `scripts/` should follow this import pattern:
```python
"""Script description."""

import standard_library_modules
from standard_library import specific_items

import third_party_modules
from third_party import specific_items

from deconcnn.module import local_imports
```

- [ ] **Commit 20**: Add unit tests for monitoring script
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

- [ ] **Commit 21**: Create failure recovery script
  - Create `scripts/recover_failed.py`:
    ```python
    """Recover failed jobs with parameter adjustments."""
    
    import json
    from pathlib import Path
    from typing import Dict, List
    
    from deconcnn.dr_exp_utils import list_decon_jobs, submit_training_job
    
    
    class FailureRecovery:
        """Handle recovery of failed experiments."""
        
        def __init__(self):
            self.failure_patterns = {
                'OOM': ['CUDA out of memory', 'OutOfMemoryError'],
                'gradient': ['gradient overflow', 'nan loss', 'inf detected'],
                'timeout': ['TimeoutError', 'SLURM timeout']
            }
            
        def detect_failure_type(self, error_log: str) -> str:
            """Detect type of failure from error log."""
            for failure_type, patterns in self.failure_patterns.items():
                if any(pattern in error_log for pattern in patterns):
                    return failure_type
            return 'unknown'
            
        def adjust_for_oom(self, params: Dict) -> Dict:
            """Adjust parameters for OOM failures."""
            adjusted = params.copy()
            # Halve batch size
            if 'batch_size' in adjusted:
                adjusted['batch_size'] = max(16, adjusted['batch_size'] // 2)
            # Reduce workers
            if 'workers' in adjusted:
                adjusted['workers'] = max(1, adjusted['workers'] // 2)
            return adjusted
            
        def should_retry(self, failure_type: str, attempts: int) -> bool:
            """Determine if job should be retried."""
            max_retries = {'OOM': 3, 'gradient': 2, 'timeout': 1}
            return attempts < max_retries.get(failure_type, 0)
    
    
    def main():
        """Main recovery logic."""
        recovery = FailureRecovery()
        
        # Get failed jobs
        try:
            jobs = list_decon_jobs(experiment="loss_lin_slope", status="failed")
        except ImportError:
            print("dr_exp not available. Cannot recover jobs.")
            return 1
            
        print(f"Found {len(jobs)} failed jobs")
        
        # Process each failed job
        recovered = 0
        for job in jobs:
            # Analyze failure and potentially retry
            # Implementation details...
            pass
            
        print(f"Recovered {recovered} jobs")
        return 0
    
    
    if __name__ == "__main__":
        import sys
        sys.exit(main())
    ```
  - Create `docs/operational_runbook_basic.md` with common failure patterns
  - Run `lint_fix` then commit: `feat: create failure recovery system`

- [ ] **Commit 22**: Add unit tests for failure recovery
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

## Phase 6: Data Pipeline (5 commits)

- [ ] **Commit 23**: Create collection script
  - Create `scripts/collect_and_archive.sh`
  - Gather results from scratch directory
  - Organize by experiment variant
  - Run `lint_fix` then commit: `feat: create result collection script`

- [ ] **Commit 24**: Create verification script
  - Create `scripts/verify_completeness.py`:
    ```python
    """Verify completeness of experiment runs."""
    
    import itertools
    from typing import List, Tuple
    
    from rich.console import Console
    from rich.table import Table
    
    from deconcnn.dr_exp_utils import list_decon_jobs
    
    
    class CompletenessVerifier:
        """Verify all expected runs are complete."""
        
        def __init__(self):
            self.variants = ['base', 'bn_off', 'narrow', 'adamw']
            self.lrs = [0.05, 0.1, 0.2]
            self.wds = [1e-4, 1e-3, 1e-2]
            self.seeds = [0, 1, 2, 3, 4, 5]
            
        def get_expected_runs(self) -> List[Tuple]:
            """Generate all expected run configurations."""
            return list(itertools.product(
                self.variants, self.lrs, self.wds, self.seeds
            ))
            
        def find_missing(self, expected: List[Tuple], actual: List[Tuple]) -> List[Tuple]:
            """Find missing runs."""
            actual_set = set(actual)
            return [run for run in expected if run not in actual_set]
    
    
    def main():
        """Main verification logic."""
        console = Console()
        verifier = CompletenessVerifier()
        
        # Get completed jobs
        try:
            jobs = list_decon_jobs(experiment="loss_lin_slope")
        except ImportError:
            console.print("[red]dr_exp not available. Cannot verify completeness.[/red]")
            return 1
            
        # Extract configurations from completed jobs
        completed_configs = []
        for job in jobs:
            if job['status'] == 'completed':
                # Extract config from job metadata
                # Implementation depends on job structure
                pass
                
        # Check completeness
        expected = verifier.get_expected_runs()
        missing = verifier.find_missing(expected, completed_configs)
        
        # Report results
        console.print(f"\nExpected runs: {len(expected)}")
        console.print(f"Completed runs: {len(completed_configs)}")
        console.print(f"Missing runs: {len(missing)}")
        
        if missing:
            console.print("\n[yellow]Missing configurations:[/yellow]")
            for variant, lr, wd, seed in missing[:10]:  # Show first 10
                console.print(f"  {variant} lr={lr} wd={wd} seed={seed}")
            if len(missing) > 10:
                console.print(f"  ... and {len(missing) - 10} more")
                
        return 0 if not missing else 1
    
    
    if __name__ == "__main__":
        import sys
        sys.exit(main())
    ```
  - Run `lint_fix` then commit: `feat: create completeness verification script`

- [ ] **Commit 25**: Add unit tests for verification script
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

- [ ] **Commit 26**: Create export script
  - Create `scripts/prepare_dataset.py`:
    ```python
    """Prepare dataset for Tier 4 analysis."""
    
    import json
    from datetime import datetime
    from pathlib import Path
    from typing import Dict, List
    
    import pandas as pd
    
    from deconcnn.dr_exp_utils import list_decon_jobs
    
    
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

- [ ] **Commit 27**: Add unit tests for export script
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

## Phase 7: Cluster Execution (2 commits - MANUAL EXECUTION)

**Note: These commits require cluster access and will be executed manually**

- [ ] **Commit 28**: Resource verification (MANUAL)
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

- [ ] **Commit 29**: Execute experiment sweep (MANUAL)
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
- [ ] **Commit 7**: All callbacks work together without conflicts
- [ ] **Commit 9**: Base experiment config loads without errors
- [ ] **Commit 12**: All 4 variants configured correctly
- [ ] **Commit 17**: Metric validation notebook confirms proper logging formats
- [ ] **Commit 27**: All scripts created and tested locally
- [ ] **Commit 28**: Resource testing prevents OOMs on cluster
- [ ] **Commit 29**: 216-job sweep successfully submitted and completed

### Pre-submission Checklist:
- [ ] Analysis library functions created and tested
- [ ] All callbacks refactored to use library
- [ ] LossSlopeLogger created using library functions
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
- Commits 1-27: Execute in deconCNN repository (local development)
- Commits 28-29: Execute from deconCNN on cluster (GPU required, MANUAL)

### **Script Execution Pattern:**
All Python scripts should be executed with `uv run` from repository root:
- `uv run python scripts/monitor_experiment.py`
- `uv run python scripts/recover_failed.py`
- `uv run python scripts/verify_completeness.py`
- `uv run python scripts/prepare_dataset.py`
- `uv run python scripts/validate_local.py`
- `uv run python scripts/test_harness.py`

## Success Metrics

**Phase 1 (Analysis Library):**
- Library functions created and tested
- Slope calculation, power law fitting, PSD, EWMA working

**Phase 2 (Callbacks):**
- Each callback tested individually
- Callbacks use library functions correctly
- Integration test passes without conflicts

**Phase 3 (Configs):**
- All 4 variants load and run
- Validation occurs 4x per epoch

**Phase 4 (Validation):**
- Test harness confirms all variants work
- Notebook validates data formats

**Phase 5 (Job Management):**
- 216 job configurations generated
- Monitoring and recovery tested

**Phase 6 (Data Pipeline):**
- Collection and export scripts functional
- Data format ready for Tier 4

**Phase 7 (Execution):**
- Resource testing complete
- 216 jobs successfully executed
- All data collected and verified

---

Remember: Quality over speed. Test thoroughly. Fix all linting issues. Document changes.


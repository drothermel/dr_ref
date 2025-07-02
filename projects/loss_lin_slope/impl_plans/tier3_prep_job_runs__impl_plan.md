# Tier 3 Implementation Plan: Job Setup and Critical Logging Infrastructure
## Date: 2025-07-02

This document provides an atomic implementation plan that gets from 0 to running 216-job sweep in 29 focused commits, ensuring each change is testable and reversible with comprehensive unit testing.

**Atomic commits - each commit does ONE thing well, with explicit testing phases.**

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

## [x] Phase 0: Pre-Implementation Setup
- [x] **COMPLETE**: Pre-flight validation script created and all checks pass

## [x] Phase 1: Library Extraction & Refactoring (5 commits)
- [x] **Commit 1**: Extract analysis functions from NoiseMonitor and create library
- [x] **Commit 2**: Extract Hutchinson trace from CurvatureMonitor
- [X] **Commit 3**: Create slope calculation in library
- [x] **Commit 4**: Integration test extracted functions

## [x] Phase 2: Create New Callback (2 commits)

- [x] **Commit 5**: Create LossSlopeLogger using library
- [x] **Commit 6**: Integration test all callbacks

## [x] Phase 3: Experiment Configurations (5 commits)

- [x] **Commit 7**: Create base experiment config
- [x] **Commit 8**: Test base configuration
- [x] **Commit 9**: Create BN-off variant
- [x] **Commit 10**: Create narrow and AdamW variants
- [x] **Commit 11**: Create unified callback config

## [x] Phase 4: Local Validation (5 commits)

- [x] **Commit 12**: Create validation script
- [x] **Commit 13**: Add unit tests for validation script
- [x] **Commit 14**: Create test harness - IN PROGRESS
- [x] **Commit 15**: Add unit tests for test harness
- [x] **Commit 16**: Create validation notebook and comprehensive tests

## [x] Phase 5: Job Management (5 commits)

- [x] **Commit 17**: Create submission wrapper script
- [x] **Commit 18**: Create monitoring script
- [x] **Commit 19**: Add unit tests for monitoring script
- [x] **Commit 20**: Create failure recovery script
- [x] **Commit 21**: Add unit tests for failure recovery

## [ ] Phase 6: Data Pipeline (5 commits)

- [x] **Commit 22**: Create collection script
- [x] **Commit 23**: Create verification script
- [x] **Commit 24**: Add unit tests for verification script
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


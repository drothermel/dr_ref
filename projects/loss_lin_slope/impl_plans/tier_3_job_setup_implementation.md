# Implementation Plan: Tier 3 Job Creation and Verification

## ⚠️ CRITICAL PRE-IMPLEMENTATION REQUIREMENTS (Added 2025-01-01)

**NOTE**: The following logging requirements MUST be implemented before starting the 216-run sweep to enable Tier 4 analysis. These need to be integrated into the phases below.

### Required Logging Infrastructure

1. **Batch-Level Metric Logging**
   - Create `LossSlopeLogger` callback that logs EVERY batch:
     - `loss_train_nats` and `loss_train_bits` (bits = nats / ln(2))
     - `acc_train`, `lr`, `wd`
   - Without this, Tier 4 cannot compute early slopes or power law fitting

2. **Quarter-Epoch Validation**
   - Set `val_check_interval: 0.25` in trainer config
   - Ensures validation metrics logged 4x per epoch
   - Critical for fine-grained loss curve analysis

3. **Enhanced Curvature Monitoring**
   - Update CurvatureMonitor to use BackPACK for trace (already in dependencies)
   - Use hessian-eigenthings for λ_max computation
   - MUST compute every 500 steps exactly

4. **Gradient/Weight Norm Tracking**
   - Add to validation logging (every 0.25 epochs)
   - Track `grad_norm` and `weight_norm`
   - Essential for BN-off stability analysis

5. **Noise Metrics Schedule**
   - Configure NoiseMonitor with `log_every_n_epochs=2`
   - Required for PSD tail, gradient variance analysis

6. **Checkpoint Size Tracking**
   - Log `checkpoint_size_mb` when saving checkpoints
   - Update checkpoint strategy to `save_top_k=3`

### Additional Critical Updates

- **Resource Testing Phase**: Add cluster testing to determine optimal workers per GPU per variant
- **Path Fix**: Training script is at root (`python train_cnn.py`), not in scripts/
- **Dependencies**: BackPACK already added to pyproject.toml

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

## Current Status
- Last updated: 2025-07-01
- Last completed step: [none]
- Active agent: [none]
- Blocked by: [none]

## Implementation Steps

### Phase 1: Hydra Configuration Setup (deconCNN)

- [ ] **Step 1.1**: Create base experiment config
  - Create `deconCNN/configs/experiment/loss_lin_slope_base.yaml`
  - Include all defaults and fixed parameters
  - Set logging frequencies (per-batch, quarter-epoch validation)
  - Configure checkpoint strategy (save_top_k=1, monitor=loss_val_nats)
  - Testing: `cd deconCNN && python scripts/train_cnn.py --config-name experiment/loss_lin_slope_base --cfg job seed=0`

- [ ] **Step 1.2**: Create architecture variant configs
  - [ ] Create `loss_lin_slope_bn_off.yaml` (batch norm disabled, grad_clip_norm=1.0)
  - [ ] Create `loss_lin_slope_narrow.yaml` (width_multiplier=0.5)
  - Testing: Verify each variant loads with `--cfg job`

- [ ] **Step 1.3**: Create optimizer variant config
  - Create `loss_lin_slope_adamw.yaml` with AdamW optimizer
  - Set betas=[0.9, 0.999] explicitly
  - Testing: Confirm optimizer settings with `--cfg job`

- [ ] **Step 1.4**: Configure monitoring callbacks
  - Create `configs/callbacks/loss_lin_slope_metrics.yaml`
  - Configure gradient monitoring (log_every_n_epochs=0.25)
  - Configure curvature (log_every_n_steps=500, hutch_samples=50)
  - Configure noise metrics (log_every_n_epochs=2, buffer_size=512)
  - Testing: Run 10 steps and verify all metrics logged

### Phase 2: Local Validation & Testing

- [ ] **Step 2.1**: Create local validation script
  - Create `dr_ref/projects/loss_lin_slope/scripts/validate_local.py`
  - Implement `test_single_config()` function
  - Add abbreviated training (3 epochs) for quick validation
  - Include metric verification logic
  - Testing: Run script and verify output structure

- [ ] **Step 2.2**: Create metric validation notebook
  - Create `notebooks/validate_metrics.ipynb`
  - [ ] Add cell to verify all expected metrics present
  - [ ] Add cell to check logging frequencies
  - [ ] Add cell to verify nats→bits conversion (factor of ln(2))
  - [ ] Add cell to validate curvature metric ranges
  - [ ] Add cell to check noise metric computations
  - Testing: Run notebook on sample output

- [ ] **Step 2.3**: Create comprehensive test harness
  - Create `scripts/test_harness.py`
  - Test all 4 architecture/optimizer variants
  - Implement `validate_run_outputs()` function
  - Add error handling and reporting
  - Testing: `python scripts/test_harness.py` completes without errors

### Phase 3: dr_exp Integration & Job Creation

- [ ] **Step 3.1**: Create job generation script
  - Create `scripts/generate_jobs.py`
  - [ ] Define experiment variants mapping (S-core, S-adamw, S-bn-off, S-narrow)
  - [ ] Set up parameter grids (LR: [0.05, 0.1, 0.2], WD: [1e-4, 1e-3, 1e-2], seeds: 0-5)
  - [ ] Implement job config generation (216 total jobs)
  - [ ] Add proper naming convention for tracking
  - Testing: Run and verify 216 job configs generated

- [ ] **Step 3.2**: Create dr_exp submission script
  - Create `scripts/submit_jobs.py`
  - [ ] Import dr_exp API and job configs
  - [ ] Implement dry-run mode for testing
  - [ ] Add batch submission logic
  - [ ] Include progress reporting
  - Testing: `python scripts/submit_jobs.py --dry-run` shows correct job count

- [ ] **Step 3.3**: Create job monitoring utilities
  - Create `scripts/monitor_jobs.py`
  - [ ] Implement `check_job_status()` to query dr_exp
  - [ ] Add completion rate tracking by experiment type
  - [ ] Implement `collect_completed_results()` function
  - [ ] Add failed job detection and reporting
  - Testing: Mock job statuses and verify monitoring logic

### Phase 4: SLURM Cluster Execution (Using dr_exp Infrastructure)

- [ ] **Step 4.1**: Create experiment submission script
  - Create `scripts/submit_loss_lin_slope.py` using dr_exp submission utilities
  - [ ] Import JobSubmitter from dr_exp
  - [ ] Define experiment configurations with priorities
  - [ ] Generate all 216 job specifications
  - [ ] Add dry-run and test modes
  - Testing: `python submit_loss_lin_slope.py --dry-run --test-one`

- [ ] **Step 4.2**: Create SLURM worker launcher (embedded parameters)
  - Create `scripts/launch_workers_embedded.sh`
  - [ ] Use embedded parameter method for reliability
  - [ ] Configure for RTX8000 GPUs with CUDA MPS
  - [ ] Set up Singularity container execution
  - [ ] Add proper cleanup handlers
  - Testing: `./launch_workers_embedded.sh 2 1` (2 workers, 1 GPU)

- [ ] **Step 4.3**: Create real-time monitoring dashboard
  - Create `scripts/monitor_experiment.py`
  - [ ] Query dr_exp for job statuses
  - [ ] Group by experiment type and status
  - [ ] Calculate completion percentages
  - [ ] Show failure warnings
  - Testing: Run while test jobs are executing

- [ ] **Step 4.4**: Create failure recovery system
  - Create `scripts/recover_failed.py`
  - [ ] Detect and categorize failures (OOM, gradient explosion, etc.)
  - [ ] Implement intelligent retry strategies
  - [ ] Adjust batch size for OOM failures
  - [ ] Increase gradient clipping for instability
  - Testing: Manually fail a job and test recovery

- [ ] **Step 4.5**: Create efficient result collection
  - Create `scripts/collect_and_archive.sh`
  - [ ] Preserve directory structure for logs
  - [ ] Compress checkpoints individually
  - [ ] Create consolidated parquet dataset
  - [ ] Generate timestamped archives
  - Testing: Run on small subset of completed jobs

### Phase 5: Verification & Quality Checks

- [ ] **Step 5.1**: Create completeness checker
  - Create `scripts/verify_completeness.py`
  - [ ] Count completed runs vs expected (216)
  - [ ] Identify missing configurations
  - [ ] Check for crashed/incomplete runs
  - [ ] Generate rerun commands if needed
  - Testing: Run on partial dataset, verify detection

- [ ] **Step 5.2**: Create metric quality validator
  - Create `scripts/validate_metrics.py`
  - [ ] Check all logging frequencies correct
  - [ ] Verify metric value ranges reasonable
  - [ ] Detect NaN/inf values in logs
  - [ ] Validate checkpoint file sizes (~44MB)
  - Testing: Include known bad data, verify detection

- [ ] **Step 5.3**: Create analysis-ready dataset
  - Create `scripts/prepare_dataset.py`
  - [ ] Merge all CSV logs into single dataset
  - [ ] Add experiment metadata columns
  - [ ] Convert to Parquet for efficiency
  - [ ] Generate summary statistics report
  - Testing: Process subset, verify output format

### Phase 6: Integration Testing & Documentation

- [ ] **Step 6.1**: End-to-end integration test
  - Create `scripts/integration_test.py`
  - Test full pipeline with 4 jobs (one per variant)
  - Verify job submission → execution → collection
  - Check all metrics and outputs
  - Testing: Complete mini-sweep successfully

- [ ] **Step 6.2**: Create operational documentation
  - [ ] Write README for job submission process
  - [ ] Document troubleshooting steps
  - [ ] Create runbook for monitoring
  - [ ] Add examples for common tasks
  - Testing: Follow docs to run test job

- [ ] **Step 6.3**: Create handoff package
  - [ ] Generate job status summary
  - [ ] Package all scripts with instructions
  - [ ] Create data dictionary for metrics
  - [ ] Prepare for Tier 4 analysis team
  - Testing: Verify package completeness

## Critical Validation Points

### After Phase 1:
- [ ] All 4 config variants load without errors
- [ ] Metrics logged at correct frequencies
- [ ] Checkpoint strategy correctly configured

### After Phase 2:
- [ ] Local tests pass for all variants
- [ ] BN-off variant stable with gradient clipping
- [ ] All expected metrics present in outputs

### After Phase 3:
- [ ] 216 job configs generated correctly
- [ ] dr_exp submission works in dry-run mode
- [ ] Monitoring utilities functional

### After Phase 5:
- [ ] All 216 runs accounted for
- [ ] No missing metrics or NaN values
- [ ] Dataset ready for analysis

## Execution Timeline

### Day 0: Setup & Local Validation
- Morning: Complete Phase 1 (Hydra configs)
- Afternoon: Complete Phase 2 (Local testing)
- Evening: Verify all variants work locally

### Day 1: Job Submission
- Morning: Complete Phase 3 (Job creation)
- Afternoon: Complete Phase 4.1 (SLURM setup)
- Evening: Submit full job array

### Day 2-3: Execution & Monitoring
- Continuous: Monitor job progress (Phase 4.2)
- As needed: Handle failures and restarts
- Ongoing: Collect completed results (Phase 4.3)

### Day 4: Verification & Handoff
- Morning: Complete Phase 5 (Verification)
- Afternoon: Integration testing
- Evening: Prepare handoff to Tier 4

## Handoff Instructions
To continue this implementation:
1. Read the full plan and agent instructions
2. Check "Current Status" to see where we are
3. Find the first unchecked [ ] item
4. Begin implementation from that point
5. Update status when stopping work

## Retrospective Notes
[Track significant decisions and deviations here]

### Key Design Decisions:
- Using dr_exp for job management provides restart capability and monitoring
- 2 workers per GPU maximizes throughput without memory issues
- Quarter-epoch validation keeps logs manageable while providing sufficient resolution
- Gradient clipping at 1.0 for BN-off based on preliminary stability tests

### Critical Success Factors:
- Early validation prevents wasted compute on broken configs
- Monitoring infrastructure enables quick failure detection
- Automated result collection prevents data loss
- Comprehensive validation ensures analysis readiness

### Risk Mitigations:
- Test each variant locally before cluster submission
- Use job priorities to run critical experiments first
- Implement automatic restart for failed jobs
- Keep local backups of all results

### Cluster-Specific Details (from SLURM documentation):
- **Environment Variables**: Use embedded parameters method (most reliable)
- **Container**: Singularity with overlay at `$SCRATCH/drexp.ext3:ro`
- **GPUs**: RTX8000 with CUDA MPS for efficient sharing
- **Account**: CDS
- **Paths**: `/scratch/ddr8143/` for storage and logs
- **dr_exp Integration**: Use existing submission utilities and monitoring tools
- **Worker Strategy**: 3 workers per GPU recommended for optimal throughput

---
Remember: Quality over speed. Test thoroughly. Document changes.
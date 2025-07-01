# Updated Phase 4: SLURM Cluster Execution

## Overview
Based on the cluster documentation, we'll use the dr_exp submission system with embedded parameters for reliable execution.

### 4.1 Create Loss Lin Slope Submission Script
**Commit**: "feat: add loss_lin_slope experiment submission script"
```python
# dr_ref/projects/loss_lin_slope/scripts/submit_loss_lin_slope.py
#!/usr/bin/env python3
"""Submit loss_lin_slope experiments to dr_exp."""

import argparse
import itertools
import sys
from pathlib import Path

# Add dr_exp to path
sys.path.append("/scratch/ddr8143/repos/dr_exp")
from scripts.submission.submission_utils import JobSubmitter

# Experiment configurations
EXPERIMENTS = {
    "S-core": ("loss_lin_slope_base.yaml", 200),
    "S-adamw": ("loss_lin_slope_adamw.yaml", 190),
    "S-bn-off": ("loss_lin_slope_bn_off.yaml", 180),
    "S-narrow": ("loss_lin_slope_narrow.yaml", 170),
}

# Parameter grids
LRS = [0.05, 0.1, 0.2]
WDS = [1e-4, 1e-3, 1e-2]
SEEDS = list(range(6))

def generate_jobs():
    """Generate all 216 job configurations."""
    jobs = []
    for (exp_name, (config, base_priority)), lr, wd, seed in itertools.product(
        EXPERIMENTS.items(), LRS, WDS, SEEDS
    ):
        # Adjust priority based on hyperparams (prioritize middle values)
        priority_adj = 0
        if lr == 0.1:  # Expected good LR
            priority_adj += 10
        if wd == 1e-3:  # Expected good WD
            priority_adj += 10
            
        jobs.append({
            "config": config,
            "seed": seed,
            "priority": base_priority + priority_adj,
            "overrides": {
                "optim.lr": lr,
                "optim.weight_decay": wd,
            },
            "tags": [exp_name, f"lr{lr}", f"wd{wd}"]
        })
    return jobs

def main():
    parser = argparse.ArgumentParser(description="Submit loss_lin_slope experiments")
    parser.add_argument("--dry-run", action="store_true", help="Preview without submitting")
    parser.add_argument("--force", action="store_true", help="Force resubmit existing jobs")
    parser.add_argument("--subset", choices=list(EXPERIMENTS.keys()), 
                       help="Submit only one experiment type")
    parser.add_argument("--seeds", type=int, nargs="+", help="Custom seed list")
    parser.add_argument("--test-one", action="store_true", 
                       help="Submit just one job for testing")
    args = parser.parse_args()
    
    # Override seeds if specified
    if args.seeds:
        global SEEDS
        SEEDS = args.seeds
    
    # Generate jobs
    jobs = generate_jobs()
    
    # Filter by subset if requested
    if args.subset:
        jobs = [j for j in jobs if args.subset in j["tags"]]
    
    # Test mode - just one job
    if args.test_one:
        jobs = jobs[:1]
        print("TEST MODE: Submitting only 1 job")
    
    # Setup submitter
    submitter = JobSubmitter(
        base_path="/scratch/ddr8143/repos/deconCNN",
        experiment="loss_lin_slope",
        config_dir="configs/experiment",
        dry_run=args.dry_run,
        skip_existing=not args.force
    )
    
    # Submit jobs
    print(f"Submitting {len(jobs)} jobs...")
    submitter.submit_batch(jobs)
    
if __name__ == "__main__":
    main()
```

**Testing**: 
```bash
python submit_loss_lin_slope.py --dry-run
python submit_loss_lin_slope.py --test-one
```

### 4.2 Create SLURM Worker Launch Script (Embedded Parameters)
**Commit**: "feat: add SLURM launcher for loss_lin_slope workers"
```bash
# dr_ref/projects/loss_lin_slope/scripts/launch_workers_embedded.sh
#!/bin/bash
# Launch dr_exp workers for loss_lin_slope experiment
# Uses embedded parameters for reliability

# Configuration
BASE_PATH="/scratch/ddr8143/repos/deconCNN"
EXPERIMENT="loss_lin_slope"
WORKERS_PER_GPU=${1:-3}  # Default 3 workers per GPU
GPUS=${2:-8}             # Default 8 GPUs (24 total workers)
HOURS=${3:-12}           # Expected to finish in 12 hours

echo "=== Launching loss_lin_slope workers ==="
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "GPUs: $GPUS"
echo "Time limit: $HOURS hours"
echo "Total workers: $((WORKERS_PER_GPU * GPUS))"

# Create temp script with embedded values
TEMP_SCRIPT="/tmp/loss_lin_slope_workers_$$.sbatch"

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=loss_lin_slope_workers
#SBATCH --output=/scratch/ddr8143/logs/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ddr8143/logs/slurm_logs/%x_%j.err
#SBATCH --time=${HOURS}:00:00
#SBATCH --gres=gpu:rtx8000:${GPUS}
#SBATCH --mem=160G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$((6 * GPUS))
#SBATCH --account=cds

# Setup CUDA MPS for GPU sharing
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-\${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-\${SLURM_JOB_ID}"
mkdir -p "\$CUDA_MPS_PIPE_DIRECTORY" "\$CUDA_MPS_LOG_DIRECTORY"

cleanup() {
    echo "Cleaning up CUDA MPS..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "\$CUDA_MPS_PIPE_DIRECTORY" "\$CUDA_MPS_LOG_DIRECTORY"
}
trap cleanup EXIT

nvidia-cuda-mps-control -d

# Launch in singularity
singularity exec --nv --overlay \$SCRATCH/drexp.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /scratch/ddr8143/repos/dr_exp/.venv/bin/activate
cd /scratch/ddr8143/repos/dr_exp
source .env

echo '========================================'
echo 'Loss Lin Slope Worker Launch'
echo '========================================'
echo 'Time: \$(date)'
echo 'Workers per GPU: $WORKERS_PER_GPU'
echo 'Total workers: $((WORKERS_PER_GPU * GPUS))'
echo '========================================'

# Show job queue status
echo 'Checking job queue...'
uv run dr_exp --base-path $BASE_PATH --experiment $EXPERIMENT status

# Launch workers
echo 'Starting launcher...'
uv run dr_exp --base-path $BASE_PATH --experiment $EXPERIMENT system launcher \
    --workers-per-gpu $WORKERS_PER_GPU \
    --max-hours $((HOURS - 1))
"
EOF

# Submit
JOBID=$(sbatch --parsable "$TEMP_SCRIPT")
echo "Submitted SLURM job: $JOBID"
rm "$TEMP_SCRIPT"

echo ""
echo "Monitor with:"
echo "  squeue -j $JOBID"
echo "  tail -f /scratch/ddr8143/logs/slurm_logs/loss_lin_slope_workers_${JOBID}.out"
echo "  watch -n 30 'dr_exp --base-path $BASE_PATH --experiment $EXPERIMENT status'"
```

**Testing**:
```bash
# Test with 1 GPU, 2 workers
./launch_workers_embedded.sh 2 1
```

### 4.3 Create Real-time Monitoring Dashboard
**Commit**: "feat: add real-time experiment monitoring"
```python
# dr_ref/projects/loss_lin_slope/scripts/monitor_experiment.py
#!/usr/bin/env python3
"""Monitor loss_lin_slope experiment progress."""

import subprocess
import time
from collections import defaultdict
from datetime import datetime

def get_job_status():
    """Query dr_exp for current job status."""
    cmd = [
        "dr_exp", 
        "--base-path", "/scratch/ddr8143/repos/deconCNN",
        "--experiment", "loss_lin_slope",
        "list", "--status", "all", "--format", "json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
        
    # Parse JSON output
    import json
    jobs = json.loads(result.stdout)
    
    # Group by status and experiment type
    status_groups = defaultdict(lambda: defaultdict(int))
    for job in jobs:
        exp_type = job["tags"][0] if job["tags"] else "unknown"
        status_groups[job["status"]][exp_type] += 1
    
    return status_groups

def monitor_loop():
    """Main monitoring loop."""
    print("Loss Lin Slope Experiment Monitor")
    print("=" * 60)
    
    while True:
        status = get_job_status()
        if not status:
            print("Failed to query job status")
            time.sleep(30)
            continue
        
        # Clear screen and show update
        print("\033[2J\033[H")  # Clear screen
        print(f"Loss Lin Slope Monitor - {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 60)
        
        # Show totals
        total_by_status = defaultdict(int)
        for status_type, exp_counts in status.items():
            for count in exp_counts.values():
                total_by_status[status_type] += count
        
        print("\nOverall Progress:")
        for status_type in ["completed", "running", "pending", "failed"]:
            count = total_by_status.get(status_type, 0)
            print(f"  {status_type.capitalize():12} {count:4d}")
        
        # Show by experiment type
        print("\nBy Experiment Type:")
        for exp_type in ["S-core", "S-adamw", "S-bn-off", "S-narrow"]:
            print(f"\n  {exp_type}:")
            for status_type in ["completed", "running", "pending", "failed"]:
                count = status.get(status_type, {}).get(exp_type, 0)
                if count > 0:
                    print(f"    {status_type:12} {count:3d}")
        
        # Calculate completion percentage
        total = sum(total_by_status.values())
        completed = total_by_status.get("completed", 0)
        if total > 0:
            pct = 100 * completed / total
            print(f"\nCompletion: {completed}/{total} ({pct:.1f}%)")
        
        # Show any failures
        if total_by_status.get("failed", 0) > 0:
            print(f"\nWARNING: {total_by_status['failed']} failed jobs!")
            print("Run with --show-failed for details")
        
        time.sleep(30)

if __name__ == "__main__":
    monitor_loop()
```

### 4.4 Create Failure Recovery Script
**Commit**: "feat: add automatic failure detection and recovery"
```python
# dr_ref/projects/loss_lin_slope/scripts/recover_failed.py
#!/usr/bin/env python3
"""Detect and resubmit failed jobs."""

import subprocess
import json
import sys

def get_failed_jobs():
    """Get list of failed jobs."""
    cmd = [
        "dr_exp",
        "--base-path", "/scratch/ddr8143/repos/deconCNN", 
        "--experiment", "loss_lin_slope",
        "list", "--status", "failed", "--format", "json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to query jobs")
        return []
        
    return json.loads(result.stdout)

def analyze_failures(failed_jobs):
    """Analyze failure patterns."""
    failure_types = defaultdict(list)
    
    for job in failed_jobs:
        # Try to categorize failure
        if "error_message" in job:
            if "CUDA out of memory" in job["error_message"]:
                failure_types["OOM"].append(job)
            elif "gradient" in job["error_message"].lower():
                failure_types["gradient_explosion"].append(job)
            else:
                failure_types["other"].append(job)
        else:
            failure_types["unknown"].append(job)
    
    return failure_types

def resubmit_with_fixes(jobs, fix_type):
    """Resubmit jobs with appropriate fixes."""
    for job in jobs:
        config = job["config"]
        
        # Apply fixes based on failure type
        if fix_type == "OOM":
            # Reduce batch size
            config["batch_size"] = config.get("batch_size", 128) // 2
        elif fix_type == "gradient_explosion":
            # Increase gradient clipping
            config["model"]["grad_clip_norm"] = 0.5
        
        # Increase priority for retries
        config["priority"] = job.get("priority", 100) + 50
        
        # Resubmit
        print(f"Resubmitting {job['job_id']} with fixes...")
        # Implementation depends on dr_exp API

def main():
    failed = get_failed_jobs()
    if not failed:
        print("No failed jobs found!")
        return
    
    print(f"Found {len(failed)} failed jobs")
    
    # Analyze failures
    failure_types = analyze_failures(failed)
    
    for ftype, jobs in failure_types.items():
        print(f"\n{ftype}: {len(jobs)} jobs")
        if len(jobs) <= 5:
            for job in jobs:
                print(f"  - {job['job_id']}: {job.get('config_name', 'unknown')}")
    
    # Offer to resubmit
    if input("\nResubmit failed jobs? [y/N]: ").lower() == "y":
        for ftype, jobs in failure_types.items():
            if ftype in ["OOM", "gradient_explosion"]:
                resubmit_with_fixes(jobs, ftype)

if __name__ == "__main__":
    main()
```

### 4.5 Create Results Collection with Compression
**Commit**: "feat: add efficient result collection and archival"
```bash
# dr_ref/projects/loss_lin_slope/scripts/collect_and_archive.sh
#!/bin/bash
# Collect results and create compressed archives

EXPERIMENT="loss_lin_slope"
BASE_PATH="/scratch/ddr8143/repos/deconCNN"
RESULTS_DIR="/scratch/ddr8143/results/loss_lin_slope"
ARCHIVE_DIR="/scratch/ddr8143/archives/loss_lin_slope"

echo "Collecting loss_lin_slope results..."

# Create directories
mkdir -p "$RESULTS_DIR"/{logs,checkpoints,metrics}
mkdir -p "$ARCHIVE_DIR"

# Collect logs (CSV files are small, keep uncompressed)
echo "Collecting training logs..."
find "$BASE_PATH/$EXPERIMENT" -name "*.csv" -type f | while read -r file; do
    # Preserve directory structure
    rel_path="${file#$BASE_PATH/$EXPERIMENT/}"
    dest_dir="$RESULTS_DIR/logs/$(dirname "$rel_path")"
    mkdir -p "$dest_dir"
    cp "$file" "$dest_dir/"
done

# Collect best checkpoints (compress individually)
echo "Collecting and compressing checkpoints..."
find "$BASE_PATH/$EXPERIMENT" -name "best_model.ckpt" -type f | while read -r file; do
    rel_path="${file#$BASE_PATH/$EXPERIMENT/}"
    job_id=$(echo "$rel_path" | cut -d'/' -f1)
    
    # Compress checkpoint
    gzip -c "$file" > "$RESULTS_DIR/checkpoints/${job_id}_best.ckpt.gz"
done

# Create consolidated metrics file
echo "Creating consolidated metrics..."
python3 - << 'EOF'
import pandas as pd
import glob
import json
from pathlib import Path

# Find all metric files
metric_files = glob.glob("/scratch/ddr8143/results/loss_lin_slope/logs/**/metrics.csv", recursive=True)

# Combine into single dataframe
all_metrics = []
for file in metric_files:
    job_id = Path(file).parent.name
    df = pd.read_csv(file)
    df['job_id'] = job_id
    all_metrics.append(df)

if all_metrics:
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_parquet("/scratch/ddr8143/results/loss_lin_slope/all_metrics.parquet")
    print(f"Combined {len(all_metrics)} runs into single parquet file")
EOF

# Create archive
echo "Creating final archive..."
cd "$RESULTS_DIR"
tar -czf "$ARCHIVE_DIR/loss_lin_slope_$(date +%Y%m%d_%H%M%S).tar.gz" .

# Show summary
echo ""
echo "Collection complete!"
echo "Results: $RESULTS_DIR"
echo "Archive: $ARCHIVE_DIR"
du -sh "$RESULTS_DIR"
du -sh "$ARCHIVE_DIR"/*.tar.gz | tail -1
```

**Testing**:
```bash
# Test on subset of results
./collect_and_archive.sh
```

## Key Updates from Original Plan:

1. **Uses dr_exp submission system** instead of raw SLURM arrays
2. **Embedded parameters method** for reliable environment propagation
3. **Singularity containers** with proper overlay mounting
4. **CUDA MPS** for efficient GPU sharing
5. **Real-time monitoring** using dr_exp's built-in tools
6. **Automatic failure recovery** with intelligent retry strategies
7. **Efficient archival** with compression for checkpoints

## Validation Steps:
- [ ] Test job submission with single job
- [ ] Verify worker launcher starts correctly
- [ ] Confirm monitoring shows accurate status
- [ ] Test failure recovery on intentionally failed job
- [ ] Verify result collection preserves structure
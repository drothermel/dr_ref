# Working Examples from the Codebase

## Complete Job Submission Workflow

### 1. Basic Experiment Submission (from dr_exp)
```python
#!/usr/bin/env python3
# From: dr_exp/scripts/submission/submit_jobs.py

from submission_utils import JobSubmitter

# Define experiments
EXPERIMENTS = [
    ("baseline.yaml", 100),      # (config, priority)
    ("variant_a.yaml", 90),
    ("variant_b.yaml", 80),
]

# Create submitter
submitter = JobSubmitter(
    base_path="/scratch/ddr8143/repos/myproject",
    experiment="my_experiment",
    config_dir="configs/experiment",
    dry_run=False,
    skip_existing=True
)

# Generate jobs
jobs = []
for config, priority in EXPERIMENTS:
    for seed in range(3):
        jobs.append({
            "config": config,
            "seed": seed, 
            "priority": priority,
            "tags": [config.replace(".yaml", ""), f"seed{seed}"]
        })

# Submit
submitter.submit_batch(jobs)
```

### 2. Embedded Parameters SLURM Script
```bash
#!/bin/bash
# From: dr_exp/scripts/submission/submit_slurm_embedded.sh

# Configuration
BASE_PATH="/scratch/ddr8143/repos/deconCNN"
EXPERIMENT="loss_lin_slope"
WORKERS_PER_GPU=3
GPUS=8

# Create temporary script with embedded values
TEMP_SCRIPT="/tmp/dr_exp_${EXPERIMENT}_$$.sbatch"

cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=dr_exp_workers
#SBATCH --output=/scratch/ddr8143/logs/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ddr8143/logs/slurm_logs/%x_%j.err
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:rtx8000:8
#SBATCH --mem=160G
#SBATCH --cpus-per-task=48
#SBATCH --account=cds

# Setup CUDA MPS
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

cleanup() {
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}
trap cleanup EXIT

nvidia-cuda-mps-control -d

# Launch in singularity
singularity exec --nv --overlay $SCRATCH/drexp.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /scratch/ddr8143/repos/dr_exp/.venv/bin/activate
        cd /scratch/ddr8143/repos/dr_exp
        source .env
        
        uv run dr_exp --base-path $BASE_PATH --experiment $EXPERIMENT \
            system launcher --workers-per-gpu 3 --max-hours 46
    "
EOF

# Submit
sbatch "$TEMP_SCRIPT"
rm "$TEMP_SCRIPT"
```

### 3. Parameter Sweep with Smart Prioritization
```python
# From conceptual best practices
import itertools

def generate_sweep_jobs():
    """Generate parameter sweep with intelligent prioritization."""
    
    # Define search space
    lrs = [0.001, 0.01, 0.1, 1.0]
    wds = [0.0, 1e-4, 1e-3, 1e-2]
    models = ["resnet18", "resnet50"]
    seeds = range(3)
    
    # Priority heuristics
    def calculate_priority(lr, wd, model):
        base = 100
        
        # Prefer middle LR values
        if lr in [0.01, 0.1]:
            base += 20
            
        # Prefer some weight decay
        if wd in [1e-4, 1e-3]:
            base += 10
            
        # Prefer smaller model for initial exploration
        if model == "resnet18":
            base += 15
            
        return base
    
    jobs = []
    for lr, wd, model, seed in itertools.product(lrs, wds, models, seeds):
        jobs.append({
            "config": f"{model}_base.yaml",
            "overrides": {
                "optim.lr": lr,
                "optim.weight_decay": wd,
                "seed": seed
            },
            "priority": calculate_priority(lr, wd, model),
            "tags": [model, f"lr{lr}", f"wd{wd}", f"seed{seed}"]
        })
    
    # Sort by priority (highest first)
    jobs.sort(key=lambda x: x["priority"], reverse=True)
    return jobs
```

### 4. Monitoring Script with Status Breakdown
```python
#!/usr/bin/env python3
# Monitoring pattern from actual usage

import subprocess
import json
import time
from datetime import datetime
from collections import defaultdict

def monitor_experiment(experiment_name, base_path):
    """Monitor experiment with detailed status breakdown."""
    
    while True:
        # Clear screen
        print("\033[2J\033[H")
        print(f"Experiment Monitor: {experiment_name}")
        print(f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 60)
        
        # Get job status
        result = subprocess.run([
            "dr_exp", "--base-path", base_path,
            "--experiment", experiment_name,
            "list", "--status", "all", "--format", "json"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Failed to query status")
            time.sleep(30)
            continue
            
        jobs = json.loads(result.stdout)
        
        # Analyze by status and tag
        status_counts = defaultdict(int)
        tag_status = defaultdict(lambda: defaultdict(int))
        
        for job in jobs:
            status = job["status"]
            status_counts[status] += 1
            
            # Group by first tag (usually experiment variant)
            if job.get("tags"):
                main_tag = job["tags"][0]
                tag_status[main_tag][status] += 1
        
        # Overall summary
        total = sum(status_counts.values())
        completed = status_counts.get("completed", 0)
        
        print("\nOverall Progress:")
        for status in ["completed", "running", "pending", "failed"]:
            count = status_counts.get(status, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"  {status:12} {count:4d} ({pct:5.1f}%)")
        
        # By variant
        print("\nBy Experiment Variant:")
        for tag in sorted(tag_status.keys()):
            print(f"\n  {tag}:")
            tag_total = sum(tag_status[tag].values())
            for status in ["completed", "running", "pending", "failed"]:
                count = tag_status[tag].get(status, 0)
                if count > 0:
                    pct = 100 * count / tag_total
                    print(f"    {status:12} {count:3d} ({pct:5.1f}%)")
        
        # Completion estimate
        if status_counts.get("running", 0) > 0:
            rate = completed / (time.time() - start_time) * 3600  # per hour
            remaining = total - completed
            eta_hours = remaining / rate if rate > 0 else float('inf')
            print(f"\nEstimated completion: {eta_hours:.1f} hours")
        
        time.sleep(30)

if __name__ == "__main__":
    monitor_experiment("loss_lin_slope", "/scratch/ddr8143/repos/deconCNN")
```

### 5. Result Collection and Analysis
```python
# Pattern for collecting distributed results

from pathlib import Path
import pandas as pd
import json

def collect_experiment_results(base_path, experiment):
    """Collect all results from completed jobs."""
    
    exp_dir = Path(base_path) / experiment
    results = []
    
    # Iterate through job directories
    for job_dir in exp_dir.glob("job_*"):
        # Skip if not completed
        status_file = job_dir / "status.json"
        if status_file.exists():
            status = json.loads(status_file.read_text())
            if status.get("status") != "completed":
                continue
        
        # Collect metrics
        metrics_file = job_dir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            
            # Add job metadata
            config_file = job_dir / "config.json"
            if config_file.exists():
                config = json.loads(config_file.read_text())
                metrics["lr"] = config.get("optim", {}).get("lr")
                metrics["wd"] = config.get("optim", {}).get("weight_decay")
                metrics["model"] = config.get("model", {}).get("name")
            
            metrics["job_id"] = job_dir.name
            results.append(metrics)
    
    # Create dataframe
    df = pd.DataFrame(results)
    
    # Save as parquet for efficiency
    output_file = exp_dir / "all_results.parquet"
    df.to_parquet(output_file)
    print(f"Saved {len(df)} results to {output_file}")
    
    return df
```

### 6. Automated Failure Recovery
```python
# From working patterns

def recover_failed_jobs(experiment, modifications=None):
    """Detect and resubmit failed jobs with fixes."""
    
    # Get failed jobs
    result = subprocess.run([
        "dr_exp", "--experiment", experiment,
        "list", "--status", "failed", "--format", "json"
    ], capture_output=True, text=True)
    
    failed_jobs = json.loads(result.stdout)
    
    if not failed_jobs:
        print("No failed jobs found!")
        return
    
    # Analyze failure types
    failure_types = defaultdict(list)
    for job in failed_jobs:
        # Read error file
        error_file = Path(job["storage_path"]) / "error.txt"
        if error_file.exists():
            error_text = error_file.read_text()
            
            if "CUDA out of memory" in error_text:
                failure_types["OOM"].append(job)
            elif "nan" in error_text.lower():
                failure_types["NaN"].append(job)
            else:
                failure_types["Other"].append(job)
    
    # Resubmit with fixes
    for failure_type, jobs in failure_types.items():
        print(f"\n{failure_type}: {len(jobs)} jobs")
        
        for job in jobs:
            new_config = job["config"].copy()
            
            # Apply fixes based on failure type
            if failure_type == "OOM":
                new_config["batch_size"] = new_config.get("batch_size", 128) // 2
                new_config["gradient_accumulation"] = 2
            elif failure_type == "NaN":
                new_config["model"]["grad_clip_norm"] = 0.5
                new_config["optim"]["lr"] = new_config["optim"].get("lr", 0.1) * 0.5
            
            # Boost priority for retries
            new_config["priority"] = job.get("priority", 100) + 50
            
            # Resubmit
            dr_exp.submit_job(
                project=job["project"],
                experiment=experiment,
                config_name=job["config_name"],
                overrides=new_config,
                tags=job["tags"] + ["retry", failure_type]
            )
```

### 7. Complete End-to-End Example
```bash
#!/bin/bash
# Full workflow from submission to results

# 1. Submit experiments
cd /scratch/ddr8143/repos/myproject
python scripts/submit_experiments.py --dry-run  # Check first
python scripts/submit_experiments.py

# 2. Launch workers
cd /scratch/ddr8143/repos/dr_exp
./scripts/submission/submit_slurm_embedded.sh \
    /scratch/ddr8143/repos/myproject \
    my_experiment \
    3 \
    8

# 3. Monitor progress
watch -n 30 'dr_exp --experiment my_experiment status'

# 4. Check for failures
dr_exp --experiment my_experiment list --status failed

# 5. Collect results
python scripts/collect_results.py

# 6. Create summary
python scripts/analyze_results.py

# 7. Archive
tar -czf my_experiment_$(date +%Y%m%d).tar.gz my_experiment/
```

## Key Patterns to Remember

1. **Always use embedded parameters** for SLURM scripts
2. **Enable CUDA MPS** for multi-worker GPU sharing
3. **Use Singularity containers** with proper activation
4. **Implement retry logic** for transient failures
5. **Log everything** for debugging and recovery
6. **Monitor early and often** to catch issues quickly
7. **Prioritize jobs intelligently** to get useful results first
8. **Compress and archive** results promptly
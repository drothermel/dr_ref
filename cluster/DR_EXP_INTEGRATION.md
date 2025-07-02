# dr_exp Integration Guide

## Overview

dr_exp is a sophisticated experiment management system that handles job queuing, execution, monitoring, and result collection. It's designed for running large-scale ML experiments on clusters.

## Architecture

### Components
1. **JobDB**: SQLite database for job queue management
2. **Workers**: Execute jobs from the queue
3. **Launcher**: Manages multiple workers per GPU
4. **Sync System**: Background upload of results
5. **API**: Python interface for job submission

### Key Concepts
- **Job**: A single training run with specific config
- **Experiment**: Collection of related jobs
- **Worker**: Process that executes jobs
- **Launcher**: Spawns and manages workers

## Job Submission

### Using Python API
```python
import dr_exp.api as dr_exp

# Submit a single job
dr_exp.submit_job(
    project="my_project",
    experiment="my_experiment",
    config_name="configs/experiment/baseline.yaml",
    overrides={
        "seed": 42,
        "optim.lr": 0.1,
        "optim.weight_decay": 1e-4
    },
    priority=100,  # Higher = runs first (0-1000)
    tags=["baseline", "lr0.1"]
)
```

### Using JobSubmitter Utility
```python
from dr_exp.scripts.submission.submission_utils import JobSubmitter

submitter = JobSubmitter(
    base_path="/scratch/ddr8143/repos/myproject",
    experiment="my_experiment",
    config_dir="configs/experiment",
    dry_run=False,
    skip_existing=True  # Avoid duplicates
)

# Submit batch of jobs
jobs = [
    {"config": "baseline.yaml", "seed": 0, "priority": 100},
    {"config": "variant.yaml", "seed": 1, "priority": 90}
]
submitter.submit_batch(jobs)
```

### Command Line Interface
```bash
# Submit jobs
dr_exp --base-path /path/to/project --experiment my_exp submit \
    --config baseline.yaml --seed 0 --priority 100

# Check status
dr_exp --experiment my_exp status

# List all jobs
dr_exp --experiment my_exp list --status all

# List failed jobs
dr_exp --experiment my_exp list --status failed

# Monitor in real-time
watch -n 10 "dr_exp --experiment my_exp status"
```

## Worker Management

### Launching Workers
```bash
# Single worker
dr_exp --experiment my_exp system worker

# Multiple workers with launcher
dr_exp --experiment my_exp system launcher \
    --workers-per-gpu 3 \
    --max-hours 46

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,1 dr_exp --experiment my_exp system launcher \
    --workers-per-gpu 2
```

### Worker Configuration
- **Workers per GPU**: 2-4 recommended
- **Max hours**: Set 1 hour less than SLURM limit
- **Background sync**: Automatic result upload
- **Fault tolerance**: Workers restart on failure

## Integration with Training Code

### Required Trainer Interface
Your training function must follow this signature:
```python
def train_model(
    job_id: str,
    worker_id: str, 
    storage_path: str,
    **config: Any
) -> dict[str, Any]:
    """
    Args:
        job_id: Unique job identifier
        worker_id: Worker executing this job
        storage_path: Where to save outputs
        **config: All configuration parameters
        
    Returns:
        Dictionary with:
        - metrics: Final performance metrics
        - artifacts: Paths to saved files
    """
    # Your training code here
    
    return {
        "metrics": {"val_acc": 0.95, "val_loss": 0.23},
        "artifacts": {"checkpoint": "model.ckpt"}
    }
```

### Example Integration
```python
# In your project's dr_exp_trainer.py
from pathlib import Path
import traceback

def train_classification(job_id, worker_id, storage_path, **config):
    try:
        # Setup paths
        storage = Path(storage_path)
        storage.mkdir(exist_ok=True)
        
        # Initialize model, data, etc.
        model = create_model(config)
        dataloader = create_dataloader(config)
        
        # Train
        metrics = train_loop(model, dataloader, config)
        
        # Save outputs
        checkpoint_path = storage / "model.ckpt"
        save_checkpoint(model, checkpoint_path)
        
        return {
            "metrics": metrics,
            "artifacts": {"checkpoint": str(checkpoint_path)}
        }
        
    except Exception as e:
        # Save error info for debugging
        error_file = Path(storage_path) / "error.txt"
        error_file.write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        raise
```

## Monitoring & Analysis

### Real-time Monitoring
```python
# Monitor script
import subprocess
import json
from collections import defaultdict

def get_status():
    result = subprocess.run([
        "dr_exp", "--experiment", "my_exp", 
        "list", "--status", "all", "--format", "json"
    ], capture_output=True, text=True)
    
    jobs = json.loads(result.stdout)
    
    # Group by status
    status_counts = defaultdict(int)
    for job in jobs:
        status_counts[job["status"]] += 1
        
    return status_counts

# Print status every 30 seconds
import time
while True:
    status = get_status()
    print(f"Completed: {status['completed']}, "
          f"Running: {status['running']}, "
          f"Pending: {status['pending']}, "
          f"Failed: {status['failed']}")
    time.sleep(30)
```

### Collecting Results
```python
# Gather all results
from pathlib import Path
import pandas as pd

results = []
result_dir = Path("/scratch/ddr8143/repos/myproject/my_experiment")

for job_dir in result_dir.glob("job_*"):
    # Read metrics
    metrics_file = job_dir / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        metrics["job_id"] = job_dir.name
        results.append(metrics)

# Create dataframe
df = pd.DataFrame(results)
df.to_csv("all_results.csv")
```

## Best Practices

### 1. Job Priority Strategy
```python
# Prioritize based on expected importance
def calculate_priority(config):
    base = 100
    
    # Boost priority for "good" hyperparams
    if config["lr"] == 0.1:  # Expected optimal
        base += 50
    
    # Reduce priority for extreme values
    if config["lr"] > 1.0 or config["lr"] < 0.001:
        base -= 50
        
    return max(0, min(1000, base))  # Clamp to valid range
```

### 2. Structured Logging
```python
# Use dr_exp's structured logger
from dr_exp.utils import StructuredLogger

logger = StructuredLogger(job_id)
logger.log_event("epoch_complete", {
    "epoch": epoch,
    "train_loss": loss,
    "val_acc": acc
})
```

### 3. Failure Handling
```python
# Implement retry logic for transient failures
def submit_with_retry(job_config, max_retries=3):
    for attempt in range(max_retries):
        try:
            dr_exp.submit_job(**job_config)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 4. Resource Optimization
- Submit jobs in priority order
- Use `--skip-existing` to avoid duplicates
- Set appropriate `--max-hours` for workers
- Monitor GPU utilization and adjust workers

### 5. Debugging Failed Jobs
```bash
# Find error details
cat /path/to/experiment/job_*/error.txt

# Check worker logs
tail -f /scratch/ddr8143/logs/slurm_logs/worker_*.out

# Resubmit failed jobs with fixes
python recover_failed.py
```

## Common Patterns

### Parameter Sweep
```python
# Generate all combinations
import itertools

lrs = [0.01, 0.1, 1.0]
wds = [1e-4, 1e-3, 1e-2]
seeds = range(3)

jobs = []
for lr, wd, seed in itertools.product(lrs, wds, seeds):
    jobs.append({
        "config": "baseline.yaml",
        "overrides": {"lr": lr, "wd": wd, "seed": seed},
        "priority": calculate_priority({"lr": lr, "wd": wd}),
        "tags": [f"lr{lr}", f"wd{wd}"]
    })
```

### Staged Experiments
```python
# Run exploration first, then refinement
# Stage 1: Coarse sweep
stage1_jobs = generate_coarse_sweep()
submit_and_wait(stage1_jobs)

# Analyze results
best_region = analyze_stage1_results()

# Stage 2: Fine sweep around best
stage2_jobs = generate_fine_sweep(best_region)
submit_jobs(stage2_jobs)
```

## Troubleshooting

### Job Not Starting
1. Check worker status: `ps aux | grep dr_exp`
2. Verify job in queue: `dr_exp list --status pending`
3. Check for errors: `dr_exp list --status failed`

### Slow Execution
1. Monitor GPU usage: `nvidia-smi`
2. Check worker count vs GPU memory
3. Verify data loading isn't bottleneck

### Results Not Syncing
1. Check sync queue: Look for `.sync_queue.db`
2. Verify network connectivity
3. Check storage permissions

### Database Lock Errors
1. Ensure only one writer at a time
2. Use transaction logging
3. Implement retry logic
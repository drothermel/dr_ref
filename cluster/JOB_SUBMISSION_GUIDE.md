# Job Submission Guide

## Critical Issue: Environment Variable Propagation

The cluster has known issues with SLURM's `--export=ALL` not working reliably. This affects how you pass parameters to jobs.

### Reliability Ranking (Most to Least)

#### 1. **Embedded Parameters Method** ✅ (100% Reliable)
Generate a temporary script with all values hardcoded:

```bash
#!/bin/bash
# Create temp script with embedded values
TEMP_SCRIPT="/tmp/myjob_$$.sbatch"

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1

# Values are embedded, not from environment
EXPERIMENT="my_experiment"
PARAM1="value1"

echo "Running with \$EXPERIMENT and \$PARAM1"
# Your actual commands here
EOF

# Submit
sbatch "$TEMP_SCRIPT"
rm "$TEMP_SCRIPT"
```

#### 2. **Explicit Export Wrapper** ✅ (Very Reliable)
Export variables AND pass them explicitly:

```bash
#!/bin/bash
export PARAM1="value1"
export PARAM2="value2"

sbatch --export=ALL,PARAM1="$PARAM1",PARAM2="$PARAM2" myscript.sbatch
```

#### 3. **Direct --export Flag** ⚠️ (Usually Works)
```bash
sbatch --export=ALL,PARAM1=value1,PARAM2=value2 script.sbatch
```

#### 4. **Environment Export Only** ❌ (Often Fails)
```bash
export PARAM1=value1  # This alone is NOT reliable!
sbatch script.sbatch
```

## Submission Patterns

### Basic SLURM Script Template
```bash
#!/bin/bash
#SBATCH --job-name="experiment_name"
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/logs/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ddr8143/logs/slurm_logs/%x_%j.err
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --account=cds

# Your job commands here
```

### Singularity Execution Pattern
```bash
singularity exec --nv \
    --overlay $SCRATCH/myenv.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /path/to/venv/bin/activate
        cd /path/to/project
        python script.py
    "
```

## Common Submission Scenarios

### 1. Single Job with Parameters
```bash
# Using embedded method
./submit_embedded.sh /path/to/exp experiment_name 3 2
```

### 2. Array Jobs
```bash
#!/bin/bash
#SBATCH --array=0-99
#SBATCH --job-name=sweep_%a

# Access array index with $SLURM_ARRAY_TASK_ID
CONFIGS=(config1.yaml config2.yaml config3.yaml)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
```

### 3. Multi-GPU with Workers
```bash
# Calculate workers based on GPUs
GPUS=4
WORKERS_PER_GPU=3
TOTAL_WORKERS=$((GPUS * WORKERS_PER_GPU))
```

## Safety Features

### 1. Dry Run Testing
Always test submissions without actually submitting:
```python
# In Python scripts
parser.add_argument("--dry-run", action="store_true")
if args.dry_run:
    print(f"Would submit: {job_config}")
    return
```

### 2. Duplicate Detection
Check for existing jobs before submitting:
```bash
# Check if job already exists
existing=$(squeue -u $USER -n "job_name" -h)
if [[ -n "$existing" ]]; then
    echo "Job already running!"
    exit 1
fi
```

### 3. Transaction Logging
Keep records of all submissions:
```python
# Log submissions
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "job_id": job_id,
    "config": config,
    "success": True
}
with open("submission_log.json", "a") as f:
    json.dump(log_entry, f)
```

## Resource Guidelines

### GPU Allocation
- **Development**: 1 GPU, 2-4 workers
- **Small experiments**: 2-4 GPUs
- **Full sweeps**: 8-16 GPUs
- **Maximum practical**: 24 GPUs

### Time Limits
- **Quick tests**: 1-2 hours
- **Standard runs**: 12-24 hours
- **Long runs**: 47 hours (just under 48h limit)

### Memory per GPU
- **Minimal**: 20GB
- **Standard**: 40GB
- **Heavy models**: 60-80GB

## Debugging Failed Submissions

### Common Issues

1. **"Command not found"**
   - Environment not activated in Singularity
   - Wrong path to virtual environment

2. **"CUDA out of memory"**
   - Too many workers per GPU
   - Batch size too large
   - Enable CUDA MPS

3. **"Module not found"**
   - Missing `source .venv/bin/activate`
   - Wrong Python environment

4. **"Permission denied"**
   - Script not executable: `chmod +x script.sh`
   - Wrong overlay permissions

### Debug Commands
```bash
# Check why job failed
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed

# Get detailed job info
scontrol show job <job_id>

# View error logs
cat /scratch/ddr8143/logs/slurm_logs/*_<job_id>.err

# Interactive debugging session
srun --gres=gpu:1 --pty bash
```

## Best Practices Summary

1. **Always use embedded parameters** for critical values
2. **Test with --dry-run** before bulk submissions
3. **Log all submissions** for recovery
4. **Monitor early jobs** before scaling up
5. **Use meaningful job names** for easy tracking
6. **Set up proper cleanup handlers** for resources
7. **Check existing jobs** to avoid duplicates
8. **Start small** (1 GPU) then scale up
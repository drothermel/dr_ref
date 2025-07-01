# NYU HPC Cluster Overview

## Hardware Resources

### GPUs
- **Type**: NVIDIA RTX 8000
- **Memory**: 48GB per GPU
- **Typical allocation**: 1-8 GPUs per job
- **GPU sharing**: CUDA MPS enabled for efficient multi-worker setups

### CPU & Memory
- **Default memory**: 40-80GB depending on GPU count
- **CPUs**: 6-12 cores (scales with GPU count)
- **Recommended**: 10GB memory per worker

### Storage Paths
- **Scratch space**: `/scratch/ddr8143/` (primary working directory)
- **Logs**: `/scratch/ddr8143/logs/slurm_logs/`
- **Repos**: `/scratch/ddr8143/repos/`
- **Results**: `/scratch/ddr8143/results/`
- **Archives**: `/scratch/ddr8143/archives/`

## Software Environment

### Container System
- **Type**: Singularity (not Docker)
- **Base image**: `/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`
- **Overlays**: Custom environments in `.ext3` format
  - dr_exp: `$SCRATCH/drexp.ext3:ro`
  - Other projects may have their own overlays

### Python Environment
- **Manager**: UV (fast Python package manager)
- **Virtual envs**: Typically in `.venv` within repo
- **Activation**: `source /path/to/repo/.venv/bin/activate`

## SLURM Configuration

### Account & Partitions
- **Account**: `cds` (Center for Data Science)
- **Partition**: `gpu` (for GPU jobs)
- **Time limits**: Up to 48 hours (`--time=47:00:00` recommended)

### Job Naming & Logging
```bash
#SBATCH --job-name="descriptive_name"
#SBATCH --output=/scratch/ddr8143/logs/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ddr8143/logs/slurm_logs/%x_%j.err
```
- `%x`: Job name
- `%j`: Job ID
- `%A`: Array job ID
- `%a`: Array task ID

### Resource Specifications
```bash
#SBATCH --gres=gpu:rtx8000:2    # 2 RTX8000 GPUs
#SBATCH --mem=80G               # Total memory
#SBATCH --cpus-per-task=12      # CPU cores
#SBATCH --nodes=1               # Single node
#SBATCH --tasks-per-node=1      # One task per node
```

## GPU Sharing with CUDA MPS

CUDA MPS (Multi-Process Service) allows efficient GPU sharing between multiple processes:

```bash
# Setup MPS
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Start daemon
nvidia-cuda-mps-control -d

# Cleanup on exit
cleanup() {
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}
trap cleanup EXIT
```

## Best Practices

### Directory Structure
```
/scratch/ddr8143/
├── repos/              # Git repositories
│   ├── dr_exp/        # Experiment management system
│   ├── deconCNN/      # Model training code
│   └── dr_ref/        # Reference/planning repo
├── logs/              # All logs
│   └── slurm_logs/    # SLURM output files
├── results/           # Experimental results
└── archives/          # Compressed backups
```

### Monitoring Commands
```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel job
scancel <job_id>

# Check GPU usage
nvidia-smi

# Monitor job output
tail -f /scratch/ddr8143/logs/slurm_logs/<job_name>_<job_id>.out
```

### Performance Tips
1. Use 2-4 workers per GPU for optimal throughput
2. Enable CUDA MPS for multi-worker setups
3. Pre-download datasets to avoid repeated downloads
4. Use local scratch for temporary files
5. Compress large artifacts before long-term storage
# NYU HPC Cluster Documentation

This directory contains comprehensive documentation for using the NYU HPC cluster effectively, with a focus on running machine learning experiments using dr_exp.

## Quick Start

If you're new to the cluster, read these in order:
1. [CLUSTER_OVERVIEW.md](CLUSTER_OVERVIEW.md) - Hardware, software, and basic setup
2. [JOB_SUBMISSION_GUIDE.md](JOB_SUBMISSION_GUIDE.md) - How to submit jobs reliably
3. [DR_EXP_INTEGRATION.md](DR_EXP_INTEGRATION.md) - Using the experiment management system
4. [EXAMPLES.md](EXAMPLES.md) - Real working examples from production code

For troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Document Overview

### 📋 [CLUSTER_OVERVIEW.md](CLUSTER_OVERVIEW.md)
- Hardware specifications (RTX 8000 GPUs)
- Storage paths and organization
- Software environment (Singularity, UV)
- SLURM basics and resource allocation
- CUDA MPS for GPU sharing

### 🚀 [JOB_SUBMISSION_GUIDE.md](JOB_SUBMISSION_GUIDE.md)
- **Critical**: Environment variable propagation issues
- Submission methods ranked by reliability
- Safety features (dry-run, duplicate detection)
- Resource allocation guidelines
- Debugging failed submissions

### 🔧 [DR_EXP_INTEGRATION.md](DR_EXP_INTEGRATION.md)
- Architecture and components
- Python API and CLI usage
- Worker management strategies
- Integration with training code
- Monitoring and result collection

### 🔍 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Common issues and solutions
- GPU memory problems
- Worker failures
- Performance optimization
- Emergency recovery procedures

### 💡 [EXAMPLES.md](EXAMPLES.md)
- Complete submission workflows
- Parameter sweep patterns
- Monitoring scripts
- Result collection
- Failure recovery

## Essential Commands

### Job Submission (Most Reliable Method)
```bash
# Use embedded parameters
./submit_slurm_embedded.sh /path/to/project experiment_name workers_per_gpu num_gpus
```

### Monitoring
```bash
# Check SLURM jobs
squeue -u $USER

# Monitor dr_exp experiment
watch -n 30 'dr_exp --experiment my_exp status'

# Check specific job
tail -f /scratch/ddr8143/logs/slurm_logs/jobname_*.out
```

### Debugging
```bash
# List failed jobs
dr_exp --experiment my_exp list --status failed

# Check error details
cat /path/to/experiment/job_*/error.txt

# GPU usage
nvidia-smi
```

## Critical Knowledge

### 1. Environment Variable Issue
The cluster's `--export=ALL` is unreliable. **Always use the embedded parameters method** for passing values to SLURM jobs.

### 2. GPU Sharing
Use CUDA MPS for efficient multi-worker GPU usage:
- 2-4 workers per GPU recommended
- Must set up MPS in SLURM script
- Clean up MPS on exit

### 3. Container Execution
All jobs run in Singularity containers:
- Base image: CUDA 11.8 with cuDNN
- Custom environments via `.ext3` overlays
- Must activate virtual environment inside container

### 4. Storage Strategy
- `/scratch/`: Primary working directory
- `/tmp/`: Fast local storage (job-specific)
- Compress checkpoints before archiving
- Use Parquet for large datasets

## Best Practices Checklist

Before submitting experiments:
- [ ] Test with single job first
- [ ] Use embedded parameters method
- [ ] Set up proper error handling
- [ ] Enable structured logging
- [ ] Configure checkpoint saving
- [ ] Plan monitoring strategy
- [ ] Prepare recovery procedures

## Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first
2. Review [EXAMPLES.md](EXAMPLES.md) for working patterns
3. Check SLURM logs: `/scratch/ddr8143/logs/slurm_logs/`
4. Use dr_exp debugging commands
5. Monitor system resources (GPU, memory, disk)

## Contributing

When you discover new patterns or solutions:
1. Add them to the appropriate guide
2. Include working examples
3. Document any cluster-specific quirks
4. Update troubleshooting if you solve new issues

Remember: The cluster environment has specific quirks (especially around environment variables). When in doubt, use the patterns that are marked as "tested and working" in these guides.
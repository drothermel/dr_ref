# Cluster Troubleshooting Guide

## Common Issues and Solutions

### Environment Variable Issues

#### Problem: Parameters not passing to SLURM jobs
```bash
# Symptom: Job uses default values instead of your parameters
echo $MY_PARAM  # Shows correct value
sbatch myjob.sbatch
# Inside job: echo $MY_PARAM shows nothing
```

**Solution**: Use embedded parameters method
```bash
# Don't rely on --export=ALL
# Instead, generate script with values embedded
cat > temp.sbatch << EOF
#!/bin/bash
#SBATCH --job-name=myjob
MY_PARAM="actual_value"  # Hardcoded, not \$MY_PARAM
echo "\$MY_PARAM"
EOF
sbatch temp.sbatch
```

### GPU Memory Issues

#### Problem: CUDA out of memory
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:
1. Reduce batch size
2. Reduce number of workers per GPU
3. Enable gradient checkpointing
4. Use mixed precision training

```python
# In your config
if "OOM" in previous_error:
    config["batch_size"] = config["batch_size"] // 2
    config["gradient_accumulation"] = 2  # Maintain effective batch size
```

### Worker Failures

#### Problem: Workers die silently
**Diagnosis**:
```bash
# Check worker processes
ps aux | grep dr_exp

# Check SLURM logs
tail -n 100 /scratch/ddr8143/logs/slurm_logs/worker_*.err

# Check dr_exp logs
find /path/to/experiment -name "*.log" -mtime -1 | xargs tail
```

**Solutions**:
1. Increase memory allocation
2. Add error handling in trainer
3. Check for silent Python errors

### Gradient Explosion (BN-off runs)

#### Problem: NaN losses with batch norm disabled
```
Step 100: loss=1.23
Step 101: loss=456.78  
Step 102: loss=nan
```

**Solution**: Add gradient clipping
```yaml
# In your config
model:
  use_batch_norm: false
  grad_clip_norm: 1.0  # or even 0.5 for stability
  
# Consider also:
optim:
  lr: 0.05  # Lower than with BN
```

### Singularity Container Issues

#### Problem: Module/Command not found
```
/bin/bash: python: command not found
ModuleNotFoundError: No module named 'torch'
```

**Solution**: Ensure proper activation
```bash
singularity exec --nv --overlay $SCRATCH/myenv.ext3:ro \
    /path/to/cuda.sif /bin/bash -c "
    source /path/to/.venv/bin/activate  # Critical!
    which python  # Should show venv python
    python script.py
"
```

### SLURM Job Stuck in Pending

#### Problem: Job won't start
```bash
squeue -u $USER
# Shows job in PD (pending) state forever
```

**Diagnosis**:
```bash
# Check why job is pending
squeue -j <job_id> -o "%T %r"

# Common reasons:
# - Resources: Not enough GPUs available
# - Priority: Other jobs ahead in queue
# - ReqNodeNotAvail: Requested resources impossible
```

**Solutions**:
1. Request fewer resources
2. Check cluster status: `sinfo -p gpu`
3. Use different GPU type if available

### Data Loading Bottlenecks

#### Problem: GPU utilization low (<50%)
```bash
nvidia-smi
# Shows low GPU usage despite training
```

**Solutions**:
1. Increase dataloader workers
2. Use local scratch for dataset
3. Implement prefetching

```python
# Copy data to local scratch
local_data = f"/tmp/{os.environ['SLURM_JOB_ID']}/data"
os.makedirs(local_data, exist_ok=True)
shutil.copytree(remote_data, local_data)

# Use more workers
DataLoader(dataset, num_workers=8, pin_memory=True, prefetch_factor=2)
```

### Network/Storage Issues

#### Problem: Slow read/write to /scratch
**Diagnosis**:
```bash
# Test write speed
time dd if=/dev/zero of=/scratch/test.tmp bs=1G count=1

# Test read speed  
time dd if=/scratch/test.tmp of=/dev/null bs=1G
```

**Solutions**:
1. Use local `/tmp` for temporary files
2. Batch small file operations
3. Use compressed formats (parquet vs CSV)

### dr_exp Specific Issues

#### Problem: Database locked
```
sqlite3.OperationalError: database is locked
```

**Solution**: Add retry logic
```python
import time
import sqlite3

def execute_with_retry(func, max_retries=5):
    for i in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and i < max_retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

#### Problem: Jobs not being picked up
**Check**:
1. Workers running: `ps aux | grep "dr_exp.*worker"`
2. Jobs in queue: `dr_exp list --status pending`
3. No failed jobs blocking: `dr_exp list --status failed`

### Performance Optimization

#### Slow Training
1. **Profile your code**:
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Training step
    pass
prof.export_chrome_trace("trace.json")
```

2. **Check batch size efficiency**:
```python
# Find optimal batch size
for bs in [32, 64, 128, 256]:
    try:
        # Time one epoch with batch size bs
        time_taken = train_one_epoch(bs)
        print(f"BS {bs}: {time_taken:.2f}s")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"BS {bs}: OOM")
            break
```

### Emergency Recovery

#### When everything fails:
1. **Kill all your jobs**:
```bash
scancel -u $USER
```

2. **Clean up processes**:
```bash
# Find stuck processes
ps aux | grep $USER | grep python

# Kill specific processes
kill -9 <pid>
```

3. **Clear temporary files**:
```bash
# Clean job-specific temp dirs
rm -rf /tmp/nvidia-mps-*
rm -rf /tmp/nvidia-log-*

# Clear your scratch tmp
find /scratch/$USER -name "*.tmp" -mtime +1 -delete
```

4. **Reset dr_exp state**:
```bash
# Mark stuck jobs as failed
# This requires database access - be careful!
# Better to use dr_exp CLI if possible
```

## Preventive Measures

1. **Always test with single job first**
2. **Monitor early iterations closely**
3. **Set up alerts for failures**
4. **Keep submission logs**
5. **Use defensive coding**:
```python
# Save intermediate results
if epoch % 10 == 0:
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
    
# Catch and log errors
try:
    result = train_step()
except Exception as e:
    logger.error(f"Training failed: {e}")
    save_debug_info()
    raise
```

6. **Resource limits**:
```python
# Set memory growth for TF
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# Limit PyTorch memory
torch.cuda.set_per_process_memory_fraction(0.9)
```
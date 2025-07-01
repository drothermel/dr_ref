# Tier 3 Implementation Plan: Consolidated Updates

## Date: 2025-01-01

This document consolidates all findings and updates needed for the Tier 3 Job Setup and Verification implementation plan.

## 1. Library Updates

### Add to deconCNN dependencies:
```toml
[project.dependencies]
# Analysis and visualization (already added)
"pandas>=2.3.0",
"matplotlib>=3.10.3", 
"statsmodels>=0.14.4",
"rich>=14.0.0",
"backpack-for-pytorch>=1.6.0",  # NEW: For efficient trace estimation

[dependency-groups.analysis]
"jupyter>=1.1.1",
"notebook>=7.4.4",
"plotly>=6.2.0",
"seaborn>=0.13.2",
```

### Library Usage Strategy:
- **BackPACK**: Hutchinson trace estimation (faster, less memory)
- **hessian-eigenthings**: Largest eigenvalue computation
- **PyHessian**: Reserved for future spectral analysis

## 2. Critical Implementation Updates

### Phase 1 Updates

**Step 1.1**: Fix path reference
- Change: `python scripts/train_cnn.py` → `python train_cnn.py`
- The training script is at repository root, not in scripts/

**Step 1.2**: Create comprehensive logging callback
```python
# src/deconcnn/callbacks/loss_slope_logger.py
class LossSlopeLogger(Callback):
    """Logs batch-level metrics for loss slope analysis."""
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log every batch
        metrics = {
            "loss_train_nats": outputs["loss"],
            "loss_train_bits": outputs["loss"] / math.log(2),
            "acc_train": outputs["acc"],
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "wd": trainer.optimizers[0].param_groups[0]["weight_decay"],
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
        }
        
        # Log to CSV for Tier 4 compatibility
        self.csv_logger.log(metrics)
        
        # Also log to dr_exp
        pl_module.log_dict(metrics, on_step=True)
```

**Step 1.3**: Update checkpoint strategy
```yaml
checkpoint:
  save_top_k: 3  # Updated from 1
  monitor: loss_val_nats
  mode: min
  save_last: true
```

### Phase 2 Updates

**NEW Step 2.0**: Create analysis library functions FIRST
```python
# dr_ref/projects/loss_lin_slope/analysis_lib/__init__.py
# dr_ref/projects/loss_lin_slope/analysis_lib/knee_detector.py
# dr_ref/projects/loss_lin_slope/analysis_lib/power_law_fitter.py
# dr_ref/projects/loss_lin_slope/analysis_lib/metric_validator.py
```

**Step 2.2**: Add CSV export to validation
- Ensure metrics are exported in both JSON (dr_exp) and CSV format
- CSV required for Tier 4 pandas-based analysis

### Phase 3 Updates

**Step 3.1**: Update worker configuration
```python
# Use 3 workers per GPU as recommended by dr_exp docs
WORKERS_PER_GPU = 3  # Updated from 2
```

### NEW Phase 3.5: Cluster Resource Testing

**Step 3.5.1**: Create resource optimization script
```python
# scripts/test_resource_configs.py
test_configs = [
    {"variant": "S-core", "workers": [2, 3, 4]},
    {"variant": "S-bn-off", "workers": [1, 2, 3]},  # May need fewer
    {"variant": "S-narrow", "workers": [3, 4, 5]},  # Can handle more
    {"variant": "S-adamw", "workers": [2, 3, 4]},
]

# Submit 10-epoch test runs with different worker configs
# Monitor GPU utilization, memory, throughput
# Determine optimal workers per variant
```

### Phase 4 Updates

**Step 4.1**: Use variant-specific worker counts
```bash
# Based on resource testing results
declare -A WORKERS_PER_VARIANT
WORKERS_PER_VARIANT[S-core]=3
WORKERS_PER_VARIANT[S-bn-off]=2  # If testing shows memory constraints
WORKERS_PER_VARIANT[S-narrow]=4  # If testing shows headroom
WORKERS_PER_VARIANT[S-adamw]=3
```

### Phase 5 Updates

**Step 5.3**: Add data format conversion
```python
# scripts/convert_to_analysis_format.py
def convert_metrics_jsonl_to_csv(experiment_dir):
    """Convert dr_exp metrics.jsonl to CSV for Tier 4 analysis."""
    for job_dir in experiment_dir.glob("job_*"):
        jsonl_path = job_dir / "metrics.jsonl"
        csv_path = job_dir / "metrics.csv"
        
        # Parse JSONL and flatten to CSV
        df = parse_jsonl_metrics(jsonl_path)
        df.to_csv(csv_path, index=False)
```

## 3. Curvature Monitor Updates

Update the CurvatureMonitor to use BackPACK for trace:

```python
# src/deconcnn/callbacks/curvature_monitor.py
from backpack import backpack, extensions
from hessian_eigenthings import compute_hessian_eigenthings

class CurvatureMonitor(BaseMonitor):
    def _compute_hutchinson_trace(self, model, data, targets):
        """Use BackPACK for efficient trace estimation."""
        model.zero_grad()
        
        with backpack(extensions.DiagHessian()):
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
        
        trace = sum(p.diag_h.sum().item() for p in model.parameters() 
                   if hasattr(p, 'diag_h'))
        return trace
    
    def _estimate_largest_eigenvalue(self, model, data, targets):
        """Use hessian-eigenthings for eigenvalue."""
        # Create minimal dataloader for efficiency
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=len(data))
        
        eigenvals, _ = compute_hessian_eigenthings(
            model, dataloader,
            loss=F.cross_entropy,
            num_eigenthings=1,
            use_cuda=data.is_cuda
        )
        return eigenvals[0]
```

## 4. Metric Logging Requirements

### Required Metrics and Frequencies:

| Metric | Frequency | Implementation |
|--------|-----------|----------------|
| loss_train_nats/bits | Every batch | LossSlopeLogger |
| acc_train, lr, wd | Every batch | LossSlopeLogger |
| loss_val_nats, acc_val | ¼ epoch | PyTorch Lightning val |
| grad_norm, weight_norm | ¼ epoch | LossSlopeLogger |
| lambda_max, hutch_trace | Every 500 steps | CurvatureMonitor |
| psd_tail, grad_var, resid_power | Every 2 epochs | NoiseMonitor |

### Data Export Requirements:
- All metrics must be available in CSV format
- Maintain dr_exp JSON format for compatibility
- Include timestamps and step numbers

## 5. Pre-implementation Checklist

- [ ] Install BackPACK: `uv add backpack-for-pytorch`
- [ ] Implement analysis library functions (knee detection, power law fitting)
- [ ] Create LossSlopeLogger for batch-level metrics
- [ ] Update CurvatureMonitor to use BackPACK
- [ ] Create CSV export utilities
- [ ] Test resource configurations on cluster
- [ ] Validate all metrics are logged at correct frequencies

## 6. Risk Mitigation Updates

### Memory Management:
- Use variant-specific worker counts based on testing
- Implement batch size reduction for OOM recovery
- Monitor memory usage during submission

### Data Integrity:
- Dual logging (JSON + CSV) ensures compatibility
- Validate metric completeness before analysis
- Implement checkpointing for long runs

### Performance:
- BackPACK reduces trace computation overhead
- Resource testing prevents oversubscription
- Chunked processing for large-scale analysis

## 7. Timeline Adjustments

### Day -1: Pre-implementation
- Install dependencies
- Implement analysis functions
- Create logging infrastructure

### Day 0: Setup & Validation
- Morning: Resource testing on cluster (4 hours)
- Afternoon: Update configs based on results
- Evening: Local validation of all components

### Days 1-4: Continue as planned
- With optimized worker configurations
- With proper logging infrastructure
- With CSV export for Tier 4

This consolidated update ensures Tier 3 properly sets up everything needed for successful Tier 4 analysis implementation.
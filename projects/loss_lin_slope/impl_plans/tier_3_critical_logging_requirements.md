# Tier 3 Critical Logging Requirements for Tier 4 Success

## Date: 2025-01-01

This document lists ONLY the essential logging and foundational code changes that MUST be implemented in Tier 3 before starting the 216-run sweep to enable Tier 4 analysis.

## 1. Batch-Level Metric Logging (CRITICAL)

### Required Implementation: LossSlopeLogger Callback

**What Tier 4 needs**: Per-batch metrics for power law fitting and early slope analysis

**Implementation**:
```python
# src/deconcnn/callbacks/loss_slope_logger.py
class LossSlopeLogger(Callback):
    """Logs metrics at batch level for loss slope analysis."""
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # MUST log these at EVERY batch
        pl_module.log_dict({
            "loss_train_nats": outputs["loss"],
            "loss_train_bits": outputs["loss"] / math.log(2),  # Tier 4 needs both
            "acc_train": outputs["acc"],
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "wd": trainer.optimizers[0].param_groups[0]["weight_decay"],
        }, on_step=True, on_epoch=False)
```

**Config addition**:
```yaml
# configs/callbacks/loss_lin_slope_metrics.yaml
defaults:
  - loss_slope_logger
```

## 2. Sub-Epoch Validation Logging

### Required: Quarter-Epoch Validation Frequency

**What Tier 4 needs**: Validation metrics at 0.25 epoch intervals for fine-grained analysis

**Implementation**:
```yaml
# In experiment configs
trainer:
  val_check_interval: 0.25  # Validate 4x per epoch
```

## 3. Enhanced Curvature Monitoring

### Required Updates to CurvatureMonitor

**What Tier 4 needs**: λ_max and trace every 500 steps for curvature analysis

**Implementation**:
```python
# Update src/deconcnn/callbacks/curvature_monitor.py
from backpack import backpack, extensions
from hessian_eigenthings import compute_hessian_eigenthings

class CurvatureMonitor(BaseMonitor):
    def __init__(self, compute_every_n_steps: int = 500, ...):
        # MUST be 500 steps for Tier 4 analysis
        
    def _compute_hutchinson_trace(self, model, data, targets):
        """Use BackPACK for efficient trace - required by Tier 4."""
        with backpack(extensions.DiagHessian()):
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
        
        trace = sum(p.diag_h.sum().item() for p in model.parameters() 
                   if hasattr(p, 'diag_h'))
        return trace
    
    def _estimate_largest_eigenvalue(self, model, data, targets):
        """Required for λ_max tracking."""
        # Use hessian-eigenthings
        eigenvals, _ = compute_hessian_eigenthings(
            model, dataloader,
            num_eigenthings=1,
            use_cuda=True
        )
        return eigenvals[0]
```

## 4. Gradient and Weight Norm Tracking

### Required: Add to LossSlopeLogger

**What Tier 4 needs**: grad_norm and weight_norm every 0.25 epochs

**Implementation**:
```python
def on_validation_epoch_end(self, trainer, pl_module):
    # Compute and log norms at validation frequency (0.25 epochs)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        pl_module.parameters(), float('inf')
    )
    weight_norm = torch.norm(
        torch.stack([p.norm() for p in pl_module.parameters()])
    )
    
    pl_module.log_dict({
        "grad_norm": grad_norm,
        "weight_norm": weight_norm,
    }, on_epoch=True)
```

## 5. Noise Metrics Schedule

### Required: Update NoiseMonitor Frequency

**What Tier 4 needs**: PSD tail, gradient variance, residual power every 2 epochs

**Implementation**:
```python
# In NoiseMonitor configuration
noise_monitor = NoiseMonitor(
    log_every_n_epochs=2,  # MUST be 2 for Tier 4
    # Other params...
)
```

## 6. Checkpoint Size Tracking

### Required: Add to checkpoint callback

**What Tier 4 needs**: Track checkpoint sizes for resource planning

**Implementation**:
```python
# Add to dr_exp_metrics callback or custom callback
def on_checkpoint_save(self, trainer, pl_module, checkpoint):
    size_mb = os.path.getsize(checkpoint_path) / 1e6
    pl_module.log("checkpoint_size_mb", size_mb, on_epoch=True)
```

## Critical Configuration Summary

```yaml
# configs/experiment/loss_lin_slope_base.yaml
defaults:
  - override /callbacks: loss_lin_slope_monitors

trainer:
  val_check_interval: 0.25  # Quarter-epoch validation

callbacks:
  loss_slope_logger:
    _target_: deconcnn.callbacks.loss_slope_logger.LossSlopeLogger
  
  curvature_monitor:
    _target_: deconcnn.callbacks.curvature_monitor.CurvatureMonitor
    compute_every_n_steps: 500  # MUST be 500
    
  noise_monitor:
    _target_: deconcnn.callbacks.noise_monitor.NoiseMonitor
    log_every_n_epochs: 2  # MUST be 2
```

## What We're NOT Doing in Tier 3

1. ❌ CSV export (Tier 4 will handle JSON→CSV conversion)
2. ❌ Analysis functions (Tier 4 will implement these)
3. ❌ Deciding final workers per GPU (resource testing will determine)

## Pre-Launch Checklist

Before starting the 216-run sweep, verify:

- [ ] LossSlopeLogger logs ALL batch-level metrics
- [ ] Validation runs at 0.25 epoch intervals
- [ ] CurvatureMonitor uses BackPACK for trace
- [ ] CurvatureMonitor computes λ_max every 500 steps
- [ ] NoiseMonitor logs every 2 epochs
- [ ] All metrics include step/epoch timestamps
- [ ] Resource testing completed to determine workers/GPU

## Why These Are Critical

Without these exact logging frequencies and metrics, Tier 4 cannot:
- Compute early slope (epochs 5-15) → needs batch-level loss
- Detect burn-in via AIC → needs fine-grained loss curve
- Analyze curvature timing → needs 500-step λ_max
- Validate units invariance → needs both nats and bits
- Generate all required plots → needs complete metric set

These logging changes are the MINIMUM required to enable Tier 4's analysis suite.
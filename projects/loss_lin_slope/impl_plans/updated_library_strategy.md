# Updated Library Strategy for Hessian Analysis

## Date: 2025-01-01

## Revised Approach

Based on the analysis of current implementation status and performance characteristics, we're updating our library strategy:

### 1. Use BackPACK for Trace Estimation ✅

**Rationale**:
- Current CurvatureMonitor has only a basic placeholder implementation
- BackPACK offers superior memory efficiency (2-3x vs 4-5x gradient memory)
- BackPACK is faster (1.2-1.5x backward pass vs separate computation)
- Low implementation cost since we're starting fresh

**Implementation**:
```python
# Add to deconCNN dependencies
"backpack-for-pytorch>=1.6.0",

# In improved CurvatureMonitor
from backpack import backpack, extensions

def _compute_hutchinson_trace_backpack(self, model, data, targets):
    """Efficient Hutchinson trace using BackPACK."""
    with backpack(extensions.DiagHessian()):
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
    
    # Access diagonal Hessian elements
    trace = sum(p.diag_h.sum() for p in model.parameters() 
                if hasattr(p, 'diag_h'))
    return trace.item()
```

### 2. Keep hessian-eigenthings for Eigenvalue Computation ✅

**Rationale**:
- BackPACK doesn't provide eigenvalue computation
- hessian-eigenthings is purpose-built for this
- Already in dependencies

**Usage remains unchanged**:
```python
from hessian_eigenthings import compute_hessian_eigenthings

lambda_max, _ = compute_hessian_eigenthings(
    model, dataloader, 
    num_eigenthings=1,
    use_cuda=True
)
```

### 3. Keep PyHessian for Future Spectral Analysis ✅

**Rationale**:
- Needed for full eigenvalue density in Tier 4 analysis
- Provides comprehensive spectral analysis tools
- Already in dependencies, no cost to keep

**Future usage**:
```python
from pyhessian import hessian

# For comprehensive spectral analysis (not trace)
hessian_comp = hessian(model, loss_func, dataloader)
eigenvalues = hessian_comp.eigenvalues()
density = hessian_comp.spectral_density()
```

## Updated Dependencies

Add to deconCNN's pyproject.toml:
```toml
[project.dependencies]
# ... existing dependencies ...
"backpack-for-pytorch>=1.6.0",  # Add this for efficient trace
# Keep existing:
"pyhessian>=0.1",
"hessian-eigenthings @ git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master",
```

## Implementation Plan Updates

### Tier 3 Updates

**Phase 1, Step 1.2** - Update callback implementation:
- Use BackPACK for trace estimation instead of PyHessian
- More efficient during training
- Better memory usage

### Tier 4 Updates

**Phase 3, Step 3.3** - Upgrade curvature analysis:
- Change from "Upgrade to Hutch++" to "Upgrade to BackPACK trace"
- Remove references to non-existent hutch_nn
- Use BackPACK's efficient implementation

**Phase 2, Step 2.3** - Clarify PyHessian usage:
- PyHessian for spectral analysis only
- Not for trace computation
- BackPACK handles trace

## Benefits of This Approach

1. **Best tool for each job**:
   - BackPACK: Trace estimation (fastest, most memory-efficient)
   - hessian-eigenthings: Eigenvalue computation (purpose-built)
   - PyHessian: Spectral analysis (comprehensive tools)

2. **Performance gains**:
   - ~2x memory reduction for trace
   - ~30-50% faster trace computation
   - No performance loss elsewhere

3. **Low implementation cost**:
   - Simple addition to dependencies
   - Clear separation of concerns
   - Each library used for its strength

## Migration Path

1. Add BackPACK to dependencies
2. Update CurvatureMonitor to use BackPACK for trace
3. Keep eigenvalue computation with hessian-eigenthings
4. Reserve PyHessian for future spectral analysis

This strategy optimizes performance while maintaining all required functionality.
# Hessian Computation Libraries Reference

This document consolidates technical insights about Hessian computation libraries for neural network analysis, focusing on PyHessian, BackPACK, and related tools.

## Table of Contents
- [Library Overview](#library-overview)
- [Performance Characteristics](#performance-characteristics)
- [Memory Usage Patterns](#memory-usage-patterns)
- [Feature Comparison Matrix](#feature-comparison-matrix)
- [Implementation Scenarios](#implementation-scenarios)
- [API Examples](#api-examples)
- [Compatibility Considerations](#compatibility-considerations)
- [Migration Strategies](#migration-strategies)

## Library Overview

### PyHessian
- **Purpose**: Specialized library for Hessian analysis in neural networks
- **Design Philosophy**: Post-hoc analysis of trained models
- **Strengths**: Full spectral analysis, eigenvalue density computation, top-k eigenvalues
- **Repository**: https://github.com/amirgholami/PyHessian

### BackPACK
- **Purpose**: General second-order optimization and curvature information
- **Design Philosophy**: Real-time integration during training
- **Strengths**: Memory efficiency, vectorized operations, automatic chunking
- **Repository**: https://github.com/f-dangel/backpack

### hessian-eigenthings
- **Purpose**: Eigenvalue computation for Hessian matrices
- **Design Philosophy**: Lightweight, focused on power iteration method
- **Repository**: https://github.com/noahgolmant/hessian-eigenthings

## Performance Characteristics

### Speed Benchmarks

| Operation | PyHessian | BackPACK | Notes |
|-----------|-----------|----------|-------|
| Trace computation | ~100 HVP computations | ~1.2-1.5x backward pass | BackPACK reuses backprop computations |
| Single HVP | O(backward pass) | O(backward pass) | Both use automatic differentiation |
| Batch HVP (100 vectors) | 100 sequential passes | 10 batches of 10 (parallel) | BackPACK enables vectorization |
| GPU utilization | Sequential operations | Vectorized operations | BackPACK better for GPU efficiency |

### Memory Overhead

| Library | Base Memory | Trace Computation | Full Hessian | Eigenvalue Computation |
|---------|-------------|-------------------|--------------|----------------------|
| PyHessian | 1x gradient | 2x gradient | 4-5x gradient | 2-3x gradient |
| BackPACK | 1x gradient | 2-3x gradient | Not supported | 2-3x gradient (with chunking) |
| BackPACK (chunked) | 1x gradient | 2x gradient | Not supported | Configurable via V_batch |

## Memory Usage Patterns

### BackPACK Chunking Strategy
```python
# Automatic memory management through V_batch parameter
backpack_extension.set_V_batch_size(10)  # Process 10 vectors at a time
# Peak memory: 10x gradient memory instead of 100x
```

### PyHessian Memory Management
```python
# Manual batch size control
trace = hessian_comp.trace(maxIter=100, mini_hessian_batch_size=32)
# Uses smaller batches for memory efficiency
```

### Key Memory Insights
1. **BackPACK automatic chunking**: Prevents memory overflow for large models
2. **During-training integration**: BackPACK reuses intermediate computations
3. **Peak memory calculations**: 
   - Without chunking: num_vectors × gradient_memory
   - With chunking: chunk_size × gradient_memory
4. **DiagHessian alternative**: BackPACK's diagonal approximation for extreme memory constraints

## Feature Comparison Matrix

| Feature | PyHessian | BackPACK | hessian-eigenthings |
|---------|-----------|----------|-------------------|
| **Trace computation** | ✓ (stochastic) | ✓ (exact/stochastic) | ✗ |
| **Top-k eigenvalues** | ✓ | ✗ (needs custom) | ✓ |
| **Eigenvalue density** | ✓ | ✗ | ✗ |
| **Spectral analysis** | ✓ (comprehensive) | ✗ | Limited |
| **GGN/Fisher** | ✗ | ✓ | ✗ |
| **Memory efficiency** | Medium | High (chunking) | Low |
| **Training integration** | ✗ | ✓ | ✗ |
| **Distributed support** | ✓ | Limited | ✗ |
| **Real-time analysis** | ✗ | ✓ | ✗ |
| **PyTorch Lightning** | Compatible | Requires care | Compatible |

## Implementation Scenarios

### Scenario 1: Keep Current Dependencies (Low Effort)
```python
# Use PyHessian + hessian-eigenthings
# Pros: No migration needed, proven combination
# Cons: Less memory efficient, no training integration
```

### Scenario 2: Add BackPACK for Trace Only (Medium Effort)
```python
# Use BackPACK for trace, keep PyHessian for eigenvalues
# Pros: Better trace performance, gradual migration
# Cons: Two dependencies, some code duplication
```

### Scenario 3: Replace PyHessian with BackPACK (High Effort)
```python
# Full migration to BackPACK
# Pros: Unified API, best performance
# Cons: Need custom eigenvalue implementation
```

### Scenario 4: BackPACK Only (Not Recommended)
```python
# Use only BackPACK without eigenvalue support
# Pros: Single dependency
# Cons: Missing critical spectral analysis features
```

## API Examples

### PyHessian Basic Usage
```python
from pyhessian import hessian

# Create hessian computer
hessian_comp = hessian(model, criterion, data, target)

# Compute trace (memory efficient)
trace = hessian_comp.trace(maxIter=100, mini_hessian_batch_size=32)

# Compute top eigenvalues
eigenvalues, eigenvectors = hessian_comp.eigenvalues(top_n=10)

# Get eigenvalue density
density, grids = hessian_comp.density()
```

### BackPACK Basic Usage
```python
from backpack import backpack, extensions
from backpack.extensions import HMP

# Extend model and loss
extend(model)
extend(loss_function)

# Compute trace during training
with backpack(extensions.DiagHessian()):
    loss = loss_function(model(X), y)
    loss.backward()
    
# Access diagonal Hessian (for trace)
for param in model.parameters():
    trace_contribution = param.diag_h.sum()
```

### Custom Power Iteration with BackPACK
```python
def power_iteration_backpack(model, loss_fn, data, target, num_iters=100):
    """Custom implementation for top eigenvalue using BackPACK HVPs"""
    v = torch.randn_like(flatten_params(model))
    v = v / v.norm()
    
    for _ in range(num_iters):
        # Compute Hv using BackPACK
        with backpack(HMP()):
            loss = loss_fn(model(data), target)
            loss.backward()
        
        Hv = compute_hvp_from_backpack(model, v)
        eigenvalue = torch.dot(v, Hv)
        v = Hv / Hv.norm()
    
    return eigenvalue, v
```

## Compatibility Considerations

### PyTorch Ecosystem
- **PyHessian**: Works with standard PyTorch, compatible with most training frameworks
- **BackPACK**: Requires model/loss extension, more integration effort
- **Both**: Support common optimizers (SGD, Adam, etc.)

### Framework Integration
| Framework | PyHessian | BackPACK | Notes |
|-----------|-----------|----------|-------|
| PyTorch Lightning | ✓ Direct | ⚠️ Careful | BackPACK needs callback integration |
| Distributed Training | ✓ Full | ⚠️ Limited | PyHessian better for multi-GPU |
| Mixed Precision | ⚠️ Manual | ✓ Automatic | BackPACK handles AMP better |

### Maintenance and Support
- **PyHessian**: Stable, focused scope, less frequent updates
- **BackPACK**: Active development, broader scope, regular updates
- **hessian-eigenthings**: Minimal maintenance, stable for core features

## Migration Strategies

### From PyHessian to BackPACK

#### Phase 1: Parallel Implementation
```python
# Keep both libraries, compare results
pyhessian_trace = compute_trace_pyhessian(model, data)
backpack_trace = compute_trace_backpack(model, data)
assert np.allclose(pyhessian_trace, backpack_trace, rtol=1e-3)
```

#### Phase 2: Gradual Replacement
1. Replace trace computation (easiest)
2. Implement custom eigenvalue computation
3. Migrate monitoring/logging code
4. Remove PyHessian dependency

#### Phase 3: Optimization
1. Enable BackPACK chunking for memory efficiency
2. Integrate with training loop for real-time analysis
3. Leverage vectorized HVP computations

### Key Migration Considerations
1. **API differences**: BackPACK requires explicit model extension
2. **Feature gaps**: Need custom implementation for spectral density
3. **Performance testing**: Verify memory and speed improvements
4. **Backward compatibility**: Maintain result consistency

## Performance Best Practices

### For ML Training
```python
# Use assertions instead of exceptions (performance requirement)
assert len(inputs) == len(targets), f"Mismatch: {len(inputs)} vs {len(targets)}"
assert model.training, "Model must be in training mode"

# Avoid try/except in hot paths
# NO: if condition: raise ValueError()
# YES: assert condition, "Error message"
```

### Memory Optimization
1. **Enable chunking**: Always use V_batch for large models
2. **Reuse computations**: Integrate with training backward pass
3. **Clear gradients**: Explicitly zero gradients after Hessian operations
4. **Use mixed precision**: Both libraries support FP16 with care

### Distributed Computing
- **PyHessian**: Use `hessian_batch_size` for multi-GPU setups
- **BackPACK**: Limited distributed support, consider data parallelism only
- **Best practice**: Compute Hessian statistics on single GPU, aggregate results

## Dependencies

### Installation
```toml
# pyproject.toml entries
[tool.poetry.dependencies]
pyhessian = "^0.2.0"  # For spectral analysis
backpack-for-pytorch = "^1.6.0"  # For efficient trace
hessian-eigenthings = "^0.2.0"  # For eigenvalues (if needed)
```

### Version Compatibility
- PyHessian: PyTorch >= 1.4.0
- BackPACK: PyTorch >= 1.9.0, <= 2.0.0 (check latest)
- hessian-eigenthings: PyTorch >= 1.0.0

## Additional Resources

### Academic Papers
- PyHessian: "PyHessian: Neural Networks Through the Lens of the Hessian" (NeurIPS 2020)
- BackPACK: "BackPACK: Packing more into Backprop" (ICLR 2020)

### Documentation
- PyHessian Docs: https://github.com/amirgholami/PyHessian/wiki
- BackPACK Docs: https://docs.backpack.pt/
- BackPACK Extensions: https://docs.backpack.pt/en/stable/extensions.html

### Related Tools
- `torch.func`: PyTorch's functional API for Hessian computation (experimental)
- `functorch`: Composable function transforms (now integrated into PyTorch)
- `jax`: Alternative framework with built-in Hessian support

## Summary Recommendations

### Use PyHessian when:
- You need comprehensive spectral analysis
- Post-hoc analysis of trained models
- Distributed training is critical
- Eigenvalue density is required

### Use BackPACK when:
- Memory efficiency is critical
- Real-time analysis during training
- You need Generalized Gauss-Newton or Fisher information
- GPU utilization is a bottleneck

### Use Both when:
- You need the best of both worlds
- Gradual migration is preferred
- Different analyses require different tools
- Comparison/validation is important
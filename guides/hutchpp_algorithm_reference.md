# Hutch++ Algorithm Reference

## Overview

The Hutch++ algorithm is an optimal stochastic trace estimation method that significantly improves upon the standard Hutchinson estimator. Developed by Meyer, Musco, Musco, and Woodruff (2021), it achieves a quadratic improvement in sample complexity.

## Key Improvements over Standard Hutchinson

1. **Sample Complexity**: O(1/ε) vs O(1/ε²) matrix-vector products
2. **Variance**: Bounded by ε·tr(A)² vs tr(A)² for standard Hutchinson
3. **Efficiency**: Optimal for any randomized algorithm using matrix-vector products

## Mathematical Foundation

### Standard Hutchinson Estimator
For a symmetric matrix A and random vectors z with E[zz^T] = I:
```
tr(A) ≈ (1/m) Σ(z_i^T A z_i)
```
Variance: Var ≤ (2/m) ||A||_F²

### Hutch++ Algorithm
Combines low-rank approximation with stochastic estimation:
```
tr(A) = tr(Q^T A Q) + tr((I - QQ^T)A)
```
Where Q is an orthonormal basis capturing dominant eigenspaces.

## Algorithm Structure

### Three-Phase Approach
1. **Phase 1**: Compute low-rank approximation using randomized range finding
   - Generate random test matrix S (m/3 × n)
   - Compute Y = AS via matrix-vector products
   - Orthogonalize Y to get Q (via QR decomposition)
2. **Phase 2**: Exactly compute trace of low-rank component
   - Compute B = Q^T A Q (small k×k matrix)
   - Exact trace: tr(B) = tr(Q^T A Q)
3. **Phase 3**: Apply Hutchinson to the residual
   - Sample m/3 random vectors z_i
   - Estimate: tr((I-QQ^T)A) ≈ (1/s)Σ z_i^T(A - Q(Q^T A Q)Q^T)z_i

### Query Allocation
- Total queries: m
- Phase 1: m/3 queries for range finding
- Phase 2: m/3 queries for exact trace computation
- Phase 3: m/3 queries for residual estimation

## Implementation Details

### Random Vector Choice
- **Rademacher**: Entries uniformly ±1 (most common)
- **Gaussian**: N(0,1) entries (theoretically cleaner)
- **Sparse**: For very large matrices

### Numerical Considerations
1. **QR Decomposition**: Use stable methods (Householder, MGS)
2. **Memory Efficiency**: Don't form full matrices when possible
3. **Parallelization**: Matrix-vector products can be batched

### PyTorch Integration
```python
def hutchpp_pytorch(hessian_vector_product, dim, num_queries):
    """PyTorch-compatible Hutch++ for neural network Hessians"""
    # Key considerations:
    # - Use torch.cuda.empty_cache() between phases
    # - Batch matrix-vector products when possible
    # - Handle gradient accumulation properly
```

## Variance Reduction Mechanism

The variance reduction comes from:
1. **Spectrum Capture**: Low-rank approximation captures large eigenvalues
2. **Residual Properties**: (I - QQ^T)A has much smaller trace
3. **Optimal Splitting**: Theory shows m/3 split is asymptotically optimal

### Mathematical Guarantee
For PSD matrix A with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ:
- Standard Hutchinson variance: O(Σλᵢ²)
- Hutch++ variance: O(ε·(Σλᵢ)²) when m = O(1/ε)

## Best Practices

### For Neural Network Hessians
1. **Batch Computation**: Group matrix-vector products to utilize GPU
2. **Mixed Precision**: Use float32 for stability in QR decomposition
3. **Checkpointing**: Save intermediate results for large networks
4. **Warm Start**: Reuse Q basis across similar Hessians

### Error Handling
1. **Rank Deficiency**: Handle when Y doesn't have full rank
2. **Numerical Stability**: Check orthogonality of Q periodically
3. **Memory Limits**: Implement streaming version for very large matrices

## Performance Characteristics

### Computational Cost
- Matrix-vector products: m (dominant cost)
- QR decomposition: O(nm²/9) (usually negligible)
- Memory: O(nm/3) for storing basis Q

### When to Use Hutch++
- **Good for**: Matrices with decay in eigenvalues, large sparse matrices
- **Less suitable**: Full rank matrices with uniform spectrum
- **Optimal when**: Need high accuracy (small ε)

## Existing Implementations

### Reference Implementations
1. **Original MATLAB**: github.com/RaphaelArkadyMeyerNYU/HutchPlusPlus
   - Contains simplified code for easy translation to other languages
   - Includes NA-Hutch++ (non-adaptive) variant
   - Provides subspace projection algorithms
2. **PyTorch Trace**: github.com/akshayka/hessian_trace_estimation
   - Jupyter notebook demonstrating Hessian trace estimation
   - Uses automatic differentiation with PyTorch
   - Specifically designed for neural network applications
3. **Python Sketching**: github.com/11hifish/OptSketchTraceEst
   - Contains both sequential and parallel implementations
   - Includes NA-Hutch++, Hutch++, and Hutchinson algorithms
   - Provides Oracle-based implementations (matrix-vector product only)
4. **Library Integrations**: 
   - PyLops: Integrated into linear operators library
   - SciPy: Available in recent versions for sparse matrix operations
   - BackPACK: Provides Hutchinson trace estimation for PyTorch

### Integration Examples
```python
# Example: Integration with PyHessian
from pyhessian import hessian
def compute_trace_hutchpp(model, loss, data):
    H = hessian(model, loss, data)
    return hutchpp(H.hessian_vector_product, param_count, queries)
```

### Pseudocode Implementation
```python
def hutchpp(matrix_vector_product, n, m):
    """
    Hutch++ algorithm for trace estimation
    Args:
        matrix_vector_product: Function that computes Av for vector v
        n: Dimension of matrix
        m: Total number of matrix-vector queries
    Returns:
        Trace estimate
    """
    k = m // 3  # Queries per phase
    
    # Phase 1: Low-rank approximation
    S = generate_random_matrix(n, k)  # Gaussian or Rademacher
    Y = []
    for i in range(k):
        Y.append(matrix_vector_product(S[:, i]))
    Y = np.column_stack(Y)
    Q, _ = np.linalg.qr(Y)  # Orthogonalize
    
    # Phase 2: Exact trace of low-rank part
    B = np.zeros((k, k))
    for i in range(k):
        AQi = matrix_vector_product(Q[:, i])
        B[:, i] = Q.T @ AQi
    trace_lowrank = np.trace(B)
    
    # Phase 3: Hutchinson on residual
    trace_residual = 0
    for _ in range(k):
        z = generate_random_vector(n)  # Rademacher
        Az = matrix_vector_product(z)
        QQTAz = Q @ (Q.T @ Az)
        residual = Az - QQTAz
        trace_residual += z.T @ residual
    trace_residual /= k
    
    return trace_lowrank + trace_residual
```

## Common Pitfalls

1. **Incorrect Query Split**: Not using m/3 for each phase
2. **Memory Issues**: Forming full matrices instead of using products
3. **Random Seed**: Not controlling randomness for reproducibility
4. **Non-PSD Matrices**: Algorithm works but theory assumes PSD

## References

1. Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2021). "Hutch++: Optimal Stochastic Trace Estimation"
2. Avron, H., & Toledo, S. (2011). "Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix"
3. Ubaru, S., Chen, J., & Saad, Y. (2017). "Fast estimation of tr(f(A)) via stochastic Lanczos quadrature"

## Practical Tips for deconCNN Integration

1. **Callback Integration**: Hutch++ can replace Hutchinson in CurvatureMonitor
2. **Frequency**: Due to efficiency, can run more frequently than standard
3. **Validation**: Compare with exact trace on small models
4. **Monitoring**: Track variance reduction vs standard Hutchinson

## Implementation Lessons Learned

### Function Decomposition for Maintainability
Break the main algorithm into three helper functions to manage complexity:
```python
def _compute_low_rank_approximation(hvp, params, k) -> List[Tuple[Tensor, Size]]:
    """Phase 1: Randomized range finding with QR decomposition"""
    
def _compute_exact_trace(hvp, q_matrices) -> float:
    """Phase 2: Exact trace of low-rank component"""
    
def _compute_residual_trace(hvp, params, q_matrices, k) -> float:
    """Phase 3: Hutchinson on residual"""
```

### Numerical Stability Considerations
1. **QR vs SVD**: Always provide SVD fallback when QR decomposition fails
2. **Variable Overwriting**: Avoid reusing loop variables to prevent linting issues
3. **Truncation Handling**: Create separate variables for truncated matrices
4. **Memory Management**: Use parameter-wise decomposition for large networks

### PyTorch-Specific Optimizations
```python
# Efficient parameter handling
trainable_params = [p for p in model.parameters() if p.requires_grad]

# Stable random generation
s_param = torch.randint_like(p, high=2).float() * 2 - 1  # Rademacher

# Memory-efficient reshaping
q_col = q_matrix[:, i].reshape(param_shape)

# Robust QR with fallback
try:
    Q, _ = torch.linalg.qr(Y, mode="reduced")
except RuntimeError:
    U, _, _ = torch.linalg.svd(Y, full_matrices=False)
```

### Testing Strategy Insights
1. **Synthetic Models**: Use simple linear/quadratic models for exact validation
2. **Consistency Tests**: Compare order of magnitude rather than exact values
3. **Fallback Verification**: Test with large models to trigger error paths
4. **Reproducibility**: Always test with fixed random seeds
5. **Edge Cases**: Test empty models and numerical stability

### Performance vs Accuracy Trade-offs
- **30 queries minimum**: Below this, variance reduction may not be observable
- **150 queries optimal**: Good balance for most neural network applications  
- **Query allocation**: Strict 1/3 split is crucial for theoretical guarantees
- **Fallback threshold**: Monitor for RuntimeError and gracefully degrade to Hutchinson

### Variance Reduction Reality Check
In practice, variance reduction is most pronounced when:
- Matrix has significant spectral decay (neural network Hessians typically do)
- Sufficient queries (≥30) to capture low-rank structure
- Model is not too small (small models may not show clear benefits)

The theoretical O(1/ε) vs O(1/ε²) improvement is asymptotic and may not always be visible in finite-sample regimes with noisy neural network training.
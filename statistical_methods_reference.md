# Statistical Methods Reference

## Python Libraries for Statistical Analysis

### Correlation Analysis

#### 1. **Pingouin** (Recommended for Correlation)
- **Version**: 0.5.5+
- **Key Features**:
  - Simple, uniform API for statistical tests
  - Comprehensive output including confidence intervals, p-values, Bayes factors
  - Supports Pearson, Spearman, Kendall, and robust methods (bicor, percbend, shepherd)
  - Built-in partial correlation with multiple covariates
  - Handles missing data gracefully

**API Changes**:
- Since v0.4.0: Use `alternative` instead of `tail` parameter
- Methods: "two-sided" (default), "greater", "less"

**Example**:
```python
import pingouin as pg
# Basic correlation
result = pg.corr(x, y, method="pearson", alternative="two-sided")
# Partial correlation
result = pg.partial_corr(data=df, x="x", y="y", covar=["z1", "z2"])
```

#### 2. **SciPy**
- No native partial correlation implementation
- Requires manual residual calculation for partial correlation
- Good for basic correlations via scipy.stats.pearsonr, spearmanr

#### 3. **Statsmodels**
- More complex API for correlation analysis
- Used internally by Pingouin for some calculations
- Better for regression-based approaches

### Bootstrap Confidence Intervals

#### **scipy.stats.bootstrap** (v1.7.0+)
- **Key Features**:
  - Efficient vectorized implementation
  - Multiple methods: percentile, basic, BCa (bias-corrected accelerated)
  - Handles multi-sample statistics
  - Memory-efficient with batch processing

**Important Notes**:
- BCa method can fail with degenerate distributions (returns NaN)
- Percentile method more stable but less accurate for skewed distributions
- Always validate results when using BCa

**Example**:
```python
from scipy.stats import bootstrap
import numpy as np

# Single sample
res = bootstrap((data,), np.mean, confidence_level=0.95, 
                n_resamples=9999, method="percentile")
ci = (res.confidence_interval.low, res.confidence_interval.high)

# Multi-sample (e.g., correlation)
res = bootstrap((x, y), lambda x, y: np.corrcoef(x, y)[0, 1], 
                n_resamples=9999)
```

## Best Practices

### Correlation Analysis
1. **Method Selection**:
   - Pearson: Linear relationships with normal distributions
   - Spearman: Monotonic relationships, ordinal data, or outliers
   - Robust methods (bicor, percbend): When outliers present

2. **Sample Size**:
   - Minimum 30 observations for reliable estimates
   - Larger samples (50+) for confidence intervals

3. **Partial Correlation**:
   - Controls for confounding variables
   - Can use multiple covariates
   - Consider non-linear relationships with Spearman

### Bootstrap Confidence Intervals
1. **Number of Resamples**:
   - Default: 9999 (good balance of accuracy/speed)
   - For publication: 10000+
   - For quick estimates: 999-1999

2. **Method Choice**:
   - Percentile: Most stable, good for symmetric distributions
   - Basic: Similar to percentile, different coverage
   - BCa: Best coverage but can fail; use with caution

3. **Random State**:
   - Always set for reproducibility
   - Use np.random.default_rng() for modern NumPy

## Common Pitfalls

1. **Zero Variance Data**: Correlation returns NaN; check data first
2. **Array Length Mismatch**: Ensure paired data has equal lengths
3. **BCa Failures**: Falls back to percentile method or handle NaN
4. **Interpretation**: Correlation ≠ causation
5. **Multiple Testing**: Apply corrections when testing many correlations

## Performance Considerations

- Pingouin adds ~2MB to dependencies but provides cleaner API
- Bootstrap is O(n × n_resamples); consider parallelization for large datasets
- Partial correlation creates DataFrames; may be slower for very large datasets
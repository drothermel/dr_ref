# Exponential Fitting in Python: Complete Reference

## Overview

Exponential fitting is the process of fitting exponential functions to data, commonly used in scientific computing for modeling growth, decay, and other exponential processes. This reference covers scipy-based methods, best practices, and implementation patterns for robust exponential curve fitting.

## Mathematical Foundation

### Exponential Function Forms

**Simple Exponential (2 parameters):**
```
y = a * exp(b * x)
```

**Exponential with Offset (3 parameters):**
```
y = a * exp(b * x) + c
```

**Bi-exponential (5 parameters):**
```
y = a * exp(k1 * x) + b * exp(k2 * x) + c
```

Most applications use the 3-parameter form as it provides good flexibility while remaining computationally tractable.

## Core Methods

### 1. Log-Linear Regression (Fast Initial Estimates)

**Principle:** Transform exponential to linear via logarithm
```python
# For y = a * exp(b * x), take log: log(y) = log(a) + b * x
p = np.polyfit(x, np.log(y), 1)
a = np.exp(p[1])  # intercept -> a
b = p[0]          # slope -> b
```

**Advantages:**
- Fast computation
- Always converges
- Good for initial parameter estimates

**Limitations:**
- Assumes c = 0 (no offset)
- Biased toward small y values due to log transform
- Requires y > 0 (fails with negative values)
- Homoscedastic noise assumption often violated

### 2. Non-linear Curve Fitting (Most Accurate)

**Principle:** Direct optimization using scipy.optimize.curve_fit
```python
from scipy.optimize import curve_fit

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

popt, pcov = curve_fit(exponential, x, y, p0=initial_guess)
```

**Advantages:**
- Handles all exponential forms
- Accounts for actual noise characteristics
- Most statistically correct

**Limitations:**
- Requires good initial guess
- May not converge
- Sensitive to outliers
- Computationally intensive

### 3. Hybrid Approach (Recommended)

**Principle:** Combine both methods for robustness
1. Use log-linear regression for initial estimates
2. Refine with non-linear curve fitting

```python
# Step 1: Log-linear for initial guess
log_y = np.log(np.maximum(y, 1e-10))  # Handle edge cases
p = np.polyfit(x, log_y, 1, w=np.sqrt(np.maximum(y, 1e-10)))
initial_a, initial_b = np.exp(p[1]), p[0]
initial_c = np.min(y) * 0.1  # Small offset estimate

# Step 2: Non-linear refinement
popt, pcov = curve_fit(
    exponential, x, y, 
    p0=[initial_a, initial_b, initial_c],
    maxfev=5000
)
```

## Best Practices and Implementation Patterns

### Input Validation

```python
def _validate_exponential_inputs(x: np.ndarray, y: np.ndarray, method: str) -> None:
    """Comprehensive input validation for exponential fitting."""
    
    # Array validation
    x = np.asarray(x)
    y = np.asarray(y)
    
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    
    if len(x) != len(y):
        raise ValueError(f"Array length mismatch: x={len(x)}, y={len(y)}")
    
    if len(x) < 4:
        raise ValueError(f"Insufficient data points: {len(x)} (minimum 4)")
    
    # Check for NaN/inf values
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Input arrays contain NaN or infinite values")
    
    # Log-linear methods require positive y values
    if method in ['log_linear', 'weighted_log', 'hybrid'] and np.any(y <= 0):
        raise ValueError(f"Method '{method}' requires positive y values, got min={np.min(y)}")
    
    # Check for constant data
    if np.ptp(x) == 0:
        raise ValueError("Independent variable x has zero range")
    if np.ptp(y) == 0:
        raise ValueError("Dependent variable y has zero range")
```

### Weighted Log-Linear Fitting

Address bias in log-transformed data by using weights proportional to y:

```python
def _weighted_log_linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Weighted log-linear fit to reduce bias from log transformation."""
    
    # Ensure positive values
    y_safe = np.maximum(y, np.finfo(float).eps)
    log_y = np.log(y_safe)
    
    # Weights proportional to sqrt(y) reduce log-transform bias
    weights = np.sqrt(y_safe)
    
    # Weighted polynomial fit
    p = np.polyfit(x, log_y, 1, w=weights)
    
    return np.exp(p[1]), p[0]  # a, b
```

### Parameter Bounds and Constraints

```python
def _get_reasonable_bounds(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Generate reasonable parameter bounds based on data characteristics."""
    
    y_min, y_max = np.min(y), np.max(y)
    x_range = np.ptp(x)
    
    # Bounds: [a_min, b_min, c_min], [a_max, b_max, c_max]
    bounds_lower = [
        1e-10,          # a: small positive
        -10 / x_range,  # b: reasonable decay rate
        -abs(y_min)     # c: allow negative offset
    ]
    
    bounds_upper = [
        10 * y_max,     # a: generous amplitude
        10 / x_range,   # b: reasonable growth rate  
        y_max           # c: positive offset up to max y
    ]
    
    return bounds_lower, bounds_upper
```

### R-squared Calculation

Since scipy.optimize.curve_fit doesn't provide R², calculate it manually:

```python
def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination (R²)."""
    
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # R-squared
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

# Alternative using sklearn
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

### Error Analysis and Uncertainty

```python
def _compute_parameter_errors(pcov: np.ndarray) -> np.ndarray:
    """Compute one-sigma parameter uncertainties from covariance matrix."""
    
    if pcov is None or np.any(np.diag(pcov) < 0):
        return np.full(len(pcov), np.nan)
    
    return np.sqrt(np.diag(pcov))

def _compute_prediction_intervals(
    x_pred: np.ndarray, 
    params: np.ndarray, 
    pcov: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute prediction intervals for fitted exponential."""
    # Implementation involves Jacobian computation and statistical inference
    # See scipy.optimize documentation for details
```

## Modern Python Implementation (2024/2025)

### Dependencies

```python
# Core dependencies
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score
import warnings
from typing import Tuple, Dict, Any, Optional, Union
```

### Complete Implementation Pattern

```python
def fit_exponential(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'hybrid',
    initial_guess: Optional[Tuple[float, float, float]] = None,
    bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    weights: Optional[np.ndarray] = None,
    return_details: bool = False
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict[str, Any]]]:
    """
    Robust exponential fitting with multiple methods and comprehensive validation.
    
    This function implements the current best practices for exponential curve fitting
    in scientific Python, combining multiple approaches for optimal results.
    """
    
    # 1. Input validation
    _validate_exponential_inputs(x, y, method)
    
    # 2. Method dispatch
    if method == 'log_linear':
        params, details = _fit_log_linear(x, y)
    elif method == 'weighted_log':
        params, details = _fit_weighted_log_linear(x, y)
    elif method == 'curve_fit':
        params, details = _fit_curve_fit(x, y, initial_guess, bounds, weights)
    elif method == 'hybrid':
        params, details = _fit_hybrid(x, y, bounds, weights)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 3. Compute R-squared
    y_pred = _exponential_func(x, *params)
    r2 = _compute_r_squared(y, y_pred)
    
    # 4. Return results
    if return_details:
        details.update({
            'method': method,
            'r_squared': r2,
            'y_pred': y_pred,
            'residuals': y - y_pred
        })
        return params, r2, details
    else:
        return params, r2
```

## Common Pitfalls and Solutions

### 1. Numerical Overflow/Underflow

**Problem:** Large exponential values cause overflow
**Solution:** Use log-space computations and reasonable bounds

```python
# Instead of: y = a * np.exp(b * x) 
# Use bounds to prevent overflow
bounds = ([-np.inf, -50, -np.inf], [np.inf, 50, np.inf])
```

### 2. Poor Initial Guesses

**Problem:** curve_fit fails to converge
**Solution:** Always use log-linear estimates as starting point

```python
# Always compute initial guess from log-linear fit
if initial_guess is None:
    a_init, b_init = _log_linear_fit(x, y)
    c_init = np.min(y) * 0.1
    initial_guess = [a_init, b_init, c_init]
```

### 3. Negative or Zero Values

**Problem:** Log-transform fails with non-positive data
**Solution:** Offset data or use different approaches

```python
# Option 1: Offset data
y_offset = y - np.min(y) + 1e-10

# Option 2: Use curve_fit only
if np.any(y <= 0):
    method = 'curve_fit'
```

### 4. High Noise or Outliers

**Problem:** Exponential fitting sensitive to outliers
**Solution:** Use robust methods or outlier detection

```python
# Robust fitting with outlier removal
from scipy import stats
z_scores = np.abs(stats.zscore(y))
mask = z_scores < 3  # Remove outliers beyond 3 sigma
x_clean, y_clean = x[mask], y[mask]
```

## Validation and Quality Assessment

### 1. Visual Inspection

```python
import matplotlib.pyplot as plt

def plot_exponential_fit(x, y, params, r2):
    """Plot data and fitted exponential for visual validation."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale plot
    x_pred = np.linspace(np.min(x), np.max(x), 100)
    y_pred = _exponential_func(x_pred, *params)
    
    ax1.scatter(x, y, alpha=0.6, label='Data')
    ax1.plot(x_pred, y_pred, 'r-', label=f'Fit (R²={r2:.3f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.set_title('Linear Scale')
    
    # Log scale plot (if y > 0)
    if np.all(y > 0):
        ax2.scatter(x, y, alpha=0.6, label='Data')
        ax2.plot(x_pred, y_pred, 'r-', label='Fit')
        ax2.set_yscale('log')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y (log)')
        ax2.legend()
        ax2.set_title('Log Scale')
    
    plt.tight_layout()
    return fig
```

### 2. Residual Analysis

```python
def analyze_residuals(x, y, y_pred):
    """Comprehensive residual analysis for fit quality assessment."""
    
    residuals = y - y_pred
    standardized_residuals = residuals / np.std(residuals)
    
    metrics = {
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mae': np.mean(np.abs(residuals)),
        'max_residual': np.max(np.abs(residuals)),
        'durbin_watson': np.sum(np.diff(residuals)**2) / np.sum(residuals**2),
        'runs_test_p': None  # Implement runs test for independence
    }
    
    return metrics
```

## Integration with deconCNN Analysis Pipeline

For the deconCNN project specifically:

### Loss Curve Fitting Context

```python
# Example usage for loss curve analysis
def fit_loss_curves(epochs, train_loss, val_loss):
    """Fit exponential decay to training loss curves."""
    
    # Fit training loss
    train_params, train_r2 = fit_exponential(
        epochs, train_loss, 
        method='hybrid',
        return_details=False
    )
    
    # Fit validation loss (may need different approach due to overfitting)
    val_params, val_r2 = fit_exponential(
        epochs, val_loss,
        method='hybrid', 
        return_details=False
    )
    
    return {
        'train': {'params': train_params, 'r2': train_r2},
        'val': {'params': val_params, 'r2': val_r2}
    }
```

### AIC Model Comparison Integration

```python
# Compare exponential vs power law models using AIC
from .utils import compute_aic, compute_aic_weights

def compare_exponential_power_law(epochs, loss):
    """Compare exponential and power law fits using AIC."""
    
    # Fit exponential
    exp_params, exp_r2 = fit_exponential(epochs, loss)
    exp_ll = compute_log_likelihood(epochs, loss, exp_params, 'exponential')
    exp_aic = compute_aic(exp_ll, len(exp_params))
    
    # Fit power law (to be implemented)
    power_params, power_r2 = fit_power_law(epochs, loss)
    power_ll = compute_log_likelihood(epochs, loss, power_params, 'power_law')
    power_aic = compute_aic(power_ll, len(power_params))
    
    # Compare models
    aic_weights = compute_aic_weights(
        [exp_aic, power_aic], 
        names=['exponential', 'power_law']
    )
    
    return {
        'exponential': {'params': exp_params, 'r2': exp_r2, 'aic': exp_aic},
        'power_law': {'params': power_params, 'r2': power_r2, 'aic': power_aic},
        'aic_weights': aic_weights
    }
```

## References and Further Reading

### Key Papers
- Akaike, H. (1974). "A new look at the statistical model identification." IEEE Transactions on Automatic Control
- Burnham, K. P., & Anderson, D. R. (2002). "Model selection and multimodel inference"

### Python Documentation
- [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
- [sklearn.metrics.r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

### Stack Overflow Resources
- ["Getting the r-squared value using curve_fit"](https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit)
- ["How to do exponential and logarithmic curve fitting in Python"](https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly)

---

**Document Status:** Complete reference for exponential fitting in Python  
**Last Updated:** 2025-01-02  
**Target Audience:** Python developers implementing scientific curve fitting  
**Related Documents:** `noise-floor-detection-reference.md`, `welch-method-reference.md`
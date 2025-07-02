# Plateau Detection in Neural Network Training Metrics

## Overview

Plateau detection is crucial for understanding neural network training dynamics, particularly for identifying when optimization has stalled or when key metrics have stabilized. This document focuses on detecting plateaus in eigenvalue trajectories and other training metrics.

## Statistical Change Point Detection Methods

### Primary Library: ruptures

The `ruptures` library is the Python standard for offline change point detection, providing robust algorithms for segmenting time series data.

#### Key Algorithms

1. **PELT (Pruned Exact Linear Time)**
   - Complexity: O(n) for piecewise constant signals
   - Advantages: Auto-detects number of change points, exact method
   - Best for: Unknown number of plateaus
   ```python
   algo = rpt.Pelt(model="l2", min_size=10, jump=1).fit(signal)
   result = algo.predict(pen=penalty_value)
   ```

2. **Dynamic Programming (Dynp)**
   - Complexity: O(n²K) where K is number of change points
   - Advantages: Exact method, guaranteed optimal segmentation
   - Best for: Known number of change points
   ```python
   algo = rpt.Dynp(model="l2", min_size=10).fit(signal)
   result = algo.predict(n_bkps=num_breakpoints)
   ```

3. **Binary Segmentation**
   - Complexity: O(n log n)
   - Advantages: Fast approximate method
   - Best for: Real-time applications

#### Model Selection

- **"l2"**: Detects changes in mean (ideal for plateau detection)
- **"rbf"**: Detects more complex pattern changes
- **"linear"**: Detects changes in linear trend
- **"normal"**: Detects changes in mean and variance

### Key Parameters

1. **penalty (pen)**: Controls sensitivity
   - Higher values → fewer change points → longer plateaus
   - Lower values → more change points → shorter plateaus
   - Typical range: 1-1000, tune empirically

2. **min_size**: Minimum segment length
   - Prevents detecting spurious short plateaus
   - For neural networks: typically 10-50 epochs

3. **jump**: Grid spacing for possible change points
   - jump=1: considers all points (most accurate)
   - jump=k: only considers points at k intervals (faster)

## Neural Network Eigenvalue Dynamics

### Key Research Findings

1. **Eigenvalue Evolution Patterns**
   - Large eigenvalues emerge rapidly in early training
   - Stabilization indicates optimization phase transitions
   - Batch normalization significantly affects eigenvalue dynamics

2. **Plateau Characteristics**
   - **Early Training**: Rapid changes, few plateaus
   - **Mid Training**: Alternating plateaus and transitions
   - **Late Training**: Extended plateaus, slow changes

3. **Noise Considerations**
   - Eigenvalue estimates are inherently noisy (stochastic estimation)
   - Hutchinson: O(1/ε²) variance
   - Hutch++: O(1/ε) variance
   - Requires robust filtering before plateau detection

### Domain-Specific Insights

1. **Largest Eigenvalue (λ_max)**
   - Indicates optimization difficulty
   - Plateaus often correlate with training phase transitions
   - Sharp increases may indicate instability

2. **Trace (sum of eigenvalues)**
   - More stable than individual eigenvalues
   - Plateaus indicate overall curvature stabilization

## Implementation Strategy

### 1. Data Preprocessing
```python
def preprocess_eigenvalue_trajectory(values, window_size=5):
    """Smooth noisy eigenvalue estimates."""
    # Apply moving average or EWMA
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(values, size=window_size, mode='reflect')
    return smoothed
```

### 2. Change Point Detection
```python
def detect_change_points(trajectory, penalty=100, min_plateau_length=10):
    """Detect change points in eigenvalue trajectory."""
    import ruptures as rpt
    
    # Ensure 2D array for ruptures
    signal = trajectory.reshape(-1, 1)
    
    # PELT with L2 model for mean changes
    algo = rpt.Pelt(model="l2", min_size=min_plateau_length).fit(signal)
    change_points = algo.predict(pen=penalty)
    
    return change_points
```

### 3. Plateau Identification
```python
def identify_plateaus(trajectory, change_points, stability_threshold=0.1):
    """Convert change points to plateau regions."""
    plateaus = []
    start_idx = 0
    
    for end_idx in change_points:
        segment = trajectory[start_idx:end_idx]
        
        # Check if segment is stable (low variance relative to mean)
        if len(segment) > 0:
            mean_val = np.mean(segment)
            std_val = np.std(segment)
            cv = std_val / (mean_val + 1e-8)  # Coefficient of variation
            
            if cv < stability_threshold:
                plateaus.append({
                    'start': start_idx,
                    'end': end_idx,
                    'mean': mean_val,
                    'cv': cv,
                    'duration': end_idx - start_idx
                })
        
        start_idx = end_idx
    
    return plateaus
```

### 4. Plateau Start Detection
```python
def lambda_plateau_epoch(eigenvalue_trajectory, min_duration=20, penalty=100):
    """Detect first significant plateau in eigenvalue trajectory."""
    # Smooth noisy estimates
    smoothed = preprocess_eigenvalue_trajectory(eigenvalue_trajectory)
    
    # Detect change points
    change_points = detect_change_points(smoothed, penalty=penalty)
    
    # Identify plateau segments
    plateaus = identify_plateaus(smoothed, change_points)
    
    # Find first significant plateau
    for plateau in plateaus:
        if plateau['duration'] >= min_duration:
            return plateau['start']
    
    return None  # No significant plateau found
```

## Parameter Tuning Guidelines

### For Neural Network Training Metrics

1. **Penalty Selection**
   - Start with: `penalty = 10 * signal_variance`
   - Too low: Over-segmentation, false plateaus
   - Too high: Missed important transitions

2. **Minimum Size**
   - Early training: 5-10 epochs (rapid changes expected)
   - Mid training: 10-20 epochs
   - Late training: 20-50 epochs

3. **Stability Threshold**
   - Strict (CV < 0.05): Only very stable plateaus
   - Moderate (CV < 0.1): Balance between stability and detection
   - Loose (CV < 0.2): Include moderately variable plateaus

### Validation Strategy

1. **Synthetic Data Testing**
   ```python
   # Generate synthetic eigenvalue trajectory
   trajectory = np.concatenate([
       np.ones(30) * 10 + np.random.normal(0, 0.5, 30),    # Plateau 1
       np.linspace(10, 25, 20) + np.random.normal(0, 1, 20), # Transition
       np.ones(40) * 25 + np.random.normal(0, 0.8, 40),    # Plateau 2
   ])
   ```

2. **Visual Verification**
   - Always plot detected plateaus over original signal
   - Check alignment with visual intuition
   - Verify robustness to noise levels

3. **Cross-validation with Domain Knowledge**
   - Plateaus should align with known training phases
   - Duration should match expected optimization behavior

## Common Pitfalls and Solutions

### 1. Over-sensitivity to Noise
**Problem**: Detecting many short "plateaus" due to noise
**Solution**: 
- Increase smoothing window
- Increase penalty parameter
- Set larger min_size

### 2. Missing Gradual Plateaus
**Problem**: Slow transitions misclassified as plateaus
**Solution**:
- Check derivative/slope within segments
- Add trend detection post-processing
- Consider "linear" model instead of "l2"

### 3. Boundary Effects
**Problem**: First/last segments incorrectly classified
**Solution**:
- Pad signal with reflected values
- Ignore first/last segments in analysis
- Use minimum duration filters

## Integration with Neural Network Monitoring

### Data Sources
1. **CurvatureMonitor**: Logs `curvature/lambda_max` every N steps
2. **Frequency**: Typically every 500 steps (configurable)
3. **Storage**: Lightning metrics or custom logging

### Best Practices
1. **Collect sufficient data**: At least 100 points for reliable detection
2. **Consistent sampling**: Regular intervals improve detection
3. **Multiple metrics**: Cross-validate with loss plateaus, gradient norms
4. **Real-time monitoring**: Can adapt penalty based on training phase

## Alternative Approaches

### 1. Window-based Methods
- Compare statistics between sliding windows
- Simpler but less principled than change point detection

### 2. Gradient-based Detection
- Monitor derivative of smoothed signal
- Plateau when |derivative| < threshold

### 3. State-space Models
- Hidden Markov Models for regime detection
- More complex but handles uncertainty better

## References

1. **ruptures Documentation**: https://centre-borelli.github.io/ruptures-docs/
2. **Change Point Detection Survey**: Aminikhanghahi & Cook (2017) - "A Survey of Methods for Time Series Change Point Detection"
3. **Neural Network Eigenvalues**: Ghorbani et al. (2019) - "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density"
4. **Optimization Dynamics**: Xing et al. (2018) - "A Walk with SGD: How SGD Explores Regions of Deep Network Loss?"

## Implementation Checklist

- [ ] Install ruptures: `pip install ruptures`
- [ ] Implement eigenvalue smoothing function
- [ ] Create change point detection wrapper
- [ ] Add plateau identification logic
- [ ] Define plateau quality metrics
- [ ] Create comprehensive test suite
- [ ] Validate on real training data
- [ ] Document parameter choices
- [ ] Create visualization utilities
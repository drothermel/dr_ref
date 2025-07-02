# Noise Floor Detection in Power Spectral Density - Reference Guide

## Overview

Noise floor detection is a fundamental technique in signal processing for identifying the baseline noise level in power spectral density (PSD) data. It separates signal components from background noise, enabling accurate signal-to-noise ratio calculations and robust signal detection in noisy environments.

## Purpose and Applications

### Primary Use Cases
- **Signal Detection**: Identifying signals above background noise
- **SNR Calculation**: Computing signal-to-noise ratios for quality assessment
- **Dynamic Thresholding**: Setting adaptive detection thresholds
- **Spectrum Analysis**: Characterizing noise characteristics in RF/communications systems
- **Quality Control**: Validating measurement system performance
- **Research**: Analyzing experimental data with background noise

### Why Noise Floor Detection Matters
- **Robust Analysis**: Enables analysis in presence of varying noise conditions
- **Adaptive Processing**: Allows algorithms to adapt to different noise environments
- **Quality Metrics**: Provides quantitative measures of signal quality
- **Threshold Setting**: Enables automatic threshold setting for detection algorithms

## Mathematical Foundation

### Definition
The noise floor represents the measure of signal created from the sum of all noise sources and unwanted signals within a measurement system. In power spectral density:

```
Noise Floor = Statistical measure of PSD values representing pure noise
```

### Key Concepts
- **Power Spectral Density**: Distribution of signal power across frequency spectrum
- **Statistical Estimation**: Using robust statistics to estimate noise from contaminated data
- **Peak Exclusion**: Removing signal components to isolate noise
- **Bias Correction**: Accounting for statistical bias in estimators

## Detection Methods

### 1. Percentile-Based Method

**Principle**: Uses lower percentiles of PSD distribution to estimate noise floor
```python
noise_floor = np.percentile(psd_values, percentile)  # typically 10th percentile
```

**Advantages**:
- Simple and robust
- Works well when signals are sparse
- Computationally efficient
- Well-understood statistical properties

**Limitations**:
- Can be biased by signal presence
- Sensitive to outliers
- May overestimate in dense signal environments

**Best Practices**:
- Use 5th-15th percentile range
- Combine with peak exclusion for dense spectra
- Validate against known noise-only regions

### 2. Minimum Statistics Method

**Principle**: Tracks minimum values in sliding windows with bias correction
```python
# Sliding window minimums
min_values = [np.min(window) for window in sliding_windows]
# Bias correction for exponential noise distribution
bias_factor = 1.0 / (1.0 - (1.0 / window_size))
noise_floor = np.mean(min_values) * bias_factor
```

**Advantages**:
- Robust to signal presence
- Theoretically well-founded
- Adaptive to local noise characteristics
- Good for non-stationary noise

**Limitations**:
- More complex implementation
- Requires appropriate window sizing
- Sensitive to noise distribution assumptions

**Best Practices**:
- Use window size 20-100 samples
- Apply bias correction based on noise model
- Validate bias factor for your noise type

### 3. Median Filter Method

**Principle**: Applies median filtering followed by percentile estimation
```python
smoothed_psd = scipy.ndimage.median_filter(psd, size=kernel_size)
noise_floor = np.percentile(smoothed_psd, percentile)
```

**Advantages**:
- Robust to impulse noise
- Smooths spectral variations
- Preserves edge information
- Good for bursty interference

**Limitations**:
- Can blur sharp spectral features
- Kernel size affects performance
- May over-smooth narrow signals

**Best Practices**:
- Use odd kernel sizes (3-11 samples)
- Choose kernel size based on spectral resolution
- Combine with other methods for validation

### 4. Hybrid Method

**Principle**: Combines multiple methods using weighted averaging and outlier rejection
```python
estimates = [percentile_est, minimum_stats_est, median_est]
weights = [0.4, 0.35, 0.25]  # Empirically determined
# Outlier rejection using MAD (Median Absolute Deviation)
noise_floor = robust_weighted_average(estimates, weights)
```

**Advantages**:
- Most robust approach
- Combines strengths of multiple methods
- Includes outlier detection
- Provides confidence measures

**Limitations**:
- Most computationally expensive
- Requires parameter tuning
- Complex implementation

**Best Practices**:
- Validate weights for your application
- Use MAD-based outlier detection
- Monitor individual method performance

## Implementation Best Practices

### Input Validation
```python
# Essential checks
assert frequencies.ndim == 1 and psd.ndim == 1, "Arrays must be 1D"
assert len(frequencies) == len(psd), "Array lengths must match"
assert np.all(psd >= 0), "PSD values must be non-negative"
assert len(psd) >= 10, "Need sufficient data points"
```

### Peak Exclusion Strategy
```python
# Detect spectral peaks using statistical threshold
psd_median = np.median(psd)
peak_indices = signal.find_peaks(
    psd, 
    height=psd_median * peak_threshold,  # typically 2.0
    distance=len(psd) // 50  # minimum peak separation
)[0]

# Exclude peak regions for noise estimation
for peak_idx in peak_indices:
    exclude_window = max(1, len(psd) // 100)
    # Remove peak ± window from analysis
```

### Parameter Selection Guidelines

#### Percentile Selection
- **5th percentile**: Conservative, good for clean spectra
- **10th percentile**: Standard choice, good general performance
- **15th percentile**: Less conservative, tolerates more signal presence

#### Window Sizing
- **Minimum statistics**: 20-100 samples (depends on noise stationarity)
- **Median filter**: 3-11 samples (odd numbers only)
- **Peak exclusion**: 1-5% of spectrum length

#### Threshold Settings
- **Peak threshold**: 1.5-3.0 × median PSD (depends on signal sparsity)
- **Outlier threshold**: 2.0 MAD units (conservative)

### Error Handling and Validation
```python
# Validate results
if not np.isfinite(noise_floor) or noise_floor <= 0:
    warnings.warn("Invalid noise floor detected, using fallback")
    noise_floor = np.percentile(psd, 5.0)  # Conservative fallback

# Sanity checks
max_reasonable = np.percentile(psd, 50)  # Should be below median
if noise_floor > max_reasonable:
    warnings.warn("Noise floor estimate seems too high")
```

## Testing and Validation

### Synthetic Test Signals
```python
def create_noisy_signal_psd(fs=1000, signal_freqs=[50, 150], noise_power=0.1):
    """Create test signal with known noise floor."""
    t = np.arange(0, 4.0, 1/fs)
    
    # Clean signal components
    signal = sum(amp * np.sin(2*np.pi*freq*t) 
                for freq, amp in zip(signal_freqs, [2.0, 1.0]))
    
    # Add white noise with known power
    noise = np.sqrt(noise_power) * np.random.randn(len(t))
    noisy_signal = signal + noise
    
    # Compute PSD
    f, psd = signal.welch(noisy_signal, fs=fs)
    
    return f, psd, noise_power / (fs/2)  # Theoretical noise floor
```

### Validation Strategies
```python
def validate_noise_floor_detection(detector_func, test_cases):
    """Validate detector performance across test cases."""
    for case in test_cases:
        f, psd, true_noise_floor = case
        estimated_noise_floor = detector_func(f, psd)
        
        # Compute relative error
        rel_error = abs(estimated_noise_floor - true_noise_floor) / true_noise_floor
        
        # Should be within reasonable tolerance
        assert rel_error < 0.5, f"Large error: {rel_error:.2f}"
```

### Performance Metrics
- **Relative Error**: `|estimated - true| / true`
- **Bias**: `mean(estimated - true) / true`
- **Variance**: `std(estimated) / mean(estimated)`
- **Robustness**: Performance across different signal scenarios

## Common Pitfalls and Solutions

### Pitfall 1: Signal Contamination
```python
# Problem: Signals bias noise floor estimation upward
# Solution: Use peak exclusion and robust methods
noise_floor = detect_noise_floor(
    freq, psd, 
    exclude_peaks=True,
    peak_threshold=2.0,
    method='hybrid'
)
```

### Pitfall 2: Insufficient Data
```python
# Problem: Short PSD arrays give unreliable estimates
# Solution: Validate array length and warn user
if len(psd) < 50:
    warnings.warn("PSD too short for reliable noise floor estimation")
    # Consider using simpler methods or larger tolerance
```

### Pitfall 3: Non-Stationary Noise
```python
# Problem: Noise characteristics change across frequency
# Solution: Use frequency-dependent estimation
freq_bands = {'low': (0, 100), 'mid': (100, 300), 'high': (300, 500)}
noise_floors = {}
for band_name, (f_low, f_high) in freq_bands.items():
    mask = (freq >= f_low) & (freq <= f_high)
    noise_floors[band_name] = detect_noise_floor(freq[mask], psd[mask])
```

### Pitfall 4: Inappropriate Method Selection
```python
# Dense signals → Use minimum statistics or hybrid
# Sparse signals → Percentile method works well
# Impulsive noise → Use median filter method
# Unknown characteristics → Use hybrid method

method_map = {
    'dense_signals': 'minimum_stats',
    'sparse_signals': 'percentile', 
    'impulsive_noise': 'median',
    'unknown': 'hybrid'
}
```

## Performance Considerations

### Computational Complexity
- **Percentile**: O(N log N) for sorting
- **Minimum Statistics**: O(N) with appropriate windowing
- **Median Filter**: O(N × K) where K is kernel size
- **Hybrid**: Sum of individual method complexities

### Memory Requirements
- **Percentile**: O(N) for array storage
- **Minimum Statistics**: O(W) where W is window size
- **Median Filter**: O(K) for kernel storage
- **Hybrid**: O(N) for temporary arrays

### Optimization Strategies
```python
# For large arrays, consider downsampling for initial estimate
if len(psd) > 10000:
    downsample_factor = len(psd) // 5000
    psd_reduced = psd[::downsample_factor]
    # Compute initial estimate, then refine if needed

# Use caching for repeated calls with same parameters
@lru_cache(maxsize=128)
def cached_noise_floor(psd_hash, method, params):
    return detect_noise_floor(psd, method=method, **params)
```

## Integration with Analysis Pipelines

### Signal Detection Pipeline
```python
def adaptive_signal_detection(freq, psd, detection_factor=3.0):
    """Detect signals using adaptive noise floor threshold."""
    # Estimate noise floor
    noise_floor = detect_noise_floor(freq, psd, method='hybrid')
    
    # Set detection threshold
    threshold = noise_floor * detection_factor
    
    # Find peaks above threshold
    signal_indices = signal.find_peaks(psd, height=threshold)[0]
    signal_freqs = freq[signal_indices]
    signal_powers = psd[signal_indices]
    
    return signal_freqs, signal_powers, noise_floor
```

### SNR Calculation
```python
def compute_snr_db(signal_power, noise_floor):
    """Compute SNR in dB."""
    if noise_floor <= 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_floor)
```

### Quality Assessment
```python
def assess_spectrum_quality(freq, psd):
    """Assess overall spectrum quality metrics."""
    noise_floor, details = detect_noise_floor(
        freq, psd, method='hybrid', return_details=True
    )
    
    quality_metrics = {
        'noise_floor_db': 10 * np.log10(noise_floor),
        'dynamic_range_db': details['snr_estimate'],
        'peak_count': details.get('excluded_peaks', 0),
        'method_consistency': details.get('estimate_std', 0) / noise_floor
    }
    
    return quality_metrics
```

## Advanced Topics

### Frequency-Dependent Noise Floor
```python
def frequency_dependent_noise_floor(freq, psd, n_bands=5):
    """Estimate noise floor across frequency bands."""
    freq_edges = np.linspace(freq[0], freq[-1], n_bands + 1)
    noise_floors = []
    
    for i in range(n_bands):
        mask = (freq >= freq_edges[i]) & (freq < freq_edges[i+1])
        if np.sum(mask) > 10:  # Sufficient data
            nf = detect_noise_floor(freq[mask], psd[mask])
            noise_floors.append(nf)
        else:
            noise_floors.append(np.nan)
            
    return freq_edges[:-1], noise_floors
```

### Multi-Method Consensus
```python
def consensus_noise_floor(freq, psd, confidence_threshold=0.8):
    """Use multiple methods and require consensus."""
    methods = ['percentile', 'minimum_stats', 'median']
    estimates = []
    
    for method in methods:
        try:
            nf = detect_noise_floor(freq, psd, method=method)
            estimates.append(nf)
        except Exception:
            continue
            
    if len(estimates) < 2:
        raise ValueError("Insufficient reliable estimates")
        
    # Check consensus using coefficient of variation
    cv = np.std(estimates) / np.mean(estimates)
    
    if cv > (1 - confidence_threshold):
        warnings.warn(f"Low consensus among methods (CV={cv:.3f})")
        
    return np.median(estimates), cv
```

## References and Further Reading

### Key Papers
- Martin, R. (2001). "Noise power spectral density estimation based on optimal smoothing and minimum statistics"
- Ephraim, Y. & Malah, D. (1984). "Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator"
- Hirsch, H.G. & Ehrlicher, C. (1995). "Noise estimation techniques for robust speech recognition"

### Standards and Guidelines
- IEEE Standards for spectral analysis
- ITU-R recommendations for noise measurement
- ANSI standards for acoustic noise measurement

### Software Libraries
- **SciPy**: `scipy.signal` for spectral analysis tools
- **NumPy**: Statistical functions and array operations
- **scikit-learn**: Robust statistical estimators
- **librosa**: Audio signal processing library

## Summary

Noise floor detection is essential for robust signal analysis in noisy environments. Key success factors:

1. **Method Selection**: Choose appropriate method based on signal characteristics
2. **Parameter Tuning**: Adjust parameters for your specific application  
3. **Validation**: Test with synthetic and real data
4. **Error Handling**: Implement robust fallbacks for edge cases
5. **Integration**: Design for your specific analysis pipeline

The hybrid approach combining multiple methods typically provides the most robust results, while simpler percentile-based methods work well for less challenging scenarios. Always validate performance with known test cases before deploying in production systems.
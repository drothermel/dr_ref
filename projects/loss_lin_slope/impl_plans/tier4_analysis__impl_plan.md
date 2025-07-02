# Implementation Plan: Tier 4 Analysis Suite (deconCNN)

## 📋 Context from Tier 3 Implementation

**What Tier 3 Accomplished:**
- ✅ Created `src/deconcnn/analysis/` directory structure (existing from prior work)
- ✅ Implemented LossSlopeLogger callback with slope calculation (alpha_5_15, alpha_full)
- ✅ Configured CurvatureMonitor (500-step frequency) and NoiseMonitor (2-epoch frequency)
- ✅ Created monitoring script `scripts/monitor_experiment.py` for job tracking
- ✅ Set up basic failure recovery in `scripts/recover_failed.py`
- ✅ Created validation notebook `notebooks/validate_metrics_basic.ipynb` with H1/H2 checks
- ✅ Created operational documentation `docs/operational_runbook_basic.md`
- ✅ Added dependencies: pandas>=2.3.0, matplotlib>=3.10.3, statsmodels>=0.14.4, rich>=14.0.0
- ✅ Established dr_exp metrics.jsonl logging format
- ✅ Created data pipeline scripts: `collect_and_archive.sh`, `verify_completeness.py`, `prepare_dataset.py`
- ✅ Created test scripts: `validate_local.py`, `test_harness.py` 
- ✅ Created submission wrapper `scripts/submit_all_loss_slope.sh` for 216-job sweep
- ✅ Created utils module and extract EWMA
- ✅ Created fitting module and extract power law
- ✅ Created spectral module and extract PSD
- ✅ Extracted slope calculation to utils

**What Tier 4 Must Complete:**
- Analysis library functions (knee detection, power law fitting, metric validation)
- Extract and modularize existing analysis code from callbacks
- Advanced SLURM job submission and monitoring infrastructure
- Intelligent failure recovery with pattern analysis
- Efficient result collection and archival systems
- Comprehensive documentation and tutorials
- Data format conversion utilities (JSON to CSV/Parquet)
- Missing visualization dependencies (seaborn, plotly, jupyter)

## Agent Instructions
**IMPORTANT: Read these before starting implementation**

1. **Quality Gates**: Before EVERY commit:
   - Run `lint_fix` and resolve all issues
   - Run tests and ensure they pass
   - Fix any issues found, even if pre-existing

2. **Progress Tracking**: 
   - Mark each step with ✅ when complete
   - Add notes in [brackets] for any deviations
   - Update the "Current Status" section after each work session

3. **Adaptation Protocol**:
   - Follow the plan but use judgment
   - Document any necessary modifications in [brackets]
   - Flag significant changes for retrospective

4. **Commit Message Protocol**:
   - **CRITICAL**: Use the EXACT commit message provided in each step
   - Messages are marked with "Use exact message: `<message>`"
   - Do NOT modify, add to, or abbreviate the provided commit messages
   - Each commit should be atomic and focused on a single logical change
   - If a step has no commit message, no commit is needed for that step

5. **Implementation Philosophy**:
   - Build on existing foundations from Tier 3
   - Extract and consolidate existing code to avoid duplication
   - Maintain backward compatibility with existing callbacks
   - Cross-validate extensively against current behavior

## Current Status
- Last updated: 2025-07-02 - Merged advanced SLURM integration from phase_4_slurm_update.md
- Last completed step: Plan restructured with advanced SLURM features
- Active agent: ready for implementation
- Blocked by: none
- Total commits: 74 atomic commits across 6 phases

## Pre-implementation Checklist

**Environment Verification:**
- [ ] Confirm deconCNN repository accessibility from implementation session
- [ ] Verify Python 3.12+ environment active in deconCNN directory  
- [ ] Verify `uv` package manager availability
- [ ] Run `uv sync --group all` successfully
- [ ] Run `pt` and confirm all current tests pass
- [ ] Check available disk space for notebooks and analysis outputs

**Dependency Verification:**
- [ ] Add visualization dependencies to pyproject.toml
- [ ] Verify all new imports work correctly
- [ ] Test matplotlib backend for headless operation (if needed for cluster)

**Baseline Testing:**
- [ ] Run existing callbacks on sample data to establish baseline
- [ ] Document current NoiseMonitor and CurvatureMonitor output formats
- [ ] Verify existing checkpoints/logs are readable and contain expected metrics

**Integration Validation:**
- [ ] Check that `dr_exp` integration still works after changes
- [ ] Verify Hydra configuration system compatibility
- [ ] Test callback execution overhead is acceptable (<15% as per plan)

## Implementation Steps

**CRITICAL**: Follow Tier 3's atomic commit pattern:
- Each commit does ONE thing
- Implementation → Unit Test → Integration Test
- Run `lint_fix` before EVERY commit

### Phase 0: Environment Setup & Verification
- [ ] **Precommit**: Verify deconCNN environment
  - [ ] Ensure deconCNN directory is accessible
  - [ ] Verify Python 3.12+ with: `uvrp -c "import sys; print(sys.version)"`
  - [ ] Run `us` to sync dependencies
  - [ ] Run `pt` to verify existing tests pass
  - [ ] Run `lint` to check code quality baseline
  - [ ] Test import: `uvrp -c "import seaborn; print('seaborn OK')"`
  - [ ] Test imports: `uvrp -c "import plotly, jupyter; print('All imports OK')"`
  - [ ] Verify `src/deconcnn/analysis/__init__.py`, `scripts/analysis/` and `notebooks/` exist


### Phase 1: Enhanced Mathematical Functions (4 commits)

- [ ] **Commit 1**: Add AIC computation to utils
  - [ ] Add `compute_aic()` function to `utils.py`
  - [ ] Add companion function `compute_aic_weights()`
  - [ ] Add comprehensive tests to `tests/test_analysis_utils.py`
  - [ ] Add necessary imports
  - [ ] Run tests and ensure all pass
  - [ ] Run `lint_fix` before committing
  - **Commit**: `feat and test: add AIC computation to utils`

- [ ] **Commit 2**: Implement exponential fitting
  - [ ] Add `fit_exponential()` to `fitting.py`
  - [ ] Implement log-linear regression
  - [ ] Return parameters and R²
  - [ ] Unitest with synthetic exponential data
  - [ ] Compare with scipy.optimize results
  - **Commit**: `feat and test: implement exponential fitting function`

- [ ] **Commit 3**: Implement Welch's method for spectral analysis
  - [ ] Add `welch_psd()` function to `spectral.py` with the following implementation:
    ```python
    from typing import Optional, Union, Tuple
    import numpy as np
    from scipy import signal
    import warnings
    
    def welch_psd(
        x: np.ndarray, 
        fs: float = 1.0,
        window: Union[str, tuple, np.ndarray] = 'hann',
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        detrend: str = 'constant',
        return_onesided: bool = True,
        scaling: str = 'density',
        freq_bands: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        Estimate power spectral density using Welch's method with enhanced frequency band analysis.
        
        This function wraps scipy.signal.welch with additional validation, 
        automatic parameter selection, and frequency band extraction capabilities.
        
        Args:
            x: Input signal array (1D)
            fs: Sampling frequency (Hz). Default: 1.0
            window: Window function type. Default: 'hann' (recommended)
            nperseg: Length of each segment. If None, uses optimal length based on signal length
            noverlap: Number of points to overlap between segments. If None, uses window-appropriate default
            nfft: Length of the FFT used. If None, uses nperseg
            detrend: Type of detrending. Options: 'constant', 'linear', None
            return_onesided: Return one-sided PSD for real signals
            scaling: Return power spectral density ('density') or power spectrum ('spectrum')
            freq_bands: Dict of frequency bands to analyze, e.g., {'low': (0, 10), 'mid': (10, 50)}
            
        Returns:
            frequencies: Array of sample frequencies (Hz)
            psd: Power spectral density or power spectrum values
            band_powers: Dict of power in each frequency band (if freq_bands provided)
            
        Raises:
            ValueError: For invalid parameters or signal characteristics
            
        Notes:
            - For Hann window, 50% overlap (noverlap = nperseg // 2) is recommended
            - Larger nperseg provides better frequency resolution but higher variance
            - Signal length should be at least 4 * nperseg for meaningful results
            - Frequency resolution is approximately fs / nperseg
            
        Examples:
            >>> # Basic usage
            >>> f, psd = welch_psd(signal, fs=1000)[:2]
            >>> 
            >>> # With frequency band analysis
            >>> bands = {'low': (0, 50), 'high': (50, 250)}
            >>> f, psd, band_powers = welch_psd(signal, fs=1000, freq_bands=bands)
        """
        # Input validation
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"Input signal must be 1D, got {x.ndim}D")
        if len(x) < 4:
            raise ValueError(f"Signal too short: {len(x)} samples (minimum 4)")
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {fs}")
            
        # Automatic parameter selection if not provided
        if nperseg is None:
            # Rule of thumb: segment length should be ~1/8 to 1/16 of signal length
            # but at least 64 samples for reasonable frequency resolution
            nperseg = max(64, min(1024, len(x) // 8))
            # Ensure power of 2 for FFT efficiency
            nperseg = 2 ** int(np.log2(nperseg))
            
        # Validate nperseg
        if nperseg <= 0:
            raise ValueError(f"nperseg must be positive, got {nperseg}")
        if nperseg > len(x):
            warnings.warn(f"nperseg ({nperseg}) > signal length ({len(x)}). Using signal length.")
            nperseg = len(x)
            
        # Set default overlap based on window type
        if noverlap is None:
            if isinstance(window, str):
                if window in ['hann', 'hamming', 'blackman']:
                    noverlap = nperseg // 2  # 50% overlap for tapered windows
                elif window in ['boxcar', 'rect']:
                    noverlap = 0  # No overlap for rectangular window
                else:
                    noverlap = nperseg // 2  # Default to 50%
            else:
                noverlap = nperseg // 2
                
        # Validate overlap
        if noverlap < 0:
            raise ValueError(f"noverlap must be non-negative, got {noverlap}")
        if noverlap >= nperseg:
            raise ValueError(f"noverlap ({noverlap}) must be less than nperseg ({nperseg})")
            
        # Check if signal is long enough for meaningful analysis
        if len(x) < 4 * nperseg:
            warnings.warn(
                f"Signal length ({len(x)}) is less than 4 * nperseg ({4 * nperseg}). "
                "Results may have high variance."
            )
            
        # Compute PSD using scipy's implementation
        try:
            frequencies, psd = signal.welch(
                x=x,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=detrend,
                return_onesided=return_onesided,
                scaling=scaling
            )
        except Exception as e:
            raise ValueError(f"Failed to compute Welch PSD: {e}")
            
        # Compute frequency band powers if requested
        band_powers = None
        if freq_bands is not None:
            band_powers = {}
            df = frequencies[1] - frequencies[0]  # Frequency resolution
            
            for band_name, (f_low, f_high) in freq_bands.items():
                if f_low < 0 or f_high <= f_low or f_high > fs / 2:
                    warnings.warn(f"Invalid frequency band '{band_name}': ({f_low}, {f_high})")
                    band_powers[band_name] = np.nan
                    continue
                    
                # Find frequency indices for this band
                mask = (frequencies >= f_low) & (frequencies <= f_high)
                if np.sum(mask) == 0:
                    warnings.warn(f"No frequencies found in band '{band_name}': ({f_low}, {f_high})")
                    band_powers[band_name] = 0.0
                    continue
                    
                # Integrate power in band (trapezoidal rule)
                if scaling == 'density':
                    # For PSD, integrate over frequency
                    band_powers[band_name] = np.trapz(psd[mask], frequencies[mask])
                else:
                    # For power spectrum, sum the powers
                    band_powers[band_name] = np.sum(psd[mask])
                    
        return frequencies, psd, band_powers
    ```
  - [ ] Add comprehensive tests to `tests/test_spectral.py`:
    ```python
    import pytest
    import numpy as np
    from scipy import signal
    from deconcnn.analysis.spectral import welch_psd
    
    class TestWelchPSD:
        """Test suite for Welch's method PSD implementation."""
        
        @pytest.fixture
        def synthetic_signal(self):
            """Generate synthetic test signal with known frequency components."""
            fs = 1000  # 1 kHz sampling
            duration = 2.0  # 2 seconds
            t = np.arange(0, duration, 1/fs)
            
            # Multi-component signal: 50 Hz + 120 Hz + noise
            signal_clean = (
                2.0 * np.sin(2 * np.pi * 50 * t) +     # 50 Hz, amplitude 2
                1.0 * np.sin(2 * np.pi * 120 * t)      # 120 Hz, amplitude 1
            )
            
            # Add white noise (SNR ~20 dB)
            np.random.seed(42)  # Reproducible
            noise = 0.1 * np.random.randn(len(t))
            signal_noisy = signal_clean + noise
            
            return {
                'signal': signal_noisy,
                'fs': fs,
                't': t,
                'true_freqs': [50, 120],
                'true_amps': [2.0, 1.0]
            }
            
        @pytest.fixture
        def white_noise_signal(self):
            """Generate white noise for statistical testing."""
            np.random.seed(123)
            fs = 500
            duration = 4.0
            n_samples = int(fs * duration)
            signal = np.random.randn(n_samples)
            return {'signal': signal, 'fs': fs}
            
        def test_welch_psd_basic_functionality(self, synthetic_signal):
            """Test basic PSD computation and peak detection."""
            data = synthetic_signal
            f, psd, _ = welch_psd(data['signal'], fs=data['fs'])
            
            # Check output shapes and types
            assert isinstance(f, np.ndarray)
            assert isinstance(psd, np.ndarray)
            assert len(f) == len(psd)
            assert f.ndim == 1 and psd.ndim == 1
            
            # Check frequency range
            assert f[0] >= 0
            assert f[-1] <= data['fs'] / 2  # Nyquist frequency
            assert np.all(np.diff(f) > 0)  # Frequencies should be increasing
            
            # Check PSD properties
            assert np.all(psd >= 0)  # PSD must be non-negative
            assert np.all(np.isfinite(psd))  # No NaN or inf values
            
            # Detect peaks for known frequencies
            peak_indices = signal.find_peaks(psd, height=np.max(psd) * 0.1)[0]
            peak_freqs = f[peak_indices]
            
            # Should detect both frequency components (within tolerance)
            for true_freq in data['true_freqs']:
                closest_peak = peak_freqs[np.argmin(np.abs(peak_freqs - true_freq))]
                assert abs(closest_peak - true_freq) < 2.0, f"Peak at {true_freq} Hz not detected"
                
        def test_welch_psd_against_scipy_reference(self, synthetic_signal):
            """Test our implementation against scipy.signal.welch."""
            data = synthetic_signal
            
            # Use same parameters for both implementations
            params = {
                'fs': data['fs'],
                'window': 'hann',
                'nperseg': 512,
                'noverlap': 256,
                'nfft': 512,
                'detrend': 'constant'
            }
            
            # Our implementation
            f_ours, psd_ours, _ = welch_psd(data['signal'], **params)
            
            # Reference implementation
            f_ref, psd_ref = signal.welch(data['signal'], **params)
            
            # Should be nearly identical
            np.testing.assert_allclose(f_ours, f_ref, rtol=1e-12)
            np.testing.assert_allclose(psd_ours, psd_ref, rtol=1e-12)
            
        def test_frequency_resolution_control(self, white_noise_signal):
            """Test that segment length controls frequency resolution."""
            data = white_noise_signal
            
            # Test different segment lengths
            segment_lengths = [128, 256, 512]
            freq_resolutions = []
            
            for nperseg in segment_lengths:
                f, _, _ = welch_psd(data['signal'], fs=data['fs'], nperseg=nperseg)
                df = f[1] - f[0]  # Frequency resolution
                freq_resolutions.append(df)
                
                # Check theoretical resolution
                expected_df = data['fs'] / nperseg
                assert abs(df - expected_df) < 1e-10
                
            # Larger segments should give better (smaller) frequency resolution
            assert freq_resolutions[0] > freq_resolutions[1] > freq_resolutions[2]
            
        def test_frequency_band_analysis(self, synthetic_signal):
            """Test frequency band power computation."""
            data = synthetic_signal
            
            # Define frequency bands around known components
            freq_bands = {
                'low_band': (45, 55),    # Around 50 Hz
                'high_band': (115, 125), # Around 120 Hz
                'noise_band': (200, 250) # Should have less power
            }
            
            f, psd, band_powers = welch_psd(
                data['signal'], 
                fs=data['fs'], 
                freq_bands=freq_bands
            )
            
            # Check that band_powers is returned
            assert band_powers is not None
            assert isinstance(band_powers, dict)
            assert set(band_powers.keys()) == set(freq_bands.keys())
            
            # All band powers should be positive
            for band_name, power in band_powers.items():
                assert power >= 0, f"Negative power in {band_name}"
                assert np.isfinite(power), f"Invalid power in {band_name}"
                
            # Signal bands should have more power than noise band
            assert band_powers['low_band'] > band_powers['noise_band']
            assert band_powers['high_band'] > band_powers['noise_band']
            
        def test_parameter_validation(self, synthetic_signal):
            """Test input parameter validation and error handling."""
            data = synthetic_signal
            
            # Test invalid signal dimensions
            with pytest.raises(ValueError, match="must be 1D"):
                welch_psd(np.ones((10, 10)), fs=1000)
                
            # Test signal too short
            with pytest.raises(ValueError, match="Signal too short"):
                welch_psd(np.ones(3), fs=1000)
                
            # Test invalid sampling frequency
            with pytest.raises(ValueError, match="must be positive"):
                welch_psd(data['signal'], fs=-100)
                
            with pytest.raises(ValueError, match="must be positive"):
                welch_psd(data['signal'], fs=0)
                
            # Test invalid nperseg
            with pytest.raises(ValueError, match="nperseg must be positive"):
                welch_psd(data['signal'], fs=1000, nperseg=-1)
                
            # Test invalid overlap
            with pytest.raises(ValueError, match="noverlap must be non-negative"):
                welch_psd(data['signal'], fs=1000, nperseg=256, noverlap=-1)
                
            with pytest.raises(ValueError, match="must be less than nperseg"):
                welch_psd(data['signal'], fs=1000, nperseg=256, noverlap=256)
                
        def test_automatic_parameter_selection(self, synthetic_signal):
            """Test automatic parameter selection when parameters are None."""
            data = synthetic_signal
            
            # Test with minimal parameters (should use defaults)
            f, psd, _ = welch_psd(data['signal'], fs=data['fs'])
            
            # Should produce reasonable results
            assert len(f) > 10  # Should have reasonable frequency resolution
            assert np.all(psd >= 0)
            assert np.all(np.isfinite(psd))
            
            # Should handle very short signals gracefully
            short_signal = data['signal'][:100]  # Very short
            f_short, psd_short, _ = welch_psd(short_signal, fs=data['fs'])
            assert len(f_short) > 0
            assert len(psd_short) > 0
            
        def test_window_types_and_overlap(self, synthetic_signal):
            """Test different window types and their default overlaps."""
            data = synthetic_signal
            window_tests = [
                ('hann', 256, 128),      # 50% overlap expected
                ('hamming', 256, 128),   # 50% overlap expected
                ('blackman', 256, 128),  # 50% overlap expected
                ('boxcar', 256, 0),      # No overlap expected
            ]
            
            for window, nperseg, expected_noverlap in window_tests:
                # Test that our function runs without error
                f, psd, _ = welch_psd(
                    data['signal'], 
                    fs=data['fs'], 
                    window=window,
                    nperseg=nperseg
                )
                
                # Should produce valid results
                assert len(f) > 0
                assert len(psd) == len(f)
                assert np.all(psd >= 0)
                
        def test_scaling_options(self, synthetic_signal):
            """Test different scaling options (density vs spectrum)."""
            data = synthetic_signal
            
            # Test both scaling options
            f, psd_density, _ = welch_psd(data['signal'], fs=data['fs'], scaling='density')
            f, psd_spectrum, _ = welch_psd(data['signal'], fs=data['fs'], scaling='spectrum')
            
            # Both should be valid
            assert np.all(psd_density >= 0)
            assert np.all(psd_spectrum >= 0)
            assert len(psd_density) == len(psd_spectrum)
            
            # Spectrum values should generally be smaller than density
            # (since spectrum is density * df for each frequency bin)
            df = f[1] - f[0]
            np.testing.assert_allclose(psd_spectrum, psd_density * df, rtol=1e-10)
            
        def test_edge_cases_and_warnings(self, white_noise_signal):
            """Test edge cases that should produce warnings."""
            data = white_noise_signal
            
            # Test very short signal (should warn about high variance)
            short_signal = data['signal'][:200]  # Very short for default nperseg
            
            with pytest.warns(UserWarning, match="high variance"):
                f, psd, _ = welch_psd(short_signal, fs=data['fs'], nperseg=128)
                
            # Should still produce results
            assert len(f) > 0
            assert len(psd) > 0
            
            # Test nperseg > signal length (should warn and adjust)
            very_short = data['signal'][:50]
            with pytest.warns(UserWarning, match="Using signal length"):
                f, psd, _ = welch_psd(very_short, fs=data['fs'], nperseg=100)
                
        def test_invalid_frequency_bands(self, synthetic_signal):
            """Test handling of invalid frequency band specifications."""
            data = synthetic_signal
            
            # Test invalid frequency bands
            invalid_bands = {
                'negative': (-10, 50),      # Negative frequency
                'reversed': (100, 50),      # f_high < f_low  
                'too_high': (400, 600),     # Above Nyquist
                'empty': (300, 301)         # No frequencies in range
            }
            
            with pytest.warns(UserWarning):
                f, psd, band_powers = welch_psd(
                    data['signal'], 
                    fs=data['fs'],
                    freq_bands=invalid_bands
                )
                
            # Should still return results
            assert band_powers is not None
            assert len(band_powers) == len(invalid_bands)
            
            # Invalid bands should have NaN or 0 power
            for band_name, power in band_powers.items():
                assert np.isnan(power) or power == 0.0
    ```
  - [ ] Add necessary imports to top of `spectral.py`:
    ```python
    from typing import Optional, Union, Tuple
    import numpy as np
    from scipy import signal
    import warnings
    ```
  - [ ] Run tests with `pt tests/test_spectral.py::TestWelchPSD -v`
  - [ ] Verify all tests pass and edge cases are handled
  - [ ] Run `lint_fix` before committing
  - **Commit**: `feat and test: implement Welch's method for PSD`


- [ ] **Commit 4**: Add noise floor detection
  - [ ] Add `detect_noise_floor()` function to `spectral.py` with the following implementation:
    ```python
    from typing import Optional, Tuple, Union, Dict, Any
    import numpy as np
    from scipy import signal, ndimage, stats
    import warnings
    
    def detect_noise_floor(
        frequencies: np.ndarray,
        psd: np.ndarray,
        method: str = 'percentile',
        percentile: float = 10.0,
        window_size: Optional[int] = None,
        min_stat_window: int = 50,
        median_kernel: int = 5,
        exclude_peaks: bool = True,
        peak_threshold: float = 2.0,
        return_details: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Detect noise floor level in power spectral density data using statistical methods.
        
        This function implements multiple statistical approaches for robust noise floor
        estimation, including percentile-based methods, minimum statistics, and 
        median filtering approaches.
        
        Args:
            frequencies: Array of frequency values (Hz)
            psd: Power spectral density values corresponding to frequencies
            method: Detection method. Options: 'percentile', 'minimum_stats', 'median', 'hybrid'
            percentile: Percentile for percentile-based method (0-100). Default: 10.0
            window_size: Window size for sliding statistics. If None, auto-computed
            min_stat_window: Window size for minimum statistics method
            median_kernel: Kernel size for median filtering (must be odd)
            exclude_peaks: Whether to exclude spectral peaks from noise estimation
            peak_threshold: Threshold for peak exclusion (multiple of median)
            return_details: If True, return detailed analysis results
            
        Returns:
            noise_floor: Estimated noise floor level (same units as psd)
            details: Dict with detailed analysis (if return_details=True)
            
        Raises:
            ValueError: For invalid parameters or mismatched array sizes
            
        Notes:
            - 'percentile': Uses lower percentile of PSD values
            - 'minimum_stats': Implements minimum statistics approach
            - 'median': Uses median filtering followed by percentile
            - 'hybrid': Combines multiple methods for robust estimation
            
        Examples:
            >>> # Basic usage
            >>> noise_floor = detect_noise_floor(freq, psd, method='percentile')
            >>> 
            >>> # With detailed analysis
            >>> noise_floor, details = detect_noise_floor(
            ...     freq, psd, method='hybrid', return_details=True
            ... )
        """
        # Input validation
        frequencies = np.asarray(frequencies)
        psd = np.asarray(psd)
        
        if frequencies.ndim != 1 or psd.ndim != 1:
            raise ValueError("Frequencies and PSD must be 1D arrays")
        if len(frequencies) != len(psd):
            raise ValueError(f"Array size mismatch: freq {len(frequencies)} vs psd {len(psd)}")
        if len(psd) < 10:
            raise ValueError("PSD array too short for reliable noise floor estimation")
        if not np.all(psd >= 0):
            raise ValueError("PSD values must be non-negative")
        if not (0 < percentile < 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        if median_kernel % 2 == 0:
            raise ValueError(f"Median kernel size must be odd, got {median_kernel}")
            
        # Set default window size
        if window_size is None:
            window_size = max(10, len(psd) // 20)
            
        # Initialize results dictionary
        details = {
            'method': method,
            'psd_length': len(psd),
            'psd_mean': np.mean(psd),
            'psd_std': np.std(psd),
            'frequency_range': (frequencies[0], frequencies[-1])
        }
        
        # Prepare PSD for analysis (optionally exclude peaks)
        analysis_psd = psd.copy()
        excluded_indices = np.array([], dtype=int)
        
        if exclude_peaks:
            # Identify spectral peaks using statistical threshold
            psd_median = np.median(psd)
            peak_indices = signal.find_peaks(
                psd, 
                height=psd_median * peak_threshold,
                distance=max(1, len(psd) // 50)  # Minimum distance between peaks
            )[0]
            
            if len(peak_indices) > 0:
                # Exclude peak regions (peak ± small window)
                peak_window = max(1, len(psd) // 100)
                excluded_mask = np.zeros(len(psd), dtype=bool)
                
                for peak_idx in peak_indices:
                    start_idx = max(0, peak_idx - peak_window)
                    end_idx = min(len(psd), peak_idx + peak_window + 1)
                    excluded_mask[start_idx:end_idx] = True
                    
                excluded_indices = np.where(excluded_mask)[0]
                analysis_psd = psd[~excluded_mask]
                
                details['excluded_peaks'] = len(peak_indices)
                details['excluded_indices'] = excluded_indices
                details['peak_locations'] = frequencies[peak_indices]
            else:
                details['excluded_peaks'] = 0
        
        # Ensure we have enough data after peak exclusion
        if len(analysis_psd) < 5:
            warnings.warn("Too few data points after peak exclusion. Using original PSD.")
            analysis_psd = psd
            excluded_indices = np.array([], dtype=int)
            
        # Method-specific noise floor estimation
        if method == 'percentile':
            noise_floor = _percentile_noise_floor(analysis_psd, percentile)
            details['percentile_used'] = percentile
            
        elif method == 'minimum_stats':
            noise_floor, min_details = _minimum_stats_noise_floor(
                analysis_psd, window_size=min_stat_window
            )
            details.update(min_details)
            
        elif method == 'median':
            noise_floor, med_details = _median_filter_noise_floor(
                analysis_psd, kernel_size=median_kernel, percentile=percentile
            )
            details.update(med_details)
            
        elif method == 'hybrid':
            noise_floor, hybrid_details = _hybrid_noise_floor(
                analysis_psd, percentile, min_stat_window, median_kernel
            )
            details.update(hybrid_details)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'percentile', 'minimum_stats', 'median', or 'hybrid'")
            
        # Validate result
        if not np.isfinite(noise_floor) or noise_floor < 0:
            warnings.warn(f"Invalid noise floor detected: {noise_floor}. Using fallback.")
            noise_floor = np.percentile(psd, 5.0)  # Conservative fallback
            
        details['noise_floor'] = noise_floor
        details['noise_floor_db'] = 10 * np.log10(noise_floor) if noise_floor > 0 else -np.inf
        details['snr_estimate'] = 10 * np.log10(np.max(psd) / noise_floor) if noise_floor > 0 else np.inf
        
        if return_details:
            return noise_floor, details
        else:
            return noise_floor
    
    
    def _percentile_noise_floor(psd: np.ndarray, percentile: float) -> float:
        """Simple percentile-based noise floor estimation."""
        return np.percentile(psd, percentile)
    
    
    def _minimum_stats_noise_floor(psd: np.ndarray, window_size: int) -> Tuple[float, Dict[str, Any]]:
        """Minimum statistics approach for noise floor estimation."""
        if len(psd) < window_size:
            window_size = len(psd) // 2
            
        # Sliding window minimum
        min_values = []
        for i in range(0, len(psd) - window_size + 1, window_size // 2):
            window = psd[i:i + window_size]
            min_values.append(np.min(window))
            
        min_values = np.array(min_values)
        
        # Bias correction factor (theoretical for exponential distribution)
        bias_factor = 1.0 / (1.0 - (1.0 / window_size))
        corrected_min = np.mean(min_values) * bias_factor
        
        details = {
            'min_values_count': len(min_values),
            'raw_minimum': np.mean(min_values),
            'bias_factor': bias_factor,
            'window_size_used': window_size
        }
        
        return corrected_min, details
    
    
    def _median_filter_noise_floor(
        psd: np.ndarray, 
        kernel_size: int, 
        percentile: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Median filtering followed by percentile estimation."""
        # Apply median filter to smooth the spectrum
        if len(psd) < kernel_size:
            kernel_size = max(3, len(psd) // 3)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
        smoothed_psd = ndimage.median_filter(psd, size=kernel_size)
        
        # Take percentile of smoothed spectrum
        noise_floor = np.percentile(smoothed_psd, percentile)
        
        details = {
            'median_kernel_used': kernel_size,
            'smoothed_mean': np.mean(smoothed_psd),
            'smoothed_std': np.std(smoothed_psd),
            'percentile_used': percentile
        }
        
        return noise_floor, details
    
    
    def _hybrid_noise_floor(
        psd: np.ndarray, 
        percentile: float, 
        min_window: int, 
        med_kernel: int
    ) -> Tuple[float, Dict[str, Any]]:
        """Hybrid approach combining multiple methods."""
        # Get estimates from all methods
        perc_estimate = _percentile_noise_floor(psd, percentile)
        min_estimate, min_details = _minimum_stats_noise_floor(psd, min_window)
        med_estimate, med_details = _median_filter_noise_floor(psd, med_kernel, percentile)
        
        # Combine estimates using weighted average
        estimates = np.array([perc_estimate, min_estimate, med_estimate])
        
        # Weight by inverse variance (more stable methods get higher weight)
        weights = np.array([0.4, 0.35, 0.25])  # Empirically determined weights
        
        # Remove outliers using robust statistics
        median_est = np.median(estimates)
        mad = np.median(np.abs(estimates - median_est))
        
        if mad > 0:
            # Use MAD-based outlier detection
            z_scores = np.abs(estimates - median_est) / (1.4826 * mad)  # 1.4826 for normal distribution
            valid_mask = z_scores < 2.0  # Conservative threshold
            
            if np.sum(valid_mask) > 0:
                noise_floor = np.average(estimates[valid_mask], weights=weights[valid_mask])
            else:
                noise_floor = median_est
        else:
            noise_floor = median_est
            
        details = {
            'percentile_estimate': perc_estimate,
            'minimum_stats_estimate': min_estimate,
            'median_filter_estimate': med_estimate,
            'final_weights': weights.tolist(),
            'combined_estimate': noise_floor,
            'estimate_std': np.std(estimates),
            'min_details': min_details,
            'med_details': med_details
        }
        
        return noise_floor, details
    ```
  - [ ] Add comprehensive tests to `tests/test_spectral.py`:
    ```python
    import pytest
    import numpy as np
    from scipy import signal
    from deconcnn.analysis.spectral import detect_noise_floor
    
    class TestNoiseFloorDetection:
        """Test suite for noise floor detection functionality."""
        
        @pytest.fixture
        def synthetic_noisy_signal(self):
            """Generate synthetic signal with known noise floor."""
            np.random.seed(42)
            fs = 1000
            duration = 4.0
            t = np.arange(0, duration, 1/fs)
            
            # Create signal with known components
            signal_clean = (
                5.0 * np.sin(2 * np.pi * 50 * t) +     # Strong 50 Hz component
                2.0 * np.sin(2 * np.pi * 150 * t) +    # Weaker 150 Hz component
                1.0 * np.sin(2 * np.pi * 300 * t)      # Weak 300 Hz component
            )
            
            # Add white noise with known power
            noise_power = 0.1  # Known noise floor level
            noise = np.sqrt(noise_power) * np.random.randn(len(t))
            signal_noisy = signal_clean + noise
            
            # Compute PSD
            f, psd = signal.welch(signal_noisy, fs=fs, nperseg=1024)
            
            return {
                'signal': signal_noisy,
                'frequencies': f,
                'psd': psd,
                'fs': fs,
                'true_noise_floor': noise_power / (fs / 2),  # Approximate theoretical noise floor
                'signal_freqs': [50, 150, 300],
                'noise_power': noise_power
            }
            
        @pytest.fixture
        def white_noise_psd(self):
            """Generate white noise PSD for baseline testing."""
            np.random.seed(123)
            fs = 500
            duration = 8.0
            noise_power = 0.05
            
            t = np.arange(0, duration, 1/fs)
            noise = np.sqrt(noise_power) * np.random.randn(len(t))
            
            f, psd = signal.welch(noise, fs=fs, nperseg=512)
            
            return {
                'frequencies': f,
                'psd': psd,
                'true_noise_floor': noise_power / (fs / 2),
                'fs': fs
            }
            
        @pytest.fixture
        def sparse_peaks_psd(self):
            """Generate PSD with sparse peaks for peak exclusion testing."""
            np.random.seed(789)
            fs = 1000
            
            # Create frequency array
            f = np.linspace(0, fs/2, 513)
            
            # Create baseline noise floor
            noise_floor = 1e-6
            psd = noise_floor * np.ones_like(f)
            
            # Add a few strong peaks
            peak_freqs = [100, 250, 400]
            peak_powers = [1e-3, 5e-4, 2e-4]
            
            for peak_freq, peak_power in zip(peak_freqs, peak_powers):
                # Find closest frequency index
                idx = np.argmin(np.abs(f - peak_freq))
                # Add Gaussian peak
                sigma = 3  # Peak width in frequency bins
                for i in range(max(0, idx-10), min(len(f), idx+11)):
                    psd[i] += peak_power * np.exp(-0.5 * ((i - idx) / sigma) ** 2)
                    
            return {
                'frequencies': f,
                'psd': psd,
                'true_noise_floor': noise_floor,
                'peak_freqs': peak_freqs,
                'peak_powers': peak_powers
            }
            
        def test_percentile_method(self, synthetic_noisy_signal):
            """Test percentile-based noise floor detection."""
            data = synthetic_noisy_signal
            
            # Test different percentiles
            percentiles = [5, 10, 15, 20]
            noise_floors = []
            
            for perc in percentiles:
                nf = detect_noise_floor(
                    data['frequencies'], 
                    data['psd'], 
                    method='percentile',
                    percentile=perc
                )
                noise_floors.append(nf)
                
                # Basic validation
                assert isinstance(nf, float)
                assert nf > 0
                assert np.isfinite(nf)
                
            # Lower percentiles should give lower noise floors
            assert noise_floors[0] <= noise_floors[1] <= noise_floors[2] <= noise_floors[3]
            
            # Should be reasonably close to theoretical noise floor
            nf_10 = noise_floors[1]  # 10th percentile
            theoretical = data['true_noise_floor']
            
            # Allow for reasonable tolerance due to finite sample effects
            assert 0.1 * theoretical <= nf_10 <= 10 * theoretical
            
        def test_minimum_stats_method(self, synthetic_noisy_signal):
            """Test minimum statistics noise floor detection."""
            data = synthetic_noisy_signal
            
            nf, details = detect_noise_floor(
                data['frequencies'], 
                data['psd'], 
                method='minimum_stats',
                return_details=True
            )
            
            # Validate result
            assert isinstance(nf, float)
            assert nf > 0
            assert np.isfinite(nf)
            
            # Check details
            assert 'min_values_count' in details
            assert 'bias_factor' in details
            assert details['min_values_count'] > 0
            assert details['bias_factor'] > 1.0  # Should have positive bias correction
            
            # Should be reasonable compared to theoretical
            theoretical = data['true_noise_floor']
            assert 0.1 * theoretical <= nf <= 10 * theoretical
            
        def test_median_filter_method(self, synthetic_noisy_signal):
            """Test median filter noise floor detection."""
            data = synthetic_noisy_signal
            
            nf, details = detect_noise_floor(
                data['frequencies'], 
                data['psd'], 
                method='median',
                median_kernel=7,
                return_details=True
            )
            
            # Validate result
            assert isinstance(nf, float)
            assert nf > 0
            assert np.isfinite(nf)
            
            # Check details
            assert 'median_kernel_used' in details
            assert 'smoothed_mean' in details
            assert details['median_kernel_used'] >= 3  # Should use odd kernel
            assert details['median_kernel_used'] % 2 == 1
            
            # Should be reasonable
            theoretical = data['true_noise_floor']
            assert 0.1 * theoretical <= nf <= 10 * theoretical
            
        def test_hybrid_method(self, synthetic_noisy_signal):
            """Test hybrid noise floor detection method."""
            data = synthetic_noisy_signal
            
            nf, details = detect_noise_floor(
                data['frequencies'], 
                data['psd'], 
                method='hybrid',
                return_details=True
            )
            
            # Validate result
            assert isinstance(nf, float)
            assert nf > 0
            assert np.isfinite(nf)
            
            # Check that all sub-methods were used
            assert 'percentile_estimate' in details
            assert 'minimum_stats_estimate' in details
            assert 'median_filter_estimate' in details
            assert 'combined_estimate' in details
            
            # All estimates should be positive
            for key in ['percentile_estimate', 'minimum_stats_estimate', 'median_filter_estimate']:
                assert details[key] > 0
                assert np.isfinite(details[key])
                
            # Combined estimate should be close to individual estimates
            estimates = [
                details['percentile_estimate'],
                details['minimum_stats_estimate'], 
                details['median_filter_estimate']
            ]
            est_range = max(estimates) - min(estimates)
            est_mean = np.mean(estimates)
            
            # Combined should be within reasonable range of individual estimates
            assert abs(nf - est_mean) <= est_range
            
        def test_peak_exclusion(self, sparse_peaks_psd):
            """Test peak exclusion functionality."""
            data = sparse_peaks_psd
            
            # Test with peak exclusion enabled
            nf_with_exclusion = detect_noise_floor(
                data['frequencies'], 
                data['psd'], 
                method='percentile',
                exclude_peaks=True,
                peak_threshold=2.0
            )
            
            # Test with peak exclusion disabled
            nf_without_exclusion = detect_noise_floor(
                data['frequencies'], 
                data['psd'], 
                method='percentile',
                exclude_peaks=False
            )
            
            # With peak exclusion should be closer to true noise floor
            true_nf = data['true_noise_floor']
            
            error_with = abs(nf_with_exclusion - true_nf) / true_nf
            error_without = abs(nf_without_exclusion - true_nf) / true_nf
            
            # Peak exclusion should reduce error
            assert error_with < error_without
            
            # Both should be positive and finite
            assert nf_with_exclusion > 0 and np.isfinite(nf_with_exclusion)
            assert nf_without_exclusion > 0 and np.isfinite(nf_without_exclusion)
            
        def test_white_noise_baseline(self, white_noise_psd):
            """Test noise floor detection on pure white noise."""
            data = white_noise_psd
            
            # All methods should give similar results for white noise
            methods = ['percentile', 'minimum_stats', 'median', 'hybrid']
            results = {}
            
            for method in methods:
                nf = detect_noise_floor(
                    data['frequencies'], 
                    data['psd'], 
                    method=method
                )
                results[method] = nf
                
                # Should be positive and finite
                assert nf > 0
                assert np.isfinite(nf)
                
                # Should be reasonably close to theoretical
                theoretical = data['true_noise_floor']
                assert 0.1 * theoretical <= nf <= 10 * theoretical
                
            # Results should be relatively consistent across methods
            nf_values = list(results.values())
            nf_range = max(nf_values) - min(nf_values)
            nf_mean = np.mean(nf_values)
            
            # Range should be reasonable compared to mean
            assert nf_range <= 2.0 * nf_mean  # Generous tolerance
            
        def test_input_validation(self, synthetic_noisy_signal):
            """Test input validation and error handling."""
            data = synthetic_noisy_signal
            
            # Test mismatched array sizes
            with pytest.raises(ValueError, match="Array size mismatch"):
                detect_noise_floor(data['frequencies'][:-1], data['psd'])
                
            # Test non-1D arrays
            with pytest.raises(ValueError, match="must be 1D arrays"):
                detect_noise_floor(
                    data['frequencies'].reshape(-1, 1), 
                    data['psd']
                )
                
            # Test empty arrays
            with pytest.raises(ValueError, match="too short"):
                detect_noise_floor(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
                
            # Test negative PSD values
            bad_psd = data['psd'].copy()
            bad_psd[0] = -1.0
            with pytest.raises(ValueError, match="must be non-negative"):
                detect_noise_floor(data['frequencies'], bad_psd)
                
            # Test invalid percentile
            with pytest.raises(ValueError, match="Percentile must be between"):
                detect_noise_floor(
                    data['frequencies'], 
                    data['psd'], 
                    percentile=150
                )
                
            # Test even median kernel
            with pytest.raises(ValueError, match="must be odd"):
                detect_noise_floor(
                    data['frequencies'], 
                    data['psd'], 
                    method='median',
                    median_kernel=6
                )
                
            # Test unknown method
            with pytest.raises(ValueError, match="Unknown method"):
                detect_noise_floor(
                    data['frequencies'], 
                    data['psd'], 
                    method='invalid_method'
                )
                
        def test_return_details_functionality(self, synthetic_noisy_signal):
            """Test detailed return information."""
            data = synthetic_noisy_signal
            
            # Test with return_details=False (default)
            result = detect_noise_floor(data['frequencies'], data['psd'])
            assert isinstance(result, float)
            
            # Test with return_details=True
            nf, details = detect_noise_floor(
                data['frequencies'], 
                data['psd'], 
                return_details=True
            )
            
            assert isinstance(nf, float)
            assert isinstance(details, dict)
            
            # Check required fields in details
            required_fields = [
                'method', 'psd_length', 'psd_mean', 'psd_std',
                'frequency_range', 'noise_floor', 'noise_floor_db', 'snr_estimate'
            ]
            
            for field in required_fields:
                assert field in details, f"Missing field: {field}"
                
            # Validate field values
            assert details['psd_length'] == len(data['psd'])
            assert details['psd_mean'] > 0
            assert details['psd_std'] >= 0
            assert len(details['frequency_range']) == 2
            assert details['noise_floor'] == nf
            assert np.isfinite(details['snr_estimate'])
            
        def test_edge_cases_and_warnings(self, synthetic_noisy_signal):
            """Test edge cases and warning conditions."""
            data = synthetic_noisy_signal
            
            # Test very short PSD (should still work with warnings)
            short_f = data['frequencies'][:20]
            short_psd = data['psd'][:20]
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                nf = detect_noise_floor(short_f, short_psd)
                # Should produce some warnings but still return valid result
                assert nf > 0
                assert np.isfinite(nf)
                
            # Test all-peak signal (aggressive peak exclusion)
            peak_psd = np.ones_like(data['psd'])
            peak_psd[::10] = 100  # Every 10th point is a peak
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                nf = detect_noise_floor(
                    data['frequencies'], 
                    peak_psd, 
                    exclude_peaks=True,
                    peak_threshold=1.1
                )
                # Should still return valid result
                assert nf > 0
                assert np.isfinite(nf)
                
        def test_statistical_consistency(self, white_noise_psd):
            """Test statistical consistency across multiple runs."""
            data = white_noise_psd
            
            # Run detection multiple times with same parameters
            n_runs = 10
            results = []
            
            for _ in range(n_runs):
                nf = detect_noise_floor(
                    data['frequencies'], 
                    data['psd'], 
                    method='hybrid'
                )
                results.append(nf)
                
            results = np.array(results)
            
            # Results should be consistent (same data, same parameters)
            assert np.std(results) == 0, "Results should be identical for same input"
            
            # All results should be valid
            assert np.all(results > 0)
            assert np.all(np.isfinite(results))
    ```
  - [ ] Add necessary imports to top of `spectral.py`:
    ```python
    from typing import Optional, Tuple, Union, Dict, Any
    import numpy as np
    from scipy import signal, ndimage, stats
    import warnings
    ```
  - [ ] Run tests with `pt tests/test_spectral.py::TestNoiseFloorDetection -v`
  - [ ] Verify all tests pass and methods work correctly
  - [ ] Run `lint_fix` before committing
  - **Commit**: `feat and test: add noise floor detection`

### Phase 2: Detection & Analysis Functions (6 commits)

- TODO: verify that the training code outputs CSV

- [ ] **Commit 5**: Implement knee detection
  - [ ] Create `src/deconcnn/analysis/detection.py`
  - [ ] Add imports and module structure
  - [ ] Add `knee_epoch()` function
  - [ ] Use AIC to compare exponential vs power law
  - [ ] Return burn-in epoch estimate
  - [ ] Create `tests/test_detection.py`
  - [ ] Test with synthetic curves
  - [ ] Verify burn-in identification
  - **Commit**: `feat and test: implement AIC-based knee detection`

- [ ] **Commit 6**: Implement windowed slope analysis
  - [ ] Add `alpha_window()` function
  - [ ] Support flexible window parameters
  - [ ] Return slope time series
  - [ ] Unit test window sliding behavior
  - [ ] Verify slope accuracy per window
  - **Commit**: `feat and test: implement windowed slope computation`

- [ ] **Commit 7**: Implement segmented power law fitting
  - [ ] Add `fit_two_power()` to `fitting.py`
  - [ ] Use dynamic programming for changepoint
  - [ ] Return both segments and changepoint
  - [ ] unit test with known two-regime data
  - [ ] Verify changepoint detection
  - **Commit**: `feat and test: implement segmented power law fitting`

- [ ] **Commit 8**: Implement Hutch++ algorithm
  - [ ] Create `src/deconcnn/analysis/curvature.py`
  - [ ] Add module structure
  - [ ] Add `hutchpp_trace()` function
  - [ ] Improve on Hutchinson estimator
  - [ ] Include variance reduction
  - [ ] Create `tests/test_curvature.py`
  - [ ] Compare with Hutchinson baseline
  - [ ] Verify variance reduction
  - **Commit**: `feat and test: implement Hutch++ trace estimator`

- [ ] **Commit 9**: Implement plateau detection
  - [ ] Add `lambda_plateau_epoch()` function
  - [ ] Use statistical change detection
  - [ ] Return plateau start epoch
  - [ ] unit Test with synthetic eigenvalue trajectories
  - [ ] Verify detection accuracy
  - **Commit**: `feat and test: implement eigenvalue plateau detection`
  
- [ ] **Commit 10**: Add statistical utilities
  - [ ] Add correlation functions to `utils.py`
  - [ ] Include Pearson, Spearman, partial
  - [ ] Add bootstrap confidence intervals
  - [ ] Unit Test correlation functions
  - [ ] Unit Test confidence intervals
  - [ ] Unit Test edge cases
  - **Commit**: `feat: add statistical correlation utilities`

### Phase 3: Primary Analysis Scripts (6 commits)

- [ ] **Commit 11**: Implement loss mosaic visualization
  - [ ] Create `src/deconcnn/analysis/visualization.py`
  - [ ] Add basic imports and module structure
  - [ ] Add `plot_loss_mosaic()` function
  - [ ] Support 6-panel layout (train/val × 3 scales)
  - [ ] Add automatic scaling and labeling
  - **Commit**: `feat: implement loss mosaic visualization`

- [ ] **Commit 12**: Create knee detection script
  - [ ] Create `scripts/analysis/A-knee.py`
  - [ ] Use click CLI interface
  - [ ] Process all runs and detect burn-in
  - **Commit**: `feat: create knee detection analysis script`

- [ ] **Commit 13**: Create slope analysis script
  - [ ] Create `scripts/analysis/A-slope.py`
  - [ ] Compute α_early (5-15) and α_full
  - [ ] Export results to CSV
  - **Commit**: `feat: create slope analysis script`

- [ ] **Commit 14**: Create model comparison script
  - [ ] Create `scripts/analysis/A-fits.py`
  - [ ] Compare exponential/power/two-power models
  - [ ] Generate AIC comparison table
  - **Commit**: `feat: create model comparison script`

- [ ] **Commit 15**: Test primary analysis scripts
  - [ ] Create test data subset
  - [ ] Run all three scripts
  - [ ] Verify output formats
  - **Commit**: `test: verify primary analysis scripts`

- [ ] **Commit 16**: Add progress tracking to scripts
  - [ ] Add rich progress bars
  - [ ] Add error recovery on failure
  - [ ] Add logging for debugging
  - **Commit**: `feat: add progress tracking to analysis scripts`

### Phase 4: Supporting Analysis Scripts (5 commits)

- [ ] **Commit 17**: Create architecture ablation script
  - [ ] Create `scripts/analysis/A-optim_arch.py`
  - [ ] Generate ablation comparison table
  - [ ] Include effect sizes
  - **Commit**: `feat: create architecture ablation analysis`

- [ ] **Commit 18**: Create hyperparameter heatmap script
  - [ ] Create `scripts/analysis/A-grid_heat.py`
  - [ ] Generate LR×WD performance heatmap
  - [ ] Add statistical significance
  - **Commit**: `feat: create hyperparameter heatmap analysis`

- [ ] **Commit 19**: Create noise correlation script
  - [ ] Create `scripts/analysis/A-noise_corr.py`
  - [ ] Compute noise proxy correlations
  - [ ] Add confidence intervals
  - **Commit**: `feat: create noise correlation analysis`

- [ ] **Commit 20**: Create curvature timing script
  - [ ] Create `scripts/analysis/A-curvature_timing.py`
  - [ ] Analyze λ₁ plateau vs α knee timing
  - [ ] Include statistical tests
  - **Commit**: `feat: create curvature timing analysis`

- [ ] **Commit 21**: Integration test supporting scripts
  - [ ] Run all supporting scripts
  - [ ] Verify outputs compatible
  - [ ] Check performance
  - **Commit**: `test: verify supporting analysis scripts`

### Phase 5: Slide Generation Notebooks (13 commits)

- [ ] **Commit 22**: Create H1 hypothesis notebook
  - [ ] Create `notebooks/h1_primary.ipynb`
  - [ ] Add to `.gitignore`: `notebooks/*.ipynb_checkpoints`, `*.png`, `*.pdf`, `*.svg`
  - [ ] Add R² histogram visualization
  - [ ] Add loss mosaic for best/worst fits
  - **Commit**: `feat: create H1 hypothesis notebook`

- [ ] **Commit 23**: Create H2 hypothesis notebook
  - [ ] Create `notebooks/h2_primary.ipynb`
  - [ ] Add α_early vs best_val_CE scatter
  - [ ] Include correlation statistics
  - **Commit**: `feat: create H2 hypothesis notebook`

- [ ] **Commit 24**: Add interactivity to primary notebooks
  - [ ] Add plotly interactive plots
  - [ ] Add parameter selection widgets
  - [ ] Test interactivity
  - **Commit**: `feat: add interactivity to hypothesis notebooks`

- [ ] **Commit 25**: Create model comparison notebook
  - [ ] Create `notebooks/backup_b_fam.ipynb`
  - [ ] Add AIC comparison visualizations
  - [ ] Include model selection analysis
  - **Commit**: `feat: create model comparison notebook`

- [ ] **Commit 26**: Create window robustness notebook
  - [ ] Create `notebooks/backup_c_window.ipynb`
  - [ ] Add robustness heatmaps
  - [ ] Include sensitivity analysis
  - **Commit**: `feat: create window robustness notebook`

- [ ] **Commit 27**: Create architecture ablation notebook
  - [ ] Create `notebooks/backup_h_arch.ipynb`
  - [ ] Add ablation result tables
  - [ ] Include statistical comparisons
  - **Commit**: `feat: create architecture ablation notebook`

- [ ] **Commit 28**: Create curvature exploration notebook
  - [ ] Create `notebooks/curvature_explore.ipynb`
  - [ ] Add eigenvalue trajectory plots
  - [ ] Include plateau analysis
  - **Commit**: `feat: create curvature exploration notebook`

- [ ] **Commit 29**: Add export functionality to notebooks
  - [ ] Add PNG/PDF export for all plots
  - [ ] Add LaTeX table export
  - [ ] Test export quality
  - **Commit**: `feat: add export functionality to notebooks`

- [ ] **Commit 30**: Create notebook utilities
  - [ ] Create `notebooks/utils.py`
  - [ ] Add common plotting functions
  - [ ] Add data loading helpers
  - **Commit**: `feat: create notebook utility functions`

- [ ] **Commit 31**: Test all notebooks
  - [ ] Run all notebooks end-to-end
  - [ ] Verify outputs and exports
  - [ ] Check narrative flow
  - **Commit**: `test: verify all analysis notebooks`

- [ ] **Commit 32**: Create integration test suite
  - [ ] Create `tests/test_integration.py`
  - [ ] Add end-to-end pipeline tests
  - [ ] Include synthetic data generation
  - **Commit**: `test: create integration test suite`

- [ ] **Commit 33**: Expand operational documentation
  - [ ] Expand `docs/operational_runbook_basic.md`
  - [ ] Add troubleshooting scenarios
  - [ ] Include performance tips
  - **Commit**: `docs: expand operational runbook`

- [ ] **Commit 34**: Create analysis library README
  - [ ] Create `src/deconcnn/analysis/README.md`
  - [ ] Add API documentation
  - [ ] Include usage examples
  - **Commit**: `docs: create analysis library documentation`

## Technical Guidelines

### Mathematical Approach
- Use AIC = 2k - 2ln(L) for model comparison (k = parameters, L = likelihood)
- Implement Hutch++ for O(1/ε) complexity trace estimation
- Enhance power law fitting with robust outlier detection
- Use Welch's method for spectral analysis with proper windowing

### Integration Philosophy  
- Maintain native deconCNN integration within project structure
- Ensure extracted functions match existing callback behavior exactly
- Leverage Lightning's metric collection and checkpoint systems
- Use Hydra configuration for all analysis script parameters

### Performance Requirements
- Process 216 experimental runs without memory issues
- Maintain <15% overhead for callback operations
- Enable parallel processing for large-scale analysis
- Implement graceful error handling for incomplete data

### Validation Approach
- Cross-validate all extracted functions against originals
- Use synthetic data for testing new mathematical functions
- Maintain comprehensive regression test suite
- Perform end-to-end testing with real experimental data

## Risk Mitigation Updates

### **Critical Implementation Risks**
1. **Callback Compatibility Risk**: Create comprehensive regression tests before extraction
   - Mitigation: Baseline test all callback outputs before making changes
   - Verification: Side-by-side comparison of outputs before/after refactoring

2. **Memory Issues During Hutch++ Upgrade**: Monitor memory usage during curvature analysis
   - Mitigation: Test on smaller models first, implement memory monitoring
   - Verification: Ensure <15% total runtime overhead as specified in plan

3. **Directory Access Limitations**: Verify session working directory includes deconCNN
   - Mitigation: Use absolute paths, verify repository access before starting
   - Verification: Test write/read permissions in target directories

4. **Dependency Conflicts**: Test new visualization packages with existing PyTorch stack
   - Mitigation: Install incrementally, test imports after each addition
   - Verification: Ensure existing functionality continues to work

### **Quality Assurance Protocol**
- **Before each commit**: Run `lint_fix` and resolve all issues
- **After each phase**: Run `pt` to ensure tests pass
- **Before extraction**: Create baseline comparison data
- **After extraction**: Verify identical behavior with integration tests

## Handoff Instructions
To continue this implementation:
1. Read the full plan and agent instructions
2. Check "Current Status" to see where we are  
3. Use TodoWrite to plan your work session
4. Find the first unchecked [ ] commit
5. Implement exactly what that commit describes
6. Run `lint_fix` before committing
7. Move to the next commit (often a test commit)
8. Update "Current Status" when stopping work

**Critical**: Follow the atomic commit pattern - each commit should do exactly ONE thing. Don't bundle changes together.

**Key Principles**:
1. Every implementation commit is followed by a test commit
2. Each commit does exactly ONE thing
3. Functions are extracted and tested before enhancement
4. Scripts are created only after their dependencies are tested
5. Integration testing validates the complete system

**Testing Strategy**:
- Unit tests for every new function
- Integration tests for extracted functions
- End-to-end tests for analysis scripts
- Notebook verification for all visualizations
- Final integration test of complete pipeline

## Retrospective Notes
[Track significant decisions and deviations here]

---
Remember: Build on existing foundations. Cross-validate extensively. Maintain compatibility.



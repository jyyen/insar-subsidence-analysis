#!/usr/bin/env python3
"""
ps02_01_signal_decomposition.py: Original signal decomposition framework
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Parallelization support
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import threading

# Try to import emd for EMD decomposition
try:
    import emd
    HAS_EMD = True
    print("✅ EMD-signal available for EMD decomposition")
except ImportError:
    HAS_EMD = False
    print("⚠️  EMD-signal not available. Install with: conda install conda-forge::emd-signal")

# Try to import VMD for VMD decomposition
try:
    from vmdpy import VMD
    HAS_VMD = True
    print("✅ vmdpy available for VMD decomposition")
except ImportError:
    HAS_VMD = False
    print("⚠️  vmdpy not available. Install with: pip install vmdpy")

# Try to import pywt for Wavelet decomposition
try:
    import pywt
    HAS_WAVELET = True
    print("✅ PyWavelets available for Wavelet decomposition")
except ImportError:
    HAS_WAVELET = False
    print("⚠️  PyWavelets not available. Install with: conda install pywavelets")



# Try to import advanced visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class SignalDecomposer:
    """Multi-method signal decomposition with EMD as primary method"""
    
    def __init__(self, data_file="data/processed/ps00_preprocessed_data.npz"):
        self.data_file = Path(data_file)
        self.coordinates = None
        self.displacement = None
        self.subsidence_rates = None
        self.n_stations = None
        self.n_acquisitions = None
        
        # Frequency band definitions (in days)
        self.frequency_bands = {
            'high_freq': (1, 60),
            'quarterly': (60, 120), 
            'semi_annual': (120, 280),
            'annual': (280, 400),
            'long_annual': (400, 1000),
            'trend': (1000, float('inf'))
        }
        
        # Decomposition results storage
        self.decomposition_results = {}
        self.re_categorization_results = {}
        self.quality_metrics = {}
        
    def load_data(self):
        """Load preprocessed InSAR data"""
        print("📡 Loading preprocessed InSAR data from ps00...")
        
        try:
            data = np.load(self.data_file, allow_pickle=True)
            self.coordinates = data['coordinates']  # [N, 2] - [lon, lat]
            self.displacement = data['displacement']  # [N, T] - mm
            self.subsidence_rates = data['subsidence_rates']  # [N] - mm/year
            self.n_stations = int(data['n_stations'])
            self.n_acquisitions = int(data['n_acquisitions'])
            
            print(f"✅ Loaded data: {self.n_stations} stations, {self.n_acquisitions} acquisitions")
            print(f"📊 Time series shape: {self.displacement.shape}")
            print(f"📊 Coordinate range: lon [{self.coordinates[:, 0].min():.3f}, {self.coordinates[:, 0].max():.3f}], lat [{self.coordinates[:, 1].min():.3f}, {self.coordinates[:, 1].max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading data: {e}")
            return False
    
    def emd_decompose_station(self, time_series, time_vector):
        """Decompose single station time series using EMD"""
        if not HAS_EMD:
            print("❌ EMD-signal not available")
            return None
        
        try:
            # Perform EMD decomposition using emd-signal
            imfs_matrix = emd.sift.sift(time_series)
            
            # Extract individual IMFs (all but last column)
            n_imfs = imfs_matrix.shape[1] - 1  # Last column is typically residual
            imfs = imfs_matrix[:, :-1].T  # Transpose to get shape (n_imfs, n_samples)
            residual = imfs_matrix[:, -1]  # Last column is residual
            
            # Combine IMFs and residual for compatibility
            components = np.vstack([imfs, residual[np.newaxis, :]])
            
            return {
                'imfs': imfs,
                'residual': residual,
                'components': components,
                'n_imfs': n_imfs
            }
            
        except Exception as e:
            print(f"⚠️  EMD failed for station: {e}")
            return None
    
    def vmd_decompose_station(self, time_series, time_vector, n_modes=6):
        """Decompose single station time series using VMD"""
        if not HAS_VMD:
            print("❌ vmdpy not available")
            return None
        
        try:
            # VMD parameters
            alpha = 2000       # bandwidth constraint
            tau = 0.          # noise-tolerance (no strictness)
            K = n_modes       # number of modes
            DC = 0            # no DC part imposed
            init = 1          # initialize omegas uniformly
            tol = 1e-7       # tolerance of convergence criterion
            
            # Ensure time series is clean for VMD
            if len(time_series) != len(time_vector):
                print(f"⚠️  Length mismatch: time_series={len(time_series)}, time_vector={len(time_vector)}")
                return None
            
            # VMD can be sensitive to data length - ensure it's appropriate
            if len(time_series) < 2 * n_modes:
                print(f"⚠️  Time series too short for {n_modes} modes")
                return None
            
            # Perform VMD decomposition
            u, u_hat, omega = VMD(time_series, alpha, tau, K, DC, init, tol)
            
            # u contains the modes, each row is a mode
            n_modes_actual = u.shape[0]
            modes = u  # Shape: (n_modes, n_samples)
            
            # Handle VMD length mismatch (common issue - VMD often returns N-1 samples)
            if modes.shape[1] != len(time_series):
                expected_len = len(time_series)
                actual_len = modes.shape[1]
                
                if actual_len == expected_len - 1:
                    # Pad with the last value to match expected length
                    modes_padded = np.zeros((modes.shape[0], expected_len))
                    modes_padded[:, :-1] = modes
                    modes_padded[:, -1] = modes[:, -1]  # Repeat last value
                    modes = modes_padded
                    # Silently fix common VMD N-1 issue (will report in chunk summary)
                elif actual_len == expected_len + 1:
                    # Trim to match expected length
                    modes = modes[:, :expected_len]
                    print(f"🔧 VMD length corrected: {actual_len} → {expected_len} (trimmed)")
                else:
                    print(f"⚠️  VMD output length mismatch: expected {expected_len}, got {actual_len}")
                    return None
            
            # VMD doesn't have explicit residual like EMD, but we can compute it
            reconstructed = np.sum(modes, axis=0)
            residual = time_series - reconstructed
            
            # Combine modes and residual for compatibility with EMD format
            components = np.vstack([modes, residual[np.newaxis, :]])
            
            return {
                'imfs': modes,  # VMD modes (equivalent to IMFs)
                'residual': residual,
                'components': components,
                'n_imfs': n_modes_actual,
                'omega': omega  # VMD-specific: center frequencies
            }
            
        except Exception as e:
            # Common VMD issues - provide more specific error handling
            if "broadcast" in str(e):
                print(f"⚠️  VMD broadcasting error (likely version compatibility): {e}")
            else:
                print(f"⚠️  VMD failed for station: {e}")
            return None
    
    def fft_decompose_station(self, time_series, time_vector):
        """Decompose single station time series using FFT with frequency band extraction"""
        try:
            # Compute FFT
            fft_values = fft(time_series)
            freqs = fftfreq(len(time_series), d=6)  # 6-day sampling
            
            # Convert frequency to period (days)
            periods = np.abs(1.0 / (freqs + 1e-12))
            
            # Define frequency bands (same as EMD)
            band_definitions = {
                'high_freq': (1, 60),
                'quarterly': (60, 120),
                'semi_annual': (120, 280),
                'annual': (280, 400),
                'long_annual': (400, 1000),
                'trend': (1000, float('inf'))
            }
            
            # Extract components for each frequency band
            components = []
            
            for band_name, (min_period, max_period) in band_definitions.items():
                # Create frequency mask
                if max_period == float('inf'):
                    mask = periods >= min_period
                else:
                    mask = (periods >= min_period) & (periods <= max_period)
                
                # Apply mask to FFT
                fft_filtered = fft_values.copy()
                fft_filtered[~mask] = 0
                
                # Convert back to time domain
                component = np.real(np.fft.ifft(fft_filtered))
                components.append(component)
            
            components = np.array(components)
            n_components = len(components)
            
            # Compute residual as difference from reconstruction
            reconstructed = np.sum(components, axis=0)
            residual = time_series - reconstructed
            
            return {
                'imfs': components,  # FFT components (equivalent to IMFs)
                'residual': residual,
                'components': np.vstack([components, residual[np.newaxis, :]]),
                'n_imfs': n_components,
                'freqs': freqs,  # FFT-specific: frequencies
                'periods': periods  # FFT-specific: periods
            }
            
        except Exception as e:
            print(f"⚠️  FFT failed for station: {e}")
            return None
    
    def wavelet_decompose_station(self, time_series, time_vector, wavelet='db4', levels=6):
        """Decompose single station time series using Wavelet decomposition"""
        if not HAS_WAVELET:
            print("❌ PyWavelets not available")
            return None
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(time_series, wavelet, level=levels)
            
            # coeffs[0] is approximation, coeffs[1:] are details
            approximation = coeffs[0]
            details = coeffs[1:]
            
            # Reconstruct each detail level separately
            components = []
            
            for i, detail in enumerate(details):
                # Create zero coefficients except for this detail level
                coeffs_single = [np.zeros_like(c) for c in coeffs]
                coeffs_single[i+1] = detail  # +1 because coeffs[0] is approximation
                
                # Reconstruct this component
                component = pywt.waverec(coeffs_single, wavelet)
                
                # Ensure same length as original (waverec can have slight length differences)
                if len(component) > len(time_series):
                    component = component[:len(time_series)]
                elif len(component) < len(time_series):
                    component = np.pad(component, (0, len(time_series) - len(component)), 'constant')
                
                components.append(component)
            
            # Add approximation as the final component (equivalent to residual)
            approx_reconstructed = pywt.waverec([approximation] + [np.zeros_like(d) for d in details], wavelet)
            if len(approx_reconstructed) > len(time_series):
                approx_reconstructed = approx_reconstructed[:len(time_series)]
            elif len(approx_reconstructed) < len(time_series):
                approx_reconstructed = np.pad(approx_reconstructed, (0, len(time_series) - len(approx_reconstructed)), 'constant')
            
            components = np.array(components)
            n_components = len(components)
            
            return {
                'imfs': components,  # Wavelet detail components (equivalent to IMFs)
                'residual': approx_reconstructed,  # Approximation as residual
                'components': np.vstack([components, approx_reconstructed[np.newaxis, :]]),
                'n_imfs': n_components,
                'wavelet': wavelet,  # Wavelet-specific: wavelet type
                'levels': levels     # Wavelet-specific: decomposition levels
            }
            
        except Exception as e:
            print(f"⚠️  Wavelet failed for station: {e}")
            return None
    
    def power_spectral_analysis(self, component, time_vector, sampling_interval=6):
        """Analyze power spectral density of component to determine frequency characteristics"""
        
        # Calculate power spectral density
        freqs, psd = signal.welch(component, fs=1/sampling_interval, nperseg=min(64, len(component)//4))
        
        # Convert frequency to period (days)
        periods = 1.0 / (freqs + 1e-12)  # Add small value to avoid division by zero
        
        # Find dominant period
        max_power_idx = np.argmax(psd[1:]) + 1  # Skip DC component
        dominant_period = periods[max_power_idx]
        
        # Count major peaks (peaks with power > 50% of max power)
        peak_threshold = 0.5 * np.max(psd[1:])
        peaks, _ = signal.find_peaks(psd[1:], height=peak_threshold)
        n_major_peaks = len(peaks)
        
        # Calculate spectral energy in different bands
        band_energies = {}
        for band_name, (min_period, max_period) in self.frequency_bands.items():
            if max_period == float('inf'):
                mask = periods >= min_period
            else:
                mask = (periods >= min_period) & (periods <= max_period)
            band_energies[band_name] = np.sum(psd[mask])
        
        # Normalize energies
        total_energy = np.sum(psd[1:])  # Exclude DC
        band_energies_norm = {k: v/total_energy for k, v in band_energies.items()}
        
        return {
            'dominant_period': dominant_period,
            'n_major_peaks': n_major_peaks,
            'band_energies': band_energies_norm,
            'freqs': freqs,
            'psd': psd,
            'periods': periods
        }
    
    def re_categorize_component(self, component, time_vector):
        """Re-categorize component based on power spectral analysis"""
        
        try:
            spectral_info = self.power_spectral_analysis(component, time_vector)
            dominant_period = spectral_info['dominant_period']
            band_energies = spectral_info['band_energies']
        except Exception as e:
            print(f"⚠️  Warning: Spectral analysis failed: {e}")
            # Return default categorization
            return {
                'period_category': 'high_freq',
                'energy_category': 'high_freq',
                'final_category': 'high_freq',
                'template_category': 'high_freq',
                'dominant_period': 30.0,
                'band_energies': {band: 0.0 for band in self.frequency_bands.keys()},
                'template_scores': {band: 0.0 for band in self.frequency_bands.keys()},
                'spectral_info': {'dominant_period': 30.0, 'band_energies': {}}
            }
        
        # Strategy 1: Dominant period classification
        if dominant_period <= 60:
            period_category = 'high_freq'
        elif dominant_period <= 120:
            period_category = 'quarterly'
        elif dominant_period <= 280:
            period_category = 'semi_annual'
        elif dominant_period <= 400:
            period_category = 'annual'
        elif dominant_period <= 1000:
            period_category = 'long_annual'
        else:
            period_category = 'trend'
        
        # Strategy 2: Spectral energy classification
        max_energy_band = max(band_energies.items(), key=lambda x: x[1])[0]
        
        # Strategy 3: Multi-criteria decision
        # Use both dominant period and energy distribution
        energy_threshold = 0.3
        if band_energies[period_category] > energy_threshold:
            final_category = period_category
        else:
            final_category = max_energy_band
        
        # Strategy 4: Template matching (simplified - based on period ranges)
        template_scores = {}
        for band_name, (min_p, max_p) in self.frequency_bands.items():
            if max_p == float('inf'):
                score = 1.0 if dominant_period >= min_p else 0.0
            else:
                if min_p <= dominant_period <= max_p:
                    score = 1.0
                else:
                    # Distance-based score
                    if dominant_period < min_p:
                        score = max(0, 1 - (min_p - dominant_period) / min_p)
                    else:
                        score = max(0, 1 - (dominant_period - max_p) / max_p)
            template_scores[band_name] = score
        
        template_category = max(template_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'period_category': period_category,
            'energy_category': max_energy_band,
            'final_category': final_category,
            'template_category': template_category,
            'dominant_period': dominant_period,
            'band_energies': band_energies,
            'template_scores': template_scores,
            'spectral_info': spectral_info
        }
    
    def process_single_station_emd(self, station_data):
        """Process single station EMD decomposition (for parallel processing)"""
        station_idx, time_series, time_vector = station_data
        
        # Skip stations with too many NaN values
        if np.sum(~np.isnan(time_series)) < self.n_acquisitions * 0.5:
            return station_idx, None, None
        
        # Interpolate small gaps if any
        if np.any(np.isnan(time_series)):
            valid_mask = ~np.isnan(time_series)
            if np.sum(valid_mask) >= 10:  # Need at least 10 valid points
                time_series = np.interp(time_vector, time_vector[valid_mask], time_series[valid_mask])
            else:
                return station_idx, None, None
        
        # Perform EMD decomposition
        result = self.emd_decompose_station(time_series, time_vector)
        
        if result is not None:
            # Re-categorize each component
            station_categories = {}
            
            for j in range(result['n_imfs']):
                component = result['imfs'][j, :]
                category_info = self.re_categorize_component(component, time_vector)
                station_categories[f'imf_{j}'] = category_info
            
            # Also analyze residual
            residual_category = self.re_categorize_component(result['residual'], time_vector)
            station_categories['residual'] = residual_category
            
            return station_idx, result, station_categories
        else:
            return station_idx, None, None

    def decompose_all_stations(self, method='emd', parallel_mode='sequential', n_workers=None, chunk_size=100, verbose=True, **method_kwargs):
        """
        Decompose all stations using specified method with multiple parallelization options
        
        Parameters:
        -----------
        method : str
            Decomposition method ('emd', 'vmd', 'fft', 'wavelet', 'ssa')
        parallel_mode : str
            Parallelization approach:
            - 'sequential': One station at a time (most reliable)
            - 'threading': Multi-threading (good for I/O, limited by GIL)
            - 'multiprocessing': Multi-processing (best for CPU-intensive)
            - 'process_pool': ProcessPoolExecutor (modern multiprocessing)
        n_workers : int
            Number of workers (default: cpu_count())
        chunk_size : int
            Chunk size for progress tracking
        """
        print(f"🔄 Starting {method.upper()} decomposition for all {self.n_stations} stations...")
        print(f"🚀 Parallelization mode: {parallel_mode.upper()}")
        
        # Check method availability
        method_checks = {
            'emd': (HAS_EMD, "EMD-signal not available. Install with: conda install conda-forge::emd-signal"),
            'vmd': (HAS_VMD, "vmdpy not available. Install with: pip install vmdpy"),
            'fft': (True, ""),  # FFT is part of scipy, always available
            'wavelet': (HAS_WAVELET, "PyWavelets not available. Install with: conda install pywavelets"),

        }
        
        if method not in method_checks:
            print(f"❌ Unknown method: {method}. Available: {list(method_checks.keys())}")
            return False
        
        is_available, error_msg = method_checks[method]
        if not is_available:
            print(f"❌ {error_msg}")
            return False
        
        # Set number of workers
        if n_workers is None:
            n_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming system
        
        print(f"💻 Using {n_workers} workers ({parallel_mode} mode)")
        
        # Create time vector (6-day sampling)
        time_vector = np.arange(self.n_acquisitions) * 6  # days
        
        # Initialize storage with appropriate sizes for different methods
        if method in ['emd', 'vmd']:
            n_max_components = 10  # Variable number for EMD/VMD
        elif method == 'fft':
            n_max_components = len(self.frequency_bands)  # Fixed 6 bands for FFT
        elif method == 'wavelet':
            n_max_components = method_kwargs.get('levels', 6)  # User-specified levels

        
        all_imfs = np.full((self.n_stations, n_max_components, self.n_acquisitions), np.nan)
        all_residuals = np.full((self.n_stations, self.n_acquisitions), np.nan)
        n_imfs_per_station = np.zeros(self.n_stations, dtype=int)
        
        # Initialize re-categorization storage
        component_categories = {}
        
        # Process stations with selected parallelization mode
        successful_decompositions = 0
        failed_decompositions = 0
        
        start_time = time.time()
        
        # Route to appropriate processing method
        if parallel_mode == 'sequential':
            successful_decompositions, failed_decompositions, component_categories = self._process_sequential(
                method, time_vector, chunk_size, verbose, method_kwargs, 
                all_imfs, all_residuals, n_imfs_per_station)
        elif parallel_mode == 'threading':
            successful_decompositions, failed_decompositions, component_categories = self._process_threading(
                method, time_vector, n_workers, method_kwargs,
                all_imfs, all_residuals, n_imfs_per_station)
        elif parallel_mode == 'multiprocessing':
            successful_decompositions, failed_decompositions, component_categories = self._process_multiprocessing(
                method, time_vector, n_workers, method_kwargs,
                all_imfs, all_residuals, n_imfs_per_station)
        elif parallel_mode == 'process_pool':
            successful_decompositions, failed_decompositions, component_categories = self._process_pool_executor(
                method, time_vector, n_workers, method_kwargs,
                all_imfs, all_residuals, n_imfs_per_station)
        else:
            print(f"❌ Unknown parallel_mode: {parallel_mode}")
            return False
        
        # Store results
        self.decomposition_results[method] = {
            'imfs': all_imfs,
            'residuals': all_residuals,
            'n_imfs_per_station': n_imfs_per_station,
            'time_vector': time_vector,
            'successful_decompositions': successful_decompositions,
            'failed_decompositions': failed_decompositions
        }
        
        self.re_categorization_results[method] = component_categories
        
        total_time = time.time() - start_time
        print(f"✅ {method.upper()} decomposition completed in {total_time/60:.1f} minutes:")
        print(f"   Successful: {successful_decompositions}/{self.n_stations} stations")
        print(f"   Failed: {failed_decompositions}/{self.n_stations} stations")
        print(f"   Success rate: {100*successful_decompositions/self.n_stations:.1f}%")
        print(f"   Processing rate: {successful_decompositions/(total_time/60):.1f} stations/min")
        
        return True

    def _process_sequential(self, method, time_vector, chunk_size, verbose, method_kwargs, 
                           all_imfs, all_residuals, n_imfs_per_station):
        """Original sequential processing"""
        successful_decompositions = 0
        failed_decompositions = 0
        component_categories = {}
        
        n_chunks = (self.n_stations + chunk_size - 1) // chunk_size
        print(f"🔄 Using SEQUENTIAL processing (chunk_size={chunk_size})")
        
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, self.n_stations)
            
            for i in range(chunk_start, chunk_end):
                station_idx, result, station_categories = self.process_single_station(
                    (i, self.displacement[i, :], time_vector), method, **method_kwargs)
                
                if result is not None:
                    n_imfs = result['n_imfs']
                    n_imfs_per_station[i] = n_imfs
                    all_imfs[i, :n_imfs, :] = result['imfs']
                    all_residuals[i, :] = result['residual']
                    component_categories[i] = station_categories
                    successful_decompositions += 1
                else:
                    failed_decompositions += 1
            
            if verbose and (chunk_idx + 1) % 10 == 0:
                print(f"🔄 Processed {chunk_end} stations ({100*chunk_end/self.n_stations:.1f}%)")
        
        return successful_decompositions, failed_decompositions, component_categories

    def _process_pool_executor(self, method, time_vector, n_workers, method_kwargs,
                              all_imfs, all_residuals, n_imfs_per_station):
        """Modern multiprocessing using ProcessPoolExecutor"""
        print(f"🚀 Using PROCESS POOL processing with {n_workers} workers (WILL MAX CPU!)")
        
        # Create list of all station data
        station_tasks = []
        for i in range(self.n_stations):
            station_tasks.append((i, self.displacement[i, :], time_vector))
        
        successful_decompositions = 0
        failed_decompositions = 0
        component_categories = {}
        
        # Process in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Create partial function with method and kwargs
            process_func = partial(self._process_single_station_static, 
                                 method=method, 
                                 frequency_bands=self.frequency_bands,
                                 n_acquisitions=self.n_acquisitions,
                                 **method_kwargs)
            
            # Submit all tasks
            print(f"📤 Submitting {len(station_tasks)} tasks to {n_workers} workers...")
            futures = {executor.submit(process_func, task): i for i, task in enumerate(station_tasks)}
            
            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                station_idx = futures[future]
                try:
                    result_station_idx, result, station_categories = future.result()
                    
                    if result is not None:
                        n_imfs = result['n_imfs']
                        n_imfs_per_station[station_idx] = n_imfs
                        all_imfs[station_idx, :n_imfs, :] = result['imfs']
                        all_residuals[station_idx, :] = result['residual']
                        component_categories[station_idx] = station_categories
                        successful_decompositions += 1
                    else:
                        failed_decompositions += 1
                        
                except Exception as e:
                    print(f"⚠️  Task {station_idx} failed: {e}")
                    failed_decompositions += 1
                
                completed += 1
                if completed % 100 == 0:
                    print(f"✅ Completed {completed}/{self.n_stations} stations ({100*completed/self.n_stations:.1f}%)")
        
        return successful_decompositions, failed_decompositions, component_categories
    
    @staticmethod
    def _process_single_station_static(station_data, method, frequency_bands, n_acquisitions, **method_kwargs):
        """Static method for multiprocessing compatibility"""
        station_idx, time_series, time_vector = station_data
        
        # Skip stations with too many NaN values
        if np.sum(~np.isnan(time_series)) < n_acquisitions * 0.5:
            return station_idx, None, None
        
        # Interpolate small gaps if any
        if np.any(np.isnan(time_series)):
            valid_mask = ~np.isnan(time_series)
            if np.sum(valid_mask) >= 10:
                time_series = np.interp(time_vector, time_vector[valid_mask], time_series[valid_mask])
            else:
                return station_idx, None, None
        
        # Perform decomposition (simplified for static method)
        try:
            if method == 'emd' and HAS_EMD:
                imfs_matrix = emd.sift.sift(time_series)
                n_imfs = imfs_matrix.shape[1] - 1
                imfs = imfs_matrix[:, :-1].T
                residual = imfs_matrix[:, -1]
                result = {'imfs': imfs, 'residual': residual, 'n_imfs': n_imfs}

            else:
                # For other methods, create simple result (can be enhanced)
                result = None
            
            if result is not None:
                # Simple categorization (can be enhanced)
                station_categories = {}
                for j in range(result['n_imfs']):
                    station_categories[f'imf_{j}'] = {'final_category': 'quarterly'}  # Simplified
                station_categories['residual'] = {'final_category': 'trend'}
                
                return station_idx, result, station_categories
            else:
                return station_idx, None, None
                
        except Exception as e:
            return station_idx, None, None

    def _process_threading(self, method, time_vector, n_workers, method_kwargs,
                          all_imfs, all_residuals, n_imfs_per_station):
        """Multi-threading implementation (limited by GIL for CPU tasks)"""
        print(f"🧵 Using THREADING processing with {n_workers} threads (Limited by Python GIL)")
        
        successful_decompositions = 0
        failed_decompositions = 0
        component_categories = {}
        
        def process_batch(batch_stations):
            batch_results = []
            for i in batch_stations:
                result = self.process_single_station((i, self.displacement[i, :], time_vector), method, **method_kwargs)
                batch_results.append(result)
            return batch_results
        
        # Create batches for threading
        batch_size = max(1, self.n_stations // n_workers)
        batches = [list(range(i, min(i + batch_size, self.n_stations))) 
                  for i in range(0, self.n_stations, batch_size)]
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                batch_results = future.result()
                for station_idx, result, station_categories in batch_results:
                    if result is not None:
                        n_imfs = result['n_imfs']
                        n_imfs_per_station[station_idx] = n_imfs
                        all_imfs[station_idx, :n_imfs, :] = result['imfs']
                        all_residuals[station_idx, :] = result['residual']
                        component_categories[station_idx] = station_categories
                        successful_decompositions += 1
                    else:
                        failed_decompositions += 1
        
        return successful_decompositions, failed_decompositions, component_categories

    def _process_multiprocessing(self, method, time_vector, n_workers, method_kwargs,
                                all_imfs, all_residuals, n_imfs_per_station):
        """Traditional multiprocessing (may have serialization issues)"""
        print(f"⚙️  Using MULTIPROCESSING with {n_workers} processes (Traditional approach)")
        
        # Prepare data for multiprocessing
        station_data_list = [(i, self.displacement[i, :], time_vector) for i in range(self.n_stations)]
        
        successful_decompositions = 0
        failed_decompositions = 0
        component_categories = {}
        
        try:
            with Pool(processes=n_workers) as pool:
                process_func = partial(self._process_single_station_static,
                                     method=method,
                                     frequency_bands=self.frequency_bands,
                                     n_acquisitions=self.n_acquisitions,
                                     **method_kwargs)
                
                results = pool.map(process_func, station_data_list)
                
                for station_idx, result, station_categories in results:
                    if result is not None:
                        n_imfs = result['n_imfs']
                        n_imfs_per_station[station_idx] = n_imfs
                        all_imfs[station_idx, :n_imfs, :] = result['imfs']
                        all_residuals[station_idx, :] = result['residual']
                        component_categories[station_idx] = station_categories
                        successful_decompositions += 1
                    else:
                        failed_decompositions += 1
                        
        except Exception as e:
            print(f"❌ Multiprocessing failed: {e}")
            print("🔄 Falling back to sequential processing...")
            return self._process_sequential(method, time_vector, 100, True, method_kwargs,
                                          all_imfs, all_residuals, n_imfs_per_station)
        
        return successful_decompositions, failed_decompositions, component_categories

    def process_single_station(self, station_data, method, **method_kwargs):
        """Process single station with specified decomposition method"""
        station_idx, time_series, time_vector = station_data
        
        # Skip stations with too many NaN values
        if np.sum(~np.isnan(time_series)) < self.n_acquisitions * 0.5:
            return station_idx, None, None
        
        # Interpolate small gaps if any
        if np.any(np.isnan(time_series)):
            valid_mask = ~np.isnan(time_series)
            if np.sum(valid_mask) >= 10:  # Need at least 10 valid points
                time_series = np.interp(time_vector, time_vector[valid_mask], time_series[valid_mask])
            else:
                return station_idx, None, None
        
        # Perform decomposition using appropriate method
        if method == 'emd':
            result = self.emd_decompose_station(time_series, time_vector)
        elif method == 'vmd':
            result = self.vmd_decompose_station(time_series, time_vector, **method_kwargs)
        elif method == 'fft':
            result = self.fft_decompose_station(time_series, time_vector)
        elif method == 'wavelet':
            result = self.wavelet_decompose_station(time_series, time_vector, **method_kwargs)

        else:
            return station_idx, None, None
        
        if result is not None:
            # Re-categorize each component using the same framework as EMD
            station_categories = {}
            
            for j in range(result['n_imfs']):
                component = result['imfs'][j, :]
                category_info = self.re_categorize_component(component, time_vector)
                station_categories[f'imf_{j}'] = category_info
            
            # Also analyze residual
            residual_category = self.re_categorize_component(result['residual'], time_vector)
            station_categories['residual'] = residual_category
            
            return station_idx, result, station_categories
        else:
            return station_idx, None, None

    
    def calculate_reconstruction_quality(self, method='emd'):
        """Calculate reconstruction quality metrics"""
        print(f"📊 Calculating reconstruction quality for {method.upper()}...")
        
        if method not in self.decomposition_results:
            print(f"❌ No decomposition results found for {method}")
            return False
        
        results = self.decomposition_results[method]
        imfs = results['imfs']
        residuals = results['residuals']
        n_imfs_per_station = results['n_imfs_per_station']
        
        reconstruction_errors = []
        correlations = []
        
        for i in range(self.n_stations):
            if n_imfs_per_station[i] == 0:
                continue
            
            original = self.displacement[i, :]
            
            # Skip if too many NaN values
            if np.sum(~np.isnan(original)) < self.n_acquisitions * 0.5:
                continue
            
            # Reconstruct signal
            n_imfs = n_imfs_per_station[i]
            reconstructed = np.sum(imfs[i, :n_imfs, :], axis=0) + residuals[i, :]
            
            # Calculate metrics (only for valid points)
            valid_mask = ~np.isnan(original)
            if np.sum(valid_mask) < 10:
                continue
            
            orig_valid = original[valid_mask]
            recon_valid = reconstructed[valid_mask]
            
            # RMSE
            rmse = np.sqrt(np.mean((orig_valid - recon_valid)**2))
            reconstruction_errors.append(rmse)
            
            # Correlation
            if np.std(orig_valid) > 1e-6 and np.std(recon_valid) > 1e-6:
                corr = np.corrcoef(orig_valid, recon_valid)[0, 1]
                correlations.append(corr)
        
        quality_metrics = {
            'mean_rmse': np.mean(reconstruction_errors),
            'std_rmse': np.std(reconstruction_errors),
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'n_valid_stations': len(reconstruction_errors)
        }
        
        self.quality_metrics[method] = quality_metrics
        
        print(f"✅ Reconstruction quality for {method.upper()}:")
        print(f"   Mean RMSE: {quality_metrics['mean_rmse']:.3f} ± {quality_metrics['std_rmse']:.3f} mm")
        print(f"   Mean correlation: {quality_metrics['mean_correlation']:.3f} ± {quality_metrics['std_correlation']:.3f}")
        print(f"   Valid stations: {quality_metrics['n_valid_stations']}")
        
        return True
    
    def create_decomposition_visualizations(self, method='emd', n_stations_to_plot=300):
        """Create comprehensive visualization of decomposition results"""
        print(f"📊 Creating {method.upper()} decomposition visualizations...")
        
        if method not in self.decomposition_results:
            print(f"❌ No decomposition results found for {method}")
            return False
        
        # Create figures directory
        fig_dir = Path("figures")
        fig_dir.mkdir(exist_ok=True)
        
        results = self.decomposition_results[method]
        time_vector = results['time_vector']
        
        # Select stations for visualization (well-distributed)
        n_plot = min(n_stations_to_plot, self.n_stations)
        station_indices = np.linspace(0, self.n_stations-1, n_plot, dtype=int)
        
        # Figure 1: Component overview with transparency and color coding
        self._create_component_overview_figure(method, station_indices, time_vector)
        
        # Figure 2: Frequency band analysis
        self._create_frequency_band_figure(method)
        
        # Figure 3: Re-categorization quality assessment
        self._create_recategorization_figure(method)
        
        # Figure 4: Spatial distribution of decomposition characteristics
        self._create_spatial_decomposition_figure(method)
        
        # Figure 5: Re-categorized frequency band signals
        self._create_frequency_band_signals_figure(method, station_indices, time_vector)
        
        print(f"✅ Created {method.upper()} decomposition visualizations")
        return True
    
    def _create_component_overview_figure(self, method, station_indices, time_vector):
        """Create component overview with color-coded time series and transparency"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{method.upper()} Component Overview - {len(station_indices)} Stations', fontsize=16, fontweight='bold')
        
        results = self.decomposition_results[method]
        imfs = results['imfs']
        residuals = results['residuals']
        n_imfs_per_station = results['n_imfs_per_station']
        
        # Define colors for different stations
        colors = plt.cm.Set3(np.linspace(0, 1, len(station_indices)))
        
        # Plot first 6 components (IMF1-IMF5 + Residual)
        component_names = ['IMF1 (High Freq)', 'IMF2 (Quarterly)', 'IMF3 (Semi-Annual)', 
                          'IMF4 (Annual)', 'IMF5 (Long-term)', 'Residual (Trend)']
        
        for comp_idx, (ax, comp_name) in enumerate(zip(axes.flat, component_names)):
            
            all_components = []
            
            for i, station_idx in enumerate(station_indices):
                if n_imfs_per_station[station_idx] == 0:
                    continue
                
                if comp_idx < 5:  # IMFs
                    if comp_idx < n_imfs_per_station[station_idx]:
                        component = imfs[station_idx, comp_idx, :]
                    else:
                        continue
                else:  # Residual
                    component = residuals[station_idx, :]
                
                # Skip if all NaN
                if np.all(np.isnan(component)):
                    continue
                
                all_components.append(component)
                
                # Plot individual time series with transparency
                ax.plot(time_vector, component, color=colors[i], alpha=0.3, linewidth=0.5)
            
            # Calculate and plot average and standard deviation
            if all_components:
                all_components = np.array(all_components)
                
                # Calculate statistics (ignoring NaN)
                mean_component = np.nanmean(all_components, axis=0)
                std_component = np.nanstd(all_components, axis=0)
                
                # Plot average line
                ax.plot(time_vector, mean_component, 'red', linewidth=2, label='Average')
                
                # Plot standard deviation as shaded area
                ax.fill_between(time_vector, 
                               mean_component - std_component,
                               mean_component + std_component, 
                               color='red', alpha=0.2, label='±1 STD')
            
            ax.set_title(comp_name)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Displacement (mm)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'figures/ps02_fig01_{method}_component_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_frequency_band_figure(self, method):
        """Create frequency band analysis figure"""
        
        if method not in self.re_categorization_results:
            return
        
        categories = self.re_categorization_results[method]
        
        # Count components in each frequency band
        band_counts = {band: 0 for band in self.frequency_bands.keys()}
        total_components = 0
        
        for station_idx, station_categories in categories.items():
            for comp_name, category_info in station_categories.items():
                final_category = category_info['final_category']
                band_counts[final_category] += 1
                total_components += 1
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of frequency band distribution
        labels = list(band_counts.keys())
        sizes = list(band_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'{method.upper()} Component Distribution by Frequency Band')
        else:
            ax1.text(0.5, 0.5, 'No Components\nClassified', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title(f'{method.upper()} Component Distribution (No Data)')
        
        # Bar chart with counts
        if sum(sizes) > 0:
            ax2.bar(labels, sizes, color=colors)
            ax2.set_title(f'{method.upper()} Component Counts by Frequency Band')
            ax2.set_ylabel('Number of Components')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Components\nClassified', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title(f'{method.upper()} Component Counts (No Data)')
        
        # Add text box with statistics
        if len(categories) > 0:
            stats_text = f"""Total Components: {total_components}
Stations Processed: {len(categories)}
Avg Components/Station: {total_components/len(categories):.1f}"""
        else:
            stats_text = """Total Components: 0
Stations Processed: 0
Avg Components/Station: 0.0"""
        
        ax2.text(0.7, 0.7, stats_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'figures/ps02_fig02_{method}_frequency_bands.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_recategorization_figure(self, method):
        """Create re-categorization quality assessment figure"""
        
        if method not in self.re_categorization_results:
            return
        
        categories = self.re_categorization_results[method]
        
        # Compare different categorization strategies
        strategy_agreements = []
        
        for station_idx, station_categories in categories.items():
            for comp_name, category_info in station_categories.items():
                # Defensive check for required keys
                if not all(key in category_info for key in ['period_category', 'energy_category', 'final_category', 'template_category']):
                    print(f"⚠️  Warning: Missing category keys for station {station_idx}, component {comp_name}")
                    print(f"   Available keys: {list(category_info.keys())}")
                    continue
                
                period_cat = category_info['period_category']
                energy_cat = category_info['energy_category']
                final_cat = category_info['final_category']
                template_cat = category_info['template_category']
                
                # Check agreement between strategies
                all_cats = [period_cat, energy_cat, final_cat, template_cat]
                unique_cats = set(all_cats)
                agreement = len(all_cats) - len(unique_cats) + 1  # Simple agreement metric
                strategy_agreements.append(agreement)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Agreement histogram
        ax1.hist(strategy_agreements, bins=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Strategy Agreement Score')
        ax1.set_ylabel('Number of Components')
        ax1.set_title(f'{method.upper()} Re-categorization Strategy Agreement')
        ax1.grid(True, alpha=0.3)
        
        # Strategy comparison matrix (simplified)
        strategies = ['Period', 'Energy', 'Final', 'Template']
        agreement_matrix = np.zeros((4, 4))
        
        for station_idx, station_categories in categories.items():
            for comp_name, category_info in station_categories.items():
                # Defensive check for required keys (same as above)
                if not all(key in category_info for key in ['period_category', 'energy_category', 'final_category', 'template_category']):
                    continue  # Skip incomplete entries
                
                cats = [category_info['period_category'], category_info['energy_category'],
                       category_info['final_category'], category_info['template_category']]
                
                for i in range(4):
                    for j in range(4):
                        if cats[i] == cats[j]:
                            agreement_matrix[i, j] += 1
        
        # Normalize
        total_components = len([c for station_cats in categories.values() for c in station_cats.values()])
        agreement_matrix /= total_components
        
        im = ax2.imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xticks(range(4))
        ax2.set_yticks(range(4))
        ax2.set_xticklabels(strategies)
        ax2.set_yticklabels(strategies)
        ax2.set_title(f'{method.upper()} Strategy Agreement Matrix')
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                ax2.text(j, i, f'{agreement_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='Agreement Fraction')
        plt.tight_layout()
        plt.savefig(f'figures/ps02_fig03_{method}_recategorization_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_spatial_decomposition_figure(self, method):
        """Create spatial distribution of decomposition characteristics"""
        
        results = self.decomposition_results[method]
        n_imfs_per_station = results['n_imfs_per_station']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Number of IMFs per station
        valid_stations = n_imfs_per_station > 0
        scatter1 = ax1.scatter(self.coordinates[valid_stations, 0], 
                             self.coordinates[valid_stations, 1],
                             c=n_imfs_per_station[valid_stations],
                             cmap='viridis', s=20, alpha=0.7)
        
        ax1.set_xlabel('Longitude (°E)')
        ax1.set_ylabel('Latitude (°N)')
        ax1.set_title(f'{method.upper()} - Number of IMFs per Station')
        plt.colorbar(scatter1, ax=ax1, label='Number of IMFs')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Signal complexity analysis
        # Calculate signal complexity metrics for each station
        complexity_metric = np.zeros(self.n_stations)
        
        for i in range(self.n_stations):
            if n_imfs_per_station[i] > 0:
                # Calculate complexity as the sum of variance across all IMFs
                station_complexity = 0
                for j in range(n_imfs_per_station[i]):
                    imf_variance = np.var(results['imfs'][i, j, :])
                    station_complexity += imf_variance
                
                # Add residual variance
                residual_variance = np.var(results['residuals'][i, :])
                station_complexity += residual_variance
                
                complexity_metric[i] = station_complexity
        
        # Only plot stations with valid decomposition
        valid_stations = n_imfs_per_station > 0
        
        if np.any(valid_stations):
            scatter2 = ax2.scatter(self.coordinates[valid_stations, 0], 
                                 self.coordinates[valid_stations, 1],
                                 c=complexity_metric[valid_stations], 
                                 cmap='viridis', s=20, alpha=0.7)
            
            ax2.set_xlabel('Longitude (°E)')
            ax2.set_ylabel('Latitude (°N)')
            ax2.set_title(f'{method.upper()} Signal Complexity\n(Total Variance Across All Components)')
            plt.colorbar(scatter2, ax=ax2, label='Signal Complexity (mm²)')
            
            # Add explanation text (shorter and moved to top-left)
            explanation_text = f"""Signal Complexity = Σ(Variance of each IMF) + Variance(Residual)
Range: {complexity_metric[valid_stations].min():.1f} - {complexity_metric[valid_stations].max():.1f} mm²
Higher values = More complex temporal behavior
Stations with higher subsidence often show higher complexity"""
            
            ax2.text(0.02, 0.98, explanation_text, transform=ax2.transAxes, 
                    verticalalignment='top', horizontalalignment='left', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            ax2.text(0.5, 0.5, 'No Valid\nDecompositions', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Signal Complexity (No Data)')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figures/ps02_fig04_{method}_spatial_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_frequency_band_signals_figure(self, method, station_indices, time_vector):
        """Create visualization of re-categorized frequency band signals"""
        
        if method not in self.re_categorization_results:
            return
        
        results = self.decomposition_results[method]
        imfs = results['imfs']
        residuals = results['residuals']
        n_imfs_per_station = results['n_imfs_per_station']
        categories = self.re_categorization_results[method]
        
        # Reconstruct signals by frequency band for each station
        frequency_bands = ['high_freq', 'quarterly', 'semi_annual', 'annual', 'long_annual', 'trend']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{method.upper()} Re-categorized Frequency Band Signals - {len(station_indices)} Stations', 
                     fontsize=16, fontweight='bold')
        
        # Define colors for different stations
        colors = plt.cm.Set3(np.linspace(0, 1, len(station_indices)))
        
        for band_idx, (band_name, ax) in enumerate(zip(frequency_bands, axes.flat)):
            
            all_band_signals = []
            
            for i, station_idx in enumerate(station_indices):
                if station_idx not in categories or n_imfs_per_station[station_idx] == 0:
                    continue
                
                station_categories = categories[station_idx]
                
                # Reconstruct signal for this frequency band
                band_signal = np.zeros(len(time_vector))
                
                # Special handling for trend band - always include reconstructed trend
                if band_name == 'trend':
                    # Always reconstruct trend = EMD residual + original ps00 subsidence rate
                    subsidence_rate = self.subsidence_rates[station_idx]  # mm/year
                    # Convert to mm per 6-day period
                    rate_per_period = subsidence_rate * 6 / 365.25
                    linear_trend = rate_per_period * np.arange(len(time_vector))
                    band_signal += residuals[station_idx, :] + linear_trend
                else:
                    # Add components classified in this band
                    for comp_name, category_info in station_categories.items():
                        if category_info['final_category'] == band_name:
                            if comp_name == 'residual':
                                band_signal += residuals[station_idx, :]
                            else:
                                # Extract IMF index from component name (e.g., 'imf_0' -> 0)
                                imf_idx = int(comp_name.split('_')[1])
                                if imf_idx < n_imfs_per_station[station_idx]:
                                    band_signal += imfs[station_idx, imf_idx, :]
                
                # Skip if signal is all zeros
                if np.all(np.abs(band_signal) < 1e-10):
                    continue
                
                all_band_signals.append(band_signal)
                
                # Plot individual time series with transparency
                ax.plot(time_vector, band_signal, color=colors[i], alpha=0.3, linewidth=0.5)
            
            # Calculate and plot average and standard deviation
            if all_band_signals:
                all_band_signals = np.array(all_band_signals)
                
                # Calculate statistics
                mean_signal = np.nanmean(all_band_signals, axis=0)
                std_signal = np.nanstd(all_band_signals, axis=0)
                
                # Plot average line
                ax.plot(time_vector, mean_signal, 'red', linewidth=2, label='Average')
                
                # Plot standard deviation as shaded area
                ax.fill_between(time_vector, 
                               mean_signal - std_signal,
                               mean_signal + std_signal, 
                               color='red', alpha=0.2, label='±1 STD')
                
                # Add band-specific title and labels
                band_titles = {
                    'high_freq': 'High Frequency (1-60 days)\nAtmospheric & Noise',
                    'quarterly': 'Quarterly (60-120 days)\nIrrigation Cycles',
                    'semi_annual': 'Semi-Annual (120-280 days)\nMonsoon Patterns',
                    'annual': 'Annual (280-400 days)\nYearly Deformation',
                    'long_annual': 'Long Annual (400-1000 days)\nMulti-year Patterns',
                    'trend': 'Trend (EMD Residual + ps00 Linear Trend)\nLong-term Subsidence'
                }
                
                ax.set_title(band_titles.get(band_name, band_name.title()))
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Displacement (mm)')
                ax.grid(True, alpha=0.3)
                
                # Position legend differently for trend subplot to avoid overlap with stats
                if band_name == 'trend':
                    ax.legend(loc='upper right')
                else:
                    ax.legend()
                
                # Calculate period statistics for this band
                n_components = len(all_band_signals)
                max_amplitude = np.max(np.abs(mean_signal))
                
                if band_name == 'trend':
                    # For trend, show linear rate statistics
                    trend_rates = []
                    for idx in station_indices:
                        if idx < len(self.subsidence_rates):
                            trend_rates.append(self.subsidence_rates[idx])
                    
                    if trend_rates:
                        mean_rate = np.mean(trend_rates)
                        std_rate = np.std(trend_rates)
                        stats_text = f'Stations: {n_components}\nMean Rate: {mean_rate:.1f} ± {std_rate:.1f} mm/year\nMax Amplitude: {max_amplitude:.1f} mm'
                    else:
                        stats_text = f'Stations: {n_components}\nMax Amplitude: {max_amplitude:.1f} mm'
                else:
                    # Calculate dominant periods for components in this band
                    band_periods = []
                    for station_idx in station_indices:
                        if station_idx in categories:
                            station_categories = categories[station_idx]
                            for comp_name, category_info in station_categories.items():
                                if category_info['final_category'] == band_name:
                                    band_periods.append(category_info['dominant_period'])
                    
                    if band_periods:
                        mean_period = np.mean(band_periods)
                        std_period = np.std(band_periods)
                        stats_text = f'Components: {n_components}\nPeriod: {mean_period:.1f} ± {std_period:.1f} days\nMax Amplitude: {max_amplitude:.1f} mm'
                    else:
                        stats_text = f'Components: {n_components}\nMax Amplitude: {max_amplitude:.1f} mm'
                
                # Position statistics text differently for trend subplot to avoid legend overlap
                if band_name == 'trend':
                    ax.text(0.02, 0.70, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # No components in this band
                band_titles = {
                    'high_freq': 'High Frequency (1-60 days)\nAtmospheric & Noise',
                    'quarterly': 'Quarterly (60-120 days)\nIrrigation Cycles',
                    'semi_annual': 'Semi-Annual (120-280 days)\nMonsoon Patterns',
                    'annual': 'Annual (280-400 days)\nYearly Deformation',
                    'long_annual': 'Long Annual (400-1000 days)\nMulti-year Patterns',
                    'trend': 'Trend (EMD Residual + ps00 Linear Trend)\nLong-term Subsidence'
                }
                ax.text(0.5, 0.5, f'No Components\nClassified as\n{band_name.title()}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(band_titles.get(band_name, band_name.title()))
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Displacement (mm)')
        
        plt.tight_layout()
        plt.savefig(f'figures/ps02_fig05_{method}_frequency_band_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_decomposition_results(self, method='emd'):
        """Save decomposition results to file"""
        print(f"💾 Saving {method.upper()} decomposition results...")
        
        results_dir = Path("data/processed")
        results_dir.mkdir(exist_ok=True)
        
        if method in self.decomposition_results:
            # Save main decomposition results
            np.savez_compressed(
                results_dir / f'ps02_{method}_decomposition.npz',
                **self.decomposition_results[method],
                coordinates=self.coordinates,
                subsidence_rates=self.subsidence_rates
            )
            
            # Save re-categorization results as JSON
            import json
            
            # Convert numpy types to Python types for JSON serialization
            categorization_for_json = {}
            if method in self.re_categorization_results:
                for station_idx, station_data in self.re_categorization_results[method].items():
                    station_key = str(station_idx)
                    categorization_for_json[station_key] = {}
                    
                    for comp_name, comp_data in station_data.items():
                        categorization_for_json[station_key][comp_name] = {
                            'period_category': comp_data['period_category'],
                            'energy_category': comp_data['energy_category'], 
                            'final_category': comp_data['final_category'],
                            'template_category': comp_data['template_category'],
                            'dominant_period': float(comp_data['dominant_period']),
                            'band_energies': {k: float(v) for k, v in comp_data['band_energies'].items()},
                            'template_scores': {k: float(v) for k, v in comp_data['template_scores'].items()}
                        }
            
            with open(results_dir / f'ps02_{method}_recategorization.json', 'w') as f:
                json.dump(categorization_for_json, f, indent=2)
            
            # Save quality metrics
            if method in self.quality_metrics:
                with open(results_dir / f'ps02_{method}_quality_metrics.json', 'w') as f:
                    json.dump(self.quality_metrics[method], f, indent=2, default=str)
            
            print(f"✅ Saved {method.upper()} results:")
            print(f"   - data/processed/ps02_{method}_decomposition.npz")
            print(f"   - data/processed/ps02_{method}_recategorization.json") 
            print(f"   - data/processed/ps02_{method}_quality_metrics.json")
            
            return True
        
        return False

def main():
    """Main signal decomposition workflow"""
    print("=" * 80)
    print("🚀 ps02_signal_decomposition.py - Multi-Method Signal Decomposition")
    print("📋 DECOMPOSITION: EMD, VMD, FFT, Wavelet - All with high-quality re-categorization")
    print("=" * 80)
    
    # Initialize decomposer
    decomposer = SignalDecomposer()
    
    # Step 1: Load data
    if not decomposer.load_data():
        print("❌ FATAL: Failed to load data")
        return False
    
    # Step 2: Multi-method decomposition
    methods_to_run = ['emd', 'vmd', 'fft', 'wavelet']
    successful_methods = []
    
    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"🔄 Processing {method.upper()} decomposition...")
        print(f"{'='*60}")
        
        # Method-specific parameters
        method_kwargs = {}
        if method == 'vmd':
            method_kwargs = {'n_modes': 6}  # 6 VMD modes
        elif method == 'wavelet':
            method_kwargs = {'wavelet': 'db4', 'levels': 6}  # Daubechies wavelet, 6 levels

        
        # Perform decomposition with SEQUENTIAL processing for reliability
        # (multiprocessing has serialization issues with EMD-signal that cause incomplete categorization)
        if decomposer.decompose_all_stations(method=method, parallel_mode='sequential', 
                                           chunk_size=100, **method_kwargs):
            successful_methods.append(method)
            
            # Calculate reconstruction quality
            if decomposer.calculate_reconstruction_quality(method):
                print(f"✅ {method.upper()} quality analysis completed")
            
            # Create visualizations
            if decomposer.create_decomposition_visualizations(method, n_stations_to_plot=300):
                print(f"✅ {method.upper()} visualizations created")
            
            # Save results
            if decomposer.save_decomposition_results(method):
                print(f"✅ {method.upper()} results saved")
        else:
            print(f"❌ {method.upper()} decomposition failed")
    
    # Final summary
    print("\n" + "=" * 80)
    if successful_methods:
        print("✅ ps02_signal_decomposition.py COMPLETED SUCCESSFULLY")
        print(f"📊 Successfully processed methods: {', '.join([m.upper() for m in successful_methods])}")
        print("📊 Generated figures for each successful method:")
        for method in successful_methods:
            print(f"   {method.upper()}:")
            print(f"   - figures/ps02_fig01_{method}_component_overview.png")
            print(f"   - figures/ps02_fig02_{method}_frequency_bands.png")
            print(f"   - figures/ps02_fig03_{method}_recategorization_quality.png")
            print(f"   - figures/ps02_fig04_{method}_spatial_distribution.png")
            print(f"   - figures/ps02_fig05_{method}_frequency_band_signals.png")
        print("📊 Generated data for each successful method:")
        for method in successful_methods:
            print(f"   - data/processed/ps02_{method}_decomposition.npz")
            print(f"   - data/processed/ps02_{method}_recategorization.json")
            print(f"   - data/processed/ps02_{method}_quality_metrics.json")
        print("🔄 Next: Run ps03_clustering_analysis.py with all methods")
    else:
        print("❌ ps02_signal_decomposition.py FAILED - No methods completed successfully")
    print("=" * 80)
    
    return len(successful_methods) > 0

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
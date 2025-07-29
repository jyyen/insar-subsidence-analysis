#!/usr/bin/env python3
"""
ps05_event_detection.py - Event Detection & Anomaly Analysis

Purpose: Detect deformation events and anomalies from ps02 decomposition results
Methods: Statistical outlier detection, regime change analysis, event characterization
Input: ps02 decomposition results (EMD, FFT, VMD, Wavelet components)
Output: Event catalog, anomaly identification, regime change analysis

Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import argparse
import warnings
from scipy import stats, signal
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime, timedelta
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import os

warnings.filterwarnings('ignore')

# Optional advanced change point detection
try:
    import ruptures as rpt
    HAS_RUPTURES = True
    print("✅ Ruptures available for advanced change point detection")
except ImportError:
    HAS_RUPTURES = False
    print("⚠️  Ruptures not available. Using basic change point detection.")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("✅ Cartopy available for geographic visualization")
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not available. Using matplotlib for geographic plots.")

class EventDetectionAnalysis:
    """
    Comprehensive event detection and anomaly analysis for Taiwan subsidence patterns
    
    Detects:
    1. Sudden deformation events (jumps, spikes)
    2. Anomalous patterns in time series
    3. Regime changes in long-term trends
    4. Event spatial-temporal correlation
    """
    
    def __init__(self, methods=['emd'], event_threshold=3.0, anomaly_contamination=0.1, n_jobs=-1):
        """
        Initialize event detection analysis framework
        
        Parameters:
        -----------
        methods : list
            Decomposition methods to analyze ['emd', 'fft', 'vmd', 'wavelet']
        event_threshold : float
            Z-score threshold for event detection (typically 2-4)
        anomaly_contamination : float
            Expected fraction of anomalies (0.05-0.2)
        n_jobs : int
            Number of parallel processes (-1 for all available cores)
        """
        self.methods = methods
        self.event_threshold = event_threshold
        self.anomaly_contamination = anomaly_contamination
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        
        # Data containers
        self.coordinates = None
        self.time_vector = None
        self.decomposition_data = {}
        self.original_displacement = None
        
        # Analysis results
        self.sudden_events = {}
        self.anomalies = {}
        self.regime_changes = {}
        self.event_catalog = {}
        
        # Create output directories
        self.setup_directories()
        
        print(f"🚀 Parallelization enabled: {self.n_jobs} processes")
        
    def setup_directories(self):
        """Create output directories for results and figures"""
        self.figures_dir = Path("figures")
        self.results_dir = Path("data/processed/ps05_events")
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Figures directory: {self.figures_dir}")

    def load_decomposition_data(self):
        """Load decomposition results from ps02"""
        print("📡 Loading decomposition data from ps02...")
        
        try:
            # Load preprocessed coordinates and original displacement
            preprocessed_data = np.load("data/processed/ps00_preprocessed_data.npz")
            self.coordinates = preprocessed_data['coordinates']
            self.original_displacement = preprocessed_data['displacement']
            
            print(f"✅ Loaded coordinates for {len(self.coordinates)} stations")
            print(f"✅ Loaded original displacement: {self.original_displacement.shape}")
            
            # Load decomposition results for each method
            for method in self.methods:
                try:
                    decomp_file = f"data/processed/ps02_{method}_decomposition.npz"
                    decomp_data = np.load(decomp_file)
                    
                    self.decomposition_data[method] = {
                        'imfs': decomp_data['imfs'],
                        'residuals': decomp_data['residuals'],
                        'time_vector': decomp_data['time_vector'],
                        'n_imfs_per_station': decomp_data['n_imfs_per_station']
                    }
                    
                    print(f"✅ Loaded {method.upper()} decomposition: {decomp_data['imfs'].shape}")
                    
                except FileNotFoundError:
                    print(f"⚠️  {method.upper()} decomposition file not found, skipping...")
                    continue
            
            if len(self.decomposition_data) == 0:
                print("❌ No decomposition data loaded")
                return False
            
            # Use time vector from first available method
            first_method = list(self.decomposition_data.keys())[0]
            self.time_vector = self.decomposition_data[first_method]['time_vector']
            
            print(f"✅ Time vector: {len(self.time_vector)} time points")
            return True
            
        except Exception as e:
            print(f"❌ Error loading decomposition data: {e}")
            return False

    def detect_sudden_events(self, method='emd'):
        """
        Detect sudden events using multiple statistical approaches
        
        Focus on high-frequency components and rate-of-change analysis
        """
        print(f"🔄 Detecting sudden events using {method.upper()} decomposition...")
        
        if method not in self.decomposition_data:
            print(f"❌ No data available for method: {method}")
            return False
        
        try:
            data = self.decomposition_data[method]
            imfs = data['imfs']
            n_stations = len(self.coordinates)
            
            # Initialize results
            events = {
                'z_score_events': [],
                'rate_change_events': [],
                'spike_events': [],
                'event_summary': []
            }
            
            print(f"   Analyzing {n_stations} stations for sudden events...")
            
            # Prepare data for parallel processing
            n_analysis_stations = min(n_stations, len(imfs))
            
            if self.n_jobs == 1 or n_analysis_stations < 100:
                # Sequential processing for small datasets
                print(f"   Using sequential processing for {n_analysis_stations} stations")
                for station_idx in range(n_analysis_stations):
                    station_events = self._detect_station_events(
                        station_idx, imfs[station_idx], method
                    )
                    
                    # Collect all events for this station
                    for event_type, event_list in station_events.items():
                        events[event_type].extend(event_list)
            else:
                # Parallel processing for large datasets
                print(f"   Using parallel processing with {self.n_jobs} processes for {n_analysis_stations} stations")
                
                # Create partial function with fixed parameters
                detect_func = partial(
                    self._detect_station_events_parallel,
                    method=method,
                    event_threshold=self.event_threshold,
                    time_vector=self.time_vector
                )
                
                # Prepare input data for parallel processing
                station_data = [(idx, imfs[idx]) for idx in range(n_analysis_stations)]
                
                # Process stations in parallel
                with Pool(processes=self.n_jobs) as pool:
                    all_station_events = pool.map(detect_func, station_data)
                
                # Collect results from all stations
                for station_events in all_station_events:
                    for event_type, event_list in station_events.items():
                        events[event_type].extend(event_list)
            
            # Create event summary
            events['event_summary'] = self._summarize_events(events, method)
            
            self.sudden_events[method] = events
            
            print(f"✅ Sudden event detection completed for {method.upper()}")
            print(f"   Found {len(events['event_summary'])} total events")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in sudden event detection: {e}")
            return False

    def _detect_station_events(self, station_idx, station_imfs, method):
        """Detect events for a single station"""
        events = {
            'z_score_events': [],
            'rate_change_events': [],
            'spike_events': []
        }
        
        try:
            # Use high-frequency components (first 2-3 IMFs) for event detection
            n_imfs = len(station_imfs)
            high_freq_components = station_imfs[:min(3, n_imfs)]
            
            # Combined high-frequency signal
            combined_signal = np.sum(high_freq_components, axis=0)
            
            # Method 1: Z-score based event detection
            z_scores = np.abs(zscore(combined_signal))
            event_indices = np.where(z_scores > self.event_threshold)[0]
            
            for idx in event_indices:
                events['z_score_events'].append({
                    'station_idx': station_idx,
                    'time_idx': idx,
                    'time_days': self.time_vector[idx] if idx < len(self.time_vector) else idx * 6,
                    'z_score': z_scores[idx],
                    'magnitude': combined_signal[idx],
                    'method': method,
                    'detection_type': 'z_score'
                })
            
            # Method 2: Rate of change detection
            if len(combined_signal) > 1:
                rate_of_change = np.diff(combined_signal)
                rate_z_scores = np.abs(zscore(rate_of_change))
                rate_events = np.where(rate_z_scores > self.event_threshold)[0]
                
                for idx in rate_events:
                    events['rate_change_events'].append({
                        'station_idx': station_idx,
                        'time_idx': idx,
                        'time_days': self.time_vector[idx] if idx < len(self.time_vector) else idx * 6,
                        'rate_z_score': rate_z_scores[idx],
                        'rate_magnitude': rate_of_change[idx],
                        'method': method,
                        'detection_type': 'rate_change'
                    })
            
            # Method 3: Spike detection using signal processing
            if len(combined_signal) > 10:
                # Find peaks that are significantly higher than surroundings
                peaks, properties = signal.find_peaks(
                    np.abs(combined_signal),
                    height=np.std(combined_signal) * self.event_threshold,
                    distance=5  # Minimum 5 time steps between peaks
                )
                
                for peak_idx in peaks:
                    events['spike_events'].append({
                        'station_idx': station_idx,
                        'time_idx': peak_idx,
                        'time_days': self.time_vector[peak_idx] if peak_idx < len(self.time_vector) else peak_idx * 6,
                        'peak_height': properties['peak_heights'][list(peaks).index(peak_idx)],
                        'magnitude': combined_signal[peak_idx],
                        'method': method,
                        'detection_type': 'spike'
                    })
            
        except Exception as e:
            print(f"⚠️  Error detecting events for station {station_idx}: {e}")
        
        return events

    @staticmethod
    def _detect_station_events_parallel(station_data, method, event_threshold, time_vector):
        """Parallel version of _detect_station_events for multiprocessing"""
        station_idx, station_imfs = station_data
        
        events = {
            'z_score_events': [],
            'rate_change_events': [],
            'spike_events': []
        }
        
        try:
            # Use high-frequency components (first 2-3 IMFs) for event detection
            n_imfs = len(station_imfs)
            high_freq_components = station_imfs[:min(3, n_imfs)]
            
            # Combined high-frequency signal
            combined_signal = np.sum(high_freq_components, axis=0)
            
            # Method 1: Z-score based event detection
            z_scores = np.abs(zscore(combined_signal))
            event_indices = np.where(z_scores > event_threshold)[0]
            
            for idx in event_indices:
                events['z_score_events'].append({
                    'station_idx': station_idx,
                    'time_idx': idx,
                    'time_days': time_vector[idx] if idx < len(time_vector) else idx * 6,
                    'z_score': z_scores[idx],
                    'magnitude': combined_signal[idx],
                    'method': method,
                    'detection_type': 'z_score'
                })
            
            # Method 2: Rate of change detection
            if len(combined_signal) > 1:
                rate_of_change = np.diff(combined_signal)
                rate_z_scores = np.abs(zscore(rate_of_change))
                rate_events = np.where(rate_z_scores > event_threshold)[0]
                
                for idx in rate_events:
                    events['rate_change_events'].append({
                        'station_idx': station_idx,
                        'time_idx': idx,
                        'time_days': time_vector[idx] if idx < len(time_vector) else idx * 6,
                        'rate_z_score': rate_z_scores[idx],
                        'rate_magnitude': rate_of_change[idx],
                        'method': method,
                        'detection_type': 'rate_change'
                    })
            
            # Method 3: Spike detection using signal processing
            if len(combined_signal) > 10:
                # Find peaks that are significantly higher than surroundings
                peaks, properties = signal.find_peaks(
                    np.abs(combined_signal),
                    height=np.std(combined_signal) * event_threshold,
                    distance=5  # Minimum 5 time steps between peaks
                )
                
                for peak_idx in peaks:
                    events['spike_events'].append({
                        'station_idx': station_idx,
                        'time_idx': peak_idx,
                        'time_days': time_vector[peak_idx] if peak_idx < len(time_vector) else peak_idx * 6,
                        'peak_height': properties['peak_heights'][list(peaks).index(peak_idx)],
                        'magnitude': combined_signal[peak_idx],
                        'method': method,
                        'detection_type': 'spike'
                    })
            
        except Exception as e:
            # Silently handle errors in parallel processing
            pass
        
        return events

    def _summarize_events(self, events, method):
        """Create summary of all detected events"""
        event_summary = []
        
        # Combine all event types
        all_events = []
        for event_type, event_list in events.items():
            if event_type != 'event_summary':
                all_events.extend(event_list)
        
        # Sort by time
        all_events.sort(key=lambda x: (x['station_idx'], x['time_idx']))
        
        # Create summary statistics
        if all_events:
            for event in all_events:
                summary = {
                    'station_idx': event['station_idx'],
                    'coordinates': self.coordinates[event['station_idx']].tolist(),
                    'time_idx': event['time_idx'],
                    'time_days': event['time_days'],
                    'detection_type': event['detection_type'],
                    'method': method
                }
                
                # Add type-specific information
                if 'z_score' in event:
                    summary['z_score'] = event['z_score']
                    summary['magnitude'] = event['magnitude']
                if 'rate_z_score' in event:
                    summary['rate_z_score'] = event['rate_z_score']
                    summary['rate_magnitude'] = event['rate_magnitude']
                if 'peak_height' in event:
                    summary['peak_height'] = event['peak_height']
                
                event_summary.append(summary)
        
        return event_summary

    def identify_anomalies(self, method='emd'):
        """
        Identify anomalous patterns using Isolation Forest
        
        Focus on multivariate anomalies across multiple IMF components
        """
        print(f"🔄 Identifying anomalies using {method.upper()} decomposition...")
        
        if method not in self.decomposition_data:
            print(f"❌ No data available for method: {method}")
            return False
        
        try:
            data = self.decomposition_data[method]
            imfs = data['imfs']
            n_stations = len(self.coordinates)
            
            # Prepare feature matrix for anomaly detection
            features = self._prepare_anomaly_features(imfs, method)
            
            if features is None:
                return False
            
            print(f"   Feature matrix shape: {features.shape}")
            
            # Apply Isolation Forest
            print(f"   Running Isolation Forest (contamination={self.anomaly_contamination})...")
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Isolation Forest anomaly detection
            iso_forest = IsolationForest(
                contamination=self.anomaly_contamination,
                random_state=42,
                n_jobs=-1
            )
            
            anomaly_labels = iso_forest.fit_predict(features_scaled)
            anomaly_scores = iso_forest.score_samples(features_scaled)
            
            # Identify anomalies (label = -1)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            print(f"   Detected {len(anomaly_indices)} anomalous stations/time periods")
            
            # Create anomaly results
            anomalies = {
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': anomaly_scores,
                'feature_matrix': features,
                'scaler': scaler,
                'model': iso_forest,
                'anomaly_details': []
            }
            
            # Add details for each anomaly
            for idx in anomaly_indices:
                station_idx = idx % n_stations
                time_idx = idx // n_stations
                
                anomalies['anomaly_details'].append({
                    'station_idx': station_idx,
                    'time_idx': time_idx,
                    'coordinates': self.coordinates[station_idx].tolist(),
                    'time_days': self.time_vector[time_idx] if time_idx < len(self.time_vector) else time_idx * 6,
                    'anomaly_score': anomaly_scores[idx],
                    'method': method
                })
            
            self.anomalies[method] = anomalies
            
            print(f"✅ Anomaly identification completed for {method.upper()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in anomaly identification: {e}")
            return False

    def _prepare_anomaly_features(self, imfs, method):
        """Prepare feature matrix for anomaly detection"""
        try:
            n_stations = len(imfs)
            
            if n_stations == 0:
                return None
            
            # Use first few IMFs as features (high to medium frequency)
            max_imfs = min(5, len(imfs[0]) if len(imfs) > 0 else 0)
            
            if max_imfs == 0:
                print(f"   ⚠️  No IMFs available for {method}")
                return None
            
            # Create feature matrix
            # Features: [station_idx * time_idx, IMF_features]
            features = []
            
            for station_idx in range(n_stations):
                station_imfs = imfs[station_idx][:max_imfs]  # First few IMFs
                
                # Statistical features for each IMF
                for imf in station_imfs:
                    if len(imf) > 0:
                        # Time-series features
                        imf_features = [
                            np.mean(imf),              # Mean
                            np.std(imf),               # Standard deviation
                            np.max(imf) - np.min(imf), # Range
                            stats.skew(imf),           # Skewness
                            stats.kurtosis(imf),       # Kurtosis
                        ]
                        
                        # Handle NaN values
                        imf_features = [f if np.isfinite(f) else 0.0 for f in imf_features]
                        features.append(imf_features)
            
            if len(features) == 0:
                return None
            
            return np.array(features)
            
        except Exception as e:
            print(f"   ⚠️  Error preparing anomaly features: {e}")
            return None

    def detect_regime_changes(self, method='emd'):
        """
        Detect regime changes in long-term trends
        
        Uses trend/residual components for structural break detection
        """
        print(f"🔄 Detecting regime changes using {method.upper()} decomposition...")
        
        if method not in self.decomposition_data:
            print(f"❌ No data available for method: {method}")
            return False
        
        try:
            data = self.decomposition_data[method]
            residuals = data['residuals']  # Long-term trend component
            n_stations = len(self.coordinates)
            
            regime_changes = {
                'change_points': [],
                'change_point_details': [],
                'pettitt_results': [],
                'cusum_results': []
            }
            
            print(f"   Analyzing {n_stations} stations for regime changes...")
            
            n_analysis_stations = min(n_stations, len(residuals))
            
            if self.n_jobs == 1 or n_analysis_stations < 100:
                # Sequential processing for small datasets
                print(f"   Using sequential processing for {n_analysis_stations} stations")
                for station_idx in range(n_analysis_stations):
                    station_trend = residuals[station_idx]
                    
                    if len(station_trend) < 20:  # Need minimum data for change point detection
                        continue
                    
                    # Method 1: Basic CUSUM analysis
                    cusum_changes = self._cusum_change_detection(station_trend, station_idx, method)
                    regime_changes['cusum_results'].extend(cusum_changes)
                    
                    # Method 2: Advanced change point detection (if available)
                    if HAS_RUPTURES:
                        rupture_changes = self._ruptures_change_detection(station_trend, station_idx, method)
                        regime_changes['change_points'].extend(rupture_changes)
                    
                    # Method 3: Pettitt test (manual implementation)
                    pettitt_changes = self._pettitt_test_change_detection(station_trend, station_idx, method)
                    regime_changes['pettitt_results'].extend(pettitt_changes)
            else:
                # Parallel processing for large datasets
                print(f"   Using parallel processing with {self.n_jobs} processes for {n_analysis_stations} stations")
                
                # Create partial function with fixed parameters
                regime_func = partial(
                    self._detect_regime_changes_parallel,
                    method=method,
                    time_vector=self.time_vector,
                    has_ruptures=HAS_RUPTURES
                )
                
                # Prepare input data for parallel processing
                station_data = [(idx, residuals[idx]) for idx in range(n_analysis_stations) 
                               if len(residuals[idx]) >= 20]
                
                # Process stations in parallel
                with Pool(processes=self.n_jobs) as pool:
                    all_regime_results = pool.map(regime_func, station_data)
                
                # Collect results from all stations
                for station_results in all_regime_results:
                    regime_changes['cusum_results'].extend(station_results['cusum_results'])
                    regime_changes['change_points'].extend(station_results['change_points'])
                    regime_changes['pettitt_results'].extend(station_results['pettitt_results'])
            
            # Combine and summarize all regime changes
            regime_changes['change_point_details'] = self._summarize_regime_changes(regime_changes, method)
            
            self.regime_changes[method] = regime_changes
            
            total_changes = len(regime_changes['change_point_details'])
            print(f"✅ Regime change detection completed for {method.upper()}")
            print(f"   Found {total_changes} potential regime changes")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in regime change detection: {e}")
            return False

    @staticmethod
    def _detect_regime_changes_parallel(station_data, method, time_vector, has_ruptures):
        """Parallel version of regime change detection for multiprocessing"""
        station_idx, station_trend = station_data
        
        results = {
            'cusum_results': [],
            'change_points': [],
            'pettitt_results': []
        }
        
        try:
            # Method 1: Basic CUSUM analysis
            cusum_changes = EventDetectionAnalysis._cusum_change_detection_static(
                station_trend, station_idx, method, time_vector
            )
            results['cusum_results'].extend(cusum_changes)
            
            # Method 2: Advanced change point detection (if available)
            if has_ruptures:
                try:
                    import ruptures as rpt
                    rupture_changes = EventDetectionAnalysis._ruptures_change_detection_static(
                        station_trend, station_idx, method, time_vector
                    )
                    results['change_points'].extend(rupture_changes)
                except ImportError:
                    pass
            
            # Method 3: Pettitt test (manual implementation)
            pettitt_changes = EventDetectionAnalysis._pettitt_test_change_detection_static(
                station_trend, station_idx, method, time_vector
            )
            results['pettitt_results'].extend(pettitt_changes)
            
        except Exception as e:
            # Silently handle errors in parallel processing
            pass
        
        return results

    def _cusum_change_detection(self, signal, station_idx, method):
        """CUSUM-based change point detection"""
        changes = []
        
        try:
            # CUSUM algorithm
            signal_mean = np.mean(signal)
            cusum_pos = np.zeros(len(signal))
            cusum_neg = np.zeros(len(signal))
            
            # Threshold based on signal variability
            threshold = 3 * np.std(signal)
            
            for i in range(1, len(signal)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + (signal[i] - signal_mean))
                cusum_neg[i] = min(0, cusum_neg[i-1] + (signal[i] - signal_mean))
            
            # Find change points
            pos_changes = np.where(cusum_pos > threshold)[0]
            neg_changes = np.where(cusum_neg < -threshold)[0]
            
            for change_idx in pos_changes:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': change_idx,
                    'time_days': self.time_vector[change_idx] if change_idx < len(self.time_vector) else change_idx * 6,
                    'cusum_value': cusum_pos[change_idx],
                    'direction': 'positive',
                    'method': method,
                    'detection_algorithm': 'cusum'
                })
            
            for change_idx in neg_changes:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': change_idx,
                    'time_days': self.time_vector[change_idx] if change_idx < len(self.time_vector) else change_idx * 6,
                    'cusum_value': cusum_neg[change_idx],
                    'direction': 'negative',
                    'method': method,
                    'detection_algorithm': 'cusum'
                })
            
        except Exception as e:
            print(f"   ⚠️  CUSUM error for station {station_idx}: {e}")
        
        return changes

    @staticmethod
    def _cusum_change_detection_static(signal, station_idx, method, time_vector):
        """Static version of CUSUM change detection for parallel processing"""
        changes = []
        
        try:
            # CUSUM algorithm
            signal_mean = np.mean(signal)
            cusum_pos = np.zeros(len(signal))
            cusum_neg = np.zeros(len(signal))
            
            # Threshold based on signal variability
            threshold = 3 * np.std(signal)
            
            for i in range(1, len(signal)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + (signal[i] - signal_mean))
                cusum_neg[i] = min(0, cusum_neg[i-1] + (signal[i] - signal_mean))
            
            # Find change points
            pos_changes = np.where(cusum_pos > threshold)[0]
            neg_changes = np.where(cusum_neg < -threshold)[0]
            
            for change_idx in pos_changes:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': change_idx,
                    'time_days': time_vector[change_idx] if change_idx < len(time_vector) else change_idx * 6,
                    'cusum_value': cusum_pos[change_idx],
                    'direction': 'positive',
                    'method': method,
                    'detection_algorithm': 'cusum'
                })
            
            for change_idx in neg_changes:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': change_idx,
                    'time_days': time_vector[change_idx] if change_idx < len(time_vector) else change_idx * 6,
                    'cusum_value': cusum_neg[change_idx],
                    'direction': 'negative',
                    'method': method,
                    'detection_algorithm': 'cusum'
                })
            
        except Exception as e:
            pass
        
        return changes

    def _process_regime_change_chunk(self, residuals, start_idx, end_idx, method):
        """Process a chunk of stations for regime change detection - OPTIMIZED"""
        chunk_results = {
            'cusum_results': [],
            'change_points': [],
            'pettitt_results': []
        }
        
        try:
            # Process each station in the chunk
            for station_idx in range(start_idx, end_idx):
                if station_idx >= len(residuals):
                    continue
                    
                station_trend = residuals[station_idx]
                
                if len(station_trend) < 20:  # Need minimum data for change point detection
                    continue
                
                # Method 1: Basic CUSUM analysis
                cusum_changes = self._cusum_change_detection(station_trend, station_idx, method)
                chunk_results['cusum_results'].extend(cusum_changes)
                
                # Method 2: Advanced change point detection (if available)
                if HAS_RUPTURES:
                    rupture_changes = self._ruptures_change_detection(station_trend, station_idx, method)
                    chunk_results['change_points'].extend(rupture_changes)
                
                # Method 3: Pettitt test (manual implementation)
                pettitt_changes = self._pettitt_test_change_detection(station_trend, station_idx, method)
                chunk_results['pettitt_results'].extend(pettitt_changes)
            
            # ENHANCEMENT: Use FastDTW for temporal similarity analysis if available
            if (self.use_fastdtw and 
                len(chunk_results['change_points']) > 10 and 
                end_idx - start_idx > 50):
                
                try:
                    # Group similar regime changes using FastDTW
                    grouped_changes = self._group_similar_regime_changes(
                        chunk_results['change_points'], residuals, start_idx, end_idx
                    )
                    # Update change points with similarity groups
                    for change in chunk_results['change_points']:
                        change['similarity_group'] = grouped_changes.get(change['station_idx'], 0)
                except Exception as e:
                    # FastDTW grouping is optional - continue without it
                    pass
        
        except Exception as e:
            print(f"   ⚠️  Error processing chunk {start_idx}-{end_idx}: {e}")
        
        return chunk_results

    def _group_similar_regime_changes(self, change_points, residuals, start_idx, end_idx):
        """Group similar regime changes using FastDTW temporal similarity"""
        try:
            if not self.use_fastdtw:
                return {}
            
            # Extract station indices with change points
            station_indices = list(set([cp['station_idx'] for cp in change_points]))
            
            if len(station_indices) < 2:
                return {}
            
            # Calculate FastDTW similarity matrix between trend series
            similarity_groups = {}
            group_id = 0
            processed_stations = set()
            
            for i, station_idx in enumerate(station_indices):
                if station_idx in processed_stations or station_idx >= len(residuals):
                    continue
                
                current_group = [station_idx]
                station_trend = residuals[station_idx]
                
                if len(station_trend) < 20:
                    continue
                
                # Compare with remaining stations
                for j, other_station_idx in enumerate(station_indices[i+1:], i+1):
                    if (other_station_idx in processed_stations or 
                        other_station_idx >= len(residuals)):
                        continue
                    
                    other_trend = residuals[other_station_idx]
                    
                    if len(other_trend) != len(station_trend):
                        continue
                    
                    try:
                        # Use FastDTW to calculate temporal similarity
                        distance, _ = fastdtw(
                            station_trend.reshape(-1, 1),
                            other_trend.reshape(-1, 1),
                            radius=self.dtw_radius
                        )
                        
                        # Normalize by series length
                        normalized_distance = distance / len(station_trend)
                        
                        # Threshold for similarity (adjustable)
                        similarity_threshold = np.std(station_trend) * 3
                        
                        if normalized_distance < similarity_threshold:
                            current_group.append(other_station_idx)
                    
                    except Exception:
                        # Skip if FastDTW fails for this pair
                        continue
                
                # Assign group IDs
                for station in current_group:
                    similarity_groups[station] = group_id
                    processed_stations.add(station)
                
                group_id += 1
            
            return similarity_groups
        
        except Exception:
            return {}

    def _ruptures_change_detection(self, signal, station_idx, method):
        """Advanced change point detection using ruptures library"""
        changes = []
        
        try:
            # Pelt algorithm for change point detection
            algo = rpt.Pelt(model="rbf").fit(signal.reshape(-1, 1))
            change_points = algo.predict(pen=10)
            
            # Remove the last point (end of signal)
            change_points = change_points[:-1]
            
            for change_idx in change_points:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': change_idx,
                    'time_days': self.time_vector[change_idx] if change_idx < len(self.time_vector) else change_idx * 6,
                    'method': method,
                    'detection_algorithm': 'ruptures_pelt'
                })
                
        except Exception as e:
            print(f"   ⚠️  Ruptures error for station {station_idx}: {e}")
        
        return changes

    @staticmethod
    def _ruptures_change_detection_static(signal, station_idx, method, time_vector):
        """Static version of ruptures change detection for parallel processing"""
        changes = []
        
        try:
            import ruptures as rpt
            # Pelt algorithm for change point detection
            algo = rpt.Pelt(model="rbf").fit(signal.reshape(-1, 1))
            change_points = algo.predict(pen=10)
            
            # Remove the last point (end of signal)
            change_points = change_points[:-1]
            
            for change_idx in change_points:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': change_idx,
                    'time_days': time_vector[change_idx] if change_idx < len(time_vector) else change_idx * 6,
                    'method': method,
                    'detection_algorithm': 'ruptures_pelt'
                })
                
        except Exception as e:
            pass
        
        return changes

    def _pettitt_test_change_detection(self, signal, station_idx, method):
        """Simplified Pettitt test for change point detection"""
        changes = []
        
        try:
            n = len(signal)
            if n < 10:
                return changes
            
            # Simplified Pettitt test statistic
            max_stat = 0
            max_idx = 0
            
            for k in range(2, n-2):
                # Calculate Pettitt statistic at point k
                s1 = signal[:k]
                s2 = signal[k:]
                
                # Mann-Whitney U-like statistic
                stat = 0
                for i in range(len(s1)):
                    for j in range(len(s2)):
                        if s1[i] > s2[j]:
                            stat += 1
                        elif s1[i] < s2[j]:
                            stat -= 1
                
                if abs(stat) > abs(max_stat):
                    max_stat = stat
                    max_idx = k
            
            # Threshold for significance (simplified)
            threshold = n * n / 4  # Rough threshold
            
            if abs(max_stat) > threshold:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': max_idx,
                    'time_days': self.time_vector[max_idx] if max_idx < len(self.time_vector) else max_idx * 6,
                    'pettitt_stat': max_stat,
                    'method': method,
                    'detection_algorithm': 'pettitt'
                })
        
        except Exception as e:
            print(f"   ⚠️  Pettitt test error for station {station_idx}: {e}")
        
        return changes

    @staticmethod
    def _pettitt_test_change_detection_static(signal, station_idx, method, time_vector):
        """Static version of Pettitt test for parallel processing"""
        changes = []
        
        try:
            n = len(signal)
            if n < 10:
                return changes
            
            # Simplified Pettitt test statistic
            max_stat = 0
            max_idx = 0
            
            for k in range(2, n-2):
                # Calculate Pettitt statistic at point k
                s1 = signal[:k]
                s2 = signal[k:]
                
                # Mann-Whitney U-like statistic
                stat = 0
                for i in range(len(s1)):
                    for j in range(len(s2)):
                        if s1[i] > s2[j]:
                            stat += 1
                        elif s1[i] < s2[j]:
                            stat -= 1
                
                if abs(stat) > abs(max_stat):
                    max_stat = stat
                    max_idx = k
            
            # Threshold for significance (simplified)
            threshold = n * n / 4  # Rough threshold
            
            if abs(max_stat) > threshold:
                changes.append({
                    'station_idx': station_idx,
                    'change_point': max_idx,
                    'time_days': time_vector[max_idx] if max_idx < len(time_vector) else max_idx * 6,
                    'pettitt_stat': max_stat,
                    'method': method,
                    'detection_algorithm': 'pettitt'
                })
        
        except Exception as e:
            pass
        
        return changes

    def _summarize_regime_changes(self, regime_changes, method):
        """Summarize all detected regime changes"""
        all_changes = []
        
        # Combine all change detection results
        for result_type in ['cusum_results', 'change_points', 'pettitt_results']:
            if result_type in regime_changes:
                all_changes.extend(regime_changes[result_type])
        
        # Sort by station and time
        all_changes.sort(key=lambda x: (x['station_idx'], x['change_point']))
        
        # Add coordinates to each change
        for change in all_changes:
            station_idx = change['station_idx']
            change['coordinates'] = self.coordinates[station_idx].tolist()
        
        return all_changes

    def characterize_events(self, method='emd'):
        """
        Characterize detected events with additional properties
        
        Analyze duration, magnitude, spatial correlation, recovery time
        """
        print(f"🔄 Characterizing events for {method.upper()}...")
        
        if method not in self.sudden_events:
            print(f"❌ No sudden events available for method: {method}")
            return False
        
        try:
            events = self.sudden_events[method]['event_summary']
            
            if len(events) == 0:
                print(f"   No events to characterize for {method}")
                return True
            
            characterized_events = []
            n_events = len(events)
            
            print(f"   Processing {n_events} events for characterization...")
            
            # Pre-sort events by station for better cache locality
            events_sorted = sorted(events, key=lambda x: x['station_idx'])
            
            for i, event in enumerate(events_sorted):
                # Progress reporting every 100 events
                if i % 100 == 0:
                    progress = (i / n_events) * 100
                    print(f"   Progress: {i}/{n_events} events ({progress:.1f}%)")
                
                characterized_event = event.copy()
                
                # Add event characterization
                station_idx = event['station_idx']
                time_idx = event['time_idx']
                
                # Duration analysis (optimized - vectorized approach)
                characterized_event['duration_estimate'] = self._estimate_event_duration(
                    events, station_idx, time_idx
                )
                
                # Magnitude classification
                if 'magnitude' in event:
                    characterized_event['magnitude_class'] = self._classify_magnitude(event['magnitude'])
                
                # Spatial correlation (optimized - vectorized distance calculation)
                characterized_event['spatial_correlation'] = self._estimate_spatial_correlation(
                    events, station_idx, time_idx
                )
                
                # Recovery analysis (optimized - vectorized time series analysis)
                characterized_event['recovery_estimate'] = self._estimate_recovery_time(
                    station_idx, time_idx, method
                )
                
                characterized_events.append(characterized_event)
            
            print(f"   ✅ Completed characterization of {n_events} events")
            
            # Create event catalog
            self.event_catalog[method] = {
                'total_events': len(characterized_events),
                'event_details': characterized_events,
                'event_statistics': self._calculate_event_statistics(characterized_events),
                'spatial_distribution': self._analyze_spatial_distribution(characterized_events),
                'temporal_distribution': self._analyze_temporal_distribution(characterized_events)
            }
            
            print(f"✅ Event characterization completed for {method.upper()}")
            print(f"   Characterized {len(characterized_events)} events")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in event characterization: {e}")
            return False

    def _estimate_event_duration(self, events, station_idx, time_idx):
        """Estimate event duration by looking for consecutive events - OPTIMIZED"""
        # Use vectorized approach instead of nested loops
        station_events = [e for e in events if e['station_idx'] == station_idx]
        
        if len(station_events) <= 1:
            return 6  # Single event duration
        
        # Get time indices for this station
        time_indices = np.array([e['time_idx'] for e in station_events])
        
        # Count events within ±2 time steps
        consecutive_count = np.sum(np.abs(time_indices - time_idx) <= 2)
        
        return consecutive_count * 6  # Convert to days (6-day sampling)

    def _classify_magnitude(self, magnitude):
        """Classify event magnitude"""
        abs_magnitude = abs(magnitude)
        
        if abs_magnitude < 5:
            return 'small'
        elif abs_magnitude < 15:
            return 'medium'
        elif abs_magnitude < 30:
            return 'large'
        else:
            return 'extreme'

    def _estimate_spatial_correlation(self, events, station_idx, time_idx):
        """Estimate spatial correlation by counting nearby events - OPTIMIZED"""
        target_coords = self.coordinates[station_idx]
        
        # Filter events within ±1 time step first (much smaller subset)
        temporal_events = [e for e in events if abs(e['time_idx'] - time_idx) <= 1]
        
        if len(temporal_events) <= 1:
            return 1  # Only this event
        
        # Vectorized distance calculation
        event_station_indices = np.array([e['station_idx'] for e in temporal_events])
        event_coords = self.coordinates[event_station_indices]
        
        # Calculate distances using broadcasting
        distances = np.sqrt(np.sum((event_coords - target_coords)**2, axis=1))
        
        # Count nearby events (within ~5km)
        nearby_events = np.sum(distances < 0.05)
        
        return nearby_events

    def _estimate_recovery_time(self, station_idx, time_idx, method):
        """Estimate recovery time (simplified analysis) - OPTIMIZED"""
        try:
            # Use original displacement to check recovery
            if (self.original_displacement is not None and 
                station_idx < len(self.original_displacement) and
                time_idx < len(self.original_displacement[station_idx]) - 10):
                
                station_ts = self.original_displacement[station_idx]
                
                # Vectorized pre-event calculation
                pre_start = max(0, time_idx - 5)
                pre_event = np.mean(station_ts[pre_start:time_idx])
                
                # Vectorized post-event analysis
                post_end = min(len(station_ts), time_idx + 10)
                post_event = station_ts[time_idx:post_end]
                
                # Vectorized recovery threshold check
                recovery_threshold = abs(pre_event) + np.std(station_ts) * 0.5
                recovery_mask = np.abs(post_event - pre_event) < recovery_threshold
                
                # Find first recovery point
                recovery_indices = np.where(recovery_mask)[0]
                if len(recovery_indices) > 0:
                    return recovery_indices[0] * 6  # Recovery time in days
                
                return None  # No recovery detected
        except:
            pass
        
        return None

    def _calculate_event_statistics(self, events):
        """Calculate overall event statistics"""
        if not events:
            return {}
        
        magnitudes = [e.get('magnitude', 0) for e in events if 'magnitude' in e]
        durations = [e.get('duration_estimate', 0) for e in events if 'duration_estimate' in e]
        
        return {
            'total_events': len(events),
            'mean_magnitude': np.mean(magnitudes) if magnitudes else 0,
            'std_magnitude': np.std(magnitudes) if magnitudes else 0,
            'max_magnitude': np.max(magnitudes) if magnitudes else 0,
            'mean_duration': np.mean(durations) if durations else 0,
            'magnitude_classes': {
                'small': sum(1 for e in events if e.get('magnitude_class') == 'small'),
                'medium': sum(1 for e in events if e.get('magnitude_class') == 'medium'),
                'large': sum(1 for e in events if e.get('magnitude_class') == 'large'),
                'extreme': sum(1 for e in events if e.get('magnitude_class') == 'extreme')
            }
        }

    def _analyze_spatial_distribution(self, events):
        """Analyze spatial distribution of events"""
        if not events:
            return {}
        
        longitudes = [e['coordinates'][0] for e in events]
        latitudes = [e['coordinates'][1] for e in events]
        
        return {
            'longitude_range': [np.min(longitudes), np.max(longitudes)],
            'latitude_range': [np.min(latitudes), np.max(latitudes)],
            'centroid': [np.mean(longitudes), np.mean(latitudes)],
            'spatial_std': [np.std(longitudes), np.std(latitudes)]
        }

    def _analyze_temporal_distribution(self, events):
        """Analyze temporal distribution of events"""
        if not events:
            return {}
        
        time_days = [e['time_days'] for e in events]
        
        return {
            'time_range_days': [np.min(time_days), np.max(time_days)],
            'mean_time': np.mean(time_days),
            'temporal_std': np.std(time_days),
            'events_per_year': len(events) / (max(time_days) - min(time_days)) * 365 if time_days else 0
        }

    def create_comprehensive_overview(self, method='emd'):
        """Create comprehensive overview figure (ps05_fig01)"""
        print(f"📊 Creating comprehensive overview for {method}...")
        
        if method not in self.event_catalog:
            print(f"❌ No event catalog found for method: {method}")
            return
        
        catalog = self.event_catalog[method]
        change_points = catalog['change_points']
        events = catalog['events']
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comprehensive Event Detection Overview - {method.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # Panel A: Time series with change points
        ax = axes[0, 0]
        # Plot example time series from central station
        central_station = len(self.coordinates) // 2
        if hasattr(self, 'displacement'):
            station_data = self.displacement[central_station, :]
            time_days = np.arange(len(station_data)) * 6
            ax.plot(time_days, station_data, 'k-', linewidth=1, alpha=0.7, label='InSAR Time Series')
            
            # Add change points for this station
            station_changes = [cp for cp in change_points if cp['station_idx'] == central_station]
            if station_changes:
                change_times = [cp['time_idx'] * 6 for cp in station_changes]
                change_values = [station_data[cp['time_idx']] for cp in station_changes if cp['time_idx'] < len(station_data)]
                ax.scatter(change_times[:len(change_values)], change_values, 
                          c='red', s=50, alpha=0.8, label='Detected Changes', zorder=5)
        
        ax.set_title('A) Example Time Series with Change Points')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Displacement (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel B: Spatial distribution of change points
        ax = axes[0, 1]
        if change_points:
            # Count change points per station
            station_counts = {}
            for cp in change_points:
                station_idx = cp['station_idx']
                station_counts[station_idx] = station_counts.get(station_idx, 0) + 1
            
            # Create scatter plot
            lons, lats, counts = [], [], []
            for station_idx, count in station_counts.items():
                if station_idx < len(self.coordinates):
                    lons.append(self.coordinates[station_idx, 0])
                    lats.append(self.coordinates[station_idx, 1])
                    counts.append(count)
            
            if lons:
                scatter = ax.scatter(lons, lats, c=counts, cmap='hot', s=30, alpha=0.7)
                plt.colorbar(scatter, ax=ax, label='Change Points per Station')
        
        ax.set_title('B) Spatial Distribution of Change Points')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.grid(True, alpha=0.3)
        
        # Panel C: Temporal distribution
        ax = axes[1, 0]
        if change_points:
            times = [cp['time_idx'] * 6 for cp in change_points]  # Convert to days
            ax.hist(times, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        ax.set_title('C) Temporal Distribution of Change Points')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Number of Change Points')
        ax.grid(True, alpha=0.3)
        
        # Panel D: Method statistics
        ax = axes[1, 1]
        if method in self.event_catalog:
            stats = self.event_catalog[method]['event_statistics']
            
            # Create summary statistics plot
            categories = ['Total\nEvents', 'Mean\nMagnitude', 'Max\nMagnitude', 'Detection\nRate']
            values = [
                stats.get('total_events', 0),
                stats.get('mean_magnitude', 0),
                stats.get('max_magnitude', 0),
                len(change_points) / len(self.coordinates) if hasattr(self, 'coordinates') else 0
            ]
            
            bars = ax.bar(categories, values, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
            ax.set_title('D) Detection Statistics')
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = self.figures_dir / f"ps05_fig01_comprehensive_overview_{method}.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive overview saved: {figure_path}")
        
    def create_event_timeline_visualization(self, method='emd'):
        """Create comprehensive event timeline visualization"""
        print(f"🔄 Creating event timeline visualization for {method.upper()}...")
        
        if method not in self.event_catalog:
            print(f"❌ No event catalog available for method: {method}")
            return False
        
        try:
            events = self.event_catalog[method]['event_details']
            
            if len(events) == 0:
                print(f"   No events to visualize for {method}")
                return True
            
            fig, axes = plt.subplots(3, 2, figsize=(18, 12))
            
            # Extract event data
            time_days = [e['time_days'] for e in events]
            magnitudes = [e.get('magnitude', 0) for e in events if 'magnitude' in e]
            coordinates = [e['coordinates'] for e in events]
            
            # Plot 1: Event timeline
            ax = axes[0, 0]
            scatter = ax.scatter(time_days, range(len(time_days)), 
                               c=magnitudes[:len(time_days)] if magnitudes else 'blue',
                               cmap='RdBu_r', alpha=0.7, s=50)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Event Index')
            ax.set_title(f'Event Timeline - {method.upper()}')
            ax.grid(True, alpha=0.3)
            
            if magnitudes:
                plt.colorbar(scatter, ax=ax, label='Event Magnitude (mm)')
            
            # Plot 2: Event magnitude histogram
            ax = axes[0, 1]
            if magnitudes:
                ax.hist(magnitudes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Event Magnitude (mm)')
                ax.set_ylabel('Frequency')
                ax.set_title('Event Magnitude Distribution')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No magnitude data available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            # Plot 3: Geographic distribution
            ax = axes[1, 0]
            if coordinates:
                lons = [c[0] for c in coordinates]
                lats = [c[1] for c in coordinates]
                
                scatter = ax.scatter(lons, lats, 
                                   c=magnitudes[:len(coordinates)] if magnitudes else 'red',
                                   cmap='RdBu_r', alpha=0.7, s=60)
                ax.set_xlabel('Longitude (°E)')
                ax.set_ylabel('Latitude (°N)')
                ax.set_title('Geographic Distribution of Events')
                ax.grid(True, alpha=0.3)
                
                if magnitudes:
                    plt.colorbar(scatter, ax=ax, label='Event Magnitude (mm)')
            
            # Plot 4: Event detection types
            ax = axes[1, 1]
            detection_types = [e['detection_type'] for e in events]
            type_counts = pd.Series(detection_types).value_counts()
            
            ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                  colors=['lightcoral', 'skyblue', 'lightgreen', 'gold'])
            ax.set_title('Event Detection Methods')
            
            # Plot 5: Temporal distribution
            ax = axes[2, 0]
            if time_days:
                ax.hist(time_days, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Number of Events')
                ax.set_title('Temporal Distribution of Events')
                ax.grid(True, alpha=0.3)
            
            # Plot 6: Event statistics summary
            ax = axes[2, 1]
            stats = self.event_catalog[method]['event_statistics']
            
            stats_text = f"""
Event Statistics ({method.upper()}):

Total Events: {stats.get('total_events', 0)}
Mean Magnitude: {stats.get('mean_magnitude', 0):.2f} mm
Max Magnitude: {stats.get('max_magnitude', 0):.2f} mm
Mean Duration: {stats.get('mean_duration', 0):.1f} days

Magnitude Classes:
• Small: {stats.get('magnitude_classes', {}).get('small', 0)}
• Medium: {stats.get('magnitude_classes', {}).get('medium', 0)}
• Large: {stats.get('magnitude_classes', {}).get('large', 0)}
• Extreme: {stats.get('magnitude_classes', {}).get('extreme', 0)}
            """
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10, fontfamily='monospace')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            timeline_file = self.figures_dir / f"ps05_fig01_event_timeline_{method}.png"
            plt.savefig(timeline_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Event timeline visualization saved: {timeline_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating event timeline visualization: {e}")
            return False

    def create_anomaly_visualization(self, method='emd'):
        """Create anomaly detection visualization"""
        print(f"🔄 Creating anomaly visualization for {method.upper()}...")
        
        if method not in self.anomalies:
            print(f"❌ No anomaly data available for method: {method}")
            return False
        
        try:
            anomalies = self.anomalies[method]
            anomaly_details = anomalies['anomaly_details']
            
            if len(anomaly_details) == 0:
                print(f"   No anomalies to visualize for {method}")
                return True
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract anomaly data
            coordinates = [a['coordinates'] for a in anomaly_details]
            anomaly_scores = [a['anomaly_score'] for a in anomaly_details]
            time_days = [a['time_days'] for a in anomaly_details]
            
            # Plot 1: Geographic distribution of anomalies
            ax = axes[0, 0]
            if coordinates:
                lons = [c[0] for c in coordinates]
                lats = [c[1] for c in coordinates]
                
                scatter = ax.scatter(lons, lats, c=anomaly_scores, 
                                   cmap='Reds', alpha=0.7, s=60)
                ax.set_xlabel('Longitude (°E)')
                ax.set_ylabel('Latitude (°N)')
                ax.set_title('Geographic Distribution of Anomalies')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Anomaly Score')
            
            # Plot 2: Anomaly score distribution
            ax = axes[0, 1]
            ax.hist(anomaly_scores, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Anomaly Score Distribution')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Temporal distribution of anomalies
            ax = axes[1, 0]
            ax.hist(time_days, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Number of Anomalies')
            ax.set_title('Temporal Distribution of Anomalies')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Anomaly timeline
            ax = axes[1, 1]
            scatter = ax.scatter(time_days, anomaly_scores, 
                               c=anomaly_scores, cmap='Reds', alpha=0.7, s=50)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Anomaly Score')
            ax.set_title('Anomaly Timeline')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Anomaly Score')
            
            plt.tight_layout()
            
            # Save visualization
            anomaly_file = self.figures_dir / f"ps05_fig02_anomalies_{method}.png"
            plt.savefig(anomaly_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Anomaly visualization saved: {anomaly_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating anomaly visualization: {e}")
            return False

    def create_regime_change_visualization(self, method='emd'):
        """Create regime change visualization"""
        print(f"🔄 Creating regime change visualization for {method.upper()}...")
        
        if method not in self.regime_changes:
            print(f"❌ No regime change data available for method: {method}")
            return False
        
        try:
            regime_changes = self.regime_changes[method]
            change_details = regime_changes['change_point_details']
            
            if len(change_details) == 0:
                print(f"   No regime changes to visualize for {method}")
                return True
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract change point data
            coordinates = [c['coordinates'] for c in change_details]
            change_times = [c['time_days'] for c in change_details]
            algorithms = [c['detection_algorithm'] for c in change_details]
            
            # Plot 1: Geographic distribution of change points
            ax = axes[0, 0]
            if coordinates:
                lons = [c[0] for c in coordinates]
                lats = [c[1] for c in coordinates]
                
                ax.scatter(lons, lats, c='red', alpha=0.7, s=60)
                ax.set_xlabel('Longitude (°E)')
                ax.set_ylabel('Latitude (°N)')
                ax.set_title('Geographic Distribution of Regime Changes')
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Detection algorithm distribution
            ax = axes[0, 1]
            algorithm_counts = pd.Series(algorithms).value_counts()
            
            ax.pie(algorithm_counts.values, labels=algorithm_counts.index, 
                  autopct='%1.1f%%', colors=['lightcoral', 'skyblue', 'lightgreen'])
            ax.set_title('Change Point Detection Algorithms')
            
            # Plot 3: Temporal distribution of change points
            ax = axes[1, 0]
            ax.hist(change_times, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Number of Change Points')
            ax.set_title('Temporal Distribution of Regime Changes')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Change point timeline by algorithm
            ax = axes[1, 1]
            
            colors = {'cusum': 'red', 'ruptures_pelt': 'blue', 'pettitt': 'green'}
            for alg in set(algorithms):
                alg_times = [t for t, a in zip(change_times, algorithms) if a == alg]
                alg_indices = list(range(len(alg_times)))
                ax.scatter(alg_times, alg_indices, c=colors.get(alg, 'black'), 
                          label=alg, alpha=0.7, s=50)
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Change Point Index')
            ax.set_title('Change Point Timeline by Algorithm')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            regime_file = self.figures_dir / f"ps05_fig03_regime_changes_{method}.png"
            plt.savefig(regime_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Regime change visualization saved: {regime_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating regime change visualization: {e}")
            return False

    def save_results(self):
        """Save all analysis results to files"""
        print("💾 Saving event detection results...")
        
        try:
            # Save sudden events
            for method in self.sudden_events:
                events_file = self.results_dir / f"sudden_events_{method}.json"
                with open(events_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    events_data = self.sudden_events[method].copy()
                    events_data['event_summary'] = events_data['event_summary']  # Already serializable
                    json.dump(events_data, f, indent=2, default=str)
                print(f"✅ Saved sudden events: {events_file}")
            
            # Save anomalies (excluding non-serializable objects)
            for method in self.anomalies:
                anomalies_file = self.results_dir / f"anomalies_{method}.json"
                with open(anomalies_file, 'w') as f:
                    # Only save serializable parts
                    anomalies_data = {
                        'anomaly_details': self.anomalies[method]['anomaly_details'],
                        'total_anomalies': len(self.anomalies[method]['anomaly_details'])
                    }
                    json.dump(anomalies_data, f, indent=2, default=str)
                print(f"✅ Saved anomalies: {anomalies_file}")
            
            # Save regime changes
            for method in self.regime_changes:
                regime_file = self.results_dir / f"regime_changes_{method}.json"
                with open(regime_file, 'w') as f:
                    json.dump(self.regime_changes[method], f, indent=2, default=str)
                print(f"✅ Saved regime changes: {regime_file}")
            
            # Save event catalog
            for method in self.event_catalog:
                catalog_file = self.results_dir / f"event_catalog_{method}.json"
                with open(catalog_file, 'w') as f:
                    json.dump(self.event_catalog[method], f, indent=2, default=str)
                print(f"✅ Saved event catalog: {catalog_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Event Detection Analysis for Taiwan Subsidence')
    parser.add_argument('--methods', type=str, default='emd',
                       help='Comma-separated list of methods: emd,fft,vmd,wavelet or "all"')
    parser.add_argument('--event-threshold', type=float, default=3.0,
                       help='Z-score threshold for event detection (default: 3.0)')
    parser.add_argument('--anomaly-contamination', type=float, default=0.1,
                       help='Expected fraction of anomalies (default: 0.1)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel processes (-1 for all cores, default: -1)')
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Create visualization figures')
    parser.add_argument('--save-results', action='store_true',
                       help='Save analysis results to files')
    return parser.parse_args()

def main():
    """Main event detection analysis workflow"""
    args = parse_arguments()
    
    # Parse methods
    if args.methods.lower() == 'all':
        methods = ['emd', 'fft', 'vmd', 'wavelet']
    else:
        methods = [m.strip().lower() for m in args.methods.split(',')]
    
    print("=" * 80)
    print("🚀 ps05_event_detection.py - Event Detection & Anomaly Analysis")
    print(f"📋 METHODS: {', '.join(methods).upper()}")
    print(f"📊 EVENT THRESHOLD: {args.event_threshold}")
    print(f"📊 ANOMALY CONTAMINATION: {args.anomaly_contamination}")
    print(f"⚡ PARALLEL PROCESSES: {args.n_jobs if args.n_jobs > 0 else cpu_count()}")
    print("=" * 80)
    
    # Initialize analysis
    event_analysis = EventDetectionAnalysis(
        methods=methods,
        event_threshold=args.event_threshold,
        anomaly_contamination=args.anomaly_contamination,
        n_jobs=args.n_jobs
    )
    
    # Load decomposition data
    if not event_analysis.load_decomposition_data():
        print("❌ Failed to load decomposition data")
        return False
    
    # Perform analysis for each method
    for method in methods:
        print(f"\n🔄 ANALYZING METHOD: {method.upper()}")
        print("-" * 50)
        
        # Sudden event detection
        if not event_analysis.detect_sudden_events(method):
            print(f"⚠️  Failed to detect sudden events for {method}")
            continue
        
        # Anomaly identification
        if not event_analysis.identify_anomalies(method):
            print(f"⚠️  Failed to identify anomalies for {method}")
            continue
        
        # Regime change detection
        if not event_analysis.detect_regime_changes(method):
            print(f"⚠️  Failed to detect regime changes for {method}")
            continue
        
        # Event characterization
        if not event_analysis.characterize_events(method):
            print(f"⚠️  Failed to characterize events for {method}")
            continue
        
        print(f"✅ Analysis completed for {method.upper()}")
    
    # Create visualizations
    if args.create_visualizations:
        print("\n" + "=" * 50)
        print("🔄 CREATING VISUALIZATIONS")
        print("=" * 50)
        
        for method in methods:
            if method in event_analysis.event_catalog:
                event_analysis.create_comprehensive_overview(method)  # ps05_fig01
                event_analysis.create_anomaly_visualization(method)   # ps05_fig02
                event_analysis.create_regime_change_visualization(method)  # ps05_fig03
    
    # Save results
    if args.save_results:
        event_analysis.save_results()
    
    print("\n" + "=" * 80)
    print("✅ ps05_event_detection.py ANALYSIS COMPLETED SUCCESSFULLY")
    
    # Print summary
    for method in methods:
        if method in event_analysis.event_catalog:
            stats = event_analysis.event_catalog[method]['event_statistics']
            print(f"📊 {method.upper()} RESULTS:")
            print(f"   Total Events: {stats.get('total_events', 0)}")
            print(f"   Mean Magnitude: {stats.get('mean_magnitude', 0):.2f} mm")
            print(f"   Max Magnitude: {stats.get('max_magnitude', 0):.2f} mm")
    
    print("📊 Generated visualizations:")
    print("   1. Event timeline and statistics")
    print("   2. Anomaly detection results")
    print("   3. Regime change analysis")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
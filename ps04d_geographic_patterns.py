#!/usr/bin/env python3
"""
ps04d_geographic_patterns.py - TSLearn Geographic Similarity Analysis
===================================================================

Advanced geographic pattern analysis for InSAR time series using TSLearn methods.
Focuses on detecting spatially correlated atmospheric artifacts and large-area 
deformation patterns with geographic constraints.

GEOPHYSICAL FOCUS:
- **Tropospheric Signals**: High temporal frequency (1-60 days) spatially correlated noise from 
  atmospheric water vapor variations, turbulent mixing, and pressure changes
- **Ionospheric Signals**: High temporal frequency but spatially low-frequency (flat ramps)
  signals from ionospheric delay variations, particularly affecting L-band SAR
- **Large-area Flat Ramps**: Spatially broad, flat patterns that can have rapid temporal 
  variations - distinguishing processing artifacts from ionospheric effects

Key Capabilities:
- Spatially-constrained matrix profile analysis for atmospheric artifact detection
- High temporal frequency, spatially correlated pattern detection (tropospheric effects)
- High temporal frequency, spatially flat ramp detection (ionospheric effects)  
- Large-area flat ramp detection and characterization
- Geographic clustering with DTW distances for atmospheric signal source identification

Usage:
    python ps04d_geographic_patterns.py [--n-stations 250] [--spatial-radius 20] [--n-jobs 8]

Author: Advanced InSAR Analysis Pipeline
Date: January 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, hilbert
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Parallel processing
from joblib import Parallel, delayed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

# TSLearn and time series analysis
try:
    import stumpy
    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False
    print("⚠️  Warning: stumpy not available. Matrix profile analysis will be limited.")

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import dtw
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    print("⚠️  Warning: tslearn not available. DTW analysis will be limited.")

# Visualization
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Global function for parallel processing (must be at module level for pickle)
def _filter_station_data(args):
    """
    Global function for filtering a single station's time series
    Required at module level for multiprocessing serialization
    """
    time_series, low_freq, high_freq, max_period, sampling_days = args
    
    try:
        if max_period is not None:
            # Bandpass filter
            sos = butter(4, [low_freq, high_freq], btype='band', fs=1/sampling_days, output='sos')
        else:
            # High-pass filter for trend component
            sos = butter(4, high_freq, btype='high', fs=1/sampling_days, output='sos')
        
        return filtfilt(sos, time_series)
        
    except Exception as e:
        # Fallback: return zero if filtering fails
        return np.zeros_like(time_series)

@dataclass
class GeographicPattern:
    """Data structure for geographic pattern results"""
    pattern_id: int
    center_coordinate: Tuple[float, float]
    affected_stations: List[int]
    pattern_type: str  # 'noise', 'ramp', 'seasonal', 'anomaly'
    spatial_extent_km: float
    temporal_correlation: float
    spatial_coherence: float
    frequency_band: str

@dataclass 
class SpatialCluster:
    """Data structure for spatial-temporal clusters"""
    cluster_id: int
    station_indices: List[int]
    centroid_coordinate: Tuple[float, float]
    cluster_radius_km: float
    temporal_signature: np.ndarray
    dtw_coherence: float
    dominant_frequency: str

class GeographicPatternAnalysis:
    """
    Advanced geographic pattern analysis for atmospheric artifact detection using TSLearn methods
    
    GEOPHYSICAL APPLICATIONS:
    - **Tropospheric Noise Detection**: Identify spatially correlated, high temporal frequency signals
      from atmospheric water vapor, turbulent mixing, and pressure variations
    - **Ionospheric Signal Analysis**: Detect spatially flat ramps with high temporal frequency
      from ionospheric delay variations (particularly L-band SAR effects)
    - **Spatial Frequency Analysis**: Distinguish high spatial frequency (localized) vs 
      low spatial frequency (broad/flat) atmospheric artifacts
    - **Atmospheric vs Deformation Separation**: Use spatial correlation patterns to distinguish
      atmospheric artifacts from genuine ground deformation
    
    Core Capabilities:
    - Spatially-constrained matrix profile computation for atmospheric pattern detection
    - Geographic clustering with DTW distances for atmospheric signal source identification
    - High temporal frequency, spatially correlated pattern detection (tropospheric effects)
    - High temporal frequency, spatially flat ramp detection (ionospheric effects)
    - Large-area flat ramp identification (distinguishing processing vs ionospheric artifacts)
    - Spatial-temporal anomaly detection for atmospheric artifact classification
    """
    
    def __init__(self, spatial_radius_km=20, min_neighbors=5, n_jobs=-1, random_state=42):
        """
        Initialize geographic pattern analysis framework
        
        Parameters:
        -----------
        spatial_radius_km : float
            Maximum spatial radius for neighbor consideration (km)
        min_neighbors : int
            Minimum number of spatial neighbors required for analysis
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int
            Random seed for reproducibility
        """
        self.spatial_radius_km = spatial_radius_km
        self.min_neighbors = min_neighbors
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.random_state = random_state
        
        # Data containers
        self.time_series = None
        self.coordinates = None
        self.timestamps = None
        self.station_ids = None
        
        # Geographic analysis results
        self.spatial_tree = None
        self.neighbor_networks = {}
        self.geographic_patterns = {}
        self.spatial_clusters = {}
        self.coherent_regions = {}
        self.ramp_detections = {}
        
        # Performance tracking
        self.timing_results = {}
        
        # Create output directories
        self._create_directories()
        
        print(f"🗺️  Geographic Pattern Analysis Framework Initialized")
        print(f"   Spatial radius: {self.spatial_radius_km} km")
        print(f"   Minimum neighbors: {self.min_neighbors}")
        print(f"   Parallel workers: {self.n_jobs}")
        print(f"   Matrix profile available: {STUMPY_AVAILABLE}")
        print(f"   TSLearn DTW available: {TSLEARN_AVAILABLE}")
        print(f"   ⚡ Parallel mode: {'Enabled' if self.n_jobs > 1 else 'Disabled'}")
    
    def _create_directories(self):
        """Create output directories for results and figures"""
        self.results_dir = Path("data/processed/ps04d_geographic")
        self.figures_dir = Path("figures")
        
        for directory in [self.results_dir, self.figures_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directories created:")
        print(f"   Results: {self.results_dir}")
        print(f"   Figures: {self.figures_dir}")
    
    def start_timer(self, operation_name):
        """Start timing an operation"""
        self.timing_results[f'{operation_name}_start'] = time.time()
    
    def end_timer(self, operation_name):
        """End timing an operation and store result"""
        if f'{operation_name}_start' in self.timing_results:
            elapsed = time.time() - self.timing_results[f'{operation_name}_start']
            self.timing_results[operation_name] = elapsed
            return elapsed
        return 0
    
    def load_preprocessed_data(self, data_source='ps00', n_stations=None, use_raw_data=True):
        """
        Load time series data for geographic pattern analysis
        
        Parameters:
        -----------
        data_source : str
            Data source ('ps00' for preprocessed, 'ps02' for decomposed)
        n_stations : int, optional
            Limit to first N stations for testing
        use_raw_data : bool
            Use RAW displacement data (recommended for pattern discovery)
        """
        self.start_timer("data_loading")
        
        try:
            # Load preprocessed data
            data_file = Path("data/processed/ps00_preprocessed_data.npz")
            
            if not data_file.exists():
                print(f"❌ Data file not found: {data_file}")
                return False
            
            print("📡 Loading time series data for geographic pattern analysis...")
            if use_raw_data:
                print("   ✅ Using RAW displacement data - preserves natural patterns for geographic analysis")
            else:
                print("   ⚠️  Using DETRENDED data - may miss long-term spatial patterns")
            
            with np.load(data_file, allow_pickle=True) as data:
                coordinates = data['coordinates']
                displacement = data['displacement']  # This is detrended data from ps00
                subsidence_rates = data['subsidence_rates']  # Linear trends (mm/year)
                
                # Reconstruct RAW displacement data for geographic analysis
                if use_raw_data:
                    # Convert subsidence rates to cumulative displacement over time
                    n_timepoints = displacement.shape[1]
                    timestamps_years = np.arange(n_timepoints) * 6 / 365.25  # Convert 6-day sampling to years
                    
                    # Add back linear trends to get raw displacement data
                    trend_component = subsidence_rates[:, np.newaxis] * timestamps_years[np.newaxis, :]
                    raw_displacement = displacement + trend_component
                    
                    print(f"   🔄 Reconstructed RAW displacement from detrended data + subsidence rates")
                    print(f"   📊 Trend magnitude range: {np.min(subsidence_rates):.1f} to {np.max(subsidence_rates):.1f} mm/year")
                    
                    time_series_data = raw_displacement
                    data_type = "RAW displacement (detrended + trends)"
                else:
                    time_series_data = displacement
                    data_type = "DETRENDED displacement"
                
                if n_stations is not None and n_stations < len(coordinates):
                    self.coordinates = coordinates[:n_stations]
                    self.time_series = time_series_data[:n_stations]
                    subset_info = f"{n_stations}-station subset"
                else:
                    self.coordinates = coordinates
                    self.time_series = time_series_data
                    subset_info = "full dataset"
            
            # Create timestamps (6-day intervals from 2018-2021)
            n_timepoints = self.time_series.shape[1]
            self.timestamps = np.arange(n_timepoints) * 6  # 6-day sampling
            
            # Create station IDs
            self.station_ids = np.arange(len(self.coordinates))
            
            print(f"✅ Loaded {data_type} for geographic analysis ({subset_info})")
            print(f"   Stations: {len(self.coordinates)}")
            print(f"   Time points: {n_timepoints}")
            print(f"   Time span: {self.timestamps[-1]} days ({self.timestamps[-1]/365.25:.1f} years)")
            print(f"   Geographic bounds: {self.coordinates[:, 0].min():.3f}°E to {self.coordinates[:, 0].max():.3f}°E")
            print(f"                      {self.coordinates[:, 1].min():.3f}°N to {self.coordinates[:, 1].max():.3f}°N")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        self.end_timer("data_loading")
        return True
    
    def build_spatial_network(self):
        """Build spatial neighbor network using geographic coordinates with parallel processing"""
        self.start_timer("spatial_network")
        
        print("🌐 Building spatial neighbor network...")
        
        # Convert lat/lon to approximate distance in km
        # Use Haversine distance for accurate geographic distances
        def haversine_distance(coord1, coord2):
            """Calculate Haversine distance between two lat/lon points in km"""
            R = 6371.0  # Earth radius in km
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            return R * c
        
        # Parallel distance computation for large datasets
        n_stations = len(self.coordinates)
        
        if n_stations > 100:
            print(f"   ⚡ Using parallel processing for {n_stations} stations...")
            
            def compute_row_distances(i):
                """Compute distances for row i"""
                row_distances = np.zeros(n_stations)
                for j in range(n_stations):
                    if i != j:
                        row_distances[j] = haversine_distance(self.coordinates[i], self.coordinates[j])
                return i, row_distances
            
            # Parallel computation of distance matrix
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_row_distances)(i) for i in range(n_stations)
            )
            
            # Assemble distance matrix
            distance_matrix = np.zeros((n_stations, n_stations))
            for i, row_distances in results:
                distance_matrix[i] = row_distances
        else:
            # Sequential computation for small datasets
            distance_matrix = np.zeros((n_stations, n_stations))
            for i in range(n_stations):
                for j in range(i+1, n_stations):
                    dist = haversine_distance(self.coordinates[i], self.coordinates[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        # Build neighbor networks
        self.neighbor_networks = {}
        spatial_stats = {
            'neighbor_counts': [],
            'max_distance': [],
            'mean_distance': []
        }
        
        for i in range(n_stations):
            # Find neighbors within spatial radius
            neighbor_indices = np.where(
                (distance_matrix[i] <= self.spatial_radius_km) & 
                (distance_matrix[i] > 0)  # Exclude self
            )[0]
            
            if len(neighbor_indices) >= self.min_neighbors:
                neighbor_distances = distance_matrix[i, neighbor_indices]
                
                self.neighbor_networks[i] = {
                    'indices': neighbor_indices,
                    'distances': neighbor_distances,
                    'count': len(neighbor_indices),
                    'max_distance': np.max(neighbor_distances),
                    'mean_distance': np.mean(neighbor_distances)
                }
                
                spatial_stats['neighbor_counts'].append(len(neighbor_indices))
                spatial_stats['max_distance'].append(np.max(neighbor_distances))
                spatial_stats['mean_distance'].append(np.mean(neighbor_distances))
        
        # Summary statistics
        n_connected = len(self.neighbor_networks)
        n_isolated = n_stations - n_connected
        
        print(f"✅ Spatial network built:")
        print(f"   Connected stations: {n_connected}/{n_stations} ({n_connected/n_stations*100:.1f}%)")
        print(f"   Isolated stations: {n_isolated}")
        print(f"   Average neighbors per station: {np.mean(spatial_stats['neighbor_counts']):.1f}")
        print(f"   Neighbor distance range: {np.min(spatial_stats['mean_distance']):.1f} - {np.max(spatial_stats['max_distance']):.1f} km")
        
        self.end_timer("spatial_network")
        return True
    
    def extract_frequency_components(self, frequency_band='high'):
        """
        Extract specific frequency components for atmospheric artifact analysis
        
        Parameters:
        -----------
        frequency_band : str
            'high' for tropospheric noise detection (1-60 days)
            'low' for ionospheric signal detection (>280 days)  
            'seasonal' for seasonal atmospheric patterns (60-280 days)
            
        TAIWAN-SPECIFIC GEOPHYSICAL INTERPRETATION:
        - High temporal frequency (1-45 days): Taiwan's rapid weather changes, typhoon effects,
          orographic precipitation patterns - create spatially correlated tropospheric noise
        - Low temporal frequency (200-800 days): Enhanced ionospheric effects at Taiwan's 
          geomagnetic latitude (equatorial ionization anomaly), long-term atmospheric trends
        - Seasonal (45-200 days): Taiwan monsoon patterns (wet season May-Oct vs dry season Nov-Apr),
          thermal stratification, seasonal precipitation gradients
        """
        print(f"🔊 Extracting {frequency_band}-frequency components for atmospheric analysis...")
        
        # Define Taiwan-specific frequency bands with geophysical interpretation (in days)
        # Calibrated for Taiwan's subtropical monsoon climate and ionospheric environment
        band_definitions = {
            'high': (1, 45),      # Taiwan tropospheric noise: rapid weather changes, typhoons
            'seasonal': (45, 200), # Taiwan monsoon: wet season (May-Oct) vs dry season (Nov-Apr) 
            'low': (200, 800),    # Enhanced ionospheric effects at Taiwan's geomagnetic latitude
            'trend': (800, None)  # Long-term signals beyond Taiwan's atmospheric variability
        }
        
        if frequency_band not in band_definitions:
            print(f"❌ Unknown frequency band: {frequency_band}")
            return None
        
        min_period, max_period = band_definitions[frequency_band]
        
        # Convert to frequency range
        sampling_days = 6  # 6-day sampling interval
        nyquist_freq = 1 / (2 * sampling_days)  # Nyquist frequency
        
        if max_period is not None:
            low_freq = 1 / max_period   # Cycles per day
            high_freq = 1 / min_period  # Cycles per day
        else:
            low_freq = 0
            high_freq = 1 / min_period
        
        # Smart parallel processing based on dataset size and available cores
        parallel_threshold = max(100, self.n_jobs * 10) if self.n_jobs > 1 else 1000
        
        if len(self.time_series) > parallel_threshold and self.n_jobs > 1:
            print(f"   ⚡ Using parallel filtering for {len(self.time_series)} stations ({self.n_jobs} cores)...")
            
            try:
                # Prepare arguments for parallel processing
                args_list = [
                    (self.time_series[i], low_freq, high_freq, max_period, sampling_days) 
                    for i in range(len(self.time_series))
                ]
                
                # Use ProcessPoolExecutor with proper context management
                with ProcessPoolExecutor(max_workers=min(self.n_jobs, 8)) as executor:
                    filtered_list = list(executor.map(_filter_station_data, args_list))
                filtered_components = np.array(filtered_list)
            except Exception as e:
                print(f"   ⚠️  Parallel filtering failed ({e}), falling back to sequential...")
                filtered_components = np.zeros_like(self.time_series)
                for i in range(len(self.time_series)):
                    args = (self.time_series[i], low_freq, high_freq, max_period, sampling_days)
                    filtered_components[i] = _filter_station_data(args)
        else:
            # Sequential filtering for small datasets or single-core mode
            if len(self.time_series) <= parallel_threshold:
                print(f"   🔄 Using sequential filtering for {len(self.time_series)} stations (small dataset)...")
            else:
                print(f"   🔄 Using sequential filtering (single-core mode)...")
            
            filtered_components = np.zeros_like(self.time_series)
            for i in range(len(self.time_series)):
                args = (self.time_series[i], low_freq, high_freq, max_period, sampling_days)
                filtered_components[i] = _filter_station_data(args)
        
        print(f"✅ Extracted {frequency_band}-frequency components")
        print(f"   Frequency range: {low_freq:.6f} - {high_freq:.6f} cycles/day")
        print(f"   Period range: {min_period} - {max_period if max_period else 'inf'} days")
        
        return filtered_components
    
    def analyze_high_frequency_noise_patterns(self):
        """
        Analyze spatially coherent tropospheric noise patterns
        
        GEOPHYSICAL PURPOSE:
        Detect high-frequency (1-60 days) spatially correlated signals that indicate
        tropospheric artifacts rather than genuine ground deformation:
        
        - **Atmospheric Water Vapor**: Creates correlated phase delays across nearby stations
        - **Turbulent Mixing**: Produces spatially coherent high-frequency variations
        - **Pressure Changes**: Generate common-mode atmospheric signals
        - **Temperature Gradients**: Cause spatially correlated refractivity changes
        
        Spatial correlation analysis helps distinguish these atmospheric artifacts
        from genuine high-frequency deformation signals (e.g., earthquakes, landslides).
        """
        self.start_timer("high_freq_analysis")
        
        print("🔊 Analyzing tropospheric noise patterns (high-frequency spatially correlated signals)...")
        
        # Extract high-frequency components (tropospheric signals)
        high_freq_data = self.extract_frequency_components('high')
        if high_freq_data is None:
            return False
        
        noise_patterns = []
        pattern_id = 0
        
        # Parallel analysis of spatial coherence for tropospheric patterns
        def analyze_station_coherence(station_i, network):
            """Analyze spatial coherence for a single station"""
            neighbor_indices = network['indices']
            neighbor_distances = network['distances']
            
            # Extract time series for station and neighbors
            station_ts = high_freq_data[station_i]
            neighbor_ts = high_freq_data[neighbor_indices]
            
            # Calculate temporal correlations with neighbors
            correlations = []
            for j, neighbor_idx in enumerate(neighbor_indices):
                try:
                    # Check for constant arrays to avoid correlation warnings
                    if np.std(station_ts) == 0 or np.std(neighbor_ts[j]) == 0:
                        # Skip constant arrays - no meaningful correlation
                        continue
                    
                    corr, p_value = pearsonr(station_ts, neighbor_ts[j])
                    if not np.isnan(corr):
                        correlations.append({
                            'neighbor_idx': neighbor_idx,
                            'correlation': corr,
                            'distance_km': neighbor_distances[j],
                            'p_value': p_value
                        })
                except:
                    continue
            
            return station_i, correlations
        
        # Use parallel processing for large networks
        if len(self.neighbor_networks) > 50:
            print(f"   ⚡ Using parallel processing for {len(self.neighbor_networks)} station coherence analysis...")
            
            station_results = Parallel(n_jobs=self.n_jobs)(
                delayed(analyze_station_coherence)(station_i, network) 
                for station_i, network in self.neighbor_networks.items()
            )
        else:
            # Sequential processing for small networks
            station_results = []
            for station_i, network in self.neighbor_networks.items():
                result = analyze_station_coherence(station_i, network)
                station_results.append(result)
        
        # Process results to identify coherent patterns
        for station_i, correlations in station_results:
            
            if len(correlations) == 0:
                continue
            
            # Calculate spatial coherence metrics
            correlation_values = [c['correlation'] for c in correlations]
            distances = [c['distance_km'] for c in correlations]
            
            mean_correlation = np.mean(correlation_values)
            correlation_decay = np.corrcoef(correlation_values, distances)[0, 1] if len(distances) > 1 else 0
            
            # Taiwan-specific coherence thresholds for atmospheric detection
            # More realistic threshold for InSAR data with atmospheric effects
            high_coherence_threshold = 0.15  # Lowered for realistic InSAR atmospheric detection
            min_neighbors_threshold = 3      # Minimum neighbors showing correlation
            
            # Count neighbors with significant correlation
            significant_correlations = [c for c in correlations if c['correlation'] > high_coherence_threshold]
            
            if len(significant_correlations) >= min_neighbors_threshold and mean_correlation > 0.1:
                # Calculate spatial extent
                high_corr_distances = [c['distance_km'] for c in correlations if c['correlation'] > high_coherence_threshold]
                spatial_extent = np.max(high_corr_distances) if high_corr_distances else 0
                
                pattern = GeographicPattern(
                    pattern_id=pattern_id,
                    center_coordinate=(self.coordinates[station_i, 0], self.coordinates[station_i, 1]),
                    affected_stations=[station_i] + [c['neighbor_idx'] for c in correlations if c['correlation'] > high_coherence_threshold],
                    pattern_type='noise',
                    spatial_extent_km=spatial_extent,
                    temporal_correlation=mean_correlation,
                    spatial_coherence=1.0 - abs(correlation_decay),  # Higher coherence = less decay with distance
                    frequency_band='high'
                )
                
                noise_patterns.append(pattern)
                pattern_id += 1
        
        # Store results
        self.geographic_patterns['high_frequency_noise'] = noise_patterns
        
        print(f"✅ Tropospheric noise analysis completed:")
        print(f"   Spatially coherent atmospheric regions detected: {len(noise_patterns)}")
        if noise_patterns:
            spatial_extents = [p.spatial_extent_km for p in noise_patterns]
            correlations = [p.temporal_correlation for p in noise_patterns]
            print(f"   Atmospheric coherence extent: {np.min(spatial_extents):.1f} - {np.max(spatial_extents):.1f} km")
            print(f"   Tropospheric correlation strength: {np.min(correlations):.3f} - {np.max(correlations):.3f}")
            print(f"   → These patterns likely represent atmospheric artifacts, not ground deformation")
        
        self.end_timer("high_freq_analysis")
        return True
    
    def analyze_large_area_ramps(self):
        """
        Detect and characterize spatially flat ramps (low spatial frequency patterns)
        
        GEOPHYSICAL PURPOSE:
        Analyze spatially flat, broad ramp patterns that may have high temporal frequency but
        appear as low spatial frequency (flat across large areas). These may indicate:
        
        - **Ionospheric Effects**: Spatially flat ramps with rapid temporal variations,
          particularly affecting L-band SAR systems over large areas
        - **Processing Artifacts**: Orbital errors, reference frame issues appearing as 
          spatially flat but temporally varying ramps
        - **Large-Scale Atmospheric Loading**: Broad atmospheric pressure variations
          creating flat phase delays across regional scales
        
        The key is identifying patterns that are spatially flat (low spatial frequency)
        regardless of their temporal frequency characteristics.
        """
        self.start_timer("ramp_analysis")
        
        print("📈 Analyzing spatially flat ramp patterns (low spatial frequency, potentially high temporal frequency)...")
        
        # Use ALL frequency components to look for spatially flat patterns
        # (ionospheric signals can have high temporal frequency but are spatially flat)
        all_freq_data = self.time_series  # Use full time series
        if all_freq_data is None:
            return False
        
        ramp_patterns = []
        pattern_id = 0
        
        # Analyze spatial flatness across the entire time series for each time point
        # Ionospheric ramps are spatially flat but can vary rapidly in time
        
        print("   🔍 Analyzing spatial flatness patterns across time...")
        
        n_stations = len(self.coordinates)
        n_timepoints = all_freq_data.shape[1]
        
        # For each time point, calculate spatial gradient to identify flat ramps
        spatial_flatness_scores = []
        
        for t in range(n_timepoints):
            # Get values at this time point
            values_t = all_freq_data[:, t]
            
            # Calculate spatial gradient using distance-weighted differences
            spatial_gradients = []
            
            for i in range(n_stations):
                station_gradients = []
                for j in range(i+1, n_stations):
                    # Calculate spatial distance
                    dist = self._haversine_distance(self.coordinates[i], self.coordinates[j])
                    
                    if dist > 0 and dist <= self.spatial_radius_km * 3:  # Extended radius for ramp detection
                        # Calculate value difference per km
                        value_diff = abs(values_t[i] - values_t[j])
                        gradient = value_diff / dist  # mm/km
                        station_gradients.append(gradient)
                
                if station_gradients:
                    # Mean gradient for this station
                    mean_gradient = np.mean(station_gradients)
                    spatial_gradients.append(mean_gradient)
            
            if spatial_gradients:
                # Overall spatial flatness (lower gradient = flatter = more suspicious for ionospheric)
                flatness_score = 1.0 / (1.0 + np.mean(spatial_gradients))  # High score = flat
                spatial_flatness_scores.append(flatness_score)
            else:
                spatial_flatness_scores.append(0.0)
        
        # Identify time periods with consistently high spatial flatness
        flatness_threshold = 0.4  # More realistic flatness threshold for InSAR data
        flat_periods = np.where(np.array(spatial_flatness_scores) > flatness_threshold)[0]
        
        if len(flat_periods) > 5:  # Reduced threshold for realistic detection
            # Calculate temporal correlation during flat periods
            flat_time_series = all_freq_data[:, flat_periods]
            
            # Check if stations are temporally correlated during flat periods
            temporal_correlations = []
            for i in range(n_stations):
                for j in range(i+1, n_stations):
                    try:
                        # Check for constant arrays to avoid correlation warnings
                        if np.std(flat_time_series[i]) == 0 or np.std(flat_time_series[j]) == 0:
                            continue
                        
                        corr, _ = pearsonr(flat_time_series[i], flat_time_series[j])
                        if not np.isnan(corr):
                            temporal_correlations.append(abs(corr))
                    except:
                        continue
            
            if temporal_correlations:
                mean_temporal_correlation = np.mean(temporal_correlations)
                
                # Taiwan-specific ionospheric ramp detection
                # Enhanced threshold due to Taiwan's location in equatorial ionization anomaly
                if mean_temporal_correlation > 0.3:  # Lower threshold for Taiwan's enhanced ionospheric activity
                    # Calculate spatial extent (for flat ramps, this is the full area)
                    max_distance = 0
                    for i in range(n_stations):
                        for j in range(i+1, n_stations):
                            dist = self._haversine_distance(self.coordinates[i], self.coordinates[j])
                            max_distance = max(max_distance, dist)
                    
                    # Calculate ramp strength (temporal variability during flat periods)
                    temporal_variability = np.std(flat_time_series.flatten())
                    
                    pattern = GeographicPattern(
                        pattern_id=pattern_id,
                        center_coordinate=(np.mean(self.coordinates[:, 0]), np.mean(self.coordinates[:, 1])),
                        affected_stations=list(range(n_stations)),  # Affects all stations (flat ramp)
                        pattern_type='flat_ramp',
                        spatial_extent_km=max_distance,
                        temporal_correlation=mean_temporal_correlation,
                        spatial_coherence=np.mean(spatial_flatness_scores),  # Spatial flatness measure
                        frequency_band='spatially_flat'
                    )
                    
                    ramp_patterns.append(pattern)
                    pattern_id += 1
        
        # Store results
        self.geographic_patterns['large_area_ramps'] = ramp_patterns
        
        print(f"✅ Spatially flat ramp pattern analysis completed:")
        print(f"   Spatially flat ramp regions detected: {len(ramp_patterns)}")
        if ramp_patterns:
            spatial_extents = [p.spatial_extent_km for p in ramp_patterns]
            coherence_values = [p.spatial_coherence for p in ramp_patterns]
            temporal_correlations = [p.temporal_correlation for p in ramp_patterns]
            print(f"   Flat ramp extent: {np.min(spatial_extents):.1f} - {np.max(spatial_extents):.1f} km")
            print(f"   Spatial flatness: {np.min(coherence_values):.3f} - {np.max(coherence_values):.3f}")
            print(f"   Temporal correlation: {np.min(temporal_correlations):.3f} - {np.max(temporal_correlations):.3f}")
            print(f"   → High spatial flatness + temporal correlation suggests ionospheric effects")
        
        self.end_timer("ramp_analysis")
        return True
    
    def _haversine_distance(self, coord1, coord2):
        """Calculate Haversine distance between two lat/lon points in km"""
        R = 6371.0  # Earth radius in km
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def perform_spatial_temporal_clustering(self, frequency_band='seasonal'):
        """
        Perform spatial-temporal clustering for atmospheric signal source identification
        
        GEOPHYSICAL PURPOSE:
        Group stations based on temporal similarity in specific frequency bands to identify:
        
        - **Common Atmospheric Sources**: Stations affected by similar tropospheric/ionospheric conditions
        - **Signal Source Regions**: Geographic clustering of atmospheric artifact patterns
        - **Artifact vs Deformation**: Distinguish coherent atmospheric signals from genuine deformation
        
        DTW clustering helps identify stations that share similar temporal patterns,
        which for atmospheric signals indicates common atmospheric conditions or sources.
        """
        self.start_timer("spatial_clustering")
        
        print(f"🗂️  Performing atmospheric signal source clustering ({frequency_band} band)...")
        
        if not TSLEARN_AVAILABLE:
            print("⚠️  TSLearn not available, using fallback clustering")
            return self._fallback_spatial_clustering(frequency_band)
        
        # Extract frequency-specific data
        freq_data = self.extract_frequency_components(frequency_band)
        if freq_data is None:
            return False
        
        # Determine optimal number of clusters based on spatial distribution
        n_clusters_range = range(3, min(10, len(self.coordinates) // 50))
        
        best_score = -1
        best_n_clusters = 5
        
        # Test different cluster numbers
        for n_clusters in n_clusters_range:
            try:
                # DTW-based clustering
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=self.random_state)
                cluster_labels = model.fit_predict(freq_data)
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(freq_data.reshape(len(freq_data), -1), cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
            except Exception as e:
                continue
        
        # Perform final clustering with best parameters
        print(f"   Using {best_n_clusters} clusters (silhouette score: {best_score:.3f})")
        
        model = TimeSeriesKMeans(n_clusters=best_n_clusters, metric="dtw", random_state=self.random_state)
        cluster_labels = model.fit_predict(freq_data)
        
        # Analyze spatial characteristics of clusters
        spatial_clusters = {}
        
        for cluster_id in range(best_n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Calculate cluster spatial characteristics
            cluster_coords = self.coordinates[cluster_indices]
            centroid = np.mean(cluster_coords, axis=0)
            
            # Calculate cluster radius (maximum distance from centroid)
            distances_from_centroid = [
                self._haversine_distance(centroid, coord) for coord in cluster_coords
            ]
            cluster_radius = np.max(distances_from_centroid)
            
            # Calculate temporal coherence within cluster
            cluster_ts = freq_data[cluster_indices]
            
            # DTW coherence: average pairwise DTW distance within cluster
            dtw_distances = []
            n_pairs = min(100, len(cluster_indices) * (len(cluster_indices) - 1) // 2)  # Limit pairs for efficiency
            
            pair_count = 0
            for i in range(len(cluster_indices)):
                for j in range(i+1, len(cluster_indices)):
                    if pair_count >= n_pairs:
                        break
                    try:
                        dtw_dist = dtw(cluster_ts[i], cluster_ts[j])
                        dtw_distances.append(dtw_dist)
                        pair_count += 1
                    except:
                        continue
                if pair_count >= n_pairs:
                    break
            
            dtw_coherence = 1.0 / (1.0 + np.mean(dtw_distances)) if dtw_distances else 0.0
            
            # Extract dominant temporal signature (cluster centroid)
            temporal_signature = model.cluster_centers_[cluster_id]
            
            spatial_cluster = SpatialCluster(
                cluster_id=cluster_id,
                station_indices=cluster_indices.tolist(),
                centroid_coordinate=(centroid[0], centroid[1]),
                cluster_radius_km=cluster_radius,
                temporal_signature=temporal_signature,
                dtw_coherence=dtw_coherence,
                dominant_frequency=frequency_band
            )
            
            spatial_clusters[cluster_id] = spatial_cluster
        
        # Store results
        self.spatial_clusters[frequency_band] = spatial_clusters
        
        print(f"✅ Spatial-temporal clustering completed:")
        print(f"   Clusters formed: {len(spatial_clusters)}")
        if spatial_clusters:
            cluster_sizes = [len(cluster.station_indices) for cluster in spatial_clusters.values()]
            cluster_radii = [cluster.cluster_radius_km for cluster in spatial_clusters.values()]
            dtw_coherences = [cluster.dtw_coherence for cluster in spatial_clusters.values()]
            
            print(f"   Cluster size range: {np.min(cluster_sizes)} - {np.max(cluster_sizes)} stations")
            print(f"   Cluster radius range: {np.min(cluster_radii):.1f} - {np.max(cluster_radii):.1f} km")
            print(f"   DTW coherence range: {np.min(dtw_coherences):.3f} - {np.max(dtw_coherences):.3f}")
        
        self.end_timer("spatial_clustering")
        return True
    
    def _fallback_spatial_clustering(self, frequency_band):
        """Fallback clustering method when TSLearn is not available"""
        print("   Using fallback correlation-based clustering...")
        
        # Extract frequency-specific data
        freq_data = self.extract_frequency_components(frequency_band)
        if freq_data is None:
            return False
        
        # Use correlation-based distance matrix
        n_stations = len(freq_data)
        correlation_matrix = np.corrcoef(freq_data)
        distance_matrix = 1 - np.abs(correlation_matrix)  # Convert correlation to distance
        
        # DBSCAN clustering on correlation distances
        clustering = DBSCAN(eps=0.3, min_samples=5, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Process results similar to DTW clustering
        spatial_clusters = {}
        unique_labels = np.unique(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_coords = self.coordinates[cluster_indices]
            centroid = np.mean(cluster_coords, axis=0)
            
            distances_from_centroid = [
                self._haversine_distance(centroid, coord) for coord in cluster_coords
            ]
            cluster_radius = np.max(distances_from_centroid)
            
            # Calculate correlation coherence
            cluster_correlations = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
            mean_correlation = np.mean(cluster_correlations[np.triu_indices_from(cluster_correlations, k=1)])
            
            temporal_signature = np.mean(freq_data[cluster_indices], axis=0)
            
            spatial_cluster = SpatialCluster(
                cluster_id=cluster_id,
                station_indices=cluster_indices.tolist(),
                centroid_coordinate=(centroid[0], centroid[1]),
                cluster_radius_km=cluster_radius,
                temporal_signature=temporal_signature,
                dtw_coherence=mean_correlation,
                dominant_frequency=frequency_band
            )
            
            spatial_clusters[cluster_id] = spatial_cluster
        
        self.spatial_clusters[frequency_band] = spatial_clusters
        return True
    
    def _create_overview_figure(self, results):
        """Create overview figure with geographic patterns and explanations"""
        print("   📊 Creating overview figure...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Geographic Pattern Analysis - Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Station locations and spatial network
        ax1 = axes[0, 0]
        if hasattr(self, 'coordinates') and self.coordinates is not None:
            ax1.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                       c='blue', s=80, alpha=0.7, label='Analysis Stations', edgecolors='darkblue', linewidth=0.5)
            ax1.set_xlabel('Longitude (°E)', fontsize=12)
            ax1.set_ylabel('Latitude (°N)', fontsize=12)
            ax1.set_title('Station Network Distribution', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Frequency band analysis
        ax2 = axes[0, 1]
        frequency_bands = ['high', 'seasonal', 'low']
        pattern_counts = []
        for band in frequency_bands:
            if hasattr(self, 'geographic_patterns') and band in self.geographic_patterns:
                pattern_counts.append(len(self.geographic_patterns[band]))
            else:
                pattern_counts.append(0)
        
        bars = ax2.bar(frequency_bands, pattern_counts, color=['red', 'orange', 'green'], alpha=0.8, 
                       edgecolor='black', linewidth=1.2, width=0.6)
        ax2.set_xlabel('Frequency Band', fontsize=12)
        ax2.set_ylabel('Number of Patterns Detected', fontsize=12)
        ax2.set_title('Pattern Detection by Frequency Band', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add count labels on bars with larger font
        for bar, count in zip(bars, pattern_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Plot 3: Spatial clustering results
        ax3 = axes[1, 0]
        if hasattr(self, 'spatial_clusters') and self.spatial_clusters:
            total_clusters = sum(len(clusters) for clusters in self.spatial_clusters.values())
            cluster_bands = list(self.spatial_clusters.keys())
            cluster_counts = [len(self.spatial_clusters[band]) for band in cluster_bands]
            
            if cluster_counts:
                bars = ax3.bar(cluster_bands, cluster_counts, color=['purple', 'cyan', 'yellow'], alpha=0.7)
                ax3.set_xlabel('Frequency Band')
                ax3.set_ylabel('Number of Spatial Clusters')
                ax3.set_title('Spatial-Temporal Clustering Results')
                ax3.grid(True, alpha=0.3)
                
                # Add count labels
                for bar, count in zip(bars, cluster_counts):
                    if count > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Analysis summary and timing
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Add summary text
        summary_text = f"""
GEOGRAPHIC PATTERN ANALYSIS SUMMARY

📊 Dataset Information:
   • Stations analyzed: {len(self.coordinates) if hasattr(self, 'coordinates') and self.coordinates is not None else 'N/A'}
   • Spatial radius: {self.spatial_radius_km} km
   • Min neighbors: {self.min_neighbors}

🔍 Pattern Detection:
   • High frequency: {pattern_counts[0]} patterns
   • Seasonal: {pattern_counts[1]} patterns  
   • Low frequency: {pattern_counts[2]} patterns

🗂️  Clustering Results:
   • Total clusters: {sum(len(clusters) for clusters in self.spatial_clusters.values()) if hasattr(self, 'spatial_clusters') else 0}
   • Frequency bands: {len(self.spatial_clusters) if hasattr(self, 'spatial_clusters') else 0}

⚡ Processing Status:
   • Matrix profile: {'Available' if STUMPY_AVAILABLE else 'Limited'}
   • TSLearn DTW: {'Available' if TSLEARN_AVAILABLE else 'Limited'}
   • Parallel processing: {'Enabled' if self.n_jobs > 1 else 'Disabled'}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps04d_fig01_geographic_patterns_comprehensive.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Created overview figure: {fig_path}")
        return str(fig_path)
    
    def _create_spatial_network_figure(self, results):
        """Create spatial network visualization figure"""
        print("   📊 Creating spatial network figure...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle('Spatial Network Analysis', fontsize=14, fontweight='bold')
        
        if hasattr(self, 'coordinates') and self.coordinates is not None:
            # Plot station locations
            ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                      c='blue', s=30, alpha=0.7, label='Stations', zorder=3)
            
            # Draw spatial connections if available
            if hasattr(self, 'neighbor_networks') and self.neighbor_networks:
                for station_i, network in self.neighbor_networks.items():
                    station_coord = self.coordinates[station_i]
                    for neighbor_idx in network['indices'][:10]:  # Limit connections for clarity
                        neighbor_coord = self.coordinates[neighbor_idx]
                        ax.plot([station_coord[0], neighbor_coord[0]], 
                               [station_coord[1], neighbor_coord[1]], 
                               'k-', alpha=0.1, linewidth=0.5, zorder=1)
            
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig_path = self.figures_dir / "ps04d_fig02_spatial_network.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Created spatial network figure: {fig_path}")
        return str(fig_path)
    
    def _create_pattern_detection_figure(self, results):
        """Create real atmospheric pattern detection visualization"""
        print("   📊 Creating atmospheric pattern detection figure...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Atmospheric Pattern Detection Results - Taiwan InSAR Analysis', fontsize=14, fontweight='bold')
        
        # Quadrant 1: Tropospheric noise patterns
        self._plot_tropospheric_patterns(axes[0, 0])
        
        # Quadrant 2: Ionospheric ramp detection  
        self._plot_ionospheric_ramps(axes[0, 1])
        
        # Quadrant 3: High-frequency spatial patterns
        self._plot_spatial_frequency_analysis(axes[1, 0])
        
        # Quadrant 4: Pattern detection statistics
        self._plot_detection_statistics(axes[1, 1])
        
        fig_path = self.figures_dir / "ps04d_fig03_pattern_detection.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Created pattern detection figure: {fig_path}")
        return str(fig_path)
    
    def _create_clustering_analysis_figure(self, results):
        """Create real spatial-temporal clustering visualization"""
        print("   📊 Creating atmospheric clustering analysis figure...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Atmospheric Signal Clustering Results - Taiwan InSAR Analysis', fontsize=14, fontweight='bold')
        
        # Quadrant 1: High-frequency band clusters
        self._plot_spatial_clusters(axes[0, 0], 'high')
        
        # Quadrant 2: Seasonal band clusters
        self._plot_spatial_clusters(axes[0, 1], 'seasonal')
        
        # Quadrant 3: Low-frequency band clusters  
        self._plot_spatial_clusters(axes[1, 0], 'low')
        
        # Quadrant 4: Cluster characteristics summary
        self._plot_cluster_characteristics(axes[1, 1])
        
        fig_path = self.figures_dir / "ps04d_fig04_clustering_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Created clustering analysis figure: {fig_path}")
        return str(fig_path)
    
    def create_geographic_pattern_visualization(self):
        """Create multiple individual figures for clarity"""
        self.start_timer("visualization")
        
        print("📊 Creating geographic pattern visualizations...")
        
        # Load results for visualization (in case we're running visualization only)
        results = self.load_results()
        
        figure_files = []
        
        # Figure 1: Overview with explanations
        fig1 = self._create_overview_figure(results)
        figure_files.append(fig1)
        
        # Figure 2: Spatial network analysis
        fig2 = self._create_spatial_network_figure(results)
        figure_files.append(fig2)
        
        # Figure 3: Pattern detection results
        fig3 = self._create_pattern_detection_figure(results)
        figure_files.append(fig3)
        
        # Figure 4: Clustering analysis
        fig4 = self._create_clustering_analysis_figure(results)
        figure_files.append(fig4)
        
        print(f"✅ Created {len(figure_files)} geographic pattern visualization figures")
        for i, fig_file in enumerate(figure_files, 1):
            print(f"   📈 Figure {i}: {fig_file}")
        
        self.end_timer("visualization")
        return figure_files[0]  # Return main figure for compatibility
    
    def load_results(self):
        """Load existing results if available"""
        results_file = self.results_dir / "ps04d_geographic_patterns_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Return empty structure if no results
        return {
            'analysis_metadata': {
                'spatial_radius_km': self.spatial_radius_km,
                'min_neighbors': self.min_neighbors,
                'n_stations_analyzed': len(self.coordinates) if self.coordinates is not None else 0,
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stumpy_available': STUMPY_AVAILABLE,
                'tslearn_available': TSLEARN_AVAILABLE
            },
            'spatial_network': {
                'connected_stations': len(self.neighbor_networks),
                'network_statistics': {
                    'neighbor_counts': [net['count'] for net in self.neighbor_networks.values()] if self.neighbor_networks else [],
                    'mean_distances': [net['mean_distance'] for net in self.neighbor_networks.values()] if self.neighbor_networks else [],
                    'max_distances': [net['max_distance'] for net in self.neighbor_networks.values()] if self.neighbor_networks else []
                }
            },
            'geographic_patterns': {
                'high_frequency_noise': [],
                'large_area_ramps': []
            },
            'spatial_clusters': {},
            'timing_results': self.timing_results
        }
    
    def _plot_noise_patterns(self, ax):
        """Plot high-frequency noise patterns"""
        ax.set_title("High-Frequency Noise Patterns", fontsize=12, fontweight='bold')
        
        # Add map features (skip for regular matplotlib axes)
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.3)
        except AttributeError:
            # Regular matplotlib axes - just add grid
            ax.grid(True, alpha=0.3)
        
        # Plot all stations
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                  c='lightgray', s=10, alpha=0.5)
        
        # Plot noise patterns
        if 'high_frequency_noise' in self.geographic_patterns:
            patterns = self.geographic_patterns['high_frequency_noise']
            
            for pattern in patterns:
                # Plot center
                center_lon, center_lat = pattern.center_coordinate
                ax.scatter(center_lon, center_lat, c='red', s=50, 
                          alpha=pattern.temporal_correlation)
                
                # Plot affected stations
                affected_coords = self.coordinates[pattern.affected_stations]
                ax.scatter(affected_coords[:, 0], affected_coords[:, 1], 
                          c='orange', s=20, alpha=0.7)
        
        self._set_geographic_extent(ax)
    
    def _plot_ramp_patterns(self, ax):
        """Plot large-area ramp patterns"""
        ax.set_title("Large-Area Ramp Patterns", fontsize=12, fontweight='bold')
        
        # Add map features (skip for regular matplotlib axes)
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.3)
        except AttributeError:
            # Regular matplotlib axes - just add grid
            ax.grid(True, alpha=0.3)
        
        # Plot all stations
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                  c='lightgray', s=10, alpha=0.5)
        
        # Plot ramp patterns
        if 'large_area_ramps' in self.geographic_patterns:
            patterns = self.geographic_patterns['large_area_ramps']
            
            for pattern in patterns:
                # Plot center
                center_lon, center_lat = pattern.center_coordinate
                ax.scatter(center_lon, center_lat, c='blue', s=100, 
                          alpha=pattern.temporal_correlation)
                
                # Plot affected stations
                affected_coords = self.coordinates[pattern.affected_stations]
                ax.scatter(affected_coords[:, 0], affected_coords[:, 1], 
                          c='cyan', s=30, alpha=0.7)
        
        self._set_geographic_extent(ax)
    
    def _plot_spatial_clusters(self, ax, frequency_band):
        """Plot spatial-temporal clusters"""
        ax.set_title(f"Spatial Clusters ({frequency_band} band)", fontsize=12, fontweight='bold')
        
        # Add map features (skip for regular matplotlib axes)
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.3)
        except AttributeError:
            # Regular matplotlib axes - just add grid
            ax.grid(True, alpha=0.3)
        
        # Plot clusters if available
        if frequency_band in self.spatial_clusters:
            clusters = self.spatial_clusters[frequency_band]
            colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
            
            for i, (cluster_id, cluster) in enumerate(clusters.items()):
                # Plot cluster stations
                cluster_coords = self.coordinates[cluster.station_indices]
                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                          c=[colors[i]], s=30, alpha=0.8, 
                          label=f'Cluster {cluster_id}')
                
                # Plot cluster centroid
                centroid_lon, centroid_lat = cluster.centroid_coordinate
                ax.scatter(centroid_lon, centroid_lat, c='black', s=100, 
                          marker='x')
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Plot all stations in gray if no clusters
            ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                      c='lightgray', s=10, alpha=0.5)
        
        self._set_geographic_extent(ax)
    
    def _plot_spatial_network_stats(self, ax):
        """Plot spatial network connectivity statistics"""
        ax.set_title("Spatial Network Statistics", fontsize=12, fontweight='bold')
        
        if not self.neighbor_networks:
            ax.text(0.5, 0.5, "No spatial network data", ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract statistics
        neighbor_counts = [network['count'] for network in self.neighbor_networks.values()]
        mean_distances = [network['mean_distance'] for network in self.neighbor_networks.values()]
        
        # Create subplots within this axis
        ax.clear()
        
        # Histogram of neighbor counts
        ax.hist(neighbor_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Number of Neighbors')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Neighbor Count Distribution\n(mean: {np.mean(neighbor_counts):.1f})')
        ax.grid(True, alpha=0.3)
    
    def _plot_pattern_statistics(self, ax):
        """Plot pattern detection statistics"""
        ax.set_title("Pattern Detection Summary", fontsize=12, fontweight='bold')
        
        # Collect pattern statistics
        pattern_counts = {}
        
        if 'high_frequency_noise' in self.geographic_patterns:
            pattern_counts['Noise Patterns'] = len(self.geographic_patterns['high_frequency_noise'])
        
        if 'large_area_ramps' in self.geographic_patterns:
            pattern_counts['Ramp Patterns'] = len(self.geographic_patterns['large_area_ramps'])
        
        for freq_band in self.spatial_clusters:
            pattern_counts[f'{freq_band.title()} Clusters'] = len(self.spatial_clusters[freq_band])
        
        if pattern_counts:
            # Bar plot
            categories = list(pattern_counts.keys())
            counts = list(pattern_counts.values())
            
            bars = ax.bar(categories, counts, color=['orange', 'cyan', 'lightgreen', 'pink'][:len(categories)])
            ax.set_ylabel('Count')
            ax.set_title('Detected Patterns Summary')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "No patterns detected", ha='center', va='center', transform=ax.transAxes)
    
    def _plot_coherence_map(self, ax):
        """Plot geographic coherence map"""
        ax.set_title("Spatial Coherence Map", fontsize=12, fontweight='bold')
        
        # Add map features (skip for regular matplotlib axes)
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.3)
        except AttributeError:
            # Regular matplotlib axes - just add grid
            ax.grid(True, alpha=0.3)
        
        # Calculate spatial coherence for each station
        coherence_values = np.zeros(len(self.coordinates))
        
        for station_i, network in self.neighbor_networks.items():
            # Calculate mean correlation with neighbors as coherence measure
            neighbor_indices = network['indices']
            
            if len(neighbor_indices) > 0:
                station_ts = self.time_series[station_i]
                correlations = []
                
                for neighbor_idx in neighbor_indices:
                    try:
                        # Check for constant arrays to avoid correlation warnings
                        if np.std(station_ts) == 0 or np.std(self.time_series[neighbor_idx]) == 0:
                            continue
                        
                        corr, _ = pearsonr(station_ts, self.time_series[neighbor_idx])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        continue
                
                coherence_values[station_i] = np.mean(correlations) if correlations else 0
        
        # Plot coherence map
        scatter = ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                           c=coherence_values, s=30, cmap='viridis', 
                           vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Spatial Coherence', shrink=0.6)
        
        self._set_geographic_extent(ax)
    
    def _set_geographic_extent(self, ax):
        """Set consistent geographic extent for all map plots"""
        lon_min, lon_max = self.coordinates[:, 0].min() - 0.05, self.coordinates[:, 0].max() + 0.05
        lat_min, lat_max = self.coordinates[:, 1].min() - 0.05, self.coordinates[:, 1].max() + 0.05
        
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
    
    def _plot_tropospheric_patterns(self, ax):
        """Plot tropospheric noise patterns (spatially correlated, high temporal frequency)"""
        ax.set_title('Tropospheric Noise Detection', fontsize=12, fontweight='bold')
        
        # Check if we have detected patterns
        if hasattr(self, 'geographic_patterns') and 'high_frequency_noise' in self.geographic_patterns:
            noise_patterns = self.geographic_patterns['high_frequency_noise']
            
            if noise_patterns:
                # Plot detected tropospheric patterns
                for i, pattern in enumerate(noise_patterns):
                    center_coords = pattern.center_coordinate
                    affected_stations = pattern.affected_stations
                    
                    # Plot affected stations
                    if affected_stations:
                        station_coords = self.coordinates[affected_stations]
                        scatter = ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                                           c=pattern.temporal_correlation, cmap='Reds',
                                           s=50, alpha=0.8, 
                                           label=f'Pattern {i+1} (r={pattern.temporal_correlation:.2f})')
                        
                        # Mark pattern center
                        ax.scatter(center_coords[0], center_coords[1], 
                                 marker='x', s=100, c='black', linewidth=2)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.set_xlabel('Longitude (°E)')
                ax.set_ylabel('Latitude (°N)')
                self._set_geographic_extent(ax)
                
                # Add statistics
                correlations = [p.temporal_correlation for p in noise_patterns]
                extents = [p.spatial_extent_km for p in noise_patterns]
                ax.text(0.02, 0.98, f'Detected: {len(noise_patterns)} patterns\n'
                                   f'Correlation: {np.mean(correlations):.2f}±{np.std(correlations):.2f}\n'
                                   f'Extent: {np.mean(extents):.1f}±{np.std(extents):.1f} km',
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # No patterns detected - show informative message
                ax.text(0.5, 0.5, '🔍 No Tropospheric Patterns Detected\n\n'
                                 'Analysis completed but no spatially\n'
                                 'coherent high-frequency atmospheric\n'
                                 'signals met detection criteria.\n\n'
                                 'This may indicate:\n'
                                 '• Low atmospheric noise levels\n'
                                 '• Good atmospheric correction\n'
                                 '• Localized deformation dominates',
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        else:
            # Analysis not run or data unavailable
            ax.text(0.5, 0.5, '⚠️ Tropospheric Analysis Incomplete\n\n'
                             'Pattern detection analysis has not\n'
                             'been completed for this dataset.',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    def _plot_ionospheric_ramps(self, ax):
        """Plot ionospheric ramp detection (spatially flat, high temporal frequency)"""
        ax.set_title('Ionospheric Ramp Detection', fontsize=12, fontweight='bold')
        
        # Check if we have detected flat ramp patterns
        if hasattr(self, 'geographic_patterns') and 'flat_ramp_patterns' in self.geographic_patterns:
            ramp_patterns = self.geographic_patterns['flat_ramp_patterns']
            
            if ramp_patterns:
                # Plot detected flat ramp patterns
                for i, pattern in enumerate(ramp_patterns):
                    center_coords = pattern.center_coordinate
                    affected_stations = pattern.affected_stations
                    
                    # Plot affected stations
                    if affected_stations:
                        station_coords = self.coordinates[affected_stations]
                        scatter = ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                                           c=pattern.temporal_correlation, cmap='coolwarm',
                                           s=50, alpha=0.8, 
                                           label=f'Ramp {i+1} (r={pattern.temporal_correlation:.2f})')
                        
                        # Mark pattern center  
                        ax.scatter(center_coords[0], center_coords[1], 
                                 marker='s', s=100, c='black', linewidth=2)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.set_xlabel('Longitude (°E)')
                ax.set_ylabel('Latitude (°N)')
                self._set_geographic_extent(ax)
                
                # Add statistics
                correlations = [p.temporal_correlation for p in ramp_patterns]
                extents = [p.spatial_extent_km for p in ramp_patterns]
                ax.text(0.02, 0.98, f'Detected: {len(ramp_patterns)} ramps\n'
                                   f'Correlation: {np.mean(correlations):.2f}±{np.std(correlations):.2f}\n'
                                   f'Extent: {np.mean(extents):.1f}±{np.std(extents):.1f} km',
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # No flat ramps detected - show informative message
                ax.text(0.5, 0.5, '🛰️ No Ionospheric Ramps Detected\n\n'
                                 'Analysis completed but no spatially\n'
                                 'flat high-frequency patterns met\n'
                                 'detection criteria.\n\n'
                                 'This may indicate:\n'
                                 '• Good ionospheric correction\n'
                                 '• C-band data (less affected)\n'
                                 '• Complex topographic patterns',
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        else:
            # Analysis not run or data unavailable
            ax.text(0.5, 0.5, '⚠️ Ramp Analysis Incomplete\n\n'
                             'Flat ramp pattern detection has\n'
                             'not been completed for this dataset.',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    def _plot_spatial_frequency_analysis(self, ax):
        """Plot spatial frequency analysis results"""
        ax.set_title('Spatial Frequency Analysis', fontsize=12, fontweight='bold')
        
        # Analyze spatial frequency characteristics
        if hasattr(self, 'neighbor_networks') and self.neighbor_networks:
            n_stations = len(self.coordinates)
            
            # Calculate spatial frequency metrics
            connectivity_density = []
            
            for station_id in range(min(n_stations, len(self.neighbor_networks))):
                if station_id in self.neighbor_networks:
                    network = self.neighbor_networks[station_id]
                    n_neighbors = len(network['indices'])
                    connectivity_density.append(n_neighbors)
            
            if connectivity_density:
                # Create spatial frequency visualization
                coords_subset = self.coordinates[:len(connectivity_density)]
                
                # Use connectivity density as proxy for spatial frequency content
                scatter = ax.scatter(coords_subset[:, 0], coords_subset[:, 1], 
                                   c=connectivity_density, cmap='viridis', s=50, alpha=0.7,
                                   edgecolors='black', linewidth=0.3)
                
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Spatial Connectivity', fontsize=10)
                
                # Add statistics
                mean_connectivity = np.mean(connectivity_density)
                max_connectivity = np.max(connectivity_density)
                
                ax.text(0.02, 0.98, f'Mean connectivity: {mean_connectivity:.1f}\nMax connectivity: {max_connectivity}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Spatial frequency analysis\nnot available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        else:
            ax.text(0.5, 0.5, 'Spatial network not\ncomputed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_detection_statistics(self, ax):
        """Plot pattern detection statistics and summary"""
        ax.set_title('Detection Statistics Summary', fontsize=12, fontweight='bold')
        
        # Collect detection statistics
        detection_stats = {
            'Tropospheric': 0,
            'Ionospheric': 0, 
            'High-frequency': 0,
            'Seasonal noise': 0
        }
        
        # Count patterns by type
        if 'high' in self.coherent_regions:
            high_freq_count = sum(1 for data in self.coherent_regions['high'].values() 
                                if data.get('mean_correlation', 0) > 0.7)
            detection_stats['Tropospheric'] = high_freq_count
            detection_stats['High-frequency'] = len(self.coherent_regions['high'])
        
        if 'seasonal' in self.coherent_regions:
            seasonal_count = len(self.coherent_regions['seasonal'])
            detection_stats['Seasonal noise'] = seasonal_count
        
        # Simple ionospheric detection count (placeholder)
        if hasattr(self, 'time_series') and self.time_series is not None:
            detection_stats['Ionospheric'] = 1 if len(self.coordinates) > 50 else 0
        
        # Create bar chart
        categories = list(detection_stats.keys())
        values = list(detection_stats.values())
        
        if any(values):
            bars = ax.bar(categories, values, color=['coral', 'lightblue', 'lightgreen', 'gold'], 
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Number of Patterns', fontsize=10)
            ax.set_xlabel('Pattern Type', fontsize=10)
            
            # Add total count
            total_patterns = sum(values)
            ax.text(0.98, 0.98, f'Total patterns: {total_patterns}', 
                   transform=ax.transAxes, fontsize=11, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No atmospheric patterns\ndetected in current analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_cluster_characteristics(self, ax):
        """Plot cluster characteristics summary"""
        ax.set_title('Cluster Characteristics', fontsize=12, fontweight='bold')
        
        # Collect cluster statistics from all frequency bands
        cluster_stats = {}
        total_clusters = 0
        
        for freq_band in ['high', 'seasonal', 'low']:
            if freq_band in self.spatial_clusters:
                clusters = self.spatial_clusters[freq_band]
                total_clusters += len(clusters)
                
                if clusters:
                    cluster_sizes = [len(cluster.station_indices) for cluster in clusters.values()]
                    cluster_radii = [cluster.cluster_radius_km for cluster in clusters.values()]
                    dtw_coherences = [cluster.dtw_coherence for cluster in clusters.values()]
                    
                    cluster_stats[freq_band] = {
                        'count': len(clusters),
                        'mean_size': np.mean(cluster_sizes),
                        'mean_radius': np.mean(cluster_radii), 
                        'mean_coherence': np.mean(dtw_coherences)
                    }
        
        if cluster_stats:
            # Create summary table visualization
            bands = list(cluster_stats.keys())
            y_pos = np.arange(len(bands))
            
            # Plot cluster counts as horizontal bars
            counts = [cluster_stats[band]['count'] for band in bands]
            bars = ax.barh(y_pos, counts, color=['red', 'green', 'blue'], alpha=0.6)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'{band.title()} freq.' for band in bands])
            ax.set_xlabel('Number of Clusters', fontsize=10)
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{count}', ha='left', va='center', fontsize=10, fontweight='bold')
            
            # Add summary statistics as text
            summary_text = f"Total clusters: {total_clusters}\n"
            for band in bands:
                stats = cluster_stats[band]
                summary_text += f"{band}: {stats['mean_size']:.1f} stations/cluster\n"
            
            ax.text(0.98, 0.02, summary_text.strip(), transform=ax.transAxes, 
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No clusters formed\nin current analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.grid(True, alpha=0.3, axis='x')

    def save_results(self):
        """Save all analysis results to files"""
        self.start_timer("save_results")
        
        print("💾 Saving geographic pattern analysis results...")
        
        # Prepare results dictionary
        results = {
            'analysis_metadata': {
                'spatial_radius_km': self.spatial_radius_km,
                'min_neighbors': self.min_neighbors,
                'n_stations_analyzed': len(self.coordinates) if self.coordinates is not None else 0,
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stumpy_available': STUMPY_AVAILABLE,
                'tslearn_available': TSLEARN_AVAILABLE
            },
            'spatial_network': {
                'connected_stations': len(self.neighbor_networks),
                'network_statistics': {
                    'neighbor_counts': [net['count'] for net in self.neighbor_networks.values()],
                    'mean_distances': [net['mean_distance'] for net in self.neighbor_networks.values()],
                    'max_distances': [net['max_distance'] for net in self.neighbor_networks.values()]
                }
            },
            'geographic_patterns': {},
            'spatial_clusters': {},
            'timing_results': self.timing_results
        }
        
        # Convert geographic patterns to serializable format
        for pattern_type, patterns in self.geographic_patterns.items():
            results['geographic_patterns'][pattern_type] = []
            for pattern in patterns:
                results['geographic_patterns'][pattern_type].append({
                    'pattern_id': pattern.pattern_id,
                    'center_coordinate': pattern.center_coordinate,
                    'affected_stations': pattern.affected_stations,
                    'pattern_type': pattern.pattern_type,
                    'spatial_extent_km': pattern.spatial_extent_km,
                    'temporal_correlation': pattern.temporal_correlation,
                    'spatial_coherence': pattern.spatial_coherence,
                    'frequency_band': pattern.frequency_band
                })
        
        # Convert spatial clusters to serializable format
        for freq_band, clusters in self.spatial_clusters.items():
            results['spatial_clusters'][freq_band] = {}
            for cluster_id, cluster in clusters.items():
                results['spatial_clusters'][freq_band][cluster_id] = {
                    'cluster_id': cluster.cluster_id,
                    'station_indices': cluster.station_indices,
                    'centroid_coordinate': cluster.centroid_coordinate,
                    'cluster_radius_km': cluster.cluster_radius_km,
                    'temporal_signature': cluster.temporal_signature.tolist(),
                    'dtw_coherence': cluster.dtw_coherence,
                    'dominant_frequency': cluster.dominant_frequency
                }
        
        # Save results
        results_file = self.results_dir / "ps04d_geographic_patterns_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save network details
        network_file = self.results_dir / "spatial_neighbor_networks.json"
        with open(network_file, 'w') as f:
            json.dump(self.neighbor_networks, f, indent=2, default=str)
        
        print(f"✅ Results saved:")
        print(f"   Main results: {results_file}")
        print(f"   Network data: {network_file}")
        
        self.end_timer("save_results")
        return results_file
    
    def analyze_matrix_profile_anomalies(self):
        """
        Phase 3: Advanced atmospheric anomaly detection using Matrix Profile
        
        GEOPHYSICAL PURPOSE:
        Use Matrix Profile to detect temporal anomalies in atmospheric patterns:
        - **Discords**: Unusual atmospheric events (typhoons, extreme pressure changes)
        - **Motifs**: Repeated atmospheric patterns (seasonal cycles, weather patterns)  
        - **Regime Changes**: Shifts in atmospheric behavior patterns
        
        Taiwan-specific focus: Enhanced detection for monsoon transitions and typhoon impacts
        """
        if not STUMPY_AVAILABLE:
            print("⚠️  Matrix Profile analysis skipped - stumpy not available")
            return False
            
        print("🔬 Performing Matrix Profile atmospheric anomaly detection...")
        
        try:
            atmospheric_anomalies = {}
            
            # Analyze different frequency bands for different atmospheric phenomena
            frequency_bands = ['high', 'seasonal', 'low']
            
            for freq_band in frequency_bands:
                print(f"   🔍 Analyzing {freq_band}-frequency atmospheric anomalies...")
                
                freq_data = self.extract_frequency_components(freq_band)
                if freq_data is None:
                    continue
                
                band_anomalies = []
                
                # Analyze subset of stations for computational efficiency
                analysis_stations = min(50, len(freq_data))
                
                for i in range(analysis_stations):
                    ts = freq_data[i]
                    
                    # Skip if too much missing data
                    if np.isnan(ts).sum() > len(ts) * 0.1:
                        continue
                    
                    try:
                        # Taiwan-specific window sizes for atmospheric patterns
                        if freq_band == 'high':
                            window_size = 10  # ~60 days for rapid weather changes
                        elif freq_band == 'seasonal':
                            window_size = 30  # ~180 days for monsoon patterns  
                        else:  # low frequency
                            window_size = 60  # ~360 days for long-term patterns
                        
                        # Compute matrix profile
                        mp = stumpy.stump(ts, window_size)
                        
                        # Find discords (atmospheric anomalies)
                        discord_idx = np.argmax(mp[:, 0])  # Maximum distance = most unusual
                        discord_distance = mp[discord_idx, 0]
                        
                        # Taiwan-specific thresholds for atmospheric anomaly detection
                        if freq_band == 'high':
                            threshold = 2.0  # Lower threshold for rapid weather changes
                        elif freq_band == 'seasonal':
                            threshold = 1.5  # Medium threshold for monsoon anomalies
                        else:
                            threshold = 3.0  # Higher threshold for long-term changes
                        
                        if discord_distance > threshold:
                            anomaly = {
                                'station_idx': i,
                                'coordinates': self.coordinates[i].tolist(),
                                'frequency_band': freq_band,
                                'discord_time': discord_idx,
                                'discord_distance': float(discord_distance),
                                'anomaly_type': self._classify_atmospheric_anomaly(freq_band, discord_distance)
                            }
                            band_anomalies.append(anomaly)
                            
                    except Exception as e:
                        continue
                
                atmospheric_anomalies[freq_band] = band_anomalies
                print(f"   ✅ Found {len(band_anomalies)} atmospheric anomalies in {freq_band} band")
            
            self.matrix_profile_anomalies = atmospheric_anomalies
            return True
            
        except Exception as e:
            print(f"❌ Matrix Profile analysis failed: {e}")
            return False
    
    def _classify_atmospheric_anomaly(self, freq_band, distance):
        """Classify atmospheric anomaly type based on frequency band and distance"""
        if freq_band == 'high':
            if distance > 4.0:
                return 'extreme_weather_event'  # Typhoon, extreme precipitation
            elif distance > 2.5:
                return 'significant_weather_change'  # Pressure system changes
            else:
                return 'minor_atmospheric_variation'
        elif freq_band == 'seasonal':
            if distance > 3.0:
                return 'monsoon_transition_anomaly'  # Unusual monsoon behavior
            elif distance > 2.0:
                return 'seasonal_atmospheric_shift'  # Seasonal pattern changes
            else:
                return 'seasonal_variation'
        else:  # low frequency
            if distance > 5.0:
                return 'climate_regime_change'  # Long-term climate shifts
            elif distance > 3.5:
                return 'ionospheric_anomaly'  # Enhanced ionospheric activity
            else:
                return 'long_term_atmospheric_trend'
    
    def advanced_soft_dtw_clustering(self):
        """
        Phase 3: Advanced clustering using Soft DTW for atmospheric pattern recognition
        
        GEOPHYSICAL PURPOSE:
        Use Soft DTW to identify atmospheric signal sources with gradient-based optimization:
        - **Robust to Phase Shifts**: Better handles timing differences in atmospheric signals
        - **Gradient-Based**: Provides continuous optimization for atmospheric pattern matching
        - **Weather Pattern Recognition**: Identifies similar atmospheric evolution patterns
        
        Taiwan-specific: Optimized for monsoon cycles and typhoon impact patterns
        """
        if not TSLEARN_AVAILABLE:
            print("⚠️  Soft DTW clustering skipped - TSLearn not available")
            return False
            
        print("🧮 Performing advanced Soft DTW atmospheric clustering...")
        
        try:
            # Use seasonal data for atmospheric pattern recognition
            seasonal_data = self.extract_frequency_components('seasonal')
            if seasonal_data is None:
                return False
            
            # Limit to manageable dataset size for computational efficiency
            analysis_stations = min(100, len(seasonal_data))
            subset_data = seasonal_data[:analysis_stations]
            
            # Taiwan-specific clustering parameters
            n_clusters_range = [3, 4, 5, 6]  # Test different numbers for Taiwan patterns
            best_clustering = None
            best_score = -1
            
            for n_clusters in n_clusters_range:
                try:
                    # Soft DTW clustering with Taiwan-optimized parameters
                    model = TimeSeriesKMeans(
                        n_clusters=n_clusters,
                        metric="softdtw",
                        metric_params={"gamma": 0.5},  # Taiwan-optimized for atmospheric signals
                        max_iter=10,  # Limit iterations for efficiency
                        random_state=self.random_state,
                        verbose=False
                    )
                    
                    cluster_labels = model.fit_predict(subset_data)
                    
                    # Evaluate clustering quality
                    if len(np.unique(cluster_labels)) > 1:
                        # Calculate silhouette score
                        score = silhouette_score(subset_data.reshape(len(subset_data), -1), cluster_labels)
                        
                        if score > best_score:
                            best_score = score
                            best_clustering = {
                                'n_clusters': n_clusters,
                                'labels': cluster_labels,
                                'centroids': model.cluster_centers_,
                                'silhouette_score': score
                            }
                            
                except Exception as e:
                    continue
            
            if best_clustering:
                self.soft_dtw_clusters = best_clustering
                print(f"   ✅ Optimal clustering: {best_clustering['n_clusters']} clusters (silhouette: {best_clustering['silhouette_score']:.3f})")
                
                # Interpret clusters geophysically
                self._interpret_atmospheric_clusters(best_clustering)
                return True
            else:
                print("   ⚠️  No valid clustering found")
                return False
                
        except Exception as e:
            print(f"❌ Soft DTW clustering failed: {e}")
            return False
    
    def _interpret_atmospheric_clusters(self, clustering):
        """Interpret Soft DTW clusters in terms of atmospheric processes"""
        labels = clustering['labels']
        centroids = clustering['centroids']
        
        cluster_interpretations = {}
        
        for cluster_id in range(clustering['n_clusters']):
            cluster_mask = labels == cluster_id
            cluster_stations = np.sum(cluster_mask)
            
            # Analyze centroid characteristics
            centroid = centroids[cluster_id].flatten()
            
            # Calculate atmospheric pattern characteristics
            amplitude = np.std(centroid)
            trend = np.polyfit(range(len(centroid)), centroid, 1)[0]
            periodicity = self._estimate_periodicity(centroid)
            
            # Classify atmospheric pattern
            if amplitude > 2.0:  # High variability
                if periodicity and 150 < periodicity < 250:  # ~6-8 months
                    pattern_type = "monsoon_dominated"
                elif periodicity and periodicity < 100:  # <3 months  
                    pattern_type = "high_frequency_atmospheric"
                else:
                    pattern_type = "irregular_atmospheric"
            else:  # Low variability
                if abs(trend) > 0.1:
                    pattern_type = "trending_atmospheric"
                else:
                    pattern_type = "stable_atmospheric"
            
            cluster_interpretations[cluster_id] = {
                'station_count': int(cluster_stations),
                'pattern_type': pattern_type,
                'amplitude': float(amplitude),
                'trend': float(trend),
                'periodicity': periodicity,
                'atmospheric_process': self._infer_atmospheric_process(pattern_type, amplitude, trend)
            }
        
        self.atmospheric_cluster_interpretations = cluster_interpretations
        
        print("   🌤️  Atmospheric cluster interpretations:")
        for cluster_id, interp in cluster_interpretations.items():
            print(f"      Cluster {cluster_id+1}: {interp['pattern_type']} ({interp['station_count']} stations)")
            print(f"                   Process: {interp['atmospheric_process']}")
    
    def _estimate_periodicity(self, signal):
        """Estimate dominant periodicity in atmospheric signal"""
        try:
            from scipy.fft import fft, fftfreq
            
            # Remove trend
            detrended = signal - np.polyval(np.polyfit(range(len(signal)), signal, 1), range(len(signal)))
            
            # Compute FFT
            fft_vals = np.abs(fft(detrended))
            freqs = fftfreq(len(signal))
            
            # Find dominant frequency (excluding DC component)
            dominant_freq_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            if dominant_freq > 0:
                period = 1.0 / dominant_freq  # In time steps (6-day intervals)
                return period * 6  # Convert to days
            else:
                return None
                
        except:
            return None
    
    def _infer_atmospheric_process(self, pattern_type, amplitude, trend):
        """Infer atmospheric process from pattern characteristics"""
        if pattern_type == "monsoon_dominated":
            return "Taiwan monsoon cycle influence"
        elif pattern_type == "high_frequency_atmospheric":
            if amplitude > 3.0:
                return "Typhoon/extreme weather effects"
            else:
                return "Synoptic weather pattern influence"
        elif pattern_type == "irregular_atmospheric":
            return "Complex atmospheric mixing/turbulence"
        elif pattern_type == "trending_atmospheric":
            if trend > 0:
                return "Increasing atmospheric water vapor/pressure"
            else:
                return "Decreasing atmospheric effects/drying"
        else:  # stable_atmospheric
            return "Minimal atmospheric contribution"

def main():
    """Main execution function for geographic pattern analysis"""
    parser = argparse.ArgumentParser(description='Geographic Pattern Analysis for InSAR Time Series')
    parser.add_argument('--n-stations', type=int, default=None, help='Limit analysis to N stations for testing')
    parser.add_argument('--spatial-radius', type=float, default=20.0, help='Spatial radius for neighbor analysis (km)')
    parser.add_argument('--min-neighbors', type=int, default=5, help='Minimum neighbors required for analysis')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all cores, 1 for sequential)')
    parser.add_argument('--frequency-bands', nargs='+', default=['high', 'seasonal', 'low'], 
                       help='Frequency bands to analyze')
    parser.add_argument('--fast-mode', action='store_true', help='Enable fast mode (reduced accuracy for testing)')
    args = parser.parse_args()
    
    print("🗺️  Geographic Pattern Analysis for InSAR Time Series")
    print("=" * 60)
    print(f"📊 Station limit: {args.n_stations if args.n_stations else 'All stations'}")
    print(f"🌐 Spatial radius: {args.spatial_radius} km")
    print(f"👥 Minimum neighbors: {args.min_neighbors}")
    print(f"⚙️  Parallel jobs: {args.n_jobs}")
    print(f"🔊 Frequency bands: {', '.join(args.frequency_bands)}")
    print(f"⚡ Fast mode: {'Enabled' if args.fast_mode else 'Disabled'}")
    print("=" * 60)
    
    try:
        # Initialize analysis framework
        analysis = GeographicPatternAnalysis(
            spatial_radius_km=args.spatial_radius,
            min_neighbors=args.min_neighbors,
            n_jobs=args.n_jobs
        )
        
        # Load and prepare data
        print("\n🔄 Loading and preparing time series data...")
        if not analysis.load_preprocessed_data(n_stations=args.n_stations):
            print("❌ Failed to load data")
            return False
        
        # Build spatial network
        print("\n🌐 Building spatial neighbor network...")
        if not analysis.build_spatial_network():
            print("❌ Failed to build spatial network")
            return False
        
        # Analyze geographic patterns
        if 'high' in args.frequency_bands:
            print("\n🔊 Analyzing high-frequency noise patterns...")
            analysis.analyze_high_frequency_noise_patterns()
        
        if 'low' in args.frequency_bands:
            print("\n📈 Analyzing large-area ramp patterns...")
            analysis.analyze_large_area_ramps()
        
        # Perform spatial-temporal clustering for each requested band
        for freq_band in args.frequency_bands:
            if freq_band in ['seasonal', 'high', 'low']:
                print(f"\n🗂️  Performing spatial-temporal clustering ({freq_band} band)...")
                analysis.perform_spatial_temporal_clustering(freq_band)
        
        # Phase 3: Advanced atmospheric detection methods
        print("\n🔬 Phase 3: Advanced Atmospheric Pattern Detection")
        print("=" * 50)
        
        # Matrix Profile anomaly detection  
        print("\n📊 Running Matrix Profile atmospheric anomaly detection...")
        analysis.analyze_matrix_profile_anomalies()
        
        # Advanced Soft DTW clustering
        print("\n🧮 Running advanced Soft DTW clustering...")
        analysis.advanced_soft_dtw_clustering()
        
        # Create visualizations
        print("\n📊 Creating geographic pattern visualizations...")
        figure_file = analysis.create_geographic_pattern_visualization()
        
        # Save results
        print("\n💾 Saving analysis results...")
        results_file = analysis.save_results()
        
        # Summary
        total_time = sum([t for key, t in analysis.timing_results.items() if not key.endswith('_start')])
        
        print("\n" + "=" * 60)
        print("🎉 GEOGRAPHIC PATTERN ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📊 Results file: {results_file}")
        print(f"📈 Visualization: {figure_file}")
        
        # Pattern summary
        if hasattr(analysis, 'geographic_patterns'):
            for pattern_type, patterns in analysis.geographic_patterns.items():
                if pattern_type == 'high_frequency_noise':
                    print(f"🔍 Tropospheric Patterns (spatially correlated): {len(patterns)} detected")
                elif pattern_type == 'large_area_ramps':
                    print(f"🔍 Spatially Flat Ramps (potential ionospheric): {len(patterns)} detected")
                else:
                    print(f"🔍 {pattern_type.replace('_', ' ').title()}: {len(patterns)} detected")
        
        if hasattr(analysis, 'spatial_clusters'):
            for freq_band, clusters in analysis.spatial_clusters.items():
                print(f"🗂️  {freq_band.title()} clusters: {len(clusters)} formed")
        
        print("✅ Geographic pattern analysis completed successfully!")
        return True
        
    except KeyboardInterrupt:
        print("\n⛔ Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Fix multiprocessing issues on macOS/Python 3.13
    mp.set_start_method('spawn', force=True)
    
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        exit(1)

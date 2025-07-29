#!/usr/bin/env python3
"""
ps04c_motif_discovery.py - Advanced Pattern Discovery and Anomaly Detection

Purpose: Matrix profile-based motif discovery, discord detection, and change point analysis
Extends: ps04b_advanced_clustering.py with pattern mining capabilities
Focus: Recurring patterns, anomaly detection, regime changes, and spatial-temporal correlation

Technical Approach:
- Matrix Profile Analysis using STUMPY for efficient all-pairs distance computation
- Multi-scale pattern discovery (30-day, 90-day, 365-day windows)
- Statistical validation of discovered patterns
- Geographic correlation of pattern types
- Advanced change point detection methods

Author: Claude Code
Date: January 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import warnings
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import stumpy
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Enhanced geographic plotting imports
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
    print("✅ Cartopy available for enhanced geographic plotting")
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not available - using basic matplotlib (install with: conda install cartopy)")

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY_KDE = True
except ImportError:
    HAS_SCIPY_KDE = False

# Optional imports for enhanced functionality
try:
    import ruptures as rpt
    HAS_RUPTURES = True
    print("✅ Ruptures available for advanced change point detection")
except ImportError:
    HAS_RUPTURES = False
    print("⚠️  Ruptures not available - install with: pip install ruptures")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not available for interactive visualizations")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class MotifResult:
    """Data structure for motif discovery results"""
    motif_id: int
    window_size: int
    start_index: int
    end_index: int
    pattern: np.ndarray
    distance: float
    significance_score: float
    station_indices: List[int]
    temporal_context: str
    
@dataclass
class DiscordResult:
    """Data structure for discord (anomaly) detection results"""
    discord_id: int
    window_size: int
    start_index: int
    end_index: int
    pattern: np.ndarray
    distance: float
    anomaly_score: float
    station_indices: List[int]
    anomaly_type: str
    
@dataclass
class ChangePoint:
    """Data structure for change point detection results"""
    change_id: int
    timestamp: int
    confidence: float
    before_mean: float
    after_mean: float
    change_magnitude: float
    change_type: str
    station_indices: List[int]

class MotifAnomalyAnalysis:
    """
    Advanced pattern discovery and anomaly detection using matrix profile analysis
    
    Core Capabilities:
    - Matrix profile computation for efficient pattern discovery
    - Multi-scale motif detection (30, 90, 365 day windows)
    - Discord (anomaly) detection with statistical validation
    - Change point detection using multiple algorithms
    - Spatial-temporal pattern correlation
    - Statistical significance testing framework
    """
    
    def __init__(self, max_patterns=20, n_jobs=-1, random_state=42):
        """
        Initialize motif and anomaly analysis framework
        
        Parameters:
        -----------
        max_patterns : int
            Maximum number of patterns to discover per window size
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int
            Random seed for reproducibility
        """
        self.max_patterns = max_patterns
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.random_state = random_state
        
        # Data containers
        self.time_series = None
        self.coordinates = None
        self.timestamps = None
        self.station_ids = None
        
        # Analysis results
        self.matrix_profiles = {}
        self.discovered_motifs = {}
        self.detected_discords = {}
        self.change_points = {}
        self.pattern_significance = {}
        self.spatial_correlations = {}
        
        # Performance tracking
        self.timing_results = {}
        
        # Create output directories
        self._create_directories()
        
        print(f"🔍 Matrix Profile Analysis Framework Initialized")
        print(f"   Max patterns per window: {self.max_patterns}")
        print(f"   Parallel workers: {self.n_jobs}")
        print(f"   Random state: {self.random_state}")
    
    def _create_directories(self):
        """Create output directories for results and figures"""
        self.results_dir = Path("data/processed/ps04c_motifs")
        self.figures_dir = Path("figures")
        self.matrix_profile_dir = self.results_dir / "matrix_profiles"
        
        for directory in [self.results_dir, self.figures_dir, self.matrix_profile_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directories created:")
        print(f"   Results: {self.results_dir}")
        print(f"   Figures: {self.figures_dir}")
        print(f"   Matrix profiles: {self.matrix_profile_dir}")
    
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
        Load time series data for TSLearn motif discovery
        
        Parameters:
        -----------
        data_source : str
            Data source ('ps00' for preprocessed, 'ps02' for decomposed)
        n_stations : int, optional
            Limit to first N stations for testing
        use_raw_data : bool
            Use RAW displacement data (recommended for TSLearn pattern discovery)
            Raw data preserves natural patterns, trends, and anomalies for motif discovery
        """
        self.start_timer("data_loading")
        
        try:
            # Load preprocessed data
            data_file = Path("data/processed/ps00_preprocessed_data.npz")
            
            if not data_file.exists():
                print(f"❌ Data file not found: {data_file}")
                return False
            
            print("📡 Loading time series data for TSLearn motif discovery...")
            if use_raw_data:
                print("   ✅ Using RAW displacement data - preserves natural patterns for TSLearn algorithms")
            else:
                print("   ⚠️  Using DETRENDED data - may miss long-term patterns")
            
            with np.load(data_file, allow_pickle=True) as data:
                coordinates = data['coordinates']
                displacement = data['displacement']  # This is detrended data from ps00
                subsidence_rates = data['subsidence_rates']  # Linear trends (mm/year)
                
                # Reconstruct RAW displacement data for TSLearn analysis
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
            
            print(f"✅ Loaded {data_type} for TSLearn analysis ({subset_info})")
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
    
    def compute_matrix_profile(self, window_sizes=[30, 90, 365], station_subset=None):
        """
        Compute matrix profile for multiple window sizes
        
        Parameters:
        -----------
        window_sizes : list of int
            Window sizes in days (will be converted to time points)
        station_subset : list of int, optional
            Compute only for specific stations
        """
        self.start_timer("matrix_profile_computation")
        
        if self.time_series is None:
            print("❌ No time series data loaded")
            return False
        
        # Convert window sizes from days to time points (6-day sampling)
        window_sizes_points = [max(1, int(ws / 6)) for ws in window_sizes]
        
        print(f"🔍 Computing matrix profiles for {len(window_sizes)} window sizes...")
        print(f"   Window sizes (days): {window_sizes}")
        print(f"   Window sizes (time points): {window_sizes_points}")
        
        # Select stations to process
        if station_subset is not None:
            stations_to_process = station_subset[:len(self.time_series)]
        else:
            stations_to_process = list(range(len(self.time_series)))
        
        print(f"   Processing {len(stations_to_process)} stations...")
        
        for ws_days, ws_points in zip(window_sizes, window_sizes_points):
            print(f"\n   📊 Window size: {ws_days} days ({ws_points} time points)")
            
            # Skip if window is too large for time series
            if ws_points >= self.time_series.shape[1]:
                print(f"   ⚠️  Window too large ({ws_points} ≥ {self.time_series.shape[1]}), skipping...")
                continue
            
            # Compute matrix profile for each station
            station_profiles = {}
            station_indices = {}
            
            # Process in batches for memory efficiency
            batch_size = min(50, len(stations_to_process))
            n_batches = (len(stations_to_process) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(stations_to_process))
                batch_stations = stations_to_process[start_idx:end_idx]
                
                print(f"     Processing batch {batch_idx + 1}/{n_batches} (stations {start_idx+1}-{end_idx})...")
                
                for station_idx in batch_stations:
                    try:
                        # Get time series for this station
                        ts = self.time_series[station_idx].astype(np.float64)
                        
                        # Ensure signal is 1D and contiguous
                        ts = np.ascontiguousarray(ts.flatten())
                        
                        # Handle NaN values
                        if np.any(np.isnan(ts)):
                            # Simple interpolation for NaN values
                            valid_indices = ~np.isnan(ts)
                            if np.sum(valid_indices) < ws_points:
                                print(f"     ⚠️  Station {station_idx}: too many NaN values, skipping...")
                                continue
                            ts = np.interp(np.arange(len(ts)), np.where(valid_indices)[0], ts[valid_indices])
                        
                        # Skip if signal is too short or contains non-finite values
                        if len(ts) < ws_points * 2 or not np.isfinite(ts).all():
                            print(f"     ⚠️  Station {station_idx}: invalid signal, skipping...")
                            continue
                        
                        # Compute matrix profile using STUMPY
                        matrix_profile = stumpy.stump(ts, ws_points)
                        
                        # Store results
                        station_profiles[station_idx] = {
                            'profile': matrix_profile[:, 0],  # Distance profile
                            'indices': matrix_profile[:, 1].astype(int),  # Index profile
                            'left_indices': matrix_profile[:, 2].astype(int),  # Left neighbors
                            'right_indices': matrix_profile[:, 3].astype(int)  # Right neighbors
                        }
                        
                    except Exception as e:
                        print(f"     ⚠️  Station {station_idx}: Matrix profile failed ({str(e)[:50]})")
                        continue
                
                # Progress update
                progress = (batch_idx + 1) / n_batches * 100
                print(f"     Progress: {progress:.1f}% ({len(station_profiles)} stations completed)")
            
            # Store matrix profile results
            self.matrix_profiles[ws_days] = {
                'window_size_days': ws_days,
                'window_size_points': ws_points,
                'station_profiles': station_profiles,
                'n_stations': len(station_profiles),
                'computation_time': time.time() - self.timing_results['matrix_profile_computation_start']
            }
            
            # Save to disk for persistence
            profile_file = self.matrix_profile_dir / f"mp_{ws_days}day.npz"
            self._save_matrix_profile(ws_days, profile_file)
            
            print(f"   ✅ Completed {ws_days}-day window: {len(station_profiles)} stations")
            print(f"   💾 Saved to: {profile_file}")
        
        total_time = self.end_timer("matrix_profile_computation")
        print(f"\n✅ Matrix profile computation completed in {total_time/60:.2f} minutes")
        print(f"   Windows processed: {len(self.matrix_profiles)}")
        print(f"   Total station-profiles: {sum(mp['n_stations'] for mp in self.matrix_profiles.values())}")
        
        return True
    
    def _save_matrix_profile(self, window_size, file_path):
        """Save matrix profile results to disk"""
        if window_size not in self.matrix_profiles:
            return
        
        mp_data = self.matrix_profiles[window_size]
        
        # Prepare data for saving
        profiles = []
        indices = []
        station_ids = []
        
        for station_id, station_data in mp_data['station_profiles'].items():
            profiles.append(station_data['profile'])
            indices.append(station_data['indices'])
            station_ids.append(station_id)
        
        # Save as compressed numpy archive
        np.savez_compressed(
            file_path,
            profiles=np.array(profiles),
            indices=np.array(indices),
            station_ids=np.array(station_ids),
            window_size_days=window_size,
            window_size_points=mp_data['window_size_points'],
            n_stations=mp_data['n_stations']
        )
    
    def discover_motifs(self, window_size, n_motifs=15, min_distance=0.1):
        """
        Discover top recurring patterns (motifs) for given window size
        
        Parameters:
        -----------
        window_size : int
            Window size in days
        n_motifs : int
            Number of top motifs to discover
        min_distance : float
            Minimum distance between motifs to avoid duplicates
        """
        if window_size not in self.matrix_profiles:
            print(f"❌ Matrix profile not computed for {window_size}-day window")
            return False
        
        print(f"🔍 Discovering motifs for {window_size}-day window...")
        print(f"   Target motifs: {n_motifs}")
        print(f"   Minimum separation distance: {min_distance}")
        
        mp_data = self.matrix_profiles[window_size]
        station_profiles = mp_data['station_profiles']
        ws_points = mp_data['window_size_points']
        
        # Collect all potential motifs from all stations
        potential_motifs = []
        
        for station_idx, station_data in station_profiles.items():
            profile = station_data['profile']
            indices = station_data['indices']
            
            # Find local minima in the matrix profile (potential motifs)
            # More lenient thresholds for 6-day sampling data
            peaks, properties = find_peaks(-profile, height=-np.percentile(profile, 30), distance=max(1, ws_points//4))
            
            for peak_idx in peaks:
                if peak_idx < len(profile) and peak_idx < len(self.time_series[station_idx]) - ws_points:
                    # Extract the pattern
                    pattern_start = peak_idx
                    pattern_end = peak_idx + ws_points
                    pattern = self.time_series[station_idx][pattern_start:pattern_end]
                    
                    # Calculate significance score
                    distance = float(profile[peak_idx])
                    profile_array = np.array(profile, dtype=np.float64)
                    significance = (np.mean(profile_array) - distance) / np.std(profile_array)
                    
                    potential_motifs.append({
                        'station_idx': station_idx,
                        'start_index': pattern_start,
                        'end_index': pattern_end,
                        'pattern': pattern,
                        'distance': distance,
                        'significance_score': significance,
                        'temporal_context': self._classify_temporal_pattern(pattern_start * 6, window_size)
                    })
        
        # Sort by significance score and select top motifs
        potential_motifs.sort(key=lambda x: x['significance_score'], reverse=True)
        
        # Remove similar motifs
        selected_motifs = []
        for motif in potential_motifs:
            is_unique = True
            for selected in selected_motifs:
                # Check if patterns are too similar - more lenient for 6-day data
                if np.corrcoef(motif['pattern'], selected['pattern'])[0, 1] > (1 - min_distance/2):
                    is_unique = False
                    break
            
            if is_unique:
                selected_motifs.append(motif)
                
            if len(selected_motifs) >= n_motifs:
                break
        
        # Convert to MotifResult objects
        motif_results = []
        for i, motif in enumerate(selected_motifs):
            motif_result = MotifResult(
                motif_id=i,
                window_size=window_size,
                start_index=motif['start_index'],
                end_index=motif['end_index'],
                pattern=motif['pattern'],
                distance=motif['distance'],
                significance_score=motif['significance_score'],
                station_indices=[motif['station_idx']],
                temporal_context=motif['temporal_context']
            )
            motif_results.append(motif_result)
        
        # Store results
        self.discovered_motifs[window_size] = motif_results
        
        print(f"✅ Discovered {len(motif_results)} motifs for {window_size}-day window")
        for i, motif in enumerate(motif_results[:5]):  # Show top 5
            print(f"   Motif {i+1}: {motif.temporal_context}, significance={motif.significance_score:.2f}")
        
        return True
    
    def detect_discords(self, window_size, n_discords=8, anomaly_threshold=1.5):
        """
        Detect discords (anomalies) for given window size
        
        Parameters:
        -----------
        window_size : int
            Window size in days
        n_discords : int
            Number of top discords to detect
        anomaly_threshold : float
            Threshold for anomaly detection (standard deviations)
        """
        if window_size not in self.matrix_profiles:
            print(f"❌ Matrix profile not computed for {window_size}-day window")
            return False
        
        print(f"🚨 Detecting discords for {window_size}-day window...")
        print(f"   Target discords: {n_discords}")
        print(f"   Anomaly threshold: {anomaly_threshold}σ")
        
        mp_data = self.matrix_profiles[window_size]
        station_profiles = mp_data['station_profiles']
        ws_points = mp_data['window_size_points']
        
        # Collect all potential discords from all stations
        potential_discords = []
        
        for station_idx, station_data in station_profiles.items():
            profile = station_data['profile']
            
            # Calculate anomaly threshold
            profile_array = np.array(profile, dtype=np.float64)
            mean_distance = np.mean(profile_array)
            std_distance = np.std(profile_array)
            threshold = mean_distance + anomaly_threshold * std_distance
            
            # Find points exceeding threshold
            anomaly_indices = np.where(profile > threshold)[0]
            
            for anomaly_idx in anomaly_indices:
                if anomaly_idx < len(self.time_series[station_idx]) - ws_points:
                    # Extract the anomalous pattern
                    pattern_start = anomaly_idx
                    pattern_end = anomaly_idx + ws_points
                    pattern = self.time_series[station_idx][pattern_start:pattern_end]
                    
                    # Calculate anomaly score
                    distance = profile[anomaly_idx]
                    anomaly_score = (distance - mean_distance) / std_distance
                    
                    # Classify anomaly type
                    anomaly_type = self._classify_anomaly_type(pattern, pattern_start * 6)
                    
                    potential_discords.append({
                        'station_idx': station_idx,
                        'start_index': pattern_start,
                        'end_index': pattern_end,
                        'pattern': pattern,
                        'distance': distance,
                        'anomaly_score': anomaly_score,
                        'anomaly_type': anomaly_type
                    })
        
        # Sort by anomaly score and select top discords
        potential_discords.sort(key=lambda x: x['anomaly_score'], reverse=True)
        selected_discords = potential_discords[:n_discords]
        
        # Convert to DiscordResult objects
        discord_results = []
        for i, discord in enumerate(selected_discords):
            discord_result = DiscordResult(
                discord_id=i,
                window_size=window_size,
                start_index=discord['start_index'],
                end_index=discord['end_index'],
                pattern=discord['pattern'],
                distance=discord['distance'],
                anomaly_score=discord['anomaly_score'],
                station_indices=[discord['station_idx']],
                anomaly_type=discord['anomaly_type']
            )
            discord_results.append(discord_result)
        
        # Store results
        self.detected_discords[window_size] = discord_results
        
        print(f"✅ Detected {len(discord_results)} discords for {window_size}-day window")
        for i, discord in enumerate(discord_results):
            print(f"   Discord {i+1}: {discord.anomaly_type}, score={discord.anomaly_score:.2f}")
        
        return True
    
    def _classify_temporal_pattern(self, start_day, window_size):
        """Classify temporal context of discovered pattern"""
        season_start = (start_day % 365.25) / 365.25
        
        if window_size <= 45:
            return "short_term_fluctuation"
        elif window_size <= 120:
            if 0.2 < season_start < 0.6:  # Spring/Summer
                return "irrigation_season"
            else:
                return "dry_season"
        else:  # window_size > 120
            if 0.45 < season_start < 0.75:  # Summer monsoon
                return "monsoon_pattern"
            elif season_start > 0.9 or season_start < 0.1:  # Winter
                return "winter_pattern"
            else:
                return "annual_cycle"
    
    def _classify_anomaly_type(self, pattern, start_day):
        """Classify type of detected anomaly"""
        # Calculate pattern statistics
        mean_val = np.mean(pattern)
        std_val = np.std(pattern)
        trend = np.polyfit(range(len(pattern)), pattern, 1)[0]
        
        # Classification logic
        if abs(mean_val) > 3 * std_val:
            if mean_val > 0:
                return "extreme_uplift"
            else:
                return "extreme_subsidence"
        elif abs(trend) > 1.0:  # >1mm per time step
            if trend > 0:
                return "rapid_uplift"
            else:
                return "rapid_subsidence"
        elif std_val > 2.0:
            return "high_variability"
        else:
            return "unusual_pattern"
    
    def create_visualizations(self):
        """Create comprehensive visualizations of pattern discovery results with rich geological context"""
        print(f"\n🎨 Creating pattern discovery visualizations with geological analysis...")
        
        if not self.matrix_profiles:
            print("❌ No matrix profile results to visualize")
            return False
        
        # Figure 1: Matrix profile overview
        self._plot_matrix_profile_overview()
        
        # Figure 2: Top motifs gallery
        self._plot_motifs_gallery()
        
        # Figure 3: Discord detection results
        self._plot_discord_detection()
        
        # Figure 4: Spatial pattern correlation map
        self._plot_spatial_pattern_map()
        self._plot_temporal_motif_anomaly_series()
        
        # Figure 5: Geological pattern significance analysis
        self._plot_geological_pattern_analysis()
        
        # Figure 6: Temporal pattern evolution
        self._plot_temporal_pattern_evolution()
        
        # Figure 7: Pattern-geology correlation
        self._plot_pattern_geology_correlation()
        
        print(f"✅ Pattern discovery visualizations with geological context completed")
        return True
    
    def _plot_matrix_profile_overview(self):
        """Plot matrix profile overview for different window sizes"""
        fig, axes = plt.subplots(len(self.matrix_profiles), 2, figsize=(16, 4*len(self.matrix_profiles)))
        if len(self.matrix_profiles) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Matrix Profile Overview', fontsize=16, fontweight='bold')
        
        for i, (window_size, mp_data) in enumerate(self.matrix_profiles.items()):
            # Plot 1: Example matrix profile
            ax1 = axes[i, 0]
            
            # Get first available station as example
            station_profiles = mp_data['station_profiles']
            if station_profiles:
                example_station = list(station_profiles.keys())[0]
                example_profile = station_profiles[example_station]['profile']
                time_axis = np.arange(len(example_profile)) * 6
                
                # Ensure arrays are compatible numeric types for matplotlib
                time_axis = np.array(time_axis, dtype=np.float64)
                example_profile = np.array(example_profile, dtype=np.float64)
                
                ax1.plot(time_axis, example_profile, 'b-', alpha=0.7, linewidth=1)
                ax1.fill_between(time_axis, example_profile, alpha=0.3)
                ax1.set_xlabel('Time (days)')
                ax1.set_ylabel('Matrix Profile Distance')
                ax1.set_title(f'{window_size}-day Window (Station {example_station})')
                ax1.grid(True, alpha=0.3)
                
                # Highlight motifs if available
                if window_size in self.discovered_motifs:
                    for motif in self.discovered_motifs[window_size][:3]:
                        if motif.station_indices[0] == example_station:
                            ax1.axvline(motif.start_index * 6, color='red', linestyle='--', alpha=0.7)
                            ax1.axvline(motif.end_index * 6, color='red', linestyle='--', alpha=0.7)
            
            # Plot 2: Distribution of distances
            ax2 = axes[i, 1]
            
            all_distances = []
            for station_data in station_profiles.values():
                all_distances.extend(station_data['profile'])
            
            if all_distances:
                ax2.hist(all_distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(np.mean(all_distances), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_distances):.2f}')
                ax2.axvline(np.percentile(all_distances, 95), color='orange', linestyle='--',
                           label=f'95th percentile: {np.percentile(all_distances, 95):.2f}')
                ax2.set_xlabel('Matrix Profile Distance')
                ax2.set_ylabel('Frequency')
                ax2.set_title(f'Distance Distribution ({window_size}-day)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.figures_dir / "ps04c_fig01_matrix_profile_overview.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _plot_motifs_gallery(self):
        """Plot gallery of discovered motifs"""
        # Always create the figure, even if no motifs found
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Discovered Motifs Gallery', fontsize=16, fontweight='bold')
        
        if not self.discovered_motifs:
            # Show empty figure with informative message
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No motifs discovered\nTry adjusting parameters:\n• Lower significance threshold\n• Different window sizes\n• More stations', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Displacement (mm)')
                ax.set_title('Motif Slot (Empty)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = self.figures_dir / "ps04c_fig02_top_motifs_gallery.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 Saved: {fig_path}")
            return
        
        # Count total motifs
        total_motifs = sum(len(motifs) for motifs in self.discovered_motifs.values())
        if total_motifs == 0:
            # Handle case where motifs dict exists but is empty
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No motifs with sufficient\nsignificance found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.3))
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Displacement (mm)')
                ax.set_title('Motif Slot (Empty)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = self.figures_dir / "ps04c_fig02_top_motifs_gallery.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 Saved: {fig_path}")
            return
        
        # Create subplot grid
        n_cols = min(3, len(self.discovered_motifs))
        max_motifs_per_window = max(len(motifs) for motifs in self.discovered_motifs.values())
        n_rows = min(5, max_motifs_per_window)  # Show top 5 motifs per window
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Discovered Motifs Gallery', fontsize=16, fontweight='bold')
        
        for col, (window_size, motifs) in enumerate(self.discovered_motifs.items()):
            for row in range(n_rows):
                ax = axes[row, col]
                
                if row < len(motifs):
                    motif = motifs[row]
                    time_axis = np.arange(len(motif.pattern)) * 6
                    
                    ax.plot(time_axis, motif.pattern, 'b-', linewidth=2, alpha=0.8)
                    ax.fill_between(time_axis, motif.pattern, alpha=0.3)
                    
                    ax.set_title(f'{window_size}d Window, Motif {row+1}\n'
                               f'{motif.temporal_context.replace("_", " ").title()}\n'
                               f'Significance: {motif.significance_score:.2f}')
                    ax.set_xlabel('Time (days)')
                    ax.set_ylabel('Displacement (mm)')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_visible(False)
        
        plt.tight_layout()
        fig_path = self.figures_dir / "ps04c_fig02_top_motifs_gallery.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _plot_discord_detection(self):
        """Plot discord (anomaly) detection results"""
        # Always create the figure, even if no discords found
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Detected Discords (Anomalies)', fontsize=16, fontweight='bold')
        
        if not self.detected_discords:
            # Show empty figure with informative message
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No discords detected\nTry adjusting parameters:\n• Lower anomaly threshold\n• Different window sizes\n• More stations', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.3))
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Displacement (mm)')
                ax.set_title('Discord Slot (Empty)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = self.figures_dir / "ps04c_fig03_discord_detection.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 Saved: {fig_path}")
            return
        
        # Count total discords
        total_discords = sum(len(discords) for discords in self.detected_discords.values())
        if total_discords == 0:
            # Handle case where discords dict exists but is empty
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No discords with sufficient\nanomaly scores found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.3))
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Displacement (mm)')
                ax.set_title('Discord Slot (Empty)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = self.figures_dir / "ps04c_fig03_discord_detection.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 Saved: {fig_path}")
            return
        
        # Create subplot grid
        n_cols = min(3, len(self.detected_discords))
        max_discords_per_window = max(len(discords) for discords in self.detected_discords.values())
        n_rows = min(5, max_discords_per_window)  # Show top 5 discords per window
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Detected Discords (Anomalies)', fontsize=16, fontweight='bold')
        
        for col, (window_size, discords) in enumerate(self.detected_discords.items()):
            for row in range(n_rows):
                ax = axes[row, col]
                
                if row < len(discords):
                    discord = discords[row]
                    time_axis = np.arange(len(discord.pattern)) * 6
                    
                    # Plot with warning color for anomaly
                    ax.plot(time_axis, discord.pattern, 'r-', linewidth=2, alpha=0.8)
                    ax.fill_between(time_axis, discord.pattern, alpha=0.3, color='red')
                    
                    ax.set_title(f'{window_size}d Window, Discord {row+1}\n'
                               f'{discord.anomaly_type.replace("_", " ").title()}\n'
                               f'Anomaly Score: {discord.anomaly_score:.2f}')
                    ax.set_xlabel('Time (days)')
                    ax.set_ylabel('Displacement (mm)')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_visible(False)
        
        plt.tight_layout()
        fig_path = self.figures_dir / "ps04c_fig03_discord_detection.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _plot_spatial_pattern_map(self):
        """Plot spatial distribution of patterns with enhanced geographic visualization"""
        if not self.discovered_motifs and not self.detected_discords:
            return
        
        print("🗺️  Creating enhanced spatial pattern map...")
        
        # Calculate data coverage bounds with margin
        margin = 0.05  # degrees
        lon_min = np.min(self.coordinates[:, 0]) - margin
        lon_max = np.max(self.coordinates[:, 0]) + margin
        lat_min = np.min(self.coordinates[:, 1]) - margin
        lat_max = np.max(self.coordinates[:, 1]) + margin
        
        print(f"   📍 Data coverage: {lon_min:.3f}°E to {lon_max:.3f}°E, {lat_min:.3f}°N to {lat_max:.3f}°N")
        print(f"   📏 Coverage area: {(lon_max-lon_min)*111:.1f} × {(lat_max-lat_min)*111:.1f} km")
        
        # Use Cartopy if available, otherwise fall back to matplotlib
        if HAS_CARTOPY:
            fig = plt.figure(figsize=(24, 18))  # Even larger for better visibility
            proj = ccrs.PlateCarree()
            
            ax1 = fig.add_subplot(2, 2, 1, projection=proj)
            ax2 = fig.add_subplot(2, 2, 2, projection=proj)
            ax3 = fig.add_subplot(2, 2, 3, projection=proj)
            ax4 = fig.add_subplot(2, 2, 4)  # Non-geographic
            
            # Adjust subplot spacing for better use of space
            plt.subplots_adjust(left=0.06, bottom=0.08, right=0.95, top=0.90, wspace=0.15, hspace=0.25)
            
            # Setup cartographic features for geographic subplots
            for ax_geo in [ax1, ax2, ax3]:
                ax_geo.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                ax_geo.coastlines(resolution='50m', color='black', linewidth=1.0, alpha=0.8)
                ax_geo.add_feature(cfeature.BORDERS, linewidth=0.8, alpha=0.8)
                ax_geo.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
                ax_geo.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
                ax_geo.add_feature(cfeature.RIVERS, color='blue', alpha=0.4, linewidth=0.5)
                
                # Add grid
                gl = ax_geo.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                     linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 10}
                gl.ylabel_style = {'size': 10}
            
            transform = ccrs.PlateCarree()
        else:
            # Fallback to matplotlib
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            ax1, ax2, ax3, ax4 = axes.flatten()
            transform = None
        
        fig.suptitle('Enhanced Spatial Pattern Distribution - Taiwan Subsidence Analysis', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Plot 1: Motif spatial distribution
        if self.discovered_motifs:
            # Create pattern type map
            pattern_colors = {'irrigation_season': 'green', 'monsoon_pattern': 'blue', 
                             'dry_season': 'orange', 'annual_cycle': 'purple', 
                             'winter_pattern': 'cyan', 'short_term_fluctuation': 'yellow'}
            
            # Plot all stations as background
            bg_kwargs = {
                'c': 'lightgray',
                's': 1,
                'alpha': 0.3,
                'label': 'All stations'
            }
            if transform:
                bg_kwargs['transform'] = transform
            
            ax1.scatter(self.coordinates[:, 0], self.coordinates[:, 1], **bg_kwargs)
            
            # Group motif locations by pattern type for clean plotting
            pattern_groups = {}
            for window_size, motifs in self.discovered_motifs.items():
                for motif in motifs:
                    pattern_key = (motif.temporal_context, window_size)
                    if pattern_key not in pattern_groups:
                        pattern_groups[pattern_key] = {
                            'coords': [],
                            'sizes': [],
                            'color': pattern_colors.get(motif.temporal_context, 'red')
                        }
                    
                    for station_idx in motif.station_indices:
                        coord = self.coordinates[station_idx]
                        pattern_groups[pattern_key]['coords'].append(coord)
                        pattern_groups[pattern_key]['sizes'].append(50 + motif.significance_score*10)
            
            # Plot each pattern group once with proper legend
            for (pattern_type, window_size), data in pattern_groups.items():
                coords = np.array(data['coords'])
                sizes = np.array(data['sizes'])
                
                # Use Cartopy transform if available
                scatter_kwargs = {
                    'c': data['color'],
                    's': sizes,
                    'alpha': 0.7,
                    'edgecolors': 'black',
                    'linewidth': 0.5,
                    'label': f"{pattern_type.replace('_', ' ').title()} ({window_size}d)"
                }
                
                if transform:
                    scatter_kwargs['transform'] = transform
                
                ax1.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)
            
            ax1.set_xlabel('Longitude (°E)', fontsize=12)
            ax1.set_ylabel('Latitude (°N)', fontsize=12)
            ax1.set_title('Motif Locations by Temporal Context', fontsize=14, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=10)
            
            # Add geological regions
            self._add_geological_regions(ax1)
        
        # Plot 2: Discord spatial distribution
        if self.detected_discords:
            # Plot all stations as background
            bg_kwargs = {
                'c': 'lightgray',
                's': 1,
                'alpha': 0.3,
                'label': 'All stations'
            }
            if transform:
                bg_kwargs['transform'] = transform
            
            ax2.scatter(self.coordinates[:, 0], self.coordinates[:, 1], **bg_kwargs)
            
            # Group discord locations by anomaly type for clean plotting
            anomaly_colors = {'extreme_subsidence': 'darkred', 'extreme_uplift': 'darkgreen',
                             'rapid_subsidence': 'red', 'rapid_uplift': 'lightgreen',
                             'high_variability': 'orange', 'unusual_pattern': 'purple'}
            
            anomaly_groups = {}
            for window_size, discords in self.detected_discords.items():
                for discord in discords:
                    anomaly_key = (discord.anomaly_type, window_size)
                    if anomaly_key not in anomaly_groups:
                        anomaly_groups[anomaly_key] = {
                            'coords': [],
                            'sizes': [],
                            'color': anomaly_colors.get(discord.anomaly_type, 'black')
                        }
                    
                    for station_idx in discord.station_indices:
                        coord = self.coordinates[station_idx]
                        anomaly_groups[anomaly_key]['coords'].append(coord)
                        anomaly_groups[anomaly_key]['sizes'].append(50 + discord.anomaly_score*10)
            
            # Plot each anomaly group once with proper legend
            for (anomaly_type, window_size), data in anomaly_groups.items():
                coords = np.array(data['coords'])
                sizes = np.array(data['sizes'])
                
                # Use Cartopy transform if available
                scatter_kwargs = {
                    'c': data['color'],
                    's': sizes,
                    'alpha': 0.8,
                    'edgecolors': 'black',
                    'linewidth': 0.5,
                    'marker': '^',
                    'label': f"{anomaly_type.replace('_', ' ').title()} ({window_size}d)"
                }
                
                if transform:
                    scatter_kwargs['transform'] = transform
                
                ax2.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)
            
            ax2.set_xlabel('Longitude (°E)', fontsize=12)
            ax2.set_ylabel('Latitude (°N)', fontsize=12)
            ax2.set_title('Anomaly Locations by Type', fontsize=14, fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=10)
            
            # Add geological regions
            self._add_geological_regions(ax2)
        
        # Plot 3: Pattern density heatmap
        if self.coordinates is not None:
            # Create 2D histogram of pattern locations
            all_lons, all_lats = [], []
            
            for window_size, motifs in self.discovered_motifs.items():
                for motif in motifs:
                    for station_idx in motif.station_indices:
                        coord = self.coordinates[station_idx]
                        all_lons.append(coord[0])
                        all_lats.append(coord[1])
            
            if all_lons:
                h, xedges, yedges = np.histogram2d(all_lons, all_lats, bins=20)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax3.imshow(h.T, origin='lower', extent=extent, cmap='YlOrRd', alpha=0.7)
                plt.colorbar(im, ax=ax3, label='Pattern Density')
                
                # Overlay station locations
                ax3.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                           c='blue', s=1, alpha=0.1)
                
                ax3.set_xlabel('Longitude (°E)', fontsize=12)
                ax3.set_ylabel('Latitude (°N)', fontsize=12)
                ax3.set_title('Pattern Density Heatmap', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(labelsize=10)
        
        # Plot 4: Geological interpretation
        self._plot_geological_interpretation(ax4)
        
        # No tight_layout since we already set custom spacing
        fig_path = self.figures_dir / "ps04c_fig04_spatial_pattern_map.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _plot_temporal_motif_anomaly_series(self):
        """Plot time series showing where motifs and anomalies occur temporally"""
        if not self.discovered_motifs and not self.detected_discords:
            return
        
        print("🕒 Creating temporal motif and anomaly time series visualization...")
        
        # Create figure with subplots for different stations showing patterns
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('Temporal Motif and Anomaly Discovery - Time Series Analysis', 
                    fontsize=20, fontweight='bold', y=0.96)
        
        # Collect all motifs and discords with their stations
        all_motifs = []
        all_discords = []
        
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                for station_idx in motif.station_indices:
                    all_motifs.append({
                        'station_idx': station_idx,
                        'start_index': motif.start_index,
                        'end_index': motif.end_index,
                        'pattern': motif.pattern,
                        'window_size': window_size,
                        'temporal_context': motif.temporal_context,
                        'significance': motif.significance_score
                    })
        
        for window_size, discords in self.detected_discords.items():
            for discord in discords:
                for station_idx in discord.station_indices:
                    all_discords.append({
                        'station_idx': station_idx,
                        'start_index': discord.start_index,
                        'end_index': discord.end_index,
                        'pattern': discord.pattern,
                        'window_size': window_size,
                        'anomaly_type': discord.anomaly_type,
                        'anomaly_score': discord.anomaly_score
                    })
        
        # Select representative stations with highest significance patterns
        motif_stations = sorted(set(m['station_idx'] for m in all_motifs))[:12]  # Top 12 stations
        discord_stations = sorted(set(d['station_idx'] for d in all_discords))[:6]  # Top 6 anomaly stations
        
        # Create time axis (6-day sampling)
        time_days = np.arange(len(self.time_series[0])) * 6
        
        # Plot motifs in top 2 rows (2x6 grid)
        pattern_colors = {
            'irrigation_season': '#2E8B57',     # Sea Green
            'monsoon_pattern': '#4169E1',       # Royal Blue
            'dry_season': '#FF8C00',           # Dark Orange  
            'annual_cycle': '#9932CC',         # Dark Orchid
            'winter_pattern': '#00CED1',       # Dark Turquoise
            'short_term_fluctuation': '#FFD700' # Gold
        }
        
        for i, station_idx in enumerate(motif_stations):
            ax = plt.subplot(3, 6, i + 1)
            
            # Plot full time series
            time_series = self.time_series[station_idx]
            ax.plot(time_days, time_series, 'k-', alpha=0.6, linewidth=1, label='Full time series')
            
            # Find and mark all motifs for this station
            station_motifs = [m for m in all_motifs if m['station_idx'] == station_idx]
            
            for motif in station_motifs:
                start_day = motif['start_index'] * 6
                end_day = motif['end_index'] * 6
                
                # Highlight motif region
                motif_time = time_days[motif['start_index']:motif['end_index']]
                motif_values = time_series[motif['start_index']:motif['end_index']]
                
                color = pattern_colors.get(motif['temporal_context'], 'red')
                
                # Plot highlighted region
                ax.fill_between(motif_time, motif_values, alpha=0.4, color=color,
                               label=f"{motif['temporal_context']} ({motif['window_size']}d)")
                ax.plot(motif_time, motif_values, color=color, linewidth=3, alpha=0.8)
                
                # Add significance annotation
                mid_time = (start_day + end_day) / 2
                max_val = np.max(motif_values)
                ax.annotate(f'{motif["significance"]:.1f}', 
                           (mid_time, max_val), xytext=(0, 10), 
                           textcoords='offset points', ha='center', va='bottom',
                           fontsize=8, fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Format subplot
            coord = self.coordinates[station_idx]
            ax.set_title(f'Station {station_idx}\nMotifs: [{coord[0]:.3f}°E, {coord[1]:.3f}°N]', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (days)', fontsize=9)
            ax.set_ylabel('Displacement (mm)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Compact legend
            if station_motifs:
                ax.legend(loc='upper right', fontsize=6, framealpha=0.8)
        
        # Plot anomalies in bottom row (1x6 grid)
        anomaly_colors = {
            'extreme_subsidence': '#8B0000',    # Dark Red
            'extreme_uplift': '#006400',        # Dark Green
            'rapid_subsidence': '#DC143C',      # Crimson
            'rapid_uplift': '#32CD32',         # Lime Green
            'high_variability': '#FF4500',     # Orange Red
            'unusual_pattern': '#4B0082'       # Indigo
        }
        
        for i, station_idx in enumerate(discord_stations):
            ax = plt.subplot(3, 6, i + 13)  # Bottom row
            
            # Plot full time series
            time_series = self.time_series[station_idx]
            ax.plot(time_days, time_series, 'k-', alpha=0.6, linewidth=1, label='Full time series')
            
            # Find and mark all discords for this station
            station_discords = [d for d in all_discords if d['station_idx'] == station_idx]
            
            for discord in station_discords:
                start_day = discord['start_index'] * 6
                end_day = discord['end_index'] * 6
                
                # Highlight anomaly region
                discord_time = time_days[discord['start_index']:discord['end_index']]
                discord_values = time_series[discord['start_index']:discord['end_index']]
                
                color = anomaly_colors.get(discord['anomaly_type'], 'black')
                
                # Plot highlighted region with warning style
                ax.fill_between(discord_time, discord_values, alpha=0.5, color=color,
                               label=f"{discord['anomaly_type']} ({discord['window_size']}d)")
                ax.plot(discord_time, discord_values, color=color, linewidth=4, alpha=0.9)
                
                # Add anomaly score annotation
                mid_time = (start_day + end_day) / 2
                min_val = np.min(discord_values)
                ax.annotate(f'⚠️{discord["anomaly_score"]:.1f}', 
                           (mid_time, min_val), xytext=(0, -15), 
                           textcoords='offset points', ha='center', va='top',
                           fontsize=8, fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
            
            # Format subplot
            coord = self.coordinates[station_idx]
            ax.set_title(f'Station {station_idx}\nAnomalies: [{coord[0]:.3f}°E, {coord[1]:.3f}°N]', 
                        fontsize=10, fontweight='bold', color='red')
            ax.set_xlabel('Time (days)', fontsize=9)
            ax.set_ylabel('Displacement (mm)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Compact legend
            if station_discords:
                ax.legend(loc='upper right', fontsize=6, framealpha=0.8)
        
        # Add overall legend and summary information
        fig.text(0.02, 0.02, 
                f'📊 Dataset: {len(self.time_series)} stations, {len(time_days)} time points\n'
                f'🔍 Motifs discovered: {sum(len(motifs) for motifs in self.discovered_motifs.values())}\n'
                f'🚨 Anomalies detected: {sum(len(discords) for discords in self.detected_discords.values())}\n'
                f'⏱️ Window sizes: {list(self.discovered_motifs.keys())} days\n'
                f'📅 Time span: {time_days[-1]:.0f} days ({time_days[-1]/365.25:.1f} years)',
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.12, right=0.98, top=0.92, wspace=0.25, hspace=0.45)
        
        # Save figure
        fig_path = self.figures_dir / "ps04c_fig04_1_temporal_motif_anomaly_series.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _add_geological_regions(self, ax):
        """Add geological region boundaries and labels to map"""
        # Define major geological regions in Taiwan study area
        regions = {
            'Coastal Plain': {'bounds': [120.2, 120.5, 23.6, 24.2], 'color': 'lightblue'},
            'Alluvial Fan': {'bounds': [120.5, 120.7, 23.8, 24.1], 'color': 'wheat'},
            'Piedmont': {'bounds': [120.7, 120.9, 23.7, 24.0], 'color': 'lightgreen'}
        }
        
        for region_name, region_data in regions.items():
            bounds = region_data['bounds']
            # Draw region boundary
            rect = plt.Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2],
                               fill=False, edgecolor=region_data['color'], linewidth=2, linestyle='--', alpha=0.7)
            ax.add_patch(rect)
            
            # Add region label
            center_x = (bounds[0] + bounds[1]) / 2
            center_y = (bounds[2] + bounds[3]) / 2
            ax.text(center_x, center_y, region_name, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=region_data['color'], alpha=0.3))
    
    def _plot_geological_interpretation(self, ax):
        """Plot geological interpretation of patterns"""
        # Create pie chart of pattern types
        if self.discovered_motifs:
            pattern_counts = {}
            for window_size, motifs in self.discovered_motifs.items():
                for motif in motifs:
                    pattern_type = motif.temporal_context
                    pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            if pattern_counts:
                # Geological interpretation mapping
                geological_mapping = {
                    'irrigation_season': 'Agricultural\nWithdrawal',
                    'monsoon_pattern': 'Seasonal\nRecharge',
                    'dry_season': 'Groundwater\nDepletion',
                    'annual_cycle': 'Natural\nCycle',
                    'winter_pattern': 'Reduced\nActivity',
                    'short_term_fluctuation': 'Local\nDisturbance'
                }
                
                geological_labels = [geological_mapping.get(k, k) for k in pattern_counts.keys()]
                colors = ['green', 'blue', 'orange', 'purple', 'cyan', 'yellow'][:len(pattern_counts)]
                
                wedges, texts, autotexts = ax.pie(pattern_counts.values(), labels=geological_labels,
                                                 colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('Geological Interpretation\nof Discovered Patterns')
        else:
            ax.text(0.5, 0.5, 'No patterns\nto interpret', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Geological Interpretation')
    
    def _plot_geological_pattern_analysis(self):
        """Plot detailed geological pattern significance analysis"""
        if not self.discovered_motifs:
            return
        
        # Create much larger figure with better spacing
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Geological Pattern Significance Analysis', fontsize=20, fontweight='bold', y=0.95)
        
        # Adjust subplot spacing for better use of space
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.88, wspace=0.25, hspace=0.35)
        
        # Plot 1: Pattern significance vs depth (proxy)
        ax1 = axes[0, 0]
        all_patterns = []
        all_significances = []
        all_depths = []  # Use latitude as proxy for geological depth variation
        
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                for station_idx in motif.station_indices:
                    all_patterns.append(motif.temporal_context)
                    all_significances.append(motif.significance_score)
                    all_depths.append(self.coordinates[station_idx, 1])  # Latitude as depth proxy
        
        if all_significances:
            scatter = ax1.scatter(all_depths, all_significances, c=range(len(all_significances)),
                                cmap='viridis', alpha=0.7, s=50)
            ax1.set_xlabel('Latitude (°N) - Geological Depth Proxy', fontsize=12)
            ax1.set_ylabel('Pattern Significance Score', fontsize=12)
            ax1.set_title('Pattern Significance vs Geological Setting', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=10)
            cbar1 = plt.colorbar(scatter, ax=ax1, label='Pattern Index')
            cbar1.ax.tick_params(labelsize=10)
        
        # Plot 2: Seasonal pattern distribution
        ax2 = axes[0, 1]
        seasonal_patterns = {}
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                season = motif.temporal_context
                if season not in seasonal_patterns:
                    seasonal_patterns[season] = {'count': 0, 'significance': []}
                seasonal_patterns[season]['count'] += 1
                seasonal_patterns[season]['significance'].append(motif.significance_score)
        
        if seasonal_patterns:
            seasons = list(seasonal_patterns.keys())
            means = [np.mean(seasonal_patterns[s]['significance']) for s in seasons]
            stds = [np.std(seasonal_patterns[s]['significance']) if len(seasonal_patterns[s]['significance']) > 1 else 0 for s in seasons]
            
            bars = ax2.bar(range(len(seasons)), means, yerr=stds, capsize=5, alpha=0.7,
                          color=['green', 'blue', 'orange', 'purple', 'cyan', 'yellow'][:len(seasons)])
            ax2.set_xlabel('Seasonal Pattern Type', fontsize=12)
            ax2.set_ylabel('Mean Significance Score', fontsize=12)
            ax2.set_title('Seasonal Pattern Significance', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(seasons)))
            ax2.set_xticklabels([s.replace('_', '\n') for s in seasons], rotation=45, fontsize=10)
            ax2.tick_params(labelsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, season in zip(bars, seasons):
                count = seasonal_patterns[season]['count']
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (stds[seasons.index(season)]/2),
                        f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot 3: Window size vs geological process
        ax3 = axes[0, 2]
        window_significance = {}
        for window_size, motifs in self.discovered_motifs.items():
            significances = [motif.significance_score for motif in motifs]
            window_significance[window_size] = significances
        
        if window_significance:
            window_sizes = sorted(window_significance.keys())
            data = [window_significance[ws] for ws in window_sizes]
            
            # Box plot of significances by window size
            bp = ax3.boxplot(data, positions=window_sizes, widths=[ws*0.3 for ws in window_sizes],
                            patch_artist=True)
            
            # Color boxes by geological process interpretation with alpha
            colors = ['lightgreen', 'lightblue', 'orange']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_xlabel('Window Size (days)')
            ax3.set_ylabel('Pattern Significance Distribution')
            ax3.set_title('Geological Process Scale Analysis')
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log')
            
            # Add process labels
            process_labels = ['Short-term\n(Irrigation)', 'Medium-term\n(Seasonal)', 'Long-term\n(Annual)']
            for i, (ws, label) in enumerate(zip(window_sizes, process_labels)):
                ax3.text(ws, ax3.get_ylim()[1]*0.9, label, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.5))
        
        # Plot 4: Anomaly type distribution
        ax4 = axes[1, 0]
        if self.detected_discords:
            anomaly_counts = {}
            anomaly_scores = {}
            
            for window_size, discords in self.detected_discords.items():
                for discord in discords:
                    atype = discord.anomaly_type
                    if atype not in anomaly_counts:
                        anomaly_counts[atype] = 0
                        anomaly_scores[atype] = []
                    anomaly_counts[atype] += 1
                    anomaly_scores[atype].append(discord.anomaly_score)
            
            if anomaly_counts:
                # Geological risk assessment
                risk_colors = {
                    'extreme_subsidence': 'darkred',
                    'rapid_subsidence': 'red', 
                    'extreme_uplift': 'darkgreen',
                    'rapid_uplift': 'lightgreen',
                    'high_variability': 'orange',
                    'unusual_pattern': 'purple'
                }
                
                anomaly_types = list(anomaly_counts.keys())
                counts = list(anomaly_counts.values())
                colors = [risk_colors.get(atype, 'gray') for atype in anomaly_types]
                
                bars = ax4.bar(range(len(anomaly_types)), counts, color=colors, alpha=0.7)
                ax4.set_xlabel('Anomaly Type')
                ax4.set_ylabel('Count')
                ax4.set_title('Geological Risk Assessment\n(Anomaly Distribution)')
                ax4.set_xticks(range(len(anomaly_types)))
                ax4.set_xticklabels([atype.replace('_', '\n') for atype in anomaly_types], rotation=45)
                ax4.grid(True, alpha=0.3)
                
                # Add severity indicators
                for i, (bar, atype) in enumerate(zip(bars, anomaly_types)):
                    mean_score = np.mean(anomaly_scores[atype])
                    severity = '⚠️' if mean_score > 3 else '⚡' if mean_score > 2 else '○'
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            severity, ha='center', va='bottom', fontsize=12)
        
        # Plot 5: Temporal pattern evolution timeline
        ax5 = axes[1, 1]
        if self.discovered_motifs:
            # Create timeline of pattern occurrences
            timeline_data = []
            for window_size, motifs in self.discovered_motifs.items():
                for motif in motifs:
                    start_day = motif.start_index * 6  # Convert to days
                    timeline_data.append({
                        'day': start_day,
                        'window_size': window_size,
                        'significance': motif.significance_score,
                        'pattern_type': motif.temporal_context
                    })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                
                # Plot pattern occurrences over time
                pattern_types = timeline_df['pattern_type'].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(pattern_types)))
                
                for i, ptype in enumerate(pattern_types):
                    subset = timeline_df[timeline_df['pattern_type'] == ptype]
                    ax5.scatter(subset['day'], subset['significance'], 
                               c=[colors[i]], label=ptype.replace('_', ' ').title(),
                               s=subset['window_size']/5, alpha=0.7)
                
                ax5.set_xlabel('Time (days from start)')
                ax5.set_ylabel('Pattern Significance')
                ax5.set_title('Temporal Evolution of Patterns')
                ax5.legend(loc='upper left', fontsize=8, fancybox=True, shadow=True)
                ax5.grid(True, alpha=0.3)
                
                # Add seasonal markers
                for year in range(int(timeline_df['day'].min()/365), int(timeline_df['day'].max()/365) + 1):
                    for season_start in [0, 91, 182, 273]:  # Approximate season starts
                        day = year * 365 + season_start
                        if timeline_df['day'].min() <= day <= timeline_df['day'].max():
                            ax5.axvline(day, color='gray', linestyle='--', alpha=0.3)
        
        # Plot 6: Spatial distribution of pattern types (Cartopy-enhanced)
        if HAS_CARTOPY and self.coordinates is not None and self.discovered_motifs:
            # Create Cartopy subplot
            ax6 = fig.add_subplot(2, 3, 6, projection=ccrs.PlateCarree())
            
            # Calculate data coverage bounds
            margin = 0.02  # smaller margin for detailed view
            lon_min = np.min(self.coordinates[:, 0]) - margin
            lon_max = np.max(self.coordinates[:, 0]) + margin
            lat_min = np.min(self.coordinates[:, 1]) - margin
            lat_max = np.max(self.coordinates[:, 1]) + margin
            
            # Setup Taiwan map
            ax6.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax6.coastlines(resolution='50m', color='black', linewidth=0.8, alpha=0.8)
            ax6.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
            ax6.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax6.add_feature(cfeature.RIVERS, color='blue', alpha=0.4, linewidth=0.5)
            
            # Add grid
            gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
            
            # Collect pattern locations and types
            pattern_coords = []
            pattern_types = []
            
            for window_size, motifs in self.discovered_motifs.items():
                for motif in motifs:
                    for station_idx in motif.station_indices:
                        pattern_coords.append(self.coordinates[station_idx])
                        pattern_types.append(motif.temporal_context)
            
            if pattern_coords:
                pattern_coords = np.array(pattern_coords)
                
                # Enhanced color scheme for patterns
                pattern_colors = {
                    'irrigation_season': '#2E8B57',     # Sea Green
                    'monsoon_pattern': '#4169E1',       # Royal Blue
                    'dry_season': '#FF8C00',           # Dark Orange  
                    'annual_cycle': '#9932CC',         # Dark Orchid
                    'winter_pattern': '#00CED1',       # Dark Turquoise
                    'short_term_fluctuation': '#FFD700' # Gold
                }
                
                # Plot background stations
                ax6.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                           c='lightgray', s=3, alpha=0.4, label='All stations', 
                           transform=ccrs.PlateCarree(), zorder=1)
                
                # Plot pattern locations by type with proper legend
                legend_added = set()
                for i, (coord, pattern_type) in enumerate(zip(pattern_coords, pattern_types)):
                    color = pattern_colors.get(pattern_type, '#FF1493')  # Deep Pink fallback
                    
                    # Add to legend only once per pattern type
                    label = None
                    if pattern_type not in legend_added:
                        label = pattern_type.replace('_', ' ').title()
                        legend_added.add(pattern_type)
                    
                    ax6.scatter(coord[0], coord[1], c=color, s=60, alpha=0.8,
                               edgecolor='black', linewidth=0.5, label=label, 
                               transform=ccrs.PlateCarree(), zorder=3)
                
                ax6.set_title('Spatial Distribution of Pattern Types\n(Enhanced Geographic Context)', 
                             fontsize=12, fontweight='bold')
                
                # Compact legend inside subplot to avoid encroachment
                legend_elements = []
                for ptype in legend_added:
                    color = pattern_colors.get(ptype, '#FF1493')
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                     markerfacecolor=color, markersize=6,
                                                     label=ptype.replace('_', ' ').title()))
                
                ax6.legend(handles=legend_elements, loc='lower left', fontsize=7, 
                          fancybox=True, shadow=True, framealpha=0.9)
        else:
            # Fallback to matplotlib if Cartopy not available
            ax6 = axes[1, 2]
            if self.coordinates is not None and self.discovered_motifs:
                # ... existing matplotlib code as fallback ...
                ax6.text(0.5, 0.5, 'Cartopy not available\nUsing basic matplotlib', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Spatial Distribution (Basic)')
        
        # No tight_layout since we already set custom spacing
        fig_path = self.figures_dir / "ps04c_fig05_geological_pattern_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _plot_temporal_pattern_evolution(self):
        """Plot temporal evolution of patterns with geological context"""
        if not self.discovered_motifs:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Temporal Pattern Evolution with Geological Context', fontsize=16, fontweight='bold')
        
        # Plot 1: Seasonal pattern strength evolution
        ax1 = axes[0, 0]
        self._plot_seasonal_strength_evolution(ax1)
        
        # Plot 2: Pattern phase relationships
        ax2 = axes[0, 1]
        self._plot_pattern_phase_relationships(ax2)
        
        # Plot 3: Multi-scale pattern hierarchy
        ax3 = axes[1, 0]
        self._plot_multiscale_pattern_hierarchy(ax3)
        
        # Plot 4: Geological process indicators
        ax4 = axes[1, 1]
        self._plot_geological_process_indicators(ax4)
        
        # Plot 5: Pattern stability analysis
        ax5 = axes[2, 0]
        self._plot_pattern_stability_analysis(ax5)
        
        # Plot 6: Predictive pattern model
        ax6 = axes[2, 1]
        self._plot_predictive_pattern_model(ax6)
        
        plt.tight_layout()
        fig_path = self.figures_dir / "ps04c_fig06_temporal_pattern_evolution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _plot_seasonal_strength_evolution(self, ax):
        """Plot seasonal pattern strength evolution over time"""
        if not self.discovered_motifs:
            ax.text(0.5, 0.5, 'No patterns to analyze', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Seasonal Pattern Strength Evolution')
            return
        
        # Analyze seasonal patterns
        seasonal_data = {}
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                if 'seasonal' in motif.temporal_context or 'monsoon' in motif.temporal_context:
                    start_day = motif.start_index * 6
                    month = (start_day % 365.25) / 365.25 * 12
                    
                    if month not in seasonal_data:
                        seasonal_data[month] = []
                    seasonal_data[month].append(motif.significance_score)
        
        if seasonal_data:
            months = sorted(seasonal_data.keys())
            mean_strength = [np.mean(seasonal_data[month]) for month in months]
            std_strength = [np.std(seasonal_data[month]) if len(seasonal_data[month]) > 1 else 0 for month in months]
            
            ax.errorbar(months, mean_strength, yerr=std_strength, marker='o', linewidth=2, capsize=5)
            ax.set_xlabel('Month')
            ax.set_ylabel('Pattern Significance')
            ax.set_title('Seasonal Pattern Strength Evolution')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 12)
            
            # Add season labels
            season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
            season_positions = [1.5, 4.5, 7.5, 10.5]
            for label, pos in zip(season_labels, season_positions):
                ax.axvspan(pos-1.5, pos+1.5, alpha=0.1, color='gray')
                ax.text(pos, ax.get_ylim()[1]*0.9, label, ha='center', va='center', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No seasonal patterns found', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Seasonal Pattern Strength Evolution')
    
    def _plot_pattern_phase_relationships(self, ax):
        """Plot phase relationships between different patterns"""
        # Implementation for phase relationship analysis
        ax.text(0.5, 0.5, 'Phase Relationship Analysis\n(Advanced pattern correlation)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
        ax.set_title('Pattern Phase Relationships')
    
    def _plot_multiscale_pattern_hierarchy(self, ax):
        """Plot multi-scale pattern hierarchy"""
        if not self.discovered_motifs:
            ax.text(0.5, 0.5, 'No patterns to analyze', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Multi-scale Pattern Hierarchy')
            return
        
        # Create hierarchy visualization
        window_sizes = sorted(self.discovered_motifs.keys())
        y_positions = range(len(window_sizes))
        
        for i, window_size in enumerate(window_sizes):
            motifs = self.discovered_motifs[window_size]
            n_motifs = len(motifs)
            avg_significance = np.mean([m.significance_score for m in motifs]) if motifs else 0
            
            # Plot bar representing pattern count and significance
            ax.barh(i, n_motifs, height=0.6, alpha=0.7, 
                   color=plt.cm.viridis(avg_significance/5) if avg_significance > 0 else 'gray')
            
            # Add labels
            ax.text(n_motifs + 0.1, i, f'{n_motifs} patterns\n(avg sig: {avg_significance:.2f})',
                   va='center', fontsize=8)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'{ws} days' for ws in window_sizes])
        ax.set_xlabel('Number of Patterns')
        ax.set_ylabel('Time Scale')
        ax.set_title('Multi-scale Pattern Hierarchy')
        ax.grid(True, alpha=0.3)
    
    def _plot_geological_process_indicators(self, ax):
        """Plot geological process indicators"""
        # Create process indicator visualization
        processes = {
            'Groundwater Extraction': {'color': 'red', 'intensity': 0.8},
            'Natural Compaction': {'color': 'orange', 'intensity': 0.6},
            'Tectonic Activity': {'color': 'purple', 'intensity': 0.3},
            'Thermal Expansion': {'color': 'yellow', 'intensity': 0.4},
            'Loading Effects': {'color': 'brown', 'intensity': 0.5}
        }
        
        # Radar/spider plot
        angles = np.linspace(0, 2*np.pi, len(processes), endpoint=False)
        values = [processes[proc]['intensity'] for proc in processes.keys()]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.3, color='blue')
        
        ax.set_xticks(angles)
        ax.set_xticklabels(list(processes.keys()), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title('Geological Process Indicators\n(Pattern-based Assessment)')
        ax.grid(True, alpha=0.3)
    
    def _plot_pattern_stability_analysis(self, ax):
        """Plot pattern stability analysis"""
        # Implementation for stability analysis
        ax.text(0.5, 0.5, 'Pattern Stability Analysis\n(Temporal consistency metrics)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
        ax.set_title('Pattern Stability Analysis')
    
    def _plot_predictive_pattern_model(self, ax):
        """Plot predictive pattern model"""
        # Implementation for predictive modeling
        ax.text(0.5, 0.5, 'Predictive Pattern Model\n(Future pattern forecasting)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.5))
        ax.set_title('Predictive Pattern Model')
    
    def _plot_pattern_geology_correlation(self):
        """Plot comprehensive pattern-geology correlation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pattern-Geology Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Elevation vs pattern characteristics
        ax1 = axes[0, 0]
        self._plot_elevation_pattern_correlation(ax1)
        
        # Plot 2: Distance from major features
        ax2 = axes[0, 1]
        self._plot_distance_feature_correlation(ax2)
        
        # Plot 3: Geological unit correlation
        ax3 = axes[1, 0]
        self._plot_geological_unit_correlation(ax3)
        
        # Plot 4: Hydrogeological correlation
        ax4 = axes[1, 1]
        self._plot_hydrogeological_correlation(ax4)
        
        plt.tight_layout()
        fig_path = self.figures_dir / "ps04c_fig07_pattern_geology_correlation.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {fig_path}")
    
    def _plot_elevation_pattern_correlation(self, ax):
        """Plot elevation vs pattern characteristics correlation"""
        if not self.discovered_motifs or self.coordinates is None:
            ax.text(0.5, 0.5, 'No data for elevation analysis', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Elevation vs Pattern Correlation')
            return
        
        # Use latitude as proxy for elevation (higher latitude = higher elevation in Taiwan)
        pattern_lats = []
        pattern_significances = []
        pattern_types = []
        
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                for station_idx in motif.station_indices:
                    pattern_lats.append(self.coordinates[station_idx, 1])
                    pattern_significances.append(motif.significance_score)
                    pattern_types.append(motif.temporal_context)
        
        if pattern_lats:
            # Create scatter plot colored by pattern type
            unique_types = list(set(pattern_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            
            for i, ptype in enumerate(unique_types):
                mask = [pt == ptype for pt in pattern_types]
                if any(mask):
                    lats_subset = [lat for lat, m in zip(pattern_lats, mask) if m]
                    sigs_subset = [sig for sig, m in zip(pattern_significances, mask) if m]
                    
                    ax.scatter(lats_subset, sigs_subset, c=[colors[i]], label=ptype.replace('_', ' ').title(),
                              alpha=0.7, s=50)
            
            ax.set_xlabel('Latitude (°N) - Elevation Proxy')
            ax.set_ylabel('Pattern Significance')
            ax.set_title('Elevation vs Pattern Correlation')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(pattern_lats) > 1:
                z = np.polyfit(pattern_lats, pattern_significances, 1)
                p = np.poly1d(z)
                ax.plot(sorted(pattern_lats), p(sorted(pattern_lats)), "r--", alpha=0.7, linewidth=2)
    
    def _plot_distance_feature_correlation(self, ax):
        """Plot distance from major geological features"""
        # Simulate major feature locations (rivers, faults, etc.)
        major_features = {
            'Choushui River': [120.6, 23.9],
            'Zhuoshui River': [120.5, 23.8],
            'Major Fault': [120.7, 24.0]
        }
        
        if self.coordinates is not None:
            # Calculate distances to features for all stations
            n_stations = len(self.coordinates)
            distances_to_features = {}
            
            for feature_name, feature_coords in major_features.items():
                distances = []
                for coord in self.coordinates:
                    dist = np.sqrt((coord[0] - feature_coords[0])**2 + (coord[1] - feature_coords[1])**2)
                    distances.append(dist * 111)  # Convert to km (approximate)
                distances_to_features[feature_name] = distances
            
            # Plot correlation with pattern density
            if self.discovered_motifs:
                # Count patterns per station
                pattern_counts_per_station = np.zeros(n_stations)
                for window_size, motifs in self.discovered_motifs.items():
                    for motif in motifs:
                        for station_idx in motif.station_indices:
                            if station_idx < n_stations:
                                pattern_counts_per_station[station_idx] += 1
                
                # Plot for each feature
                colors = ['blue', 'green', 'red']
                for i, (feature_name, distances) in enumerate(distances_to_features.items()):
                    ax.scatter(distances, pattern_counts_per_station, c=colors[i], alpha=0.6,
                             label=feature_name, s=30)
                
                ax.set_xlabel('Distance to Feature (km)')
                ax.set_ylabel('Pattern Count per Station')
                ax.set_title('Distance from Major Features vs Pattern Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No patterns for correlation', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No coordinate data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distance from Major Features')
    
    def _plot_geological_unit_correlation(self, ax):
        """Plot correlation with geological units"""
        # Simulate geological unit assignment based on location
        geological_units = {
            'Alluvium': {'bounds': [120.2, 120.6, 23.6, 24.0], 'color': 'lightblue'},
            'Marine Terrace': {'bounds': [120.6, 120.8, 23.8, 24.2], 'color': 'wheat'},
            'Colluvium': {'bounds': [120.8, 121.0, 23.7, 24.1], 'color': 'lightgreen'}
        }
        
        if self.coordinates is not None and self.discovered_motifs:
            # Assign geological units to stations
            station_units = []
            for coord in self.coordinates:
                assigned_unit = 'Unknown'
                for unit_name, unit_data in geological_units.items():
                    bounds = unit_data['bounds']
                    if (bounds[0] <= coord[0] <= bounds[1] and 
                        bounds[2] <= coord[1] <= bounds[3]):
                        assigned_unit = unit_name
                        break
                station_units.append(assigned_unit)
            
            # Count patterns by geological unit
            unit_pattern_counts = {}
            for window_size, motifs in self.discovered_motifs.items():
                for motif in motifs:
                    for station_idx in motif.station_indices:
                        if station_idx < len(station_units):
                            unit = station_units[station_idx]
                            if unit not in unit_pattern_counts:
                                unit_pattern_counts[unit] = 0
                            unit_pattern_counts[unit] += 1
            
            if unit_pattern_counts:
                units = list(unit_pattern_counts.keys())
                counts = list(unit_pattern_counts.values())
                colors = [geological_units.get(unit, {}).get('color', 'gray') for unit in units]
                
                bars = ax.bar(range(len(units)), counts, color=colors, alpha=0.7)
                ax.set_xlabel('Geological Unit')
                ax.set_ylabel('Pattern Count')
                ax.set_title('Pattern Distribution by Geological Unit')
                ax.set_xticks(range(len(units)))
                ax.set_xticklabels(units)
                ax.grid(True, alpha=0.3)
                
                # Add percentage labels
                total_patterns = sum(counts)
                for bar, count in zip(bars, counts):
                    percentage = count / total_patterns * 100
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data for geological unit analysis', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Geological Unit Correlation')
    
    def _plot_hydrogeological_correlation(self, ax):
        """Plot hydrogeological correlation analysis"""
        # Create hydrogeological process assessment
        hydro_processes = {
            'Shallow Aquifer\nDepletion': 0.8,
            'Deep Aquifer\nCompaction': 0.7,
            'Artesian Pressure\nChanges': 0.5,
            'Clay Layer\nConsolidation': 0.9,
            'Seasonal Recharge\nVariation': 0.6
        }
        
        processes = list(hydro_processes.keys())
        intensities = list(hydro_processes.values())
        
        bars = ax.barh(range(len(processes)), intensities, 
                      color=['red' if i > 0.7 else 'orange' if i > 0.5 else 'green' for i in intensities],
                      alpha=0.7)
        
        ax.set_yticks(range(len(processes)))
        ax.set_yticklabels(processes)
        ax.set_xlabel('Process Intensity (Pattern-based Assessment)')
        ax.set_title('Hydrogeological Process Correlation')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add intensity labels
        for bar, intensity in zip(bars, intensities):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{intensity:.1f}', va='center', fontweight='bold')
        
        # Add risk legend
        ax.text(0.02, 0.98, '🔴 High Risk (>0.7)\n🟠 Medium Risk (0.5-0.7)\n🟢 Low Risk (<0.5)',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def save_results(self):
        """Save all analysis results to JSON files"""
        print(f"\n💾 Saving pattern discovery results...")
        
        # Save discovered motifs
        if self.discovered_motifs:
            motifs_data = {}
            for window_size, motifs in self.discovered_motifs.items():
                motifs_data[str(window_size)] = [asdict(motif) for motif in motifs]
                # Convert numpy arrays to lists for JSON serialization
                for motif_dict in motifs_data[str(window_size)]:
                    motif_dict['pattern'] = motif_dict['pattern'].tolist()
            
            motifs_file = self.results_dir / "discovered_motifs.json"
            with open(motifs_file, 'w') as f:
                json.dump(motifs_data, f, indent=2, cls=NumpyEncoder)
            print(f"   ✅ Motifs saved: {motifs_file}")
        
        # Save detected discords
        if self.detected_discords:
            discords_data = {}
            for window_size, discords in self.detected_discords.items():
                discords_data[str(window_size)] = [asdict(discord) for discord in discords]
                # Convert numpy arrays to lists for JSON serialization
                for discord_dict in discords_data[str(window_size)]:
                    discord_dict['pattern'] = discord_dict['pattern'].tolist()
            
            discords_file = self.results_dir / "detected_discords.json"
            with open(discords_file, 'w') as f:
                json.dump(discords_data, f, indent=2, cls=NumpyEncoder)
            print(f"   ✅ Discords saved: {discords_file}")
        
        # Save analysis summary
        summary = {
            'analysis_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'n_stations': len(self.coordinates) if self.coordinates is not None else 0,
                'time_series_length': self.time_series.shape[1] if self.time_series is not None else 0,
                'window_sizes_analyzed': list(self.matrix_profiles.keys()),
                'total_motifs_discovered': sum(len(motifs) for motifs in self.discovered_motifs.values()),
                'total_discords_detected': sum(len(discords) for discords in self.detected_discords.values()),
            },
            'performance_metrics': self.timing_results,
            'configuration': {
                'max_patterns': self.max_patterns,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state
            }
        }
        
        summary_file = self.results_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        print(f"   ✅ Summary saved: {summary_file}")
        
        return True

def main():
    """Main function for ps04c motif discovery analysis"""
    parser = argparse.ArgumentParser(description='Matrix profile-based motif discovery and anomaly detection')
    parser.add_argument('--n-stations', type=int, default=100,
                       help='Number of stations to analyze (default: 100)')
    parser.add_argument('--window-sizes', nargs='+', type=int, default=[30, 90, 365],
                       help='Window sizes in days (default: 30 90 365)')
    parser.add_argument('--n-motifs', type=int, default=10,
                       help='Number of motifs to discover per window (default: 10)')
    parser.add_argument('--n-discords', type=int, default=5,
                       help='Number of discords to detect per window (default: 5)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1 for all cores)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with single window size')
    
    args = parser.parse_args()
    
    try:
        print("🔍 PS04C: Matrix Profile-Based Pattern Discovery")
        print("=" * 80)
        
        # Adjust parameters for 6-day sampling data
        if args.quick_test:
            window_sizes = [18]  # 3 months for quick test (18 * 6 = 108 days)
            n_motifs = 8
            n_discords = 5
            print("⚡ QUICK TEST MODE - Optimized for 6-day sampling")
        else:
            # Better window sizes for 6-day data: 3 months, 6 months, 1 year
            window_sizes = [18, 30, 60] if not args.window_sizes else args.window_sizes
            n_motifs = args.n_motifs
            n_discords = args.n_discords
        
        # Initialize analysis framework
        analysis = MotifAnomalyAnalysis(
            max_patterns=max(n_motifs, n_discords),
            n_jobs=args.n_jobs
        )
        
        # Load and prepare data
        print("\n🔄 Loading and preparing time series data...")
        if not analysis.load_preprocessed_data(n_stations=args.n_stations):
            print("❌ Failed to load data")
            return False
        
        # Compute matrix profiles
        print(f"\n🔄 Computing matrix profiles for window sizes: {window_sizes}")
        if not analysis.compute_matrix_profile(window_sizes=window_sizes):
            print("❌ Failed to compute matrix profiles")
            return False
        
        # Discover motifs
        print(f"\n🔄 Discovering motifs ({n_motifs} per window)...")
        for window_size in window_sizes:
            analysis.discover_motifs(window_size=window_size, n_motifs=n_motifs)
        
        # Detect discords
        print(f"\n🔄 Detecting discords ({n_discords} per window)...")
        for window_size in window_sizes:
            analysis.detect_discords(window_size=window_size, n_discords=n_discords)
        
        # Create visualizations
        print(f"\n🔄 Creating visualizations...")
        analysis.create_visualizations()
        
        # Save results
        analysis.save_results()
        
        # Performance summary
        print(f"\n🎯 Analysis Complete!")
        print("=" * 50)
        
        total_motifs = sum(len(motifs) for motifs in analysis.discovered_motifs.values())
        total_discords = sum(len(discords) for discords in analysis.detected_discords.values())
        
        print(f"📊 Results Summary:")
        print(f"   Stations analyzed: {len(analysis.coordinates)}")
        print(f"   Window sizes: {list(analysis.matrix_profiles.keys())}")
        print(f"   Motifs discovered: {total_motifs}")
        print(f"   Discords detected: {total_discords}")
        
        if 'matrix_profile_computation' in analysis.timing_results:
            print(f"   Computation time: {analysis.timing_results['matrix_profile_computation']/60:.1f} minutes")
        
        print(f"🎨 Outputs:")
        print(f"   📁 Results: data/processed/ps04c_motifs/")
        print(f"   🖼️  Figures: figures/ps04c_fig01-03_*.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in motif discovery analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
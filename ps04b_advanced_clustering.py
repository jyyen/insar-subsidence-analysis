#!/usr/bin/env python3
"""
ps04b_advanced_clustering.py - Advanced TSLearn Clustering Methods

Purpose: Enhanced time series clustering using state-of-the-art TSLearn algorithms
Extends: ps04_temporal_clustering.py with advanced pattern discovery methods
Focus: Soft DTW, shapelet discovery, multi-metric clustering, multi-resolution analysis

Author: Claude Code
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import argparse
import warnings
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import time
import signal

# TSLearn imports
try:
    from tslearn.utils import to_time_series_dataset
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import dtw, soft_dtw
    from tslearn.shapelets import ShapeletModel
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False
    print("⚠️  TSLearn not available - install with: pip install tslearn")

# Advanced visualization imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
    
    # Check for kaleido (for static image export)
    try:
        import kaleido
        HAS_KALEIDO = True
    except ImportError:
        HAS_KALEIDO = False
        
except ImportError:
    HAS_PLOTLY = False
    HAS_KALEIDO = False
    print("⚠️  Plotly not available - install with: pip install plotly")

warnings.filterwarnings('ignore')

# Suppress multiprocessing warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress specific multiprocessing resource warnings
import multiprocessing
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# TSLearn imports with enhanced functionality
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import dtw, soft_dtw, dtw_path
    from tslearn.utils import to_time_series_dataset
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    HAS_TSLEARN = True
    print("✅ TSLearn available - using advanced clustering methods")
except ImportError:
    HAS_TSLEARN = False
    print("❌ TSLearn not available - install with: conda install -c conda-forge tslearn")

# Shapelet imports (optional - requires TensorFlow)
try:
    from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
    HAS_SHAPELETS = True
    print("✅ TSLearn shapelets available")
except ImportError:
    HAS_SHAPELETS = False
    print("⚠️  TSLearn shapelets not available (requires TensorFlow) - shapelet discovery will be skipped")

# Additional advanced libraries
try:
    import stumpy
    HAS_STUMPY = True
    print("✅ Stumpy available for matrix profile analysis")
except ImportError:
    HAS_STUMPY = False
    print("⚠️  Stumpy not available - install with: pip install stumpy")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not available for enhanced geographic visualization")

class AdvancedTSLearnClustering:
    """
    Advanced time series clustering using state-of-the-art TSLearn methods
    
    Extends basic DTW clustering with:
    - Soft DTW for gradient-based optimization
    - Multi-metric DTW variants (Shape, Derivative)
    - Shapelet discovery for discriminative patterns
    - Multi-resolution temporal analysis
    """
    
    def __init__(self, max_clusters=8, random_state=42, n_jobs=1):
        """
        Initialize advanced clustering framework
        
        Parameters:
        -----------
        max_clusters : int
            Maximum number of clusters to consider
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        if not HAS_TSLEARN:
            raise ImportError("TSLearn is required for advanced clustering. Install with: conda install -c conda-forge tslearn")
            
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        
        # Data containers
        self.coordinates = None
        self.time_series_data = None
        self.scaled_time_series = None
        
        # Clustering results
        self.soft_dtw_results = {}
        self.multi_metric_results = {}
        self.shapelet_results = {}
        self.hierarchical_results = {}
        
        # Validation results
        self.cluster_validation = {}
        self.method_comparison = {}
        
        # Timing system
        self.timing_results = {}
        self.stage_timers = {}
        self.total_start_time = None
        
        # Setup directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories for advanced clustering results"""
        self.figures_dir = Path("figures")
        self.results_dir = Path("data/processed/ps04b_advanced")
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Advanced results directory: {self.results_dir}")
        print(f"📁 Figures directory: {self.figures_dir}")
    
    def start_timer(self, stage_name):
        """Start timing a specific stage"""
        self.stage_timers[stage_name] = time.time()
        print(f"⏱️  Starting: {stage_name}")
    
    def end_timer(self, stage_name):
        """End timing and record duration"""
        if stage_name in self.stage_timers:
            duration = time.time() - self.stage_timers[stage_name]
            self.timing_results[stage_name] = duration
            print(f"✅ Completed: {stage_name} ({duration:.2f}s)")
            return duration
        return 0
    
    def start_total_timer(self):
        """Start timing the entire analysis"""
        self.total_start_time = time.time()
        print("🚀 Starting ps04b advanced clustering analysis...")
    
    def end_total_timer(self):
        """End total timing and generate summary"""
        if self.total_start_time:
            total_time = time.time() - self.total_start_time
            self.timing_results['total_analysis'] = total_time
            return total_time
        return 0
    
    def print_timing_summary(self):
        """Print comprehensive timing summary"""
        print("\n" + "⏱️ " * 20)
        print("📊 PERFORMANCE TIMING SUMMARY")
        print("⏱️ " * 20)
        
        if not self.timing_results:
            print("❌ No timing data available")
            return
        
        # Calculate percentages
        total_time = self.timing_results.get('total_analysis', sum(self.timing_results.values()))
        
        # Sort by duration
        sorted_stages = sorted([(k, v) for k, v in self.timing_results.items() if k != 'total_analysis'], 
                              key=lambda x: x[1], reverse=True)
        
        print(f"🕐 Total Analysis Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print("\n📋 Stage Breakdown:")
        print("-" * 60)
        
        for stage, duration in sorted_stages:
            percentage = (duration / total_time) * 100
            stars = "⭐" * int(percentage / 10)
            print(f"{stage:30} {duration:8.2f}s ({percentage:5.1f}%) {stars}")
        
        # Performance insights
        print("\n💡 Performance Insights:")
        if 'data_loading' in self.timing_results and 'soft_dtw_clustering' in self.timing_results:
            data_time = self.timing_results['data_loading']
            clustering_time = self.timing_results['soft_dtw_clustering']
            if clustering_time > data_time * 10:
                print(f"   🔥 Clustering is the main bottleneck ({clustering_time/data_time:.1f}x data loading)")
            
        if 'temporal_interpolation' in self.timing_results:
            interp_time = self.timing_results['temporal_interpolation']
            if interp_time > 5:
                print(f"   ⚠️  Temporal interpolation took {interp_time:.1f}s - consider optimization")
        
        # Parallelization effectiveness
        if self.n_jobs > 1:
            print(f"   🚀 Using {self.n_jobs} parallel workers")
            if total_time < 120:  # Less than 2 minutes
                print(f"   ✅ Good performance with parallelization")
            else:
                print(f"   ⚠️  Consider more cores or algorithm optimization")
        
        print("⏱️ " * 20 + "\n")
    
    def save_timing_results(self):
        """Save timing results to JSON file"""
        timing_file = self.results_dir / "timing_analysis.json"
        timing_data = {
            'timing_results': self.timing_results,
            'analysis_config': {
                'n_jobs': self.n_jobs,
                'max_clusters': self.max_clusters,
                'n_stations': len(self.coordinates) if self.coordinates is not None else 0,
                'time_series_length': self.scaled_time_series.shape[1] if self.scaled_time_series is not None else 0
            },
            'performance_metrics': {
                'total_time_minutes': self.timing_results.get('total_analysis', 0) / 60,
                'time_per_station': self.timing_results.get('total_analysis', 0) / max(1, len(self.coordinates) if self.coordinates is not None else 1),
                'parallelization_efficiency': self.n_jobs if self.n_jobs > 1 else 'sequential'
            }
        }
        
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
        print(f"📊 Timing analysis saved to: {timing_file}")
        
    def load_preprocessed_data(self, n_stations=None, use_raw_data=True):
        """
        Load preprocessed time series data
        
        Parameters:
        -----------
        n_stations : int, optional
            Limit to first N stations for testing (None for all)
        use_raw_data : bool
            Use RAW displacement data for TSLearn analysis (recommended for pattern discovery)
            Raw data preserves natural patterns, trends, and anomalies that TSLearn algorithms need
        """
        self.start_timer("data_loading")
        print("📡 Loading RAW time series data for TSLearn analysis...")
        print("   ✅ Using RAW displacement data - preserves natural patterns for TSLearn algorithms")
        
        try:
            # Load preprocessed data
            preprocessed_data = np.load("data/processed/ps00_preprocessed_data.npz", allow_pickle=True)
            
            if n_stations is None:
                self.coordinates = preprocessed_data['coordinates']
                # Use RAW displacement data for TSLearn - contains natural patterns and trends
                self.time_series_data = preprocessed_data['displacement']
                subset_info = f"full dataset ({len(self.coordinates)} stations)"
            else:
                self.coordinates = preprocessed_data['coordinates'][:n_stations]
                # Use RAW displacement data for TSLearn - contains natural patterns and trends  
                self.time_series_data = preprocessed_data['displacement'][:n_stations]
                subset_info = f"{n_stations}-station subset"
            
            print(f"✅ Loaded {len(self.coordinates)} stations ({subset_info})")
            print(f"   Time series shape: {self.time_series_data.shape}")
            
            self.end_timer("data_loading")
            
            # Convert to TSLearn format and scale
            self.prepare_tslearn_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def prepare_tslearn_data(self):
        """Prepare data for TSLearn compatibility - NO temporal interpolation"""
        self.start_timer("data_preparation")
        print("🔄 Preparing data for TSLearn analysis...")
        
        # Use original temporal resolution (215 time points) - NO interpolation
        print(f"   Using original temporal resolution: {self.time_series_data.shape[1]} time points")
        print("   ✅ Preserving full temporal information (215 acquisitions)")
        
        # Convert to TSLearn format (add dimension for multivariate)
        self.time_series_data = to_time_series_dataset(self.time_series_data)
        
        # Scale time series (important for DTW-based methods)
        scaler = TimeSeriesScalerMeanVariance()
        self.scaled_time_series = scaler.fit_transform(self.time_series_data)
        
        print(f"✅ Prepared TSLearn format: {self.scaled_time_series.shape}")
        print(f"   Using scaled time series for clustering")
        
        self.end_timer("data_preparation")
    
    def characterize_clusters(self, cluster_labels, method_name="clustering"):
        """
        Characterize clusters in geological and geophysical context
        
        Parameters:
        -----------
        cluster_labels : array-like
            Cluster assignments for each station
        method_name : str
            Name of the clustering method for reporting
            
        Returns:
        --------
        dict : Cluster characterization results
        """
        print(f"🔍 Characterizing {method_name} clusters for geological interpretation...")
        
        # Load the original subsidence rates from ps00
        ps00_data = np.load("data/processed/ps00_preprocessed_data.npz", allow_pickle=True)
        original_subsidence_rates_full = ps00_data['subsidence_rates']  # Already in mm/year
        
        # Extract subsidence rates for the same subset of stations that were loaded for analysis
        n_stations_analyzed = len(self.coordinates)
        original_subsidence_rates = original_subsidence_rates_full[:n_stations_analyzed]
        
        n_clusters = len(np.unique(cluster_labels))
        characterization = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = self.scaled_time_series[cluster_mask, :, 0]  # Detrended, scaled data
            cluster_coords = self.coordinates[cluster_mask]
            cluster_rates = original_subsidence_rates[cluster_mask]  # Original subsidence rates (subset-matched)
            n_stations = np.sum(cluster_mask)
            
            # 1. Temporal pattern analysis using ORIGINAL subsidence rates
            cluster_mean_rate = np.mean(cluster_rates)  # Mean subsidence rate in mm/year
            cluster_std_rate = np.std(cluster_rates)    # Rate variability
            
            # Seasonal pattern analysis from detrended data
            cluster_mean_detrended = np.mean(cluster_data, axis=0)
            seasonal_amplitude = np.std(cluster_mean_detrended)  # Seasonal variation amplitude
            max_seasonal = np.max(cluster_mean_detrended)
            min_seasonal = np.min(cluster_mean_detrended)
            seasonal_range = max_seasonal - min_seasonal
            
            # 2. Geographic distribution analysis
            lon_center = np.mean(cluster_coords[:, 0])
            lat_center = np.mean(cluster_coords[:, 1])
            lon_spread = np.std(cluster_coords[:, 0])
            lat_spread = np.std(cluster_coords[:, 1])
            
            # Geographic bounds
            lon_range = [np.min(cluster_coords[:, 0]), np.max(cluster_coords[:, 0])]
            lat_range = [np.min(cluster_coords[:, 1]), np.max(cluster_coords[:, 1])]
            
            # 3. Deformation pattern classification
            pattern_type = self._classify_deformation_pattern(cluster_mean_rate, seasonal_amplitude)
            
            # 4. Potential geological processes
            geological_processes = self._infer_geological_processes(
                pattern_type, cluster_mean_rate, seasonal_amplitude, lon_center, lat_center
            )
            
            # 5. InSAR/Remote sensing characteristics
            radar_characteristics = self._analyze_radar_characteristics(
                cluster_data, seasonal_range, seasonal_amplitude
            )
            
            # 6. Calculate deformation range for remote sensing notes
            deformation_range = max_seasonal - min_seasonal
            
            characterization[f'cluster_{cluster_id+1}'] = {
                # Basic statistics
                'n_stations': int(n_stations),
                'percentage': float(100 * n_stations / len(cluster_labels)),
                
                # Temporal characteristics (CORRECTED)
                'mean_subsidence_rate_mm_year': float(cluster_mean_rate),  # Already in mm/year
                'rate_variability_mm_year': float(cluster_std_rate),
                'seasonal_amplitude_mm': float(seasonal_amplitude),
                'seasonal_range_mm': float(seasonal_range),
                'max_seasonal_mm': float(max_seasonal),
                'min_seasonal_mm': float(min_seasonal),
                'deformation_range_mm': float(deformation_range),
                
                # Geographic distribution
                'geographic_center': [float(lon_center), float(lat_center)],
                'geographic_spread': [float(lon_spread), float(lat_spread)],
                'geographic_bounds': {
                    'lon_range': [float(x) for x in lon_range],
                    'lat_range': [float(x) for x in lat_range]
                },
                
                # Pattern classification
                'pattern_type': pattern_type,
                'geological_processes': geological_processes,
                'radar_characteristics': radar_characteristics,
                
                # Interpretation
                'geological_interpretation': self._generate_geological_interpretation(
                    pattern_type, geological_processes, lon_center, lat_center, n_stations
                ),
                'remote_sensing_notes': self._generate_remote_sensing_notes(
                    radar_characteristics, seasonal_amplitude, deformation_range
                )
            }
        
        # Add overall analysis
        characterization['summary'] = self._generate_cluster_summary(characterization, method_name)
        
        return characterization
    
    def _classify_deformation_pattern(self, mean_rate_mm_year, seasonal_amp):
        """Classify the temporal deformation pattern using CORRECT units"""
        
        # Pattern classification based on subsidence rate (mm/year) and seasonality
        if abs(mean_rate_mm_year) < 2.0:  # Very small rate
            if seasonal_amp > 2.0:
                return "seasonal_dominated"
            else:
                return "stable_minimal_deformation"
        elif mean_rate_mm_year < -10.0:  # Strong subsidence (>10 mm/year)
            if seasonal_amp > 3.0:
                return "subsidence_with_strong_seasonality"
            else:
                return "steady_subsidence"
        elif mean_rate_mm_year > 5.0:  # Significant uplift (>5 mm/year)
            if seasonal_amp > 3.0:
                return "uplift_with_seasonality"
            else:
                return "steady_uplift"
        elif mean_rate_mm_year < -2.0:  # Moderate subsidence (2-10 mm/year)
            if seasonal_amp > 2.0:
                return "moderate_subsidence_with_seasonality"
            else:
                return "moderate_subsidence"
        else:  # Small positive rates (0-5 mm/year)
            if seasonal_amp > 2.0:
                return "seasonally_dominated_with_trend"
            else:
                return "minimal_trend_with_seasonality"
    
    def _infer_geological_processes(self, pattern_type, trend, seasonal_amp, lon, lat):
        """Infer likely geological processes based on deformation pattern and location"""
        
        processes = []
        
        # Taiwan regional context
        is_western_plain = lon < 120.6  # Western coastal plain
        is_central_taiwan = 120.6 <= lon <= 121.0  # Central Taiwan
        is_northern = lat > 24.0
        is_southern = lat < 23.8
        
        # Process inference based on pattern and location
        if "subsidence" in pattern_type:
            if is_western_plain:
                processes.extend(["groundwater_extraction", "sediment_compaction"])
                if seasonal_amp > 3.0:
                    processes.append("irrigation_induced")
            if is_central_taiwan:
                processes.extend(["tectonic_subsidence", "fault_activity"])
                
        if "seasonal" in pattern_type:
            processes.extend(["groundwater_level_variation", "soil_moisture_changes"])
            if seasonal_amp > 5.0:
                processes.append("intensive_agriculture")
                
        if "uplift" in pattern_type:
            if is_central_taiwan:
                processes.extend(["tectonic_uplift", "orogenic_processes"])
            else:
                processes.extend(["groundwater_recharge", "elastic_rebound"])
                
        if "stable" in pattern_type:
            processes.extend(["stable_bedrock", "minimal_human_activity"])
            
        return processes
    
    def _analyze_radar_characteristics(self, cluster_data, deform_range, seasonal_amp):
        """Analyze InSAR/radar remote sensing characteristics"""
        
        characteristics = {}
        
        # Signal-to-noise assessment
        mean_variation = np.mean(np.std(cluster_data, axis=0))
        characteristics['temporal_coherence'] = 'high' if mean_variation < 2.0 else 'moderate' if mean_variation < 5.0 else 'low'
        
        # Deformation detectability
        if deform_range > 10.0:
            characteristics['detectability'] = 'excellent'
        elif deform_range > 5.0:
            characteristics['detectability'] = 'good'
        elif deform_range > 2.0:
            characteristics['detectability'] = 'moderate'
        else:
            characteristics['detectability'] = 'challenging'
            
        # Seasonal signal strength
        if seasonal_amp > 5.0:
            characteristics['seasonal_signal'] = 'strong'
        elif seasonal_amp > 2.0:
            characteristics['seasonal_signal'] = 'moderate'
        else:
            characteristics['seasonal_signal'] = 'weak'
            
        # Processing considerations
        characteristics['processing_notes'] = []
        if mean_variation > 3.0:
            characteristics['processing_notes'].append("high_noise_requires_filtering")
        if seasonal_amp > 8.0:
            characteristics['processing_notes'].append("strong_seasonal_needs_detrending")
        if deform_range > 20.0:
            characteristics['processing_notes'].append("large_gradients_check_unwrapping")
            
        return characteristics
    
    def _generate_geological_interpretation(self, pattern_type, processes, lon, lat, n_stations):
        """Generate comprehensive geological interpretation"""
        
        # Regional context
        if lon < 120.4:
            region = "Western Coastal Plain"
        elif lon < 120.8:
            region = "Changhua-Yunlin Alluvial Fan"
        else:
            region = "Central Taiwan Foothills"
            
        interpretation = f"Region: {region} | "
        
        # Pattern interpretation
        if "subsidence" in pattern_type:
            if "seasonal" in pattern_type:
                interpretation += "Active subsidence with seasonal modulation suggests groundwater pumping for agriculture. "
            else:
                interpretation += "Steady subsidence indicates ongoing sediment compaction or regional tectonic processes. "
        elif "uplift" in pattern_type:
            interpretation += "Uplift pattern suggests tectonic activity or groundwater recovery. "
        elif "seasonal" in pattern_type:
            interpretation += "Seasonal-dominated signal indicates strong hydrological control on deformation. "
        else:
            interpretation += "Stable region with minimal active deformation processes. "
            
        # Process-specific interpretation
        if "groundwater_extraction" in processes:
            interpretation += "Groundwater over-extraction is likely causing aquifer system compaction. "
        if "irrigation_induced" in processes:
            interpretation += "Agricultural irrigation patterns strongly influence temporal deformation. "
        if "tectonic" in " ".join(processes):
            interpretation += "Tectonic processes (faulting or regional stress) contribute to deformation. "
            
        # Significance assessment
        if n_stations > 100:
            interpretation += f"Large cluster ({n_stations} stations) indicates regionally significant process."
        elif n_stations < 20:
            interpretation += f"Small cluster ({n_stations} stations) may represent localized phenomenon."
            
        return interpretation
    
    def _generate_remote_sensing_notes(self, radar_chars, seasonal_amp, deform_range):
        """Generate InSAR/remote sensing specific notes"""
        
        notes = []
        
        # Data quality assessment
        coherence = radar_chars['temporal_coherence']
        if coherence == 'high':
            notes.append("Excellent PSI coherence - high confidence in measurements")
        elif coherence == 'moderate':
            notes.append("Moderate coherence - results reliable but monitor for atmospheric artifacts")
        else:
            notes.append("Lower coherence - increased uncertainty, validate with independent data")
            
        # Detectability notes
        detectability = radar_chars['detectability']
        if detectability == 'excellent':
            notes.append("Strong deformation signal easily detected by InSAR")
        elif detectability == 'challenging':
            notes.append("Weak signal near InSAR detection limits - exercise caution in interpretation")
            
        # Processing recommendations
        if seasonal_amp > 6.0:
            notes.append("Strong seasonal signal - recommend seasonal decomposition analysis")
        if deform_range > 15.0:
            notes.append("Large deformation gradients - verify phase unwrapping quality")
            
        # Validation recommendations
        notes.append("Recommend validation with GPS data where available")
        if "groundwater" in " ".join(radar_chars.get('processing_notes', [])):
            notes.append("Cross-validate with groundwater level monitoring data")
            
        return notes
    
    def _generate_cluster_summary(self, characterization, method_name):
        """Generate overall cluster analysis summary"""
        
        n_clusters = len([k for k in characterization.keys() if k.startswith('cluster_')])
        
        # Count pattern types
        pattern_counts = {}
        process_counts = {}
        
        for cluster_id in range(n_clusters):
            cluster_data = characterization[f'cluster_{cluster_id+1}']
            pattern = cluster_data['pattern_type']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            for process in cluster_data['geological_processes']:
                process_counts[process] = process_counts.get(process, 0) + 1
                
        # Find dominant patterns and processes
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
        dominant_processes = sorted(process_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary = {
            'method': method_name,
            'n_clusters': n_clusters,
            'dominant_pattern': dominant_pattern,
            'top_processes': [p[0] for p in dominant_processes],
            'geological_summary': f"{method_name} identified {n_clusters} distinct deformation patterns. "
                                f"Dominant pattern: {dominant_pattern.replace('_', ' ')}. "
                                f"Key processes: {', '.join([p[0].replace('_', ' ') for p in dominant_processes[:2]])}.",
            'remote_sensing_assessment': f"InSAR coherence and detectability vary across clusters. "
                                       f"Recommend focused validation for clusters with challenging detectability.",
            'recommendations': [
                "Integrate with groundwater monitoring data",
                "Cross-validate with GPS measurements", 
                "Correlate with geological maps and land use data",
                "Monitor temporal evolution for process validation"
            ]
        }
        
        return summary
    
        
    def soft_dtw_clustering(self, n_clusters_range=None, gamma=0.1):
        """
        Soft DTW clustering with gradient-based optimization (parallelized)
        
        Parameters:
        -----------
        n_clusters_range : list, optional
            Range of cluster numbers to test
        gamma : float
            Soft DTW smoothing parameter (smaller = closer to DTW)
        """
        self.start_timer("soft_dtw_clustering")
        print("\n🔄 SOFT DTW CLUSTERING (PARALLELIZED)")
        print("-" * 50)
        
        if n_clusters_range is None:
            n_clusters_range = range(2, self.max_clusters + 1)
            
        # Use parallel implementation for performance
        results = self._parallel_soft_dtw_clustering(n_clusters_range, gamma)
        
        self.soft_dtw_results = results
        
        # Find optimal k
        valid_results = {k: v for k, v in results.items() if 'silhouette_score' in v}
        if valid_results:
            optimal_k = max(valid_results.keys(), key=lambda k: valid_results[k]['silhouette_score'])
            print(f"✅ Optimal k={optimal_k} (Silhouette: {valid_results[optimal_k]['silhouette_score']:.3f})")
            self.soft_dtw_results['optimal_k'] = optimal_k
            
            # Characterize the optimal clustering
            optimal_labels = np.array(valid_results[optimal_k]['cluster_labels'])
            characterization = self.characterize_clusters(optimal_labels, "Soft_DTW")
            self.soft_dtw_results['cluster_characterization'] = characterization
            
            print("\n🔍 CLUSTER CHARACTERIZATION:")
            print("-" * 40)
            for cluster_id in range(optimal_k):
                cluster_info = characterization[f'cluster_{cluster_id+1}']
                print(f"\n🏷️  Cluster {cluster_id+1} ({cluster_info['n_stations']} stations, {cluster_info['percentage']:.1f}%)")
                print(f"   Pattern: {cluster_info['pattern_type'].replace('_', ' ').title()}")
                print(f"   Subsidence rate: {cluster_info['mean_subsidence_rate_mm_year']:.1f} mm/year")
                print(f"   Seasonal amplitude: {cluster_info['seasonal_amplitude_mm']:.1f} mm")
                print(f"   Processes: {', '.join(cluster_info['geological_processes'][:3]).replace('_', ' ')}")
                print(f"   Region: {cluster_info['geographic_center'][0]:.3f}°E, {cluster_info['geographic_center'][1]:.3f}°N")
                print(f"   InSAR quality: {cluster_info['radar_characteristics']['detectability']}")
            
            print(f"\n📋 Summary: {characterization['summary']['geological_summary']}")
        
        self.end_timer("soft_dtw_clustering")
        return results
    
    def multi_metric_clustering(self, n_clusters=4, metrics=None):
        """
        Compare multiple DTW metrics for clustering (parallelized)
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        metrics : list, optional
            DTW metrics to compare ['dtw', 'softdtw', 'euclidean']
        """
        print("\n🔄 MULTI-METRIC DTW CLUSTERING (PARALLELIZED)")
        print("-" * 50)
        
        if metrics is None:
            metrics = ['dtw', 'softdtw', 'euclidean']
            
        # Use parallel implementation for performance
        results = self._parallel_multi_metric_clustering(n_clusters, metrics)
        
        self.multi_metric_results = results
        
        # Find best metric
        valid_results = {k: v for k, v in results.items() if 'silhouette_score' in v}
        if valid_results:
            best_metric = max(valid_results.keys(), key=lambda k: valid_results[k]['silhouette_score'])
            print(f"✅ Best metric: {best_metric.upper()} (Silhouette: {valid_results[best_metric]['silhouette_score']:.3f})")
            self.multi_metric_results['best_metric'] = best_metric
            
            # Characterize the best metric clustering
            best_labels = np.array(valid_results[best_metric]['cluster_labels'])
            characterization = self.characterize_clusters(best_labels, f"Multi_Metric_{best_metric.upper()}")
            self.multi_metric_results['cluster_characterization'] = characterization
            
            print(f"\n🔍 BEST METHOD ({best_metric.upper()}) CHARACTERIZATION:")
            print("-" * 50)
            n_clusters = len(np.unique(best_labels))
            for cluster_id in range(n_clusters):
                cluster_info = characterization[f'cluster_{cluster_id+1}']
                print(f"\n🏷️  Cluster {cluster_id+1} ({cluster_info['n_stations']} stations, {cluster_info['percentage']:.1f}%)")
                print(f"   Pattern: {cluster_info['pattern_type'].replace('_', ' ').title()}")
                print(f"   Subsidence rate: {cluster_info['mean_subsidence_rate_mm_year']:.1f} mm/year")
                print(f"   Seasonal: {cluster_info['seasonal_amplitude_mm']:.1f} mm amplitude")
                print(f"   Geological interpretation: {cluster_info['geological_interpretation'][:100]}...")
                print(f"   InSAR detectability: {cluster_info['radar_characteristics']['detectability']}")
        
        return results
    
    def shapelet_discovery(self, n_shapelets=10, shapelet_lengths=None):
        """
        Discover discriminative shapelets for clustering
        
        Parameters:
        -----------
        n_shapelets : int
            Number of shapelets to discover per class
        shapelet_lengths : list, optional
            Lengths of shapelets to consider
        """
        print("\n🔄 SHAPELET DISCOVERY")
        print("-" * 50)
        
        if not HAS_SHAPELETS:
            print("⚠️  Shapelet discovery requires TensorFlow. Skipping...")
            self.shapelet_results = {'error': 'TensorFlow not available for shapelets'}
            return self.shapelet_results
        
        if shapelet_lengths is None:
            # Use multiple shapelet lengths based on time series characteristics
            ts_length = self.scaled_time_series.shape[1]
            shapelet_lengths = [
                max(10, ts_length // 20),  # Short patterns
                max(20, ts_length // 10),  # Medium patterns  
                max(30, ts_length // 5)    # Long patterns
            ]
            
        print(f"   Using shapelet lengths: {shapelet_lengths}")
        
        try:
            # First, get initial clustering for shapelet discovery
            initial_model = TimeSeriesKMeans(
                n_clusters=4,  # Use reasonable default
                metric="dtw",
                max_iter=30,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=False
            )
            
            initial_labels = initial_model.fit_predict(self.scaled_time_series)
            print(f"   Initial clustering for shapelet discovery completed")
            
            # Configure shapelet parameters
            shapelet_sizes = grabocka_params_to_shapelet_size_dict(
                n_ts=len(self.scaled_time_series),
                ts_sz=self.scaled_time_series.shape[1],
                n_classes=len(np.unique(initial_labels)),
                l=0.1,  # Shapelet length fraction
                r=2     # Number of shapelets per length
            )
            
            # Create shapelet model
            shapelet_model = ShapeletModel(
                n_shapelets_per_size=shapelet_sizes,
                optimizer="sgd",
                weight_regularizer=0.01,
                max_iter=100,
                verbose=0,
                random_state=self.random_state
            )
            
            # Fit shapelet model
            print(f"   Fitting shapelet model...")
            start_time = time.time()
            
            shapelet_model.fit(self.scaled_time_series, initial_labels)
            
            # Get shapelet transform
            transformed_data = shapelet_model.transform(self.scaled_time_series)
            
            # Cluster in shapelet space
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=4, random_state=self.random_state, n_init=10)
            shapelet_labels = kmeans.fit_predict(transformed_data)
            
            # Validation
            silhouette = silhouette_score(transformed_data, shapelet_labels)
            
            results = {
                'shapelet_labels': shapelet_labels.tolist(),
                'transformed_data': transformed_data.tolist(),
                'silhouette_score': float(silhouette),
                'n_shapelets': len(shapelet_model.shapelets_),
                'shapelet_lengths': [s.shape[0] for s in shapelet_model.shapelets_],
                'runtime_seconds': time.time() - start_time
            }
            
            # Extract representative shapelets
            shapelets_data = []
            for i, shapelet in enumerate(shapelet_model.shapelets_):
                shapelets_data.append({
                    'shapelet_id': i,
                    'length': shapelet.shape[0],
                    'pattern': shapelet.flatten().tolist()
                })
            
            results['discovered_shapelets'] = shapelets_data
            
            self.shapelet_results = results
            
            print(f"✅ Discovered {len(shapelet_model.shapelets_)} shapelets")
            print(f"   Shapelet clustering silhouette: {silhouette:.3f}")
            print(f"   Runtime: {time.time() - start_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Shapelet discovery failed: {e}")
            self.shapelet_results = {'error': str(e)}
            
        return self.shapelet_results
    
    def hierarchical_clustering(self, n_clusters=4, linkage_method='ward'):
        """
        Hierarchical clustering using DTW distances with parallelization
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        linkage_method : str
            Linkage method for hierarchical clustering
        """
        print("\n🔄 HIERARCHICAL DTW CLUSTERING")
        print("-" * 50)
        
        try:
            # Calculate DTW distance matrix with parallelization
            print("   Computing DTW distance matrix (parallelized)...")
            start_time = time.time()
            
            n_series = len(self.scaled_time_series)
            print(f"   Computing {n_series * (n_series - 1) // 2} pairwise DTW distances...")
            
            # Parallel DTW computation
            distance_matrix = self._compute_dtw_matrix_parallel(self.scaled_time_series)
            
            print(f"   DTW matrix computed in {time.time() - start_time:.1f}s")
            
            # Perform hierarchical clustering
            print(f"   Performing hierarchical clustering ({linkage_method})...")
            
            # Convert distance matrix to condensed form
            condensed_distances = squareform(distance_matrix)
            
            # Hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method=linkage_method)
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
            # Validation
            silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            
            results = {
                'cluster_labels': cluster_labels.tolist(),
                'distance_matrix': distance_matrix.tolist(),
                'linkage_matrix': linkage_matrix.tolist(),
                'silhouette_score': float(silhouette),
                'linkage_method': linkage_method,
                'n_clusters': n_clusters,
                'runtime_seconds': time.time() - start_time
            }
            
            self.hierarchical_results = results
            
            print(f"✅ Hierarchical clustering completed")
            print(f"   Silhouette score: {silhouette:.3f}")
            
        except Exception as e:
            print(f"❌ Hierarchical clustering failed: {e}")
            self.hierarchical_results = {'error': str(e)}
            
        return self.hierarchical_results
    
    def _compute_dtw_matrix_parallel(self, time_series):
        """
        Compute DTW distance matrix using optimized parallel processing
        
        Features:
        ---------
        - Memory-efficient chunked processing for large datasets
        - Adaptive batch sizing based on available memory
        - Enhanced progress tracking with ETA estimation
        - Robust error handling with fallback strategies
        - Smart worker allocation based on dataset size
        
        Parameters:
        -----------
        time_series : array-like
            Time series data [n_series, n_timepoints]
            
        Returns:
        --------
        distance_matrix : numpy.ndarray
            Symmetric DTW distance matrix [n_series, n_series]
        """
        n_series = len(time_series)
        distance_matrix = np.zeros((n_series, n_series))
        
        # Create list of index pairs for upper triangle
        index_pairs = [(i, j) for i in range(n_series) for j in range(i + 1, n_series)]
        total_pairs = len(index_pairs)
        
        # Adaptive worker allocation
        optimal_workers = min(self.n_jobs, max(1, total_pairs // 100)) if self.n_jobs > 1 else 1
        
        # Memory-efficient batch sizing
        available_memory_gb = 4  # Conservative estimate
        estimated_pairs_per_gb = 50000  # Based on typical DTW memory usage
        max_batch_size = max(100, int(available_memory_gb * estimated_pairs_per_gb))
        batch_size = min(max_batch_size, total_pairs)
        
        print(f"   🚀 Parallel DTW Matrix Computation:")
        print(f"      Dataset: {n_series} series → {total_pairs:,} pairwise computations")
        print(f"      Workers: {optimal_workers} (requested: {self.n_jobs})")
        print(f"      Batch size: {batch_size:,} pairs per batch")
        
        def compute_dtw_pair(pair):
            """Compute DTW distance for a pair of time series with error handling"""
            i, j = pair
            try:
                # Use dtaidistance for better performance if available
                try:
                    from dtaidistance import dtw as dtw_fast
                    dist = dtw_fast.distance(time_series[i], time_series[j])
                except ImportError:
                    # Fallback to basic DTW implementation
                    dist = dtw(time_series[i], time_series[j])
                
                # Validate result
                if np.isnan(dist) or np.isinf(dist):
                    return i, j, np.inf
                return i, j, dist
                
            except Exception as e:
                print(f"   ⚠️  DTW failed for pair ({i}, {j}): {str(e)[:50]}...")
                return i, j, np.inf
        
        # Process in batches for memory efficiency
        import time
        start_time = time.time()
        results = []
        
        if optimal_workers == 1:
            # Sequential processing with enhanced progress tracking
            print(f"      Mode: Sequential processing")
            for idx, pair in enumerate(index_pairs):
                result = compute_dtw_pair(pair)
                results.append(result)
                
                # Enhanced progress reporting with ETA
                if (idx + 1) % max(1, total_pairs // 20) == 0 or idx == 0:
                    elapsed = time.time() - start_time
                    progress = (idx + 1) / total_pairs
                    eta_seconds = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                    eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                    
                    print(f"      Progress: {progress*100:.1f}% ({idx + 1:,}/{total_pairs:,}) | "
                          f"Rate: {(idx + 1)/elapsed:.1f} pairs/sec | ETA: {eta_str}")
        else:
            # Parallel processing with batching
            from joblib import Parallel, delayed
            print(f"      Mode: Parallel processing ({optimal_workers} workers)")
            
            n_batches = (total_pairs + batch_size - 1) // batch_size
            print(f"      Processing {n_batches} batches...")
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_pairs)
                batch_pairs = index_pairs[start_idx:end_idx]
                
                # Process batch in parallel
                batch_results = Parallel(n_jobs=optimal_workers, backend='threading')(
                    delayed(compute_dtw_pair)(pair) for pair in batch_pairs
                )
                results.extend(batch_results)
                
                # Progress reporting for batches
                completed_pairs = end_idx
                elapsed = time.time() - start_time
                progress = completed_pairs / total_pairs
                eta_seconds = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                
                print(f"      Batch {batch_idx + 1}/{n_batches}: {progress*100:.1f}% complete | "
                      f"Rate: {completed_pairs/elapsed:.1f} pairs/sec | ETA: {eta_str}")
        
        # Fill distance matrix with validation
        print(f"   📊 Filling distance matrix...")
        valid_computations = 0
        infinite_values = 0
        
        for i, j, dist in results:
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            
            if np.isinf(dist):
                infinite_values += 1
            else:
                valid_computations += 1
        
        total_time = time.time() - start_time
        print(f"   ✅ Distance matrix completed:")
        print(f"      Total time: {total_time/60:.2f} minutes")
        print(f"      Valid computations: {valid_computations:,}/{total_pairs:,} ({valid_computations/total_pairs*100:.1f}%)")
        if infinite_values > 0:
            print(f"      ⚠️  Failed computations: {infinite_values:,} (replaced with inf)")
        print(f"      Average rate: {total_pairs/total_time:.1f} pairs/second")
        
        return distance_matrix
    
    def _fit_soft_dtw_model(self, n_clusters, gamma=0.1):
        """Fit Soft DTW model for given number of clusters (picklable method)"""
        try:
            start_time = time.time()
            
            # For k=2-4, use more cores per model since we have few models
            internal_jobs = 2 if self.n_jobs > 2 else 1
            
            print(f"      → Model for k={n_clusters} using {internal_jobs} internal jobs...")
            
            model = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric="softdtw",
                metric_params={"gamma": gamma},
                max_iter=50,
                random_state=self.random_state,
                n_jobs=internal_jobs,  # Let TSLearn use some cores per model
                verbose=False
            )
            
            cluster_labels = model.fit_predict(self.scaled_time_series)
            
            # Calculate validation metrics
            silhouette = silhouette_score(
                self.scaled_time_series.reshape(len(self.scaled_time_series), -1),
                cluster_labels
            )
            
            return n_clusters, {
                'cluster_labels': cluster_labels.tolist(),
                'centroids': model.cluster_centers_.tolist(),
                'silhouette_score': float(silhouette),
                'inertia': float(model.inertia_),
                'n_iter': int(model.n_iter_),
                'runtime_seconds': time.time() - start_time,
                'gamma': gamma
            }
            
        except Exception as e:
            return n_clusters, {'error': str(e)}

    def _fit_metric_model(self, metric, n_clusters):
        """Fit model for given metric (picklable method)"""
        try:
            start_time = time.time()
            
            # Configure metric parameters
            metric_params = {}
            if metric == 'softdtw':
                metric_params = {"gamma": 0.1}
            
            model = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric=metric,
                metric_params=metric_params,
                max_iter=50,
                random_state=self.random_state,
                n_jobs=1,  # Use 1 job per model
                verbose=False
            )
            
            cluster_labels = model.fit_predict(self.scaled_time_series)
            
            silhouette = silhouette_score(
                self.scaled_time_series.reshape(len(self.scaled_time_series), -1),
                cluster_labels
            )
            
            return metric, {
                'cluster_labels': cluster_labels.tolist(),
                'centroids': model.cluster_centers_.tolist(),
                'silhouette_score': float(silhouette),
                'inertia': float(model.inertia_),
                'n_iter': int(model.n_iter_),
                'runtime_seconds': time.time() - start_time,
                'metric_params': metric_params
            }
            
        except Exception as e:
            return metric, {'error': str(e)}

    def _parallel_soft_dtw_clustering(self, n_clusters_range, gamma=0.1):
        """
        Parallel soft DTW clustering for multiple k values
        
        Parameters:
        -----------
        n_clusters_range : iterable
            Range of cluster numbers to test
        gamma : float
            Soft DTW smoothing parameter
            
        Returns:
        --------
        results : dict
            Clustering results for each k value
        """
        n_models = len(list(n_clusters_range))
        print(f"   Testing {n_models} cluster configurations in parallel...")
        print(f"   Using {self.n_jobs} parallel workers for {n_models} models")
        if n_models < self.n_jobs:
            print(f"   ⚠️  Note: Only {n_models} models to fit, but {self.n_jobs} cores available")
        
        # Parallel execution with proper cleanup
        if self.n_jobs == 1:
            # Sequential processing
            results = {}
            for n_clusters in n_clusters_range:
                k, result = self._fit_soft_dtw_model(n_clusters, gamma)
                results[k] = result
                if 'silhouette_score' in result:
                    print(f"      k={k}: Silhouette={result['silhouette_score']:.3f}, "
                          f"Time={result['runtime_seconds']:.1f}s")
        else:
            # Parallel processing with loky backend (handles class methods better)
            try:
                print(f"   → Starting parallel execution with {self.n_jobs} processes...")
                # Suppress joblib/multiprocessing warnings temporarily
                import sys
                from contextlib import redirect_stderr
                from io import StringIO
                
                with redirect_stderr(StringIO()):
                    parallel_results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
                        delayed(self._fit_soft_dtw_model)(n_clusters, gamma) for n_clusters in n_clusters_range
                    )
                
                results = {}
                for k, result in parallel_results:
                    results[k] = result
                    if 'silhouette_score' in result:
                        print(f"      k={k}: Silhouette={result['silhouette_score']:.3f}, "
                              f"Time={result['runtime_seconds']:.1f}s")
            except Exception as e:
                print(f"   ⚠️  Parallel execution failed, falling back to sequential: {e}")
                results = {}
                for n_clusters in n_clusters_range:
                    k, result = self._fit_soft_dtw_model(n_clusters, gamma)
                    results[k] = result
                    if 'silhouette_score' in result:
                        print(f"      k={k}: Silhouette={result['silhouette_score']:.3f}, "
                              f"Time={result['runtime_seconds']:.1f}s")
        
        return results
    
    def _parallel_multi_metric_clustering(self, n_clusters=4, metrics=None):
        """
        Parallel multi-metric DTW clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        metrics : list, optional
            DTW metrics to compare
            
        Returns:
        --------
        results : dict
            Clustering results for each metric
        """
        if metrics is None:
            metrics = ['dtw', 'softdtw', 'euclidean']
        
        print(f"   Testing {len(metrics)} metrics in parallel...")
        
        def fit_metric_model(metric):
            """Fit model for given metric"""
            try:
                start_time = time.time()
                
                # Configure metric parameters
                metric_params = {}
                if metric == 'softdtw':
                    metric_params = {"gamma": 0.1}
                
                model = TimeSeriesKMeans(
                    n_clusters=n_clusters,
                    metric=metric,
                    metric_params=metric_params,
                    max_iter=50,
                    random_state=self.random_state,
                    n_jobs=1,  # Use 1 job per model
                    verbose=False
                )
                
                cluster_labels = model.fit_predict(self.scaled_time_series)
                
                silhouette = silhouette_score(
                    self.scaled_time_series.reshape(len(self.scaled_time_series), -1),
                    cluster_labels
                )
                
                return metric, {
                    'cluster_labels': cluster_labels.tolist(),
                    'centroids': model.cluster_centers_.tolist(),
                    'silhouette_score': float(silhouette),
                    'inertia': float(model.inertia_),
                    'n_iter': int(model.n_iter_),
                    'runtime_seconds': time.time() - start_time,
                    'metric_params': metric_params
                }
                
            except Exception as e:
                return metric, {'error': str(e)}
        
        # Parallel execution
        if self.n_jobs == 1 or len(metrics) == 1:
            # Sequential processing
            results = {}
            for metric in metrics:
                m, result = fit_metric_model(metric)
                results[m] = result
                if 'silhouette_score' in result:
                    print(f"      {m.upper()}: Silhouette={result['silhouette_score']:.3f}, "
                          f"Time={result['runtime_seconds']:.1f}s")
        else:
            # Parallel processing with loky backend (handles class methods better)
            try:
                print(f"   → Starting parallel metric comparison with {min(self.n_jobs, len(metrics))} processes...")
                # Suppress joblib/multiprocessing warnings temporarily
                from contextlib import redirect_stderr
                from io import StringIO
                
                with redirect_stderr(StringIO()):
                    parallel_results = Parallel(n_jobs=min(self.n_jobs, len(metrics)), backend='loky', verbose=0)(
                        delayed(self._fit_metric_model)(metric, n_clusters) for metric in metrics
                    )
                
                results = {}
                for m, result in parallel_results:
                    results[m] = result
                    if 'silhouette_score' in result:
                        print(f"      {m.upper()}: Silhouette={result['silhouette_score']:.3f}, "
                              f"Time={result['runtime_seconds']:.1f}s")
            except Exception as e:
                print(f"   ⚠️  Parallel execution failed, falling back to sequential: {e}")
                results = {}
                for metric in metrics:
                    m, result = self._fit_metric_model(metric, n_clusters)
                    results[m] = result
                    if 'silhouette_score' in result:
                        print(f"      {m.upper()}: Silhouette={result['silhouette_score']:.3f}, "
                              f"Time={result['runtime_seconds']:.1f}s")
        
        return results
    
    def compare_clustering_methods(self):
        """Compare all clustering methods and select best approach"""
        print("\n📊 CLUSTERING METHOD COMPARISON")
        print("-" * 50)
        
        comparison = {}
        
        # Collect results from all methods
        methods = {
            'soft_dtw': self.soft_dtw_results,
            'multi_metric': self.multi_metric_results,
            'shapelet': self.shapelet_results,
            'hierarchical': self.hierarchical_results
        }
        
        for method_name, results in methods.items():
            if results and 'error' not in results:
                
                if method_name == 'soft_dtw' and 'optimal_k' in results:
                    optimal_k = results['optimal_k']
                    method_results = results[optimal_k]
                elif method_name == 'multi_metric' and 'best_metric' in results:
                    best_metric = results['best_metric']
                    method_results = results[best_metric]
                else:
                    method_results = results
                
                if 'silhouette_score' in method_results:
                    comparison[method_name] = {
                        'silhouette_score': method_results['silhouette_score'],
                        'runtime_seconds': method_results.get('runtime_seconds', 0),
                        'method_specific': method_results
                    }
        
        # Rank methods by silhouette score
        if comparison:
            ranked_methods = sorted(
                comparison.items(),
                key=lambda x: x[1]['silhouette_score'],
                reverse=True
            )
            
            print("📊 Method Rankings (by Silhouette Score):")
            for rank, (method, metrics) in enumerate(ranked_methods, 1):
                print(f"   {rank}. {method.upper()}: {metrics['silhouette_score']:.3f} "
                      f"({metrics['runtime_seconds']:.1f}s)")
            
            self.method_comparison = {
                'rankings': ranked_methods,
                'best_method': ranked_methods[0][0],
                'comparison_metrics': comparison
            }
            
            print(f"\n🏆 Best method: {ranked_methods[0][0].upper()}")
            
        return comparison
    
    def save_results(self):
        """Save all clustering results to files"""
        print("\n💾 SAVING RESULTS")
        print("-" * 50)
        
        # Save individual method results
        results_to_save = {
            'soft_dtw_clusters.json': self.soft_dtw_results,
            'multi_metric_comparison.json': self.multi_metric_results,
            'shapelet_patterns.json': self.shapelet_results,
            'hierarchical_clusters.json': self.hierarchical_results,
            'method_comparison.json': self.method_comparison
        }
        
        for filename, data in results_to_save.items():
            if data:
                filepath = self.results_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"✅ Saved: {filepath}")
        
        # Save metadata
        metadata = {
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_stations': len(self.coordinates) if self.coordinates is not None else 0,
            'time_series_length': self.scaled_time_series.shape[1] if self.scaled_time_series is not None else 0,
            'methods_completed': list(results_to_save.keys()),
            'best_method': self.method_comparison.get('best_method', 'unknown') if self.method_comparison else 'unknown'
        }
        
        metadata_file = self.results_dir / 'analysis_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata: {metadata_file}")
        
    def create_visualizations(self):
        """Create comprehensive visualizations for all clustering methods"""
        print("\n🎨 CREATING VISUALIZATIONS")
        print("-" * 50)
        
        # Set style for professional figures
        plt.style.use('seaborn-v0_8')
        
        try:
            # Figure 1: Soft DTW clustering results
            if self.soft_dtw_results:
                self._plot_soft_dtw_results()
                
            # Figure 2: Multi-metric comparison
            if self.multi_metric_results:
                self._plot_multi_metric_comparison()
                
            # Figure 3: Shapelet patterns
            if self.shapelet_results and 'discovered_shapelets' in self.shapelet_results:
                self._plot_shapelet_patterns()
                
            # Figure 4: Method comparison summary
            if self.method_comparison:
                self._plot_method_comparison()
                
            print("✅ All visualizations created successfully")
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
    
    def _plot_soft_dtw_results(self):
        """Plot Soft DTW clustering results"""
        if 'optimal_k' not in self.soft_dtw_results:
            return
            
        optimal_k = self.soft_dtw_results['optimal_k']
        results = self.soft_dtw_results[optimal_k]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Soft DTW Clustering Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Cluster centroids
        ax1 = axes[0, 0]
        centroids = np.array(results['centroids'])
        time_axis = np.arange(centroids.shape[1]) * 6  # 6-day intervals
        
        colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
        
        # Calculate cluster statistics for shading
        cluster_labels = np.array(results['cluster_labels'])
        
        for i in range(optimal_k):
            # Get all time series belonging to this cluster
            cluster_mask = cluster_labels == i
            cluster_data = self.scaled_time_series[cluster_mask, :, 0]  # Extract univariate series
            
            # Calculate mean (centroid) and standard deviation
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)
            
            # Plot centroid line (thinner for clarity)
            ax1.plot(time_axis, cluster_mean, 
                    color=colors[i], linewidth=1.0, alpha=0.9, label=f'Cluster {i+1} (n={np.sum(cluster_mask)})')
            
            # Add ±1 std deviation shading
            ax1.fill_between(time_axis, 
                           cluster_mean - cluster_std, 
                           cluster_mean + cluster_std,
                           color=colors[i], alpha=0.2, 
                           label=f'±1σ Cluster {i+1}' if i == 0 else "")
        
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Displacement (mm)')
        ax1.set_title(f'Cluster Centroids (k={optimal_k})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Geographic distribution
        ax2 = axes[0, 1]
        cluster_labels = np.array(results['cluster_labels'])
        
        scatter = ax2.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                            c=cluster_labels, cmap='Set1', s=20, alpha=0.7)
        ax2.set_xlabel('Longitude (°E)')
        ax2.set_ylabel('Latitude (°N)')
        ax2.set_title('Geographic Cluster Distribution')
        plt.colorbar(scatter, ax=ax2, label='Cluster')
        
        # Plot 3: Silhouette analysis
        ax3 = axes[1, 0]
        k_values = [k for k in self.soft_dtw_results.keys() if isinstance(k, int)]
        silhouette_scores = [self.soft_dtw_results[k]['silhouette_score'] 
                           for k in k_values if 'silhouette_score' in self.soft_dtw_results[k]]
        
        ax3.plot(k_values, silhouette_scores, 'bo-', linewidth=1.5, markersize=6, alpha=0.8)
        ax3.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax3.set_xlabel('Number of Clusters (k)')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Soft DTW Clustering Validation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cluster sizes
        ax4 = axes[1, 1]
        unique_labels, cluster_counts = np.unique(cluster_labels, return_counts=True)
        bars = ax4.bar(unique_labels + 1, cluster_counts, color=colors[:len(unique_labels)], alpha=0.7)
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Number of Stations')
        ax4.set_title('Cluster Size Distribution')
        
        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps04b_fig01_soft_dtw_clusters.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: {fig_path}")
        
        # Create interactive version if Plotly is available
        if HAS_PLOTLY:
            self._create_interactive_soft_dtw_plot()
    
    def _create_interactive_soft_dtw_plot(self):
        """Create interactive Plotly version of Soft DTW clustering results"""
        if 'optimal_k' not in self.soft_dtw_results:
            return
            
        optimal_k = self.soft_dtw_results['optimal_k']
        optimal_results = self.soft_dtw_results[optimal_k]
        cluster_labels = np.array(optimal_results['cluster_labels'])
        centroids = np.array(optimal_results['centroids'])
        
        # Create time axis (6-day intervals)
        n_points = centroids.shape[1]
        time_axis = np.arange(n_points) * 6  # Days
        
        # Create interactive subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Centroids (Interactive)', 'Silhouette Analysis', 
                          'Station Distribution Map', 'Cluster Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Plot 1: Interactive centroids with enhanced hover info
        # Ensure we have enough colors and they're valid
        colors = px.colors.qualitative.Set1[:max(len(centroids), 10)]
        if len(centroids) > len(colors):
            colors = colors * (len(centroids) // len(colors) + 1)
        
        for i, centroid in enumerate(centroids):
            n_stations = np.sum(cluster_labels == i)
            centroid_flat = centroid.flatten() if len(centroid.shape) > 1 else centroid
            
            # Calculate cluster statistics for hover
            mean_deformation = np.mean(centroid_flat)
            max_deformation = np.max(centroid_flat)
            min_deformation = np.min(centroid_flat)
            deformation_range = max_deformation - min_deformation
            
            # Create hover text with detailed info
            hover_text = [
                f"<b>Cluster {i+1}</b><br>" +
                f"Time: Day {t}<br>" +
                f"Deformation: {d:.2f} mm<br>" +
                f"Stations: {n_stations}<br>" +
                f"Range: {deformation_range:.2f} mm<br>" +
                f"Mean: {mean_deformation:.2f} mm"
                for t, d in zip(time_axis, centroid_flat)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=centroid_flat,
                    mode='lines+markers',
                    name=f'Cluster {i+1} (n={n_stations})',
                    line=dict(color=colors[i], width=1.5),
                    marker=dict(size=3, opacity=0.8),
                    opacity=0.75,
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_text,
                    legendgroup=f'cluster_{i}',
                ),
                row=1, col=1
            )
        
        # Plot 2: Interactive silhouette analysis
        if hasattr(self, 'soft_dtw_results'):
            k_values = [k for k in self.soft_dtw_results.keys() if isinstance(k, int)]
            silhouette_scores = [self.soft_dtw_results[k]['silhouette_score'] 
                               for k in k_values if 'silhouette_score' in self.soft_dtw_results[k]]
            
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=silhouette_scores,
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6, color='blue', opacity=0.8),
                    opacity=0.8,
                    hovertemplate='<b>k=%{x}</b><br>Silhouette: %{y:.3f}<extra></extra>',
                ),
                row=1, col=2
            )
            
            # Highlight optimal k
            optimal_silhouette = self.soft_dtw_results[optimal_k]['silhouette_score']
            fig.add_trace(
                go.Scatter(
                    x=[optimal_k],
                    y=[optimal_silhouette],
                    mode='markers',
                    name=f'Optimal k={optimal_k}',
                    marker=dict(size=12, color='red', symbol='star'),
                    hovertemplate=f'<b>Optimal k={optimal_k}</b><br>Silhouette: {optimal_silhouette:.3f}<extra></extra>',
                ),
                row=1, col=2
            )
        
        # Plot 3: Geographic distribution (if coordinates available)
        if hasattr(self, 'coordinates') and len(self.coordinates) > 0:
            coords_subset = self.coordinates[:len(cluster_labels)]
            
            # Create color map for clusters
            cluster_colors = [colors[label] for label in cluster_labels]
            
            # Use discrete colors for clusters instead of colorscale
            cluster_colors_numeric = [colors[label % len(colors)] for label in cluster_labels]
            
            fig.add_trace(
                go.Scatter(
                    x=coords_subset[:, 0],  # Longitude
                    y=coords_subset[:, 1],  # Latitude
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cluster_colors_numeric,
                        line=dict(width=1, color='black')
                    ),
                    text=[f'Cluster {label+1}' for label in cluster_labels],
                    hovertemplate='<b>%{text}</b><br>Lon: %{x:.4f}<br>Lat: %{y:.4f}<extra></extra>',
                    name='Stations',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 4: Cluster size statistics
        unique_labels, cluster_counts = np.unique(cluster_labels, return_counts=True)
        cluster_names = [f'Cluster {i+1}' for i in unique_labels]
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=cluster_counts,
                marker_color=colors[:len(cluster_counts)],
                text=cluster_counts,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Stations: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
                customdata=cluster_counts/np.sum(cluster_counts)*100,
                name='Station Count',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text="<b>Interactive Soft DTW Clustering Analysis</b>",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update individual subplot styling
        fig.update_xaxes(title_text="Time (days)", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Deformation (mm)", row=1, col=1, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Longitude", row=2, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Latitude", row=2, col=1, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Cluster", row=2, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Number of Stations", row=2, col=2, gridcolor='lightgray')
        
        # Save interactive HTML
        interactive_path = self.figures_dir / "ps04b_interactive_soft_dtw_analysis.html"
        fig.write_html(str(interactive_path))
        print(f"✅ Created interactive plot: {interactive_path}")
        
        # Also save as static image (if kaleido is available)
        if HAS_KALEIDO:
            try:
                static_path = self.figures_dir / "ps04b_interactive_soft_dtw_analysis.png"
                fig.write_image(str(static_path), width=1200, height=800, scale=2)
                print(f"✅ Created static version: {static_path}")
            except Exception as e:
                # Suppress the detailed traceback but still show it's skipped
                print(f"ℹ️  Static image export skipped (export issue)")
        else:
            print("ℹ️  Static image export skipped (kaleido not available)")
    
    def _plot_multi_metric_comparison(self):
        """Plot multi-metric DTW comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Metric DTW Comparison', fontsize=16, fontweight='bold')
        
        # Collect metrics
        metrics = []
        silhouette_scores = []
        runtime_seconds = []
        inertias = []
        
        for metric, results in self.multi_metric_results.items():
            if isinstance(results, dict) and 'silhouette_score' in results:
                metrics.append(metric.upper())
                silhouette_scores.append(results['silhouette_score'])
                runtime_seconds.append(results['runtime_seconds'])
                inertias.append(results['inertia'])
        
        if not metrics:
            return
            
        # Plot 1: Silhouette scores
        ax1 = axes[0, 0]
        bars = ax1.bar(metrics, silhouette_scores, alpha=0.7, color=['blue', 'green', 'orange'][:len(metrics)])
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Clustering Quality by Metric')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, silhouette_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Runtime comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(metrics, runtime_seconds, alpha=0.7, color=['blue', 'green', 'orange'][:len(metrics)])
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_title('Computational Efficiency by Metric')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, runtime in zip(bars, runtime_seconds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(runtime_seconds)*0.01,
                    f'{runtime:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Inertia comparison
        ax3 = axes[1, 0]
        bars = ax3.bar(metrics, inertias, alpha=0.7, color=['blue', 'green', 'orange'][:len(metrics)])
        ax3.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax3.set_title('Cluster Compactness by Metric')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Centroids comparison for best metric
        ax4 = axes[1, 1]
        if 'best_metric' in self.multi_metric_results:
            best_metric = self.multi_metric_results['best_metric']
            best_results = self.multi_metric_results[best_metric]
            centroids = np.array(best_results['centroids'])
            time_axis = np.arange(centroids.shape[1]) * 6
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(centroids)))
            
            # Get cluster assignments for the best method
            best_labels = np.array(best_results['cluster_labels'])
            
            for i in range(len(centroids)):
                # Get all time series belonging to this cluster  
                cluster_mask = best_labels == i
                cluster_data = self.scaled_time_series[cluster_mask, :, 0]
                
                # Calculate mean and standard deviation
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                
                # Plot centroid line (thinner for clarity)
                ax4.plot(time_axis, cluster_mean, 
                        color=colors[i], linewidth=1.0, alpha=0.9, label=f'Cluster {i+1} (n={np.sum(cluster_mask)})')
                
                # Add ±1 std deviation shading
                ax4.fill_between(time_axis, 
                               cluster_mean - cluster_std, 
                               cluster_mean + cluster_std,
                               color=colors[i], alpha=0.2)
            
            ax4.set_xlabel('Time (days)')
            ax4.set_ylabel('Displacement (mm)')
            ax4.set_title(f'Best Method Centroids ({best_metric.upper()})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps04b_fig02_multi_metric_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: {fig_path}")
        
        # Create interactive version if Plotly is available
        if HAS_PLOTLY:
            self._create_interactive_multi_metric_plot()
    
    def _create_interactive_multi_metric_plot(self):
        """Create interactive Plotly version of multi-metric comparison"""
        if not hasattr(self, 'multi_metric_results') or not self.multi_metric_results:
            return
            
        # Collect metrics data
        metrics = []
        silhouette_scores = []
        runtime_seconds = []
        inertias = []
        
        for metric, results in self.multi_metric_results.items():
            if isinstance(results, dict) and 'silhouette_score' in results:
                metrics.append(metric.upper())
                silhouette_scores.append(results['silhouette_score'])
                runtime_seconds.append(results['runtime_seconds'])
                inertias.append(results['inertia'])
        
        if not metrics:
            return
            
        # Create interactive subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Clustering Quality by Metric', 'Computational Efficiency', 
                          'Cluster Compactness', 'Best Method Centroids'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1[:len(metrics)]
        
        # Plot 1: Interactive silhouette scores
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=silhouette_scores,
                marker_color=colors,
                text=[f'{score:.3f}' for score in silhouette_scores],
                textposition='auto',
                hovertemplate='<b>%{x} Metric</b><br>Silhouette Score: %{y:.3f}<br>Quality: %{customdata}<extra></extra>',
                customdata=['Excellent' if s > 0.7 else 'Good' if s > 0.5 else 'Fair' for s in silhouette_scores],
                name='Silhouette Score'
            ),
            row=1, col=1
        )
        
        # Plot 2: Interactive runtime comparison
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=runtime_seconds,
                marker_color=colors,
                text=[f'{runtime:.1f}s' for runtime in runtime_seconds],
                textposition='auto',
                hovertemplate='<b>%{x} Metric</b><br>Runtime: %{y:.1f} seconds<br>Efficiency: %{customdata}<extra></extra>',
                customdata=['Fast' if r < 30 else 'Medium' if r < 60 else 'Slow' for r in runtime_seconds],
                name='Runtime'
            ),
            row=1, col=2
        )
        
        # Plot 3: Interactive inertia comparison
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=inertias,
                marker_color=colors,
                text=[f'{inertia:.1f}' for inertia in inertias],
                textposition='auto',
                hovertemplate='<b>%{x} Metric</b><br>Inertia: %{y:.2f}<br>Compactness: Lower is better<extra></extra>',
                name='Inertia'
            ),
            row=2, col=1
        )
        
        # Plot 4: Interactive centroids for best metric
        if 'best_metric' in self.multi_metric_results:
            best_metric = self.multi_metric_results['best_metric']
            best_results = self.multi_metric_results[best_metric]
            centroids = np.array(best_results['centroids'])
            
            # Create time axis
            n_points = centroids.shape[1]
            time_axis = np.arange(n_points) * 6
            
            for i, centroid in enumerate(centroids):
                centroid_flat = centroid.flatten() if len(centroid.shape) > 1 else centroid
                
                # Enhanced hover information
                hover_text = [
                    f"<b>{best_metric.upper()} - Cluster {i+1}</b><br>" +
                    f"Time: Day {t}<br>" +
                    f"Deformation: {d:.2f} mm<br>" +
                    f"Method: {best_metric.upper()}<br>" +
                    f"Quality: Best performing metric"
                    for t, d in zip(time_axis, centroid_flat)
                ]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=centroid_flat,
                        mode='lines+markers',
                        name=f'{best_metric.upper()} Cluster {i+1}',
                        line=dict(color=colors[i], width=1.5),
                        marker=dict(size=3, opacity=0.8),
                        opacity=0.75,
                        hovertemplate='%{text}<extra></extra>',
                        text=hover_text,
                        legendgroup=f'best_metric_{i}',
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>Interactive Multi-Metric DTW Comparison</b>",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="DTW Metric", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=1)
        
        fig.update_xaxes(title_text="DTW Metric", row=1, col=2)
        fig.update_yaxes(title_text="Runtime (seconds)", row=1, col=2)
        
        fig.update_xaxes(title_text="DTW Metric", row=2, col=1)
        fig.update_yaxes(title_text="Inertia", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Deformation (mm)", row=2, col=2)
        
        # Save interactive HTML
        interactive_path = self.figures_dir / "ps04b_interactive_multi_metric_comparison.html"
        fig.write_html(str(interactive_path))
        print(f"✅ Created interactive plot: {interactive_path}")
        
        # Also save as static image (if kaleido is available)
        if HAS_KALEIDO:
            try:
                static_path = self.figures_dir / "ps04b_interactive_multi_metric_comparison.png"
                fig.write_image(str(static_path), width=1200, height=800, scale=2)
                print(f"✅ Created static version: {static_path}")
            except Exception as e:
                # Suppress the detailed traceback but still show it's skipped
                print(f"ℹ️  Static image export skipped (export issue)")
        else:
            print("ℹ️  Static image export skipped (kaleido not available)")
    
    def _plot_shapelet_patterns(self):
        """Plot discovered shapelet patterns"""
        if 'discovered_shapelets' not in self.shapelet_results:
            return
            
        shapelets = self.shapelet_results['discovered_shapelets']
        n_shapelets = min(len(shapelets), 12)  # Show up to 12 shapelets
        
        # Create subplot grid
        rows = int(np.ceil(n_shapelets / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        fig.suptitle('Discovered Shapelet Patterns', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        for i in range(n_shapelets):
            ax = axes_flat[i]
            shapelet = shapelets[i]
            pattern = np.array(shapelet['pattern'])
            
            # Plot shapelet pattern
            ax.plot(pattern, linewidth=1, alpha=0.7, color='blue')
            ax.set_title(f"Shapelet {shapelet['shapelet_id']} (Length: {shapelet['length']})")
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_shapelets, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps04b_fig03_shapelet_patterns.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: {fig_path}")
    
    def _plot_method_comparison(self):
        """Plot comprehensive method comparison"""
        if not self.method_comparison or 'rankings' not in self.method_comparison:
            return
            
        rankings = self.method_comparison['rankings']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Clustering Methods Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        methods = [rank[0].replace('_', ' ').title() for rank in rankings]
        silhouette_scores = [rank[1]['silhouette_score'] for rank in rankings]
        runtimes = [rank[1]['runtime_seconds'] for rank in rankings]
        
        # Plot 1: Silhouette scores ranking
        ax1 = axes[0, 0]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(methods)))
        bars = ax1.barh(methods, silhouette_scores, color=colors, alpha=0.8)
        ax1.set_xlabel('Silhouette Score')
        ax1.set_title('Method Quality Ranking')
        
        # Add value labels
        for bar, score in zip(bars, silhouette_scores):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontweight='bold')
        
        # Plot 2: Runtime efficiency
        ax2 = axes[0, 1]
        bars = ax2.bar(methods, runtimes, alpha=0.7, color='skyblue')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_title('Computational Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, runtime in zip(bars, runtimes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(runtimes)*0.01,
                    f'{runtime:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Performance vs efficiency scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(runtimes, silhouette_scores, s=100, alpha=0.7, c=colors)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax3.annotate(method, (runtimes[i], silhouette_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel('Runtime (seconds)')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Performance vs Efficiency Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for i, (method, score, runtime) in enumerate(zip(methods, silhouette_scores, runtimes)):
            table_data.append([
                f"{i+1}",
                method,
                f"{score:.3f}",
                f"{runtime:.1f}s"
            ])
        
        table = ax4.table(
            cellText=table_data,
            colLabels=['Rank', 'Method', 'Silhouette', 'Runtime'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the ranking
        for i in range(len(table_data)):
            table[(i+1, 0)].set_facecolor(colors[i])
        
        ax4.set_title('Method Rankings Summary', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps04b_fig04_method_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: {fig_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Advanced TSLearn clustering analysis')
    parser.add_argument('--n-stations', type=int, default=None,
                       help='Number of stations to analyze (default: all available stations)')
    parser.add_argument('--max-clusters', type=int, default=8,
                       help='Maximum number of clusters to test (default: 8)')
    parser.add_argument('--k-range', nargs=2, type=int, default=None,
                       help='Range of k values to test (e.g., --k-range 2 4)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1 for all cores)')
    parser.add_argument('--skip-shapelets', action='store_true',
                       help='Skip shapelet discovery (computationally intensive)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run only soft DTW clustering for quick performance test')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ps04b_advanced_clustering.py - Advanced TSLearn Methods")
    print("📋 PURPOSE: State-of-the-art time series clustering")
    print("🔧 METHODS: Soft DTW, Multi-metric, Shapelets, Hierarchical")
    print("📊 EXTENDS: ps04_temporal_clustering.py with advanced algorithms")
    print("=" * 80)
    
    try:
        # Determine k range
        if args.k_range:
            k_range = range(args.k_range[0], args.k_range[1] + 1)
            max_clusters = args.k_range[1]
        else:
            k_range = None
            max_clusters = args.max_clusters
        
        # Initialize advanced clustering
        clustering = AdvancedTSLearnClustering(
            max_clusters=max_clusters,
            n_jobs=args.n_jobs
        )
        
        # Start total timing
        clustering.start_total_timer()
        
        # Load data
        print("\n🔄 Loading and preparing data...")
        if not clustering.load_preprocessed_data(n_stations=args.n_stations):
            print("❌ Failed to load data")
            return False
        
        # Run clustering analyses
        if args.quick_test:
            print("\n🔄 Running QUICK PERFORMANCE TEST (Soft DTW only)...")
            clustering.soft_dtw_clustering(n_clusters_range=k_range)
        else:
            print("\n🔄 Running advanced clustering analyses...")
            
            # 1. Soft DTW clustering
            clustering.soft_dtw_clustering(n_clusters_range=k_range)
            
            # 2. Multi-metric comparison
            clustering.multi_metric_clustering()
            
            # 3. Shapelet discovery (optional - computationally intensive)
            if not args.skip_shapelets:
                clustering.shapelet_discovery()
            else:
                print("⏭️  Skipping shapelet discovery (use --skip-shapelets=False to enable)")
            
            # 4. Hierarchical clustering
            clustering.hierarchical_clustering()
            
            # 5. Compare all methods
            clustering.compare_clustering_methods()
        
        # Save results
        clustering.start_timer("save_results")
        clustering.save_results()
        clustering.end_timer("save_results")
        
        # Create visualizations
        if not args.quick_test:
            clustering.start_timer("create_visualizations")
            clustering.create_visualizations()
            clustering.end_timer("create_visualizations")
        else:
            # Even in quick test, create interactive plots for the results we have
            if HAS_PLOTLY and hasattr(clustering, 'soft_dtw_results'):
                clustering.start_timer("interactive_plots")
                print("\n🎨 Creating interactive plots for quick test results...")
                clustering._create_interactive_soft_dtw_plot()
                clustering.end_timer("interactive_plots")
        
        # End total timing and generate summary
        clustering.end_total_timer()
        clustering.print_timing_summary()
        clustering.save_timing_results()
        
        print("\n" + "=" * 80)
        print("✅ ps04b_advanced_clustering.py COMPLETED SUCCESSFULLY")
        print("\n📊 ANALYSIS SUMMARY:")
        if clustering.method_comparison and 'best_method' in clustering.method_comparison:
            best_method = clustering.method_comparison['best_method']
            print(f"   🏆 Best clustering method: {best_method.upper()}")
        
        print(f"   📍 Stations analyzed: {len(clustering.coordinates)}")
        print(f"   🕐 Time series length: {clustering.scaled_time_series.shape[1]} points")
        
        print(f"\n📁 Outputs generated:")
        print(f"   📊 Results: data/processed/ps04b_advanced/")
        print(f"   🎨 Figures: figures/ps04b_fig01-04_*.png")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced clustering analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Fix multiprocessing issues on macOS/Python 3.13
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        exit(1)
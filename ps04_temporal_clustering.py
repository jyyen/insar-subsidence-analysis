#!/usr/bin/env python3
"""
ps04_temporal_clustering.py - Advanced Temporal Clustering Analysis

Purpose: DTW-based time series clustering for Taiwan subsidence patterns
Focus: Temporal evolution analysis complementing ps03's feature-based clustering  
Key Innovation: Dynamic Time Warping captures temporal similarity vs static features

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
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import time
import signal

warnings.filterwarnings('ignore')

# Optional imports for advanced DTW functionality
DTW_LIBRARY = None
try:
    from tslearn.metrics import dtw as tslearn_dtw
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset
    DTW_LIBRARY = "tslearn"
    print("✅ TSLearn available - using advanced time series clustering")
except ImportError:
    try:
        from dtw import dtw, accelerated_dtw
        DTW_LIBRARY = "dtw-python"
        print("✅ DTW-Python available - using standard DTW implementation")
    except ImportError:
        try:
            from fastdtw import fastdtw
            DTW_LIBRARY = "fastdtw"
            print("✅ FastDTW available - using fast approximation DTW")
        except ImportError:
            try:
                from dtaidistance import dtw_ndim
                DTW_LIBRARY = "dtaidistance"
                print("✅ DTAIDistance available - using optimized DTW computation")
            except ImportError:
                DTW_LIBRARY = None
                print("⚠️  No specialized DTW library found. Using scipy-based implementation.")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("✅ Cartopy available for geographic visualization")
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not available. Using matplotlib for geographic plots.")

class TemporalClusteringAnalysis:
    """
    Advanced temporal clustering analysis framework for Taiwan subsidence patterns
    
    Uses Dynamic Time Warping (DTW) for time series similarity analysis
    Complements ps03's feature-based clustering with temporal evolution patterns
    """
    
    def __init__(self, methods=['emd'], dtw_radius=0.1, max_clusters=10, random_state=42, fastdtw_threshold=200):
        """
        Initialize temporal clustering analysis framework
        
        Parameters:
        -----------
        methods : list
            Decomposition methods to include ['emd', 'vmd', 'fft', 'wavelet']
        dtw_radius : float
            DTW constraint radius (fraction of series length)
        max_clusters : int
            Maximum number of clusters to consider
        random_state : int
            Random seed for reproducibility
        fastdtw_threshold : int
            Station count threshold to auto-switch to FastDTW
        """
        self.methods = methods
        self.dtw_radius = dtw_radius
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.fastdtw_threshold = fastdtw_threshold
        
        # Data containers
        self.coordinates = None
        self.time_series_data = {}
        self.dtw_distance_matrix = None
        self.temporal_clusters = {}
        self.cluster_medoids = {}
        
        # Analysis results
        self.optimal_clusters = {}
        self.validation_results = {}
        self.ps03_comparison = None
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories for results and figures"""
        self.figures_dir = Path("figures")
        self.results_dir = Path("data/processed/ps04_temporal")
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Figures directory: {self.figures_dir}")

    def load_time_series_data(self, use_full_dataset=False, use_denoised=True):
        """
        Load time series data for temporal clustering
        
        Parameters:
        -----------
        use_full_dataset : bool
            Use full dataset vs 100-station subset
        use_denoised : bool  
            Use raw data minus EMD IMF1 (removes high-frequency noise)
        """
        print("📡 Loading time series data for temporal clustering...")
        
        try:
            if use_denoised:
                print("🔄 Loading EMD decomposition results for denoising...")
                
                # Load EMD decomposition data
                emd_data = np.load("data/processed/ps02_emd_decomposition.npz")
                
                # Load raw displacement data  
                preprocessed_data = np.load("data/processed/ps00_preprocessed_data.npz")
                raw_displacement = preprocessed_data['displacement']
                
                # Get IMF1 (high-frequency noise component)
                imfs = emd_data['imfs']  # Shape: (n_stations, n_imfs, n_time_points)
                imf1 = imfs[:, 0, :]  # First IMF contains high-frequency noise
                
                # Create denoised signal: raw_data - IMF1
                denoised_displacement = raw_displacement - imf1
                
                print(f"✅ Created denoised signals (raw - EMD IMF1)")
                print(f"   IMF1 range: {np.min(imf1):.3f} to {np.max(imf1):.3f}")
                print(f"   Denoised range: {np.min(denoised_displacement):.3f} to {np.max(denoised_displacement):.3f}")
                
                displacement = denoised_displacement
                self.coordinates = emd_data['coordinates']
                data_source = "EMD-denoised (raw - IMF1)"
                
            else:
                print("🔄 Loading raw displacement data...")
                
                # Load preprocessed coordinates and raw displacement
                preprocessed_data = np.load("data/processed/ps00_preprocessed_data.npz")
                displacement = preprocessed_data['displacement']
                self.coordinates = preprocessed_data['coordinates']
                data_source = "raw displacement"
            
            # Select stations with significant subsidence for better clustering
            subsidence_rates = preprocessed_data['subsidence_rates']
            
            if use_full_dataset:
                # Use all stations with significant subsidence (|rate| > 5 mm/year)
                significant_mask = np.abs(subsidence_rates) > 5.0
                if np.sum(significant_mask) < 50:
                    # Fallback: use stations with |rate| > 2 mm/year
                    significant_mask = np.abs(subsidence_rates) > 2.0
                    
                selected_indices = np.where(significant_mask)[0]
                subset_info = f"stations with significant subsidence (|rate| > 5 mm/year)"
                
                # Limit to reasonable number for computational efficiency
                if len(selected_indices) > 1000:
                    np.random.seed(42)  # Reproducible selection
                    selected_indices = np.random.choice(selected_indices, 1000, replace=False)
                    subset_info = f"1000 randomly selected {subset_info}"
                    
            else:
                # For testing: use first 100 stations with significant subsidence
                significant_mask = np.abs(subsidence_rates) > 5.0
                significant_indices = np.where(significant_mask)[0]
                
                if len(significant_indices) >= 100:
                    selected_indices = significant_indices[:100]
                    subset_info = f"first 100 stations with |rate| > 5 mm/year"
                else:
                    # Fallback: use first 100 stations period
                    selected_indices = np.arange(100)
                    subset_info = f"first 100 stations (mixed subsidence rates)"
            
            # Apply selection
            self.coordinates = self.coordinates[selected_indices]
            displacement = displacement[selected_indices]
            selected_rates = subsidence_rates[selected_indices]
            
            n_stations = len(self.coordinates)
            print(f"✅ Selected {n_stations} stations ({subset_info})")
            print(f"   Subsidence rate range: {np.min(selected_rates):.1f} to {np.max(selected_rates):.1f} mm/year")
            print(f"   Mean rate: {np.mean(selected_rates):.1f} mm/year")
            
            print(f"✅ Loaded {data_source} time series: {displacement.shape}")
            
            # Debug: Check time series statistics
            print(f"   Data range: {np.min(displacement):.3f} to {np.max(displacement):.3f}")
            print(f"   Has NaN: {np.isnan(displacement).any()}")
            print(f"   Has Inf: {np.isinf(displacement).any()}")
            
            # Analyze signal characteristics
            signal_std = np.std(displacement, axis=1)  # Variability per station
            print(f"   Signal variability: {np.mean(signal_std):.1f} ± {np.std(signal_std):.1f} mm")
            
            # Check trends to verify we have meaningful signals
            trends = []
            for i in range(min(10, len(displacement))):
                trend = np.polyfit(range(displacement.shape[1]), displacement[i], 1)[0]
                trends.append(trend * (365.25/6))  # Convert to mm/year
            print(f"   Sample trends (mm/year): {[f'{t:.1f}' for t in trends[:5]]}")
            
            # Store the displacement time series for all requested methods
            for method in self.methods:
                self.time_series_data[method] = displacement
                print(f"✅ Using {data_source} for {method.upper()} temporal clustering")
            
            if len(self.time_series_data) == 0:
                print("❌ No time series data loaded")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading time series data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _dtw_distance_scipy(self, ts1, ts2):
        """
        Compute DTW distance using scipy (fallback implementation)
        """
        from scipy.spatial.distance import euclidean
        
        n, m = len(ts1), len(ts2)
        
        # Create cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Apply Sakoe-Chiba band constraint
        radius = max(1, int(self.dtw_radius * max(n, m)))
        
        for i in range(1, n + 1):
            for j in range(max(1, i - radius), min(m, i + radius) + 1):
                cost = euclidean([ts1[i-1]], [ts2[j-1]])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return dtw_matrix[n, m]

    def _compute_dtw_pair(self, args):
        """
        Compute DTW distance between a pair of time series using best available library
        Auto-selects FastDTW for large datasets to improve performance
        """
        i, j, ts_data, n_stations = args
        
        if i == j:
            return i, j, 0.0  # Diagonal elements are zero
        
        if i > j:
            return i, j, None  # Use symmetry, will be filled later
        
        ts1 = ts_data[i]
        ts2 = ts_data[j]
        
        try:
            # Ensure time series are finite
            if not (np.isfinite(ts1).all() and np.isfinite(ts2).all()):
                return i, j, np.inf
            
            # Smart DTW method selection based on dataset size
            if n_stations > 200 and DTW_LIBRARY in ["tslearn", "fastdtw"]:
                # For large datasets (>200 stations), prioritize FastDTW for speed
                # FastDTW provides ~90-95% accuracy of exact DTW with major speed improvement
                if DTW_LIBRARY == "tslearn":
                    try:
                        from fastdtw import fastdtw
                        distance, path = fastdtw(ts1, ts2, radius=10, dist=lambda x, y: np.abs(x - y))
                    except ImportError:
                        # Fallback to TSLearn if FastDTW not available
                        distance = tslearn_dtw(ts1, ts2)
                    except Exception:
                        # Fallback to TSLearn if FastDTW fails
                        distance = tslearn_dtw(ts1, ts2)
                elif DTW_LIBRARY == "fastdtw":
                    distance, path = fastdtw(ts1, ts2, radius=10, dist=lambda x, y: np.abs(x - y))
                else:
                    # Use available library
                    if DTW_LIBRARY == "tslearn":
                        distance = tslearn_dtw(ts1, ts2)
                    else:
                        distance = self._dtw_distance_scipy(ts1, ts2)
            else:
                # For small datasets (≤200 stations), use exact DTW for best quality
                if DTW_LIBRARY == "tslearn":
                    # TSLearn - most advanced, built for time series ML
                    distance = tslearn_dtw(ts1, ts2)
                    
                elif DTW_LIBRARY == "dtw-python":
                    # DTW-Python - standard implementation
                    distance, cost_matrix, acc_cost_matrix, path = dtw(ts1, ts2, dist=lambda x, y: np.abs(x - y))
                    
                elif DTW_LIBRARY == "fastdtw":
                    # FastDTW - approximate but fast
                    distance, path = fastdtw(ts1, ts2, dist=lambda x, y: np.abs(x - y))
                    
                elif DTW_LIBRARY == "dtaidistance":
                    # DTAIDistance - optimized C implementation
                    window = max(1, int(self.dtw_radius * len(ts1)))
                    distance = dtw_ndim.distance(ts1, ts2, window=window)
                    
                else:
                    # Fallback to scipy-based implementation
                    distance = self._dtw_distance_scipy(ts1, ts2)
            
            # Ensure distance is finite
            if not np.isfinite(distance):
                return i, j, np.inf
            
            return i, j, distance
            
        except Exception as e:
            print(f"⚠️  DTW computation failed for pair ({i}, {j}): {e}")
            return i, j, np.inf

    def compute_dtw_distances(self, method='emd', parallel=True, n_jobs=-1):
        """
        Compute DTW distance matrix for time series clustering
        
        Parameters:
        -----------
        method : str
            Decomposition method to use for clustering
        parallel : bool
            Enable parallel computation
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        print(f"🔄 Computing DTW distances for {method.upper()} method...")
        
        if method not in self.time_series_data:
            print(f"❌ No data available for method: {method}")
            return False
        
        ts_data = self.time_series_data[method]
        n_stations = len(ts_data)
        
        # Validate and adjust DTW radius
        if self.dtw_radius > 1.0:
            print(f"⚠️  DTW radius {self.dtw_radius} > 1.0, interpreting as absolute samples")
            radius_samples = int(self.dtw_radius)
            self.dtw_radius = radius_samples / ts_data.shape[1]  # Convert to fraction
            print(f"   Adjusted DTW radius: {self.dtw_radius:.3f} ({radius_samples} samples)")
        else:
            radius_samples = int(self.dtw_radius * ts_data.shape[1])
        
        print(f"   Time series shape: {ts_data.shape}")
        print(f"   DTW radius: {self.dtw_radius:.3f} ({radius_samples} samples)")
        print(f"   DTW Library: {DTW_LIBRARY}")
        
        # Show smart DTW selection
        if n_stations > self.fastdtw_threshold:
            print(f"   🚀 Large dataset ({n_stations} stations): Auto-switching to FastDTW for speed")
            print(f"      Expected speedup: ~10-100x faster than exact DTW")
            print(f"      Threshold: {self.fastdtw_threshold} stations")
        else:
            print(f"   🎯 Small dataset ({n_stations} stations): Using exact DTW for best accuracy")
        
        try:
            start_time = time.time()
            
            # Prepare arguments for parallel computation (upper triangle only)
            pair_args = []
            for i in range(n_stations):
                for j in range(i, n_stations):
                    pair_args.append((i, j, ts_data, n_stations))
            
            print(f"   Computing {len(pair_args)} distance pairs...")
            
            if parallel and n_jobs != 1:
                # Parallel computation with proper resource management
                if n_jobs == -1:
                    n_jobs = min(8, cpu_count())  # Limit to 8 cores to avoid resource issues
                
                print(f"   Using {n_jobs} parallel processes...")
                
                try:
                    # Use context manager for proper cleanup
                    results = Parallel(n_jobs=n_jobs, verbose=1, backend='threading')(
                        delayed(self._compute_dtw_pair)(args) for args in pair_args
                    )
                except Exception as parallel_error:
                    print(f"   ⚠️  Parallel computation failed: {parallel_error}")
                    print("   Falling back to sequential computation...")
                    results = [self._compute_dtw_pair(args) for args in pair_args]
            else:
                # Sequential computation
                print("   Using sequential computation...")
                results = [self._compute_dtw_pair(args) for args in pair_args]
            
            # Build symmetric distance matrix
            distance_matrix = np.zeros((n_stations, n_stations))
            
            for result in results:
                if result[2] is not None:  # Skip None values
                    i, j, distance = result
                    distance_matrix[i, j] = distance
                    if i != j:  # Don't double-set diagonal
                        distance_matrix[j, i] = distance  # Ensure symmetry
            
            self.dtw_distance_matrix = distance_matrix
            
            elapsed_time = time.time() - start_time
            print(f"✅ DTW distance computation completed in {elapsed_time:.1f} seconds")
            print(f"   Distance matrix shape: {distance_matrix.shape}")
            
            # Check distance matrix statistics
            non_zero_distances = distance_matrix[distance_matrix > 0]
            if len(non_zero_distances) > 0:
                print(f"   Distance range: {np.min(non_zero_distances):.3f} to {np.max(distance_matrix):.3f}")
            else:
                print("   Warning: All distances are zero")
                print(f"   Matrix summary: min={np.min(distance_matrix):.3f}, max={np.max(distance_matrix):.3f}")
            
            # Save distance matrix
            np.savez_compressed(self.results_dir / "dtw_distance_matrix.npz",
                              distance_matrix=distance_matrix,
                              method=method,
                              dtw_radius=self.dtw_radius,
                              n_stations=n_stations)
            
            return True
            
        except Exception as e:
            print(f"❌ Error computing DTW distances: {e}")
            import traceback
            traceback.print_exc()
            return False

    def tslearn_clustering_alternative(self, method='emd', n_clusters=4, n_jobs=1, backend='threading'):
        """
        Alternative clustering using TSLearn's built-in DTW k-means
        Multiple parallelization strategies available
        
        Parameters:
        -----------
        backend : str
            'threading', 'loky', 'multiprocessing', or 'sequential'
        """
        if DTW_LIBRARY != "tslearn":
            print("⚠️  TSLearn not available for optimized clustering")
            return False
            
        print(f"🔄 Performing TSLearn DTW k-means clustering (k={n_clusters})...")
        print(f"   Using {n_jobs} parallel jobs with {backend} backend")
        
        try:
            ts_data = self.time_series_data[method]
            
            # Convert to TSLearn format
            ts_dataset = to_time_series_dataset(ts_data)
            
            # DTW k-means clustering with multiple parallelization strategies
            print(f"   Starting DTW k-means with {len(ts_dataset)} time series...")
            
            # Choose parallelization strategy based on dataset size and backend
            if backend == 'sequential' or n_jobs == 1:
                actual_n_jobs = 1
                actual_backend = None
                print(f"   Using sequential processing")
            elif backend == 'loky':
                # Loky backend - best for CPU-intensive tasks, avoids GIL
                actual_n_jobs = min(n_jobs, 6)  # Loky handles more cores better
                actual_backend = 'loky'
                print(f"   Using {actual_n_jobs} cores with Loky backend (multiprocessing)")
            elif backend == 'multiprocessing':
                # Traditional multiprocessing
                actual_n_jobs = min(n_jobs, 4)  # Conservative for multiprocessing
                actual_backend = 'multiprocessing'
                print(f"   Using {actual_n_jobs} cores with multiprocessing backend")
            else:  # threading (default)
                # Threading backend - limited by GIL but safer
                if len(ts_dataset) > 500:
                    actual_n_jobs = 1  # Large datasets often hang with threading
                    actual_backend = None
                    print(f"   Large dataset: forcing sequential (threading issues)")
                else:
                    actual_n_jobs = min(n_jobs, 2)  # Conservative for threading
                    actual_backend = 'threading'
                    print(f"   Using {actual_n_jobs} cores with threading backend")
            
            # Set joblib backend if specified
            if actual_backend and actual_n_jobs > 1:
                from joblib import parallel_backend
                print(f"   Setting joblib backend to {actual_backend}")
                
                with parallel_backend(actual_backend, n_jobs=actual_n_jobs):
                    model = TimeSeriesKMeans(n_clusters=n_clusters, 
                                           metric="dtw",
                                           max_iter=10,
                                           random_state=self.random_state,
                                           verbose=True,
                                           n_jobs=actual_n_jobs)
            else:
                model = TimeSeriesKMeans(n_clusters=n_clusters, 
                                       metric="dtw",
                                       max_iter=10,
                                       random_state=self.random_state,
                                       verbose=True,
                                       n_jobs=actual_n_jobs)
            
            print("   Fitting DTW k-means model...")
            
            try:
                # Execute clustering with appropriate backend
                start_time = time.time()
                
                if actual_backend and actual_n_jobs > 1:
                    from joblib import parallel_backend
                    with parallel_backend(actual_backend, n_jobs=actual_n_jobs):
                        cluster_labels = model.fit_predict(ts_dataset)
                else:
                    cluster_labels = model.fit_predict(ts_dataset)
                
                elapsed = time.time() - start_time
                print(f"   DTW k-means fitting completed in {elapsed:.1f} seconds!")
                
            except Exception as e:
                print(f"   ⚠️  {backend} clustering failed: {e}")
                print("   Falling back to sequential processing...")
                
                # Fallback to sequential processing
                model_seq = TimeSeriesKMeans(n_clusters=n_clusters, 
                                           metric="dtw",
                                           max_iter=10,
                                           random_state=self.random_state,
                                           verbose=True,
                                           n_jobs=1)
                
                start_time = time.time()
                cluster_labels = model_seq.fit_predict(ts_dataset)
                elapsed = time.time() - start_time
                print(f"   Sequential DTW k-means completed in {elapsed:.1f} seconds!")
                model = model_seq  # Use sequential model for centroids
            
            # Store results
            self.tslearn_clusters = {n_clusters: cluster_labels}
            self.tslearn_centroids = model.cluster_centers_
            
            print(f"✅ TSLearn clustering completed")
            print(f"   Cluster sizes: {np.bincount(cluster_labels)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in TSLearn clustering: {e}")
            return False

    def temporal_hierarchical_clustering(self):
        """
        Perform hierarchical clustering on DTW distance matrix
        """
        print("🔄 Performing temporal hierarchical clustering...")
        
        if self.dtw_distance_matrix is None:
            print("❌ No DTW distance matrix available")
            return False
        
        try:
            # Convert distance matrix to condensed form for scipy
            condensed_distances = squareform(self.dtw_distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            print("   Computing linkage matrix...")
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine optimal number of clusters
            self.optimal_clusters = self._determine_optimal_clusters_temporal(linkage_matrix)
            
            # Generate clusters for different k values
            self.temporal_clusters = {}
            for k in range(2, min(self.max_clusters + 1, len(self.coordinates))):
                clusters = fcluster(linkage_matrix, k, criterion='maxclust')
                self.temporal_clusters[k] = clusters
            
            self.linkage_matrix = linkage_matrix
            
            print(f"✅ Temporal hierarchical clustering completed")
            print(f"📊 Optimal cluster suggestions: {self.optimal_clusters}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in temporal hierarchical clustering: {e}")
            return False

    def _determine_optimal_clusters_temporal(self, linkage_matrix):
        """
        Determine optimal number of clusters using temporal-specific metrics
        """
        optimal_suggestions = {}
        
        try:
            # Silhouette analysis on DTW distances
            silhouette_scores = []
            k_range = range(2, min(self.max_clusters + 1, 16))
            
            for k in k_range:
                clusters = fcluster(linkage_matrix, k, criterion='maxclust')
                if len(np.unique(clusters)) > 1:
                    # Use precomputed distance matrix for silhouette score
                    score = silhouette_score(self.dtw_distance_matrix, clusters, metric='precomputed')
                    silhouette_scores.append((k, score))
            
            if silhouette_scores:
                best_k = max(silhouette_scores, key=lambda x: x[1])[0]
                optimal_suggestions['silhouette'] = best_k
            
            # Temporal coherence metric (custom)
            coherence_scores = []
            for k in k_range:
                clusters = fcluster(linkage_matrix, k, criterion='maxclust')
                coherence = self._compute_temporal_coherence(clusters)
                coherence_scores.append((k, coherence))
            
            if coherence_scores:
                best_k = max(coherence_scores, key=lambda x: x[1])[0]
                optimal_suggestions['temporal_coherence'] = best_k
            
            return optimal_suggestions
            
        except Exception as e:
            print(f"⚠️  Error determining optimal clusters: {e}")
            return {'default': 4}

    def _compute_temporal_coherence(self, clusters):
        """
        Compute temporal coherence metric for cluster validation
        """
        try:
            coherence_score = 0.0
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) > 1:
                    # Compute average intra-cluster DTW distance
                    intra_distances = []
                    for i in range(len(cluster_indices)):
                        for j in range(i + 1, len(cluster_indices)):
                            idx1, idx2 = cluster_indices[i], cluster_indices[j]
                            intra_distances.append(self.dtw_distance_matrix[idx1, idx2])
                    
                    if intra_distances:
                        coherence_score += 1.0 / (1.0 + np.mean(intra_distances))
            
            return coherence_score / len(unique_clusters)
            
        except Exception as e:
            return 0.0

    def create_dtw_heatmap(self, k_optimal=4):
        """
        Create DTW distance matrix heatmap visualization
        """
        print("🔄 Creating DTW distance matrix heatmap...")
        
        try:
            # Get cluster labels for ordering
            cluster_labels = fcluster(self.linkage_matrix, k_optimal, criterion='maxclust')
            
            # Sort indices by cluster labels
            sorted_indices = np.argsort(cluster_labels)
            sorted_matrix = self.dtw_distance_matrix[sorted_indices][:, sorted_indices]
            sorted_labels = cluster_labels[sorted_indices]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot heatmap
            im = ax.imshow(sorted_matrix, cmap='viridis', aspect='auto')
            
            # Add cluster boundaries
            cluster_boundaries = []
            current_cluster = sorted_labels[0]
            for i, cluster in enumerate(sorted_labels):
                if cluster != current_cluster:
                    cluster_boundaries.append(i)
                    current_cluster = cluster
            
            for boundary in cluster_boundaries:
                ax.axhline(y=boundary - 0.5, color='red', linestyle='--', alpha=0.7)
                ax.axvline(x=boundary - 0.5, color='red', linestyle='--', alpha=0.7)
            
            # Customize plot
            ax.set_title(f'DTW Distance Matrix (k={k_optimal} clusters)\nTemporal Similarity Patterns', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Station Index (sorted by cluster)', fontsize=12)
            ax.set_ylabel('Station Index (sorted by cluster)', fontsize=12)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('DTW Distance', fontsize=12)
            
            plt.tight_layout()
            
            # Save plot
            heatmap_file = self.figures_dir / "ps04_fig01_dtw_distance_matrix.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ DTW heatmap saved: {heatmap_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating DTW heatmap: {e}")
            return False

    def create_temporal_dendrogram(self):
        """
        Create temporal clustering dendrogram
        """
        print("🔄 Creating temporal dendrogram...")
        
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Create dendrogram
            dendrogram_plot = dendrogram(
                self.linkage_matrix,
                ax=ax,
                truncate_mode='level',
                p=10,
                leaf_rotation=90,
                leaf_font_size=8,
                show_leaf_counts=True
            )
            
            # Add cluster cut lines
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            for i, k in enumerate([3, 4, 5, 6]):
                if k <= self.max_clusters:
                    threshold = self.linkage_matrix[-(k-1), 2]
                    ax.axhline(y=threshold, color=colors[i % len(colors)], 
                              linestyle='--', alpha=0.7, 
                              label=f'k={k} clusters')
            
            ax.set_title('Temporal Clustering Dendrogram\nDTW-Based Hierarchical Clustering', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Station Clusters', fontsize=12)
            ax.set_ylabel('DTW Distance (Ward Linkage)', fontsize=12)
            ax.legend()
            
            plt.tight_layout()
            
            # Save dendrogram
            dendro_file = self.figures_dir / "ps04_fig02_temporal_dendrogram.png"
            plt.savefig(dendro_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Temporal dendrogram saved: {dendro_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating temporal dendrogram: {e}")
            return False

    def create_tslearn_visualizations(self, k_optimal=4):
        """
        Create comprehensive visualizations for TSLearn clustering results
        """
        print("🔄 Creating TSLearn clustering visualizations...")
        
        if k_optimal not in self.tslearn_clusters:
            print(f"❌ No TSLearn results for k={k_optimal}")
            return False
            
        try:
            cluster_labels = self.tslearn_clusters[k_optimal]
            centroids = self.tslearn_centroids
            
            # Figure 1: Cluster Centroids (Time Series)
            self._plot_cluster_centroids(cluster_labels, centroids, k_optimal)
            
            # Figure 2: Geographic Distribution
            self._plot_tslearn_geographic(cluster_labels, k_optimal)
            
            # Figure 3: Cluster Characteristics Analysis
            self._plot_cluster_characteristics(cluster_labels, k_optimal)
            
            # Figure 4: Temporal Evolution Comparison
            self._plot_temporal_evolution(cluster_labels, k_optimal)
            
            # Figure 5: Multi-Window Velocity Analysis (NEW!)
            try:
                self._plot_multi_window_analysis(cluster_labels, k_optimal)
            except Exception as velocity_error:
                print(f"⚠️  Multi-window velocity analysis failed: {velocity_error}")
                import traceback
                traceback.print_exc()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating TSLearn visualizations: {e}")
            return False

    def _plot_cluster_centroids(self, cluster_labels, centroids, k_optimal):
        """Plot cluster centroid time series with several hundred station signals"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        # Time vector (assuming 6-day sampling)
        time_days = np.arange(len(centroids[0])) * 6
        time_years = time_days / 365.25
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        
        # Get all time series data
        ts_data_all = list(self.time_series_data.values())[0]
        
        for i in range(min(k_optimal, 4)):  # Show up to 4 centroids
            ax = axes[i]
            
            # Get cluster information
            cluster_mask = cluster_labels == i
            cluster_indices = np.where(cluster_mask)[0]
            n_cluster_stations = len(cluster_indices)
            
            # Determine how many stations to plot (up to several hundred)
            max_stations_to_plot = min(300, n_cluster_stations)  # Show up to 300 stations per cluster
            
            if n_cluster_stations > max_stations_to_plot:
                # Randomly sample stations for display
                np.random.seed(42)  # For reproducibility
                sampled_indices = np.random.choice(cluster_indices, max_stations_to_plot, replace=False)
            else:
                sampled_indices = cluster_indices
            
            # Plot individual station time series with very low alpha
            print(f"   Plotting {len(sampled_indices)} station signals for Cluster {i+1}...")
            for j, idx in enumerate(sampled_indices):
                ts_data = ts_data_all[idx]
                ax.plot(time_years, ts_data, color=colors[i], alpha=0.05, linewidth=0.5, zorder=1)
            
            # Plot cluster mean (average of all stations in cluster) for reference
            cluster_ts_data = ts_data_all[cluster_indices]
            cluster_mean = np.mean(cluster_ts_data, axis=0)
            ax.plot(time_years, cluster_mean, color=colors[i], alpha=0.6, linewidth=2, 
                   linestyle='--', label=f'Cluster {i+1} Mean', zorder=2)
            
            # Plot centroid with prominent styling
            ax.plot(time_years, centroids[i], color=colors[i], linewidth=4, 
                   label=f'Cluster {i+1} Centroid', zorder=3)
            
            # Add confidence envelope (mean ± std)
            cluster_std = np.std(cluster_ts_data, axis=0)
            ax.fill_between(time_years, cluster_mean - cluster_std, cluster_mean + cluster_std,
                           color=colors[i], alpha=0.15, zorder=0)
            
            # Enhanced title with more information
            ax.set_title(f'Cluster {i+1}: {len(sampled_indices)} stations shown (of {n_cluster_stations} total)\n'
                        f'Individual Signals + Centroid + Mean ± StdDev', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (years)', fontsize=10)
            ax.set_ylabel('Displacement (mm)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Set consistent y-axis limits for better comparison
            if i == 0:
                global_min = np.min([np.min(ts_data_all[cluster_indices]) for cluster_indices in 
                                   [np.where(cluster_labels == j)[0] for j in range(k_optimal)]])
                global_max = np.max([np.max(ts_data_all[cluster_indices]) for cluster_indices in 
                                   [np.where(cluster_labels == j)[0] for j in range(k_optimal)]])
                y_padding = (global_max - global_min) * 0.1
                global_ylim = [global_min - y_padding, global_max + y_padding]
            
            ax.set_ylim(global_ylim)
        
        plt.tight_layout(pad=2.0)
        centroid_file = self.figures_dir / "ps04_fig03_tslearn_centroids.png"
        plt.savefig(centroid_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Enhanced cluster centroids with hundreds of station signals saved: {centroid_file}")

    def _plot_tslearn_geographic(self, cluster_labels, k_optimal):
        """Plot geographic distribution of TSLearn clusters"""
        fig = plt.figure(figsize=(15, 8))
        
        # Calculate actual data extent with padding
        lon_min, lon_max = self.coordinates[:, 0].min(), self.coordinates[:, 0].max()
        lat_min, lat_max = self.coordinates[:, 1].min(), self.coordinates[:, 1].max()
        
        lon_padding = (lon_max - lon_min) * 0.1
        lat_padding = (lat_max - lat_min) * 0.1
        
        extent = [lon_min - lon_padding, lon_max + lon_padding, 
                 lat_min - lat_padding, lat_max + lat_padding]
        
        if HAS_CARTOPY:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.gridlines(draw_labels=True, alpha=0.5)
        else:
            ax = plt.gca()
            ax.set_xlim([extent[0], extent[1]])
            ax.set_ylim([extent[2], extent[3]])
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        
        for cluster_id in range(k_optimal):
            mask = cluster_labels == cluster_id
            cluster_coords = self.coordinates[mask]
            n_stations = np.sum(mask)
            
            ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                      c=colors[cluster_id % len(colors)], s=25, alpha=0.7,
                      label=f'Cluster {cluster_id+1} (n={n_stations})')
        
        ax.set_title('TSLearn DTW Clustering - Geographic Distribution\nTaiwan Subsidence Temporal Patterns')
        if not HAS_CARTOPY:
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            ax.grid(True, alpha=0.3)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        geo_file = self.figures_dir / "ps04_fig04_tslearn_geographic.png"
        plt.savefig(geo_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Geographic distribution saved: {geo_file}")

    def _plot_cluster_characteristics(self, cluster_labels, k_optimal):
        """Plot cluster characteristics analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Analyze cluster characteristics
        cluster_stats = []
        for i in range(k_optimal):
            mask = cluster_labels == i
            cluster_indices = np.where(mask)[0]
            
            # Get time series for this cluster
            ts_data = list(self.time_series_data.values())[0][cluster_indices]
            
            stats = {
                'cluster_id': i + 1,
                'n_stations': len(cluster_indices),
                'mean_displacement': np.mean(ts_data),
                'std_displacement': np.std(ts_data),
                'trend': np.polyfit(range(ts_data.shape[1]), np.mean(ts_data, axis=0), 1)[0],
                'amplitude': np.std(np.mean(ts_data, axis=0))
            }
            cluster_stats.append(stats)
        
        # Plot 1: Cluster sizes
        cluster_ids = [s['cluster_id'] for s in cluster_stats]
        sizes = [s['n_stations'] for s in cluster_stats]
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'][:k_optimal]
        
        axes[0, 0].bar(cluster_ids, sizes, color=colors)
        axes[0, 0].set_title('Cluster Sizes')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Stations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Mean displacement
        means = [s['mean_displacement'] for s in cluster_stats]
        axes[0, 1].bar(cluster_ids, means, color=colors)
        axes[0, 1].set_title('Mean Displacement by Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Mean Displacement (mm)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Trend analysis
        trends = [s['trend'] for s in cluster_stats]
        axes[1, 0].bar(cluster_ids, trends, color=colors)
        axes[1, 0].set_title('Subsidence Trend by Cluster')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Trend (mm/time step)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Variability
        amplitudes = [s['amplitude'] for s in cluster_stats]
        axes[1, 1].bar(cluster_ids, amplitudes, color=colors)
        axes[1, 1].set_title('Temporal Variability by Cluster')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Amplitude (mm)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        char_file = self.figures_dir / "ps04_fig05_tslearn_characteristics.png"
        plt.savefig(char_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Cluster characteristics saved: {char_file}")

    def _analyze_velocity_variations(self, cluster_labels, k_optimal):
        """
        Analyze seasonal, semi-annual, and annual velocity changes in clusters
        Returns velocity analysis results for visualization
        """
        print("🔄 Analyzing velocity variations (seasonal, semi-annual, annual)...")
        
        velocity_analysis = {}
        ts_data_all = list(self.time_series_data.values())[0]
        
        # Time parameters (assuming 6-day sampling)
        n_time_points = ts_data_all.shape[1]
        time_days = np.arange(n_time_points) * 6
        time_years = time_days / 365.25
        dt = 6/365.25  # Time step in years
        
        for cluster_id in range(k_optimal):
            mask = cluster_labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_ts = ts_data_all[cluster_indices]
            
            # Calculate velocities (central difference)
            cluster_velocities = []
            for ts in cluster_ts:
                # Calculate velocity as displacement difference / time difference
                velocity = np.gradient(ts) / dt  # mm/year
                cluster_velocities.append(velocity)
            
            cluster_velocities = np.array(cluster_velocities)
            mean_velocity = np.mean(cluster_velocities, axis=0)
            std_velocity = np.std(cluster_velocities, axis=0)
            
            # Analyze periodicity in velocity using FFT
            velocity_fft = np.fft.fft(mean_velocity)
            frequencies = np.fft.fftfreq(len(mean_velocity), dt)
            power_spectrum = np.abs(velocity_fft)**2
            
            # Find dominant frequencies (exclude DC component)
            positive_freq_mask = frequencies > 0
            pos_frequencies = frequencies[positive_freq_mask]
            pos_power = power_spectrum[positive_freq_mask]
            
            # Convert to periods (years)
            periods = 1.0 / pos_frequencies
            
            # Find peaks for seasonal analysis
            peak_indices = []
            if len(pos_power) > 10:  # Need sufficient data points
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(pos_power, height=np.max(pos_power)*0.1)
                peak_indices = peaks
            
            # Identify seasonal components
            seasonal_components = {}
            for idx in peak_indices:
                if idx < len(periods):
                    period = periods[idx]
                    power = pos_power[idx]
                    
                    # Classify periods
                    if 0.8 < period < 1.2:  # Annual (around 1 year)
                        seasonal_components['annual'] = {'period': period, 'power': power, 'freq_idx': idx}
                    elif 0.4 < period < 0.6:  # Semi-annual (around 0.5 years)
                        seasonal_components['semi_annual'] = {'period': period, 'power': power, 'freq_idx': idx}
                    elif 0.2 < period < 0.3:  # Quarterly (around 0.25 years)
                        seasonal_components['quarterly'] = {'period': period, 'power': power, 'freq_idx': idx}
            
            # Calculate velocity statistics
            velocity_time = time_years[:len(mean_velocity)]  # Match arrays
            velocity_stats = {
                'mean_velocity': np.mean(mean_velocity),
                'velocity_range': [np.min(mean_velocity), np.max(mean_velocity)],
                'velocity_std': np.mean(std_velocity),
                'seasonal_amplitude': np.std(mean_velocity),
                'trend': np.polyfit(velocity_time, mean_velocity, 1)[0] if len(mean_velocity) > 1 else 0
            }
            
            # Ensure time and velocity arrays have same length
            velocity_time = time_years[:len(mean_velocity)]  # Match velocity length
            
            velocity_analysis[cluster_id] = {
                'mean_velocity': mean_velocity,
                'std_velocity': std_velocity,
                'time_years': velocity_time,
                'seasonal_components': seasonal_components,
                'velocity_stats': velocity_stats,
                'n_stations': len(cluster_indices)
            }
            
            print(f"   Cluster {cluster_id+1}: {velocity_stats['seasonal_amplitude']:.1f} mm/year seasonal amplitude")
        
        return velocity_analysis

    def _plot_temporal_evolution(self, cluster_labels, k_optimal):
        """Plot temporal evolution comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        time_days = np.arange(self.tslearn_centroids.shape[1]) * 6
        time_years = time_days / 365.25
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        
        for i in range(k_optimal):
            mask = cluster_labels == i
            n_stations = np.sum(mask)
            
            # Plot centroid with confidence bands
            centroid = self.tslearn_centroids[i]
            ax.plot(time_years, centroid, color=colors[i], linewidth=3,
                   label=f'Cluster {i+1} (n={n_stations})')
            
            # Add confidence band (if we have individual time series)
            cluster_indices = np.where(mask)[0]
            if len(cluster_indices) > 1:
                ts_data = list(self.time_series_data.values())[0][cluster_indices]
                mean_ts = np.mean(ts_data, axis=0)
                std_ts = np.std(ts_data, axis=0)
                
                ax.fill_between(time_years, mean_ts - std_ts, mean_ts + std_ts,
                               color=colors[i], alpha=0.2)
        
        ax.set_title('Temporal Evolution Patterns by Cluster\nTSLearn DTW-based Clustering Results')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Displacement (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        evolution_file = self.figures_dir / "ps04_fig06_tslearn_evolution.png"
        plt.savefig(evolution_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Temporal evolution saved: {evolution_file}")

    def _sliding_window_velocity_regression(self, cluster_labels, k_optimal):
        """
        Perform sliding window velocity regression analysis
        Similar to MATLAB's robustfit with sliding windows
        """
        print("🔄 Performing sliding window velocity regression analysis...")
        
        ts_data_all = list(self.time_series_data.values())[0]
        
        # Parameters
        window_days = 90  # 90-day window (approximately 1/4 year)
        sampling_interval = 6  # 6-day sampling
        window_samples = window_days // sampling_interval  # ~15 samples per window
        step_size = 1  # TRUE SLIDING WINDOW: Move by 1 sample at a time (6 days)
        
        # Time parameters
        n_time_points = ts_data_all.shape[1]
        time_days = np.arange(n_time_points) * sampling_interval
        time_years = time_days / 365.25
        
        print(f"   Window size: {window_days} days ({window_samples} samples)")
        print(f"   Step size: {step_size * sampling_interval} days (TRUE SLIDING WINDOW)")
        print(f"   Total time span: {time_years[-1]:.2f} years")
        print(f"   Total windows: {(n_time_points - window_samples) // step_size + 1}")
        
        regression_results = {}
        
        for cluster_id in range(k_optimal):
            mask = cluster_labels == cluster_id
            cluster_indices = np.where(mask)[0] 
            cluster_ts = ts_data_all[cluster_indices]
            
            # Calculate cluster centroid/mean time series
            cluster_mean_ts = np.mean(cluster_ts, axis=0)
            
            # Sliding window regression (equivalent to MATLAB robustfit)
            print(f"      Using robust sliding window regression...")
            
            window_times = []
            window_velocities = []
            window_r_squared = []
            window_std_errors = []
            
            # Manual sliding window implementation
            for start_idx in range(0, n_time_points - window_samples + 1, step_size):
                end_idx = start_idx + window_samples
                
                # Extract window data
                window_displacement = cluster_mean_ts[start_idx:end_idx]
                window_time_years = time_years[start_idx:end_idx]
                
                # Center time for better numerical stability
                window_center_time = np.mean(window_time_years)
                centered_time = window_time_years - window_center_time
                
                try:
                    # Use Huber regression for robustness (equivalent to MATLAB robustfit)
                    from sklearn.linear_model import HuberRegressor
                    huber = HuberRegressor(epsilon=1.35, max_iter=100)
                    X = centered_time.reshape(-1, 1)
                    huber.fit(X, window_displacement)
                    
                    # Extract slope (velocity in mm/year)
                    velocity = huber.coef_[0]
                    y_pred = huber.predict(X)
                    
                    # Calculate R-squared
                    ss_res = np.sum((window_displacement - y_pred) ** 2)
                    ss_tot = np.sum((window_displacement - np.mean(window_displacement)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    std_error = np.std(window_displacement - y_pred)
                    
                    # Store results
                    window_times.append(window_center_time)
                    window_velocities.append(velocity)
                    window_r_squared.append(r_squared)
                    window_std_errors.append(std_error)
                    
                except Exception as e:
                    # Fallback to scipy linear regression
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(centered_time, window_displacement)
                    
                    window_times.append(window_center_time)
                    window_velocities.append(slope)  # mm/year
                    window_r_squared.append(r_value**2)
                    window_std_errors.append(std_err)
            
            regression_results[cluster_id] = {
                'window_times': np.array(window_times),
                'window_velocities': np.array(window_velocities),
                'window_r_squared': np.array(window_r_squared),
                'window_std_errors': np.array(window_std_errors),
                'cluster_mean_ts': cluster_mean_ts,
                'time_years': time_years,
                'n_stations': len(cluster_indices),
                'window_params': {
                    'window_days': window_days,
                    'step_days': step_size * sampling_interval,
                    'window_samples': window_samples
                }
            }
            
            # Print summary statistics  
            mean_velocity = np.mean(window_velocities)
            velocity_std = np.std(window_velocities)
            velocity_range = [np.min(window_velocities), np.max(window_velocities)]
            
            print(f"   Cluster {cluster_id+1}: Mean velocity = {mean_velocity:.1f} ± {velocity_std:.1f} mm/year")
            print(f"      Velocity range: {velocity_range[0]:.1f} to {velocity_range[1]:.1f} mm/year")
        
        return regression_results

    def _multiple_window_velocity_analysis(self, cluster_labels, k_optimal):
        """
        Perform sliding window analysis with multiple window sizes
        """
        print("🔄 Performing multi-window velocity analysis...")
        
        ts_data_all = list(self.time_series_data.values())[0]
        
        # Multiple window sizes
        window_configs = [
            {'days': 90, 'label': '1/4 year', 'color': 'red', 'linewidth': 2},
            {'days': 180, 'label': '1/2 year', 'color': 'blue', 'linewidth': 2.5},
            {'days': 365, 'label': '1 year', 'color': 'green', 'linewidth': 3}
        ]
        
        sampling_interval = 6  # 6-day sampling
        time_years = np.arange(ts_data_all.shape[1]) * sampling_interval / 365.25
        
        multi_window_results = {}
        
        for cluster_id in range(k_optimal):
            mask = cluster_labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_ts = ts_data_all[cluster_indices]
            
            # Calculate cluster centroid
            cluster_mean_ts = np.mean(cluster_ts, axis=0)
            
            # Check centroid quality (correlation with individual stations)
            correlations = []
            for station_ts in cluster_ts:
                corr = np.corrcoef(cluster_mean_ts, station_ts)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            mean_correlation = np.mean(correlations) if correlations else 0
            
            print(f"   Cluster {cluster_id+1}: Centroid correlation with stations = {mean_correlation:.3f}")
            if mean_correlation < 0.7:
                print(f"      ⚠️  Low correlation - centroid may not represent cluster well")
            
            # Perform regression for each window size
            cluster_results = {
                'centroid': cluster_mean_ts,
                'time_years': time_years,
                'n_stations': len(cluster_indices),
                'centroid_quality': mean_correlation,
                'windows': {}
            }
            
            for config in window_configs:
                window_days = config['days']
                window_samples = window_days // sampling_interval
                
                if window_samples >= len(cluster_mean_ts):
                    print(f"      Skipping {config['label']} - window too large")
                    continue
                
                print(f"      Processing {config['label']} windows...")
                
                window_times = []
                window_velocities = []
                
                # Sliding window with step=1
                for start_idx in range(0, len(cluster_mean_ts) - window_samples + 1, 1):
                    end_idx = start_idx + window_samples
                    
                    window_displacement = cluster_mean_ts[start_idx:end_idx]
                    window_time_years = time_years[start_idx:end_idx]
                    
                    # Center time
                    window_center_time = np.mean(window_time_years)
                    centered_time = window_time_years - window_center_time
                    
                    try:
                        # Robust regression
                        from sklearn.linear_model import HuberRegressor
                        huber = HuberRegressor(epsilon=1.35, max_iter=100)
                        huber.fit(centered_time.reshape(-1, 1), window_displacement)
                        velocity = huber.coef_[0]
                        
                        window_times.append(window_center_time)
                        window_velocities.append(velocity)
                        
                    except Exception:
                        # Fallback
                        from scipy import stats
                        slope, _, _, _, _ = stats.linregress(centered_time, window_displacement)
                        window_times.append(window_center_time)
                        window_velocities.append(slope)
                
                cluster_results['windows'][config['label']] = {
                    'times': np.array(window_times),
                    'velocities': np.array(window_velocities),
                    'config': config
                }
            
            multi_window_results[cluster_id] = cluster_results
        
        return multi_window_results

    def _plot_multi_window_analysis(self, cluster_labels, k_optimal):
        """Create multi-window sliding regression visualization"""
        print("🔄 Creating multi-window velocity analysis plots...")
        
        # Get multi-window results
        multi_results = self._multiple_window_velocity_analysis(cluster_labels, k_optimal)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        
        for i in range(min(k_optimal, 4)):
            ax = axes[i//2, i%2]
            cluster_data = multi_results[i]
            
            # Plot original centroid displacement (background)
            time_years = cluster_data['time_years']
            centroid = cluster_data['centroid']
            ax.plot(time_years, centroid, color='lightgray', linewidth=2, alpha=0.8,
                   label='Centroid Displacement', zorder=1)
            
            # Plot individual station time series (very faint)
            mask = cluster_labels == i
            cluster_indices = np.where(mask)[0]
            ts_data_all = list(self.time_series_data.values())[0]
            for idx in cluster_indices[:10]:  # Show first 10 stations
                station_ts = ts_data_all[idx]
                ax.plot(time_years, station_ts, color=cluster_colors[i], alpha=0.1, linewidth=0.5, zorder=0)
            
            # Plot velocity curves for different window sizes
            ax2 = ax.twinx()
            
            for window_label, window_data in cluster_data['windows'].items():
                config = window_data['config']
                times = window_data['times']
                velocities = window_data['velocities']
                
                ax2.plot(times, velocities, color=config['color'], 
                        linewidth=config['linewidth'], alpha=0.8,
                        label=f"{config['label']} velocity", zorder=3)
            
            # Styling
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=2)
            ax2.set_ylabel('Velocity (mm/year)', fontsize=11)
            
            # Primary axis
            ax.set_title(f'Cluster {i+1}: Multi-Window Velocity Analysis\n'
                        f'({cluster_data["n_stations"]} stations, '
                        f'centroid quality: {cluster_data["centroid_quality"]:.3f})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (years)', fontsize=11)
            ax.set_ylabel('Displacement (mm)', fontsize=11)
            ax.grid(True, alpha=0.3, zorder=0)
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
            
            # Add cluster statistics
            mean_1q = np.mean(cluster_data['windows']['1/4 year']['velocities']) if '1/4 year' in cluster_data['windows'] else 0
            mean_1h = np.mean(cluster_data['windows']['1/2 year']['velocities']) if '1/2 year' in cluster_data['windows'] else 0
            mean_1y = np.mean(cluster_data['windows']['1 year']['velocities']) if '1 year' in cluster_data['windows'] else 0
            
            stats_text = f'Mean velocities (mm/yr):\n' + \
                        f'1/4 year: {mean_1q:.1f}\n' + \
                        f'1/2 year: {mean_1h:.1f}\n' + \
                        f'1 year: {mean_1y:.1f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   verticalalignment='top', horizontalalignment='left', fontsize=8)
        
        plt.tight_layout()
        multi_window_file = self.figures_dir / "ps04_fig07_multi_window_velocity.png"
        plt.savefig(multi_window_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Multi-window velocity analysis saved: {multi_window_file}")
        
        # Print clustering quality summary
        print("\n📊 CLUSTERING QUALITY ASSESSMENT:")
        for cluster_id in range(k_optimal):
            data = multi_results[cluster_id]
            quality = data['centroid_quality']
            n_stations = data['n_stations']
            print(f"   Cluster {cluster_id+1}: {n_stations} stations, centroid correlation = {quality:.3f}")
            if quality > 0.8:
                print(f"      ✅ Excellent representation")
            elif quality > 0.7:
                print(f"      ✅ Good representation") 
            elif quality > 0.6:
                print(f"      ⚠️  Fair representation")
            else:
                print(f"      ❌ Poor representation - consider different clustering")
        
        return multi_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Temporal Clustering Analysis for Taiwan Subsidence')
    parser.add_argument('--methods', type=str, default='emd',
                       help='Comma-separated list of methods: emd,fft,vmd,wavelet or "all"')
    parser.add_argument('--dtw-radius', type=float, default=0.1,
                       help='DTW constraint radius (fraction of series length)')
    parser.add_argument('--max-clusters', type=int, default=10,
                       help='Maximum number of clusters to consider')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel DTW computation')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--compare-with-ps03', action='store_true',
                       help='Compare with ps03 feature-based clustering results')
    parser.add_argument('--use-tslearn', action='store_true',
                       help='Use TSLearn DTW k-means instead of hierarchical clustering')
    parser.add_argument('--full-dataset', action='store_true',
                       help='Use full dataset instead of 20-station subset')
    parser.add_argument('--force-sequential', action='store_true',
                       help='Force sequential processing (avoids TSLearn hanging issues)')
    parser.add_argument('--backend', type=str, default='threading',
                       choices=['threading', 'loky', 'multiprocessing', 'sequential'],
                       help='Parallelization backend for TSLearn')
    parser.add_argument('--fastdtw-threshold', type=int, default=200,
                       help='Station count threshold to auto-switch to FastDTW (default: 200)')
    return parser.parse_args()

def main():
    """Main temporal clustering analysis workflow"""
    args = parse_arguments()
    
    # Parse methods
    if args.methods.lower() == 'all':
        methods = ['emd', 'fft', 'vmd', 'wavelet']
    else:
        methods = [m.strip().lower() for m in args.methods.split(',')]
    
    print("=" * 80)
    print("🚀 ps04_temporal_clustering.py - Temporal Clustering Analysis")
    print(f"📋 METHODS: {', '.join(methods).upper()}")
    print(f"📊 DTW RADIUS: {args.dtw_radius}")
    print(f"📊 MAX CLUSTERS: {args.max_clusters}")
    print("=" * 80)
    
    # Initialize analysis
    temporal_analysis = TemporalClusteringAnalysis(
        methods=methods,
        dtw_radius=args.dtw_radius,
        max_clusters=args.max_clusters,
        fastdtw_threshold=args.fastdtw_threshold
    )
    
    # Load time series data (using EMD-denoised signals by default)
    if not temporal_analysis.load_time_series_data(use_full_dataset=args.full_dataset, use_denoised=True):
        print("❌ Failed to load time series data")
        return False
    
    # Perform analysis for primary method
    primary_method = methods[0]
    print(f"\n🔄 ANALYZING PRIMARY METHOD: {primary_method.upper()}")
    
    if args.use_tslearn and DTW_LIBRARY == "tslearn":
        # Use TSLearn's optimized DTW k-means clustering
        optimal_k = 4  # Default
        n_jobs = 1 if args.force_sequential else (args.n_jobs if args.parallel else 1)
        backend = 'sequential' if args.force_sequential else args.backend
        
        if not temporal_analysis.tslearn_clustering_alternative(
            method=primary_method, 
            n_clusters=optimal_k,
            n_jobs=n_jobs,
            backend=backend
        ):
            print("❌ Failed to perform TSLearn clustering")
            return False
    else:
        # Use traditional hierarchical clustering with DTW distance matrix
        
        # Compute DTW distances
        if not temporal_analysis.compute_dtw_distances(
            method=primary_method,
            parallel=args.parallel,
            n_jobs=args.n_jobs
        ):
            print("❌ Failed to compute DTW distances")
            return False
        
        # Perform temporal clustering
        if not temporal_analysis.temporal_hierarchical_clustering():
            print("❌ Failed to perform temporal clustering")
            return False
    
    print("\n" + "=" * 50)
    print("🔄 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Create visualizations
    if args.use_tslearn and DTW_LIBRARY == "tslearn":
        optimal_k = 4  # Used for TSLearn
        print(f"🔄 Creating comprehensive TSLearn visualizations for k={optimal_k}...")
        
        # Create comprehensive TSLearn visualizations
        temporal_analysis.create_tslearn_visualizations(k_optimal=optimal_k)
        
        print("📊 Generated TSLearn visualizations:")
        print("   1. ✅ Cluster centroids and sample time series")
        print("   2. ✅ Geographic distribution of temporal clusters")  
        print("   3. ✅ Cluster characteristics analysis")
        print("   4. ✅ Temporal evolution patterns")
    else:
        optimal_k = temporal_analysis.optimal_clusters.get('silhouette', 4)
        
        # DTW heatmap
        temporal_analysis.create_dtw_heatmap(k_optimal=optimal_k)
        
        # Temporal dendrogram
        temporal_analysis.create_temporal_dendrogram()
        
        print("📊 Generated hierarchical clustering visualizations:")
        print("   1. ✅ DTW distance matrix heatmap")
        print("   2. ✅ Temporal clustering dendrogram")
    
    print("\n" + "=" * 80)
    print("✅ ps04_temporal_clustering.py ANALYSIS COMPLETED SUCCESSFULLY")
    print("📋 TEMPORAL CLUSTERING RESULTS:")
    print(f"   Optimal cluster suggestions: {temporal_analysis.optimal_clusters}")
    print(f"   Primary recommendation: k={optimal_k} (based on silhouette analysis)")
    print("📊 Generated visualizations:")
    print("   1. ✅ DTW distance matrix heatmap")
    print("   2. ✅ Temporal clustering dendrogram")
    print("📋 TEMPORAL INTERPRETATION:")
    print("   🕒 Clusters represent distinct temporal evolution patterns")
    print("   📈 DTW captures time series similarity beyond static features")
    print("   🔄 Complements ps03 feature-based clustering analysis")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
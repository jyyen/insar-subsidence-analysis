#!/usr/bin/env python3
"""
Analyze velocity patterns from ps04_temporal_clustering.py results
Extract and compare sliding window velocity signals across clusters
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from sklearn.linear_model import HuberRegressor
from scipy import stats
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import time
import threading

# Try to import cartopy for geographic mapping
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    CARTOPY_AVAILABLE = True
    print("✅ Cartopy available for geographic visualization")
except ImportError:
    CARTOPY_AVAILABLE = False
    print("⚠️  Cartopy not available - using basic matplotlib plots")

# Import the temporal clustering class
sys.path.append('.')
from ps04_temporal_clustering import TemporalClusteringAnalysis

def load_clustering_results(n_stations=100, use_random=False, min_rate=5.0):
    """Load the temporal clustering results by running the analysis
    
    Parameters:
    -----------
    n_stations : int
        Number of stations to use (default: 100, max: ~7000+ available)
    use_random : bool
        If True, randomly sample stations. If False, use highest subsidence rates
    min_rate : float
        Minimum subsidence rate threshold (mm/year) when not using random sampling
    """
    
    # Initialize and run temporal clustering analysis
    print(f"🔄 Running temporal clustering analysis with {n_stations} stations...")
    
    clustering_analyzer = TemporalClusteringAnalysis()
    
    # Modify the station selection logic
    print(f"📊 Station selection: {n_stations} stations, random={use_random}, min_rate={min_rate}")
    
    # Temporarily modify the load_time_series_data method behavior
    original_load = clustering_analyzer.load_time_series_data
    
    def modified_load():
        # Load the full preprocessed dataset directly
        print("📊 Loading full preprocessed dataset...")
        data = np.load('data/processed/ps00_preprocessed_data.npz')
        
        # Get full dataset
        coords = data['coordinates']  # (7154, 2)
        displacement = data['displacement']  # (7154, 215)
        subsidence_rates = data['subsidence_rates']  # (7154,)
        
        print(f"✅ Loaded full dataset: {displacement.shape[0]} stations x {displacement.shape[1]} time points")
        print(f"   Subsidence rate range: {np.min(subsidence_rates):.1f} to {np.max(subsidence_rates):.1f} mm/year")
        
        # Apply EMD denoising (load EMD results)
        print("🔄 Loading EMD decomposition for denoising...")
        emd_data = np.load('data/processed/ps02_emd_decomposition.npz')
        imfs = emd_data['imfs']  # Shape: (n_stations, n_imfs, n_time_points)
        imf1 = imfs[:, 0, :]  # First IMF contains high-frequency noise
        
        # Create denoised signal: raw_data - IMF1 
        denoised_displacement = displacement - imf1
        print(f"✅ Applied EMD denoising (raw - IMF1)")
        print(f"   IMF1 range: {np.min(imf1):.3f} to {np.max(imf1):.3f}")
        print(f"   Denoised range: {np.min(denoised_displacement):.3f} to {np.max(denoised_displacement):.3f}")
        
        # Use denoised displacement
        displacement = denoised_displacement
        
        # Station selection logic
        if use_random:
            # Random sampling from all stations
            available_indices = np.arange(len(coords))
            if n_stations < len(available_indices):
                np.random.seed(42)  # For reproducibility
                selected_indices = np.random.choice(available_indices, n_stations, replace=False)
                selected_indices = np.sort(selected_indices)  # Keep order for reproducibility
            else:
                selected_indices = available_indices
            subset_info = f"{len(selected_indices)} randomly selected stations"
        else:
            # Select by highest subsidence rates
            significant_mask = np.abs(subsidence_rates) >= min_rate
            significant_indices = np.where(significant_mask)[0]
            
            print(f"   Found {len(significant_indices)} stations with |rate| >= {min_rate} mm/year")
            
            if len(significant_indices) >= n_stations:
                # Sort by absolute subsidence rate (descending) and take top N
                sorted_indices = significant_indices[np.argsort(-np.abs(subsidence_rates[significant_indices]))]
                selected_indices = sorted_indices[:n_stations]
                subset_info = f"top {n_stations} stations with |rate| >= {min_rate} mm/year"
            else:
                # Not enough significant stations, take all significant + fill with others
                remaining_needed = n_stations - len(significant_indices)
                non_significant_indices = np.where(~significant_mask)[0]
                
                print(f"   Need {remaining_needed} additional stations from {len(non_significant_indices)} available")
                
                if len(non_significant_indices) >= remaining_needed:
                    # Sort non-significant by absolute rate and take top remaining
                    sorted_nonsig = non_significant_indices[np.argsort(-np.abs(subsidence_rates[non_significant_indices]))]
                    additional_indices = sorted_nonsig[:remaining_needed]
                    selected_indices = np.concatenate([significant_indices, additional_indices])
                else:
                    # Take all available stations
                    selected_indices = np.arange(min(n_stations, len(coords)))
                
                subset_info = f"{len(selected_indices)} stations (mixed subsidence rates)"
        
        # Apply selection
        clustering_analyzer.coordinates = coords[selected_indices]
        clustering_analyzer.time_series_data = {
            'emd': displacement[selected_indices]
        }
        
        print(f"✅ Selected {len(selected_indices)} stations: {subset_info}")
        print(f"   Subsidence rate range: {np.min(subsidence_rates[selected_indices]):.1f} to {np.max(subsidence_rates[selected_indices]):.1f} mm/year")
        print(f"   Mean rate: {np.mean(subsidence_rates[selected_indices]):.1f} mm/year")
        
        return True
    
    # Replace the method temporarily
    clustering_analyzer.load_time_series_data = modified_load
    
    # Load time series data with modified selection
    clustering_analyzer.load_time_series_data()
    
    # Perform TSLearn clustering with optimized settings
    import os
    n_cores = min(8, os.cpu_count())
    
    # Choose optimal backend based on dataset size  
    # Use conservative settings to avoid resource cleanup errors
    if n_stations <= 150:
        backend = 'threading'
        n_jobs = min(4, n_cores)
    elif n_stations <= 300:
        backend = 'threading'  # Safer for medium datasets
        n_jobs = min(2, n_cores)  # Conservative to avoid hanging
    else:
        backend = 'threading'  # Force threading to avoid multiprocessing cleanup issues
        n_jobs = 1  # Sequential for large datasets to ensure stability
        print(f"⚡ Large dataset ({n_stations} stations): using sequential processing for stability")
    
    print(f"🔧 Optimized settings: {backend} backend with {n_jobs} cores")
    
    clustering_analyzer.tslearn_clustering_alternative(method='emd', n_clusters=4, n_jobs=n_jobs, backend=backend)
    
    # Get the results
    ts_data = list(clustering_analyzer.time_series_data.values())[0]
    cluster_labels = clustering_analyzer.tslearn_clusters[4]
    
    clustering_data = {
        'time_series_data': ts_data,
        'cluster_labels': cluster_labels,
        'analyzer': clustering_analyzer,
        'n_stations_used': len(clustering_analyzer.coordinates),
        'selection_info': 'Selected stations'
    }
    
    return clustering_data

def process_cluster_velocity(args):
    """Process velocity patterns for a single cluster - for parallel processing"""
    cluster_id, cluster_ts, time_years, window_configs, sampling_interval = args
    
    # Calculate cluster centroid
    cluster_mean_ts = np.mean(cluster_ts, axis=0)
    
    cluster_velocities = {}
    
    for config in window_configs:
        window_days = config['days']
        window_samples = window_days // sampling_interval
        
        if window_samples >= len(cluster_mean_ts):
            continue
        
        # Parallel processing of sliding windows
        window_args = []
        for start_idx in range(0, len(cluster_mean_ts) - window_samples + 1, 1):
            end_idx = start_idx + window_samples
            window_displacement = cluster_mean_ts[start_idx:end_idx]
            window_time_years = time_years[start_idx:end_idx]
            window_args.append((window_displacement, window_time_years))
        
        # Process windows with threading (safer than multiprocessing for nested calls)
        if len(window_args) > 20:  # Use threading for moderate parallelization
            with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
                window_results = list(executor.map(process_single_window, window_args))
        else:
            # Sequential processing for small number of windows
            window_results = [process_single_window(arg) for arg in window_args]
        
        # Filter successful results
        valid_results = [r for r in window_results if r is not None]
        
        if valid_results:
            window_times, window_velocities = zip(*valid_results)
            cluster_velocities[config['label']] = {
                'times': np.array(window_times),
                'velocities': np.array(window_velocities),
                'color': config['color']
            }
    
    return cluster_id, cluster_ts.shape[0], cluster_mean_ts, cluster_velocities

def process_single_window(args):
    """Process a single sliding window - for parallel processing"""
    window_displacement, window_time_years = args
    
    # Center time
    window_center_time = np.mean(window_time_years)
    centered_time = window_time_years - window_center_time
    
    try:
        # Robust regression (equivalent to MATLAB robustfit)
        huber = HuberRegressor(epsilon=1.35, max_iter=100)
        huber.fit(centered_time.reshape(-1, 1), window_displacement)
        velocity = huber.coef_[0]
        
        return window_center_time, velocity
        
    except Exception:
        return None

def extract_velocity_patterns(clustering_data, k_optimal=4, n_processes=None):
    """Extract sliding window velocity patterns for all clusters - PARALLELIZED"""
    
    # Get time series data and cluster labels
    ts_data = clustering_data['time_series_data']
    cluster_labels = clustering_data['cluster_labels']
    
    # Time parameters
    sampling_interval = 6  # 6-day sampling
    n_time_points = ts_data.shape[1]
    time_years = np.arange(n_time_points) * sampling_interval / 365.25
    
    # Window configurations
    window_configs = [
        {'days': 90, 'label': '1/4 year', 'color': 'red'},
        {'days': 180, 'label': '1/2 year', 'color': 'blue'},
        {'days': 365, 'label': '1 year', 'color': 'green'}
    ]
    
    print("🔄 Extracting velocity patterns for each cluster (PARALLELIZED)...")
    
    # Prepare arguments for parallel processing
    cluster_args = []
    for cluster_id in range(k_optimal):
        mask = cluster_labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_ts = ts_data[cluster_indices]
        
        if len(cluster_ts) > 0:  # Only process non-empty clusters
            cluster_args.append((cluster_id, cluster_ts, time_years, window_configs, sampling_interval))
    
    # Determine optimal number of processes
    if n_processes is None:
        n_processes = min(len(cluster_args), mp.cpu_count())
    
    print(f"   Using {n_processes} processes for {len(cluster_args)} clusters")
    
    velocity_results = {}
    start_time = time.time()
    
    # Use threading instead of multiprocessing to avoid nested process conflicts
    if len(cluster_args) > 2 and n_processes > 1:
        print(f"   Using threading for {len(cluster_args)} clusters (avoiding nested process conflicts)")
        with ThreadPoolExecutor(max_workers=min(n_processes, 4)) as executor:
            future_to_cluster = {executor.submit(process_cluster_velocity, args): args[0] for args in cluster_args}
            
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    cluster_id, n_stations, cluster_mean_ts, cluster_velocities = future.result()
                    
                    velocity_results[f'Cluster_{cluster_id + 1}'] = {
                        'n_stations': n_stations,
                        'centroid': cluster_mean_ts,
                        'velocities': cluster_velocities
                    }
                    
                    print(f"   ✅ Completed Cluster {cluster_id + 1} ({n_stations} stations)")
                    
                except Exception as e:
                    print(f"   ❌ Error processing Cluster {cluster_id + 1}: {e}")
    else:
        # Sequential processing for small number of clusters or single process
        print(f"   Using sequential processing for {len(cluster_args)} clusters")
        for args in cluster_args:
            try:
                cluster_id, n_stations, cluster_mean_ts, cluster_velocities = process_cluster_velocity(args)
                
                velocity_results[f'Cluster_{cluster_id + 1}'] = {
                    'n_stations': n_stations,
                    'centroid': cluster_mean_ts,
                    'velocities': cluster_velocities
                }
                
                print(f"   ✅ Completed Cluster {cluster_id + 1} ({n_stations} stations)")
                
            except Exception as e:
                print(f"   ❌ Error processing Cluster {cluster_id + 1}: {e}")
    
    elapsed = time.time() - start_time
    print(f"   ⏱️  Velocity extraction completed in {elapsed:.1f}s")
    
    return velocity_results

def compute_correlation_pair(args):
    """Compute correlation between two velocity arrays - for parallel processing"""
    vel1, vel2 = args
    
    # Truncate to minimum length
    min_len = min(len(vel1), len(vel2))
    vel1_trunc = vel1[:min_len]
    vel2_trunc = vel2[:min_len]
    
    try:
        # Calculate correlation
        corr, _ = stats.pearsonr(vel1_trunc, vel2_trunc)
        return corr
    except Exception:
        return 0.0

def create_correlation_matrices(velocity_results, n_processes=None):
    """Create correlation coefficient matrices for different periods - PARALLELIZED"""
    
    print("\n🔗 Creating correlation coefficient matrices (PARALLELIZED)...")
    
    clusters = list(velocity_results.keys())
    window_types = ['1/4 year', '1/2 year', '1 year']
    
    # Create figure with subplots for each window type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Velocity Pattern Correlation Matrices Across Window Periods', fontsize=16, fontweight='bold')
    
    correlation_matrices = {}
    
    # Determine optimal number of processes
    if n_processes is None:
        n_processes = min(4, mp.cpu_count())
    
    start_time = time.time()
    
    for idx, window_type in enumerate(window_types):
        ax = axes[idx]
        
        # Extract velocity data for all clusters
        cluster_velocities = {}
        for cluster in clusters:
            if window_type in velocity_results[cluster]['velocities']:
                velocities = velocity_results[cluster]['velocities'][window_type]['velocities']
                cluster_velocities[cluster] = velocities
        
        # Create correlation matrix
        cluster_names = list(cluster_velocities.keys())
        n_clusters = len(cluster_names)
        corr_matrix = np.zeros((n_clusters, n_clusters))
        
        # Prepare correlation computation arguments
        corr_args = []
        index_map = []
        
        for i, cluster1 in enumerate(cluster_names):
            for j, cluster2 in enumerate(cluster_names):
                vel1 = cluster_velocities[cluster1]
                vel2 = cluster_velocities[cluster2]
                corr_args.append((vel1, vel2))
                index_map.append((i, j))
        
        # Parallel correlation computation using threading
        if len(corr_args) > 8:  # Only parallelize if enough computations
            with ThreadPoolExecutor(max_workers=min(4, n_processes)) as executor:
                correlations = list(executor.map(compute_correlation_pair, corr_args))
        else:
            # Sequential for small matrices
            correlations = [compute_correlation_pair(args) for args in corr_args]
        
        # Fill correlation matrix
        for (i, j), corr in zip(index_map, correlations):
            corr_matrix[i, j] = corr
        
        correlation_matrices[window_type] = {
            'matrix': corr_matrix,
            'labels': cluster_names
        }
        
        # Plot correlation matrix
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        
        # Add correlation values as text
        for i in range(n_clusters):
            for j in range(n_clusters):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Customize plot
        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        ax.set_xticklabels([name.replace('Cluster_', 'C') for name in cluster_names])
        ax.set_yticklabels([name.replace('Cluster_', 'C') for name in cluster_names])
        ax.set_title(f'{window_type} Window Correlations')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    elapsed = time.time() - start_time
    print(f"   ⏱️  Correlation matrices completed in {elapsed:.1f}s")
    
    plt.tight_layout()
    plt.savefig('figures/velocity_correlation_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Correlation matrices saved: figures/velocity_correlation_matrices.png")
    
    return correlation_matrices

def create_cluster_geographic_map_cartopy(clustering_data):
    """Create Cartopy-based geographic distribution map of clusters with different colors"""
    
    print("\n🗺️ Creating Cartopy-based geographic distribution map of clusters...")
    
    # Extract data
    coordinates = clustering_data['analyzer'].coordinates
    cluster_labels = clustering_data['cluster_labels'] 
    
    # Define colors for each cluster
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    
    # Calculate extent for Taiwan region
    lon_min, lon_max = coordinates[:, 0].min() - 0.05, coordinates[:, 0].max() + 0.05
    lat_min, lat_max = coordinates[:, 1].min() - 0.05, coordinates[:, 1].max() + 0.05
    
    # Create figure with Cartopy projection
    fig = plt.figure(figsize=(14, 12))
    
    # Use PlateCarree projection for Taiwan
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Set map extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    # Add geographic features with enhanced river visualization
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=1, color='gray')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    # Add multiple resolution rivers and lakes for comprehensive coverage
    from cartopy.feature import NaturalEarthFeature
    
    # Try multiple approaches to ensure river visibility
    try:
        # 1. Standard Cartopy rivers
        ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.8, linewidth=1.5)
        
        # 2. High-resolution Natural Earth rivers (10m)
        rivers_10m = NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                                        edgecolor='darkblue', facecolor='none', 
                                        linewidth=2.0, alpha=0.9, zorder=3)
        ax.add_feature(rivers_10m)
        
        # 3. Medium resolution for broader coverage (50m)
        rivers_50m = NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m',
                                        edgecolor='blue', facecolor='none', 
                                        linewidth=1.5, alpha=0.7, zorder=2)
        ax.add_feature(rivers_50m)
        
        # 4. Low resolution for major rivers (110m)  
        rivers_110m = NaturalEarthFeature('physical', 'rivers_lake_centerlines', '110m',
                                         edgecolor='steelblue', facecolor='none', 
                                         linewidth=1.0, alpha=0.6, zorder=1)
        ax.add_feature(rivers_110m)
        
        # 5. Add lakes and water bodies
        lakes_10m = NaturalEarthFeature('physical', 'lakes', '10m',
                                       edgecolor='blue', facecolor='lightblue', 
                                       alpha=0.6, zorder=2)
        ax.add_feature(lakes_10m)
        
        print("   ✅ Added multi-resolution river data")
        
    except Exception as e:
        print(f"   ⚠️  River data loading issue: {e}")
        # Fallback to basic rivers
        ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.8, linewidth=1.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Plot each cluster with different color
    unique_clusters = np.unique(cluster_labels)
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = coordinates[cluster_mask]
        
        # Count stations in this cluster
        n_stations = np.sum(cluster_mask)
        
        # Plot scatter points with transform
        scatter = ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                           c=colors[i], label=f'{cluster_names[i]} (n={n_stations})',
                           alpha=0.8, s=25, edgecolors='black', linewidths=0.5,
                           transform=proj, zorder=5)
    
    # Add title
    ax.set_title('Geographic Distribution of Velocity Pattern Clusters\nTaiwan InSAR Subsidence Analysis', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                      bbox_to_anchor=(0.02, 0.98), fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add coordinate labels
    ax.text(0.02, 0.02, 'Coordinate System: WGS84 Geographic (EPSG:4326)', 
            transform=ax.transAxes, fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'figures/ps04e_fig06_cluster_geographic_cartopy_500stations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cartopy geographic distribution map saved: {output_path}")
    
    return output_path

def create_cluster_geographic_map(clustering_data):
    """Create geographic distribution map of clusters with different colors"""
    
    # Check if we should use Cartopy
    if CARTOPY_AVAILABLE:
        return create_cluster_geographic_map_cartopy(clustering_data)
    
    print("\n🗺️ Creating geographic distribution map of clusters...")
    
    # Extract data
    coordinates = clustering_data['analyzer'].coordinates
    cluster_labels = clustering_data['cluster_labels'] 
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Define colors for each cluster
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    
    # Plot each cluster with different color
    unique_clusters = np.unique(cluster_labels)
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = coordinates[cluster_mask]
        
        # Count stations in this cluster
        n_stations = np.sum(cluster_mask)
        
        # Plot scatter points
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                  c=colors[i], label=f'{cluster_names[i]} (n={n_stations})',
                  alpha=0.7, s=20, edgecolors='black', linewidths=0.5)
    
    # Customize plot
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    ax.set_title('Geographic Distribution of Velocity Pattern Clusters\nTaiwan InSAR Subsidence Analysis', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set aspect ratio and adjust limits
    ax.set_aspect('equal', adjustable='box')
    
    # Add coordinate labels on axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'figures/ps04e_fig05_cluster_geographic_distribution_500stations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Geographic distribution map saved: {output_path}")
    
    # Print cluster statistics
    print(f"\n📊 Cluster Geographic Statistics:")
    print("-" * 50)
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = coordinates[cluster_mask]
        n_stations = np.sum(cluster_mask)
        
        lon_center = np.mean(cluster_coords[:, 0])
        lat_center = np.mean(cluster_coords[:, 1])
        lon_span = np.max(cluster_coords[:, 0]) - np.min(cluster_coords[:, 0])
        lat_span = np.max(cluster_coords[:, 1]) - np.min(cluster_coords[:, 1])
        
        print(f"{cluster_names[i]}: {n_stations} stations")
        print(f"  Center: {lon_center:.3f}°E, {lat_center:.3f}°N")
        print(f"  Span: {lon_span:.3f}° × {lat_span:.3f}°")
        print()
    
    return output_path

def analyze_velocity_similarities(velocity_results):
    """Analyze similarities and differences between cluster velocity patterns"""
    
    print("\n" + "="*80)
    print("📊 VELOCITY PATTERN ANALYSIS")
    print("="*80)
    
    clusters = list(velocity_results.keys())
    window_types = ['1/4 year', '1/2 year', '1 year']
    
    # Statistical comparison for each window type
    for window_type in window_types:
        print(f"\n🔍 {window_type.upper()} WINDOW ANALYSIS:")
        print("-" * 50)
        
        # Extract velocity data for all clusters
        cluster_velocities = {}
        for cluster in clusters:
            if window_type in velocity_results[cluster]['velocities']:
                velocities = velocity_results[cluster]['velocities'][window_type]['velocities']
                cluster_velocities[cluster] = velocities
        
        # Basic statistics
        print("📈 Velocity Statistics (mm/year):")
        for cluster, velocities in cluster_velocities.items():
            mean_vel = np.mean(velocities)
            std_vel = np.std(velocities)
            min_vel = np.min(velocities)
            max_vel = np.max(velocities)
            
            print(f"   {cluster}: {mean_vel:6.1f} ± {std_vel:5.1f} "
                  f"(range: {min_vel:6.1f} to {max_vel:6.1f})")
        
        # Pairwise correlations between clusters
        print("\n🔗 Pairwise Velocity Correlations:")
        cluster_names = list(cluster_velocities.keys())
        
        for i, cluster1 in enumerate(cluster_names):
            for j, cluster2 in enumerate(cluster_names[i+1:], i+1):
                vel1 = cluster_velocities[cluster1]
                vel2 = cluster_velocities[cluster2]
                
                # Truncate to minimum length
                min_len = min(len(vel1), len(vel2))
                vel1_trunc = vel1[:min_len]
                vel2_trunc = vel2[:min_len]
                
                # Calculate correlation
                corr, p_value = stats.pearsonr(vel1_trunc, vel2_trunc)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"   {cluster1} vs {cluster2}: r = {corr:6.3f} (p = {p_value:.3f}) {significance}")
        
        # Temporal variance analysis
        print("\n📊 Temporal Variability:")
        for cluster, velocities in cluster_velocities.items():
            # Calculate coefficient of variation
            cv = np.std(velocities) / np.abs(np.mean(velocities)) * 100 if np.mean(velocities) != 0 else np.inf
            
            # Count periods of acceleration/deceleration
            velocity_changes = np.diff(velocities)
            accelerations = np.sum(velocity_changes > 1.0)  # > 1 mm/year acceleration
            decelerations = np.sum(velocity_changes < -1.0)  # > 1 mm/year deceleration
            
            print(f"   {cluster}: CV = {cv:5.1f}%, "
                  f"Accelerations: {accelerations}, Decelerations: {decelerations}")

def plot_velocity_comparison(velocity_results):
    """Create comprehensive velocity comparison plots"""
    
    print("\n🎨 Creating velocity comparison plots...")
    
    clusters = list(velocity_results.keys())
    window_types = ['1/4 year', '1/2 year', '1 year']
    
    # Create subplot for each window type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Sliding Window Velocity Comparison Across Clusters', fontsize=16, fontweight='bold')
    
    for idx, window_type in enumerate(window_types):
        ax = axes[idx]
        
        for cluster in clusters:
            if window_type in velocity_results[cluster]['velocities']:
                vel_data = velocity_results[cluster]['velocities'][window_type]
                times = vel_data['times']
                velocities = vel_data['velocities']
                color = vel_data['color']
                
                ax.plot(times, velocities, color=color, linewidth=2, 
                       label=f"{cluster} (n={velocity_results[cluster]['n_stations']})",
                       alpha=0.8)
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Velocity (mm/year)')
        ax.set_title(f'{window_type} Window')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add horizontal zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figures/velocity_pattern_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Velocity comparison plot saved: figures/velocity_pattern_comparison.png")
    
    return fig

def main():
    """Main analysis function"""
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Velocity Pattern Analysis - PARALLELIZED')
    parser.add_argument('--n-stations', type=int, default=100, 
                       help='Number of stations to analyze (default: 100, max: ~7000+)')
    parser.add_argument('--random', action='store_true', 
                       help='Use random station sampling instead of highest subsidence rates')
    parser.add_argument('--min-rate', type=float, default=5.0, 
                       help='Minimum subsidence rate threshold in mm/year (default: 5.0)')
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of parallel processes (default: auto-detect)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced parallelization for smaller datasets')
    
    args = parser.parse_args()
    
    # Determine optimal parallelization settings
    if args.n_processes is None:
        if args.quick or args.n_stations <= 50:
            n_processes = min(2, mp.cpu_count())
        else:
            n_processes = min(8, mp.cpu_count())
    else:
        n_processes = args.n_processes
    
    print("🚀 Starting velocity pattern analysis (PARALLELIZED)...")
    print(f"📊 Configuration: {args.n_stations} stations, random={args.random}, min_rate={args.min_rate}")
    print(f"⚡ Parallelization: {n_processes} processes, quick={args.quick}")
    
    try:
        # Load clustering results with specified parameters
        clustering_data = load_clustering_results(
            n_stations=args.n_stations,
            use_random=args.random,
            min_rate=args.min_rate
        )
        print(f"✅ Loaded clustering results for {clustering_data['n_stations_used']} stations")
        
        # Extract velocity patterns (parallelized)
        velocity_results = extract_velocity_patterns(clustering_data, n_processes=n_processes)
        print("✅ Extracted velocity patterns")
        
        # Create correlation matrices (parallelized)
        correlation_matrices = create_correlation_matrices(velocity_results, n_processes=min(n_processes, 4))
        
        # Create geographic distribution map
        create_cluster_geographic_map(clustering_data)
        
        # Analyze similarities and differences
        analyze_velocity_similarities(velocity_results)
        
        # Create comparison plots
        plot_velocity_comparison(velocity_results)
        
        print("\n✅ Velocity pattern analysis completed!")
        
    except Exception as e:
        print(f"❌ Error in velocity analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up multiprocessing resources to avoid resource tracker errors
        try:
            import multiprocessing
            import time
            
            print("🧹 Cleaning up parallel processes...")
            
            # Force cleanup of any remaining multiprocessing resources
            active_children = multiprocessing.active_children()
            if active_children:
                print(f"   Terminating {len(active_children)} active processes...")
                for p in active_children:
                    try:
                        p.terminate()
                        p.join(timeout=2)
                        if p.is_alive():
                            p.kill()
                    except Exception as e:
                        print(f"   Warning: Could not clean up process {p.pid}: {e}")
                
                # Brief pause to ensure cleanup
                time.sleep(0.5)
                
            print("✅ Process cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Process cleanup warning: {e}")

if __name__ == "__main__":
    main()
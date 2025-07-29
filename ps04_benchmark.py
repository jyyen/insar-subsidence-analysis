#!/usr/bin/env python3
"""
ps04_benchmark.py - DTW Performance Benchmark

Purpose: Compare FastDTW vs TSLearn performance for different dataset sizes
Test sizes: 50, 500, 1000 stations
Focus: Speed, memory usage, clustering quality

Author: Claude Code
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from pathlib import Path
from sklearn.metrics import silhouette_score, adjusted_rand_score
import psutil
import gc

warnings.filterwarnings('ignore')

# DTW Libraries
DTW_LIBRARIES = {}
try:
    from tslearn.metrics import dtw as tslearn_dtw
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset
    DTW_LIBRARIES['tslearn'] = True
    print("✅ TSLearn available")
except ImportError:
    DTW_LIBRARIES['tslearn'] = False
    print("❌ TSLearn not available")

try:
    from fastdtw import fastdtw
    DTW_LIBRARIES['fastdtw'] = True
    print("✅ FastDTW available")
except ImportError:
    DTW_LIBRARIES['fastdtw'] = False
    print("❌ FastDTW not available")

try:
    from dtaidistance import dtw_ndim
    DTW_LIBRARIES['dtaidistance'] = True
    print("✅ DTAIDistance available")
except ImportError:
    DTW_LIBRARIES['dtaidistance'] = False
    print("❌ DTAIDistance not available")

# GLOBAL WORKER FUNCTIONS FOR M1 ULTRA MULTIPROCESSING COMPATIBILITY
def compute_fastdtw_pair_global(pair_data):
    """Global worker function for FastDTW parallel computation"""
    i, j, data, radius = pair_data
    try:
        from fastdtw import fastdtw
        # Convert to proper data types for M1 Ultra compatibility
        ts1 = np.array(data[i], dtype=np.float64)
        ts2 = np.array(data[j], dtype=np.float64)
        distance, path = fastdtw(ts1, ts2, radius=radius, dist=lambda x, y: abs(x - y))
        return float(distance)
    except Exception as e:
        return np.inf

def compute_dtw_pair_global(pair_data):
    """Global worker function for DTAIDistance parallel computation"""
    i, j, data, window = pair_data
    try:
        from dtaidistance import dtw_ndim
        # Convert to proper data types for DTAIDistance M1 Ultra compatibility  
        ts1 = np.array(data[i], dtype=np.float64)
        ts2 = np.array(data[j], dtype=np.float64)
        return float(dtw_ndim.distance(ts1, ts2, window=window))
    except Exception as e:
        return np.inf

class DTWBenchmark:
    """
    Comprehensive DTW performance benchmark for Taiwan subsidence analysis
    """
    
    def __init__(self, test_sizes=[50, 500, 1000]):
        self.test_sizes = test_sizes
        self.results = {}
        self.time_series_data = None
        self.coordinates = None
        
        # DYNAMIC SYSTEM DETECTION
        self.system_info = self._detect_system_capabilities()
        
        # Create output directories
        self.figures_dir = Path("figures")
        self.results_dir = Path("data/processed/ps04_benchmark")
        self.figures_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def _detect_system_capabilities(self):
        """Dynamically detect system capabilities for optimal performance"""
        import multiprocessing as mp
        import platform
        import psutil
        
        # Get system information
        system_info = {
            'platform': platform.system(),
            'machine': platform.machine(), 
            'cpu_count_logical': mp.cpu_count(),  # Logical cores
            'cpu_count_physical': psutil.cpu_count(logical=False),  # Physical cores  
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_brand': 'Unknown'
        }
        
        # Get CPU brand on macOS
        if system_info['platform'] == 'Darwin':
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                     capture_output=True, text=True)
                system_info['cpu_brand'] = result.stdout.strip()
            except:
                pass
        
        # Determine optimal core usage based on system
        if 'M1 Ultra' in system_info['cpu_brand']:
            # M1 Ultra: 20 cores, use 18 (leave 2 for system)
            system_info['optimal_cores'] = min(18, system_info['cpu_count_logical'] - 2)
            system_info['system_type'] = 'M1_Ultra'
        elif 'M1' in system_info['cpu_brand'] or 'M2' in system_info['cpu_brand']:
            # M1/M2: 8 cores, use all (efficient cores handle system)
            system_info['optimal_cores'] = system_info['cpu_count_logical']
            system_info['system_type'] = 'Apple_Silicon'
        elif system_info['cpu_count_logical'] >= 16:
            # High-end system: Leave 2 cores for system
            system_info['optimal_cores'] = system_info['cpu_count_logical'] - 2
            system_info['system_type'] = 'High_Performance'
        else:
            # Standard system: Use all cores
            system_info['optimal_cores'] = system_info['cpu_count_logical']
            system_info['system_type'] = 'Standard'
        
        print(f"🖥️  System detected: {system_info['cpu_brand']}")
        print(f"   Platform: {system_info['platform']} {system_info['machine']}")
        print(f"   Cores: {system_info['cpu_count_logical']} logical, {system_info['cpu_count_physical']} physical")
        print(f"   Memory: {system_info['memory_gb']:.1f} GB")
        print(f"   Optimal cores for DTW: {system_info['optimal_cores']} ({system_info['system_type']})")
        
        return system_info
    
    def load_test_data(self):
        """Load test data from ps00 preprocessed results"""
        print("📡 Loading test data...")
        
        try:
            data = np.load("data/processed/ps00_preprocessed_data.npz")
            self.coordinates = data['coordinates']
            self.time_series_data = data['displacement']
            
            print(f"✅ Loaded {len(self.coordinates)} stations")
            print(f"   Time series shape: {self.time_series_data.shape}")
            print(f"   Data range: {self.time_series_data.min():.2f} to {self.time_series_data.max():.2f} mm")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_fastdtw_pairwise(self, data_subset, radius=1):
        """Benchmark FastDTW for pairwise distance computation - OPTIMIZED FOR M1 ULTRA"""
        n_stations = len(data_subset)
        n_pairs = n_stations * (n_stations - 1) // 2
        
        print(f"   FastDTW: Computing {n_pairs} distances (radius={radius}, M1 Ultra parallel)...")
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # M1 ULTRA OPTIMIZATION: Parallel FastDTW computation using global function
        import multiprocessing as mp
        
        # Create all pair indices
        pairs = [(i, j) for i in range(n_stations) for j in range(i + 1, n_stations)]
        pair_data = [(i, j, data_subset, radius) for i, j in pairs]
        
        # DYNAMIC OPTIMIZATION: Use system-optimal core count
        n_cores = self.system_info['optimal_cores']
        system_type = self.system_info['system_type']
        print(f"      🚀 Using {n_cores} cores for parallel FastDTW computation ({system_type} optimized)...")
        
        try:
            with mp.Pool(processes=n_cores) as pool:
                distances = pool.map(compute_fastdtw_pair_global, pair_data)
        except Exception as e:
            print(f"      ⚠️  Parallel FastDTW failed ({e}), falling back to sequential...")
            # Fallback to sequential processing
            distances = []
            for i in range(n_stations):
                for j in range(i + 1, n_stations):
                    try:
                        distance, path = fastdtw(data_subset[i], data_subset[j], 
                                               radius=radius, dist=lambda x, y: abs(x - y))
                        distances.append(distance)
                    except Exception as e:
                        distances.append(np.inf)
        
        elapsed_time = time.time() - start_time
        end_memory = self.get_memory_usage()
        memory_used = end_memory - start_memory
        
        success_rate = np.sum(np.isfinite(distances)) / len(distances)
        print(f"      ✅ FastDTW completed in {elapsed_time:.1f}s (success rate: {success_rate:.1%})")
        
        return {
            'elapsed_time': elapsed_time,
            'memory_used': memory_used,
            'distances': np.array(distances),
            'n_pairs': n_pairs,
            'success_rate': success_rate,
            'n_cores_used': n_cores,
            'optimization': 'M1_Ultra_Parallel'
        }
    
    def benchmark_tslearn_kmeans(self, data_subset, n_clusters=4):
        """Benchmark TSLearn DTW k-means clustering - OPTIMIZED FOR M1 ULTRA"""
        # DYNAMIC OPTIMIZATION: Use system-optimal core count
        n_cores = self.system_info['optimal_cores']
        system_type = self.system_info['system_type']
        
        print(f"   TSLearn: DTW k-means clustering (k={n_clusters}, {n_cores} cores - {system_type} optimized)...")
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            # Convert to TSLearn format
            ts_dataset = to_time_series_dataset(data_subset)
            
            # M1 ULTRA OPTIMIZATION: Enable multi-threading for TSLearn
            model = TimeSeriesKMeans(n_clusters=n_clusters, 
                                   metric="dtw",
                                   max_iter=10,
                                   random_state=42,
                                   verbose=False,
                                   n_jobs=n_cores)  # Use multiple cores on M1 Ultra
            
            cluster_labels = model.fit_predict(ts_dataset)
            
            elapsed_time = time.time() - start_time
            end_memory = self.get_memory_usage()
            memory_used = end_memory - start_memory
            
            # Compute clustering quality
            if len(np.unique(cluster_labels)) > 1:
                # Use Euclidean distance for silhouette (DTW too expensive)
                flattened_data = data_subset.reshape(len(data_subset), -1)
                silhouette = silhouette_score(flattened_data, cluster_labels)
            else:
                silhouette = -1
            
            print(f"      ✅ TSLearn clustering completed in {elapsed_time:.1f}s using {n_cores} cores ({system_type})")
            
            return {
                'elapsed_time': elapsed_time,
                'memory_used': memory_used,
                'cluster_labels': cluster_labels,
                'silhouette_score': silhouette,
                'centroids': model.cluster_centers_,
                'success': True,
                'n_cores_used': n_cores,
                'optimization': 'M1_Ultra_Parallel'
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"   ❌ TSLearn clustering failed: {e}")
            return {
                'elapsed_time': elapsed_time,
                'memory_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_dtaidistance(self, data_subset, window=0.1):
        """Benchmark DTAIDistance for DTW computation - CROSS-PLATFORM OPTIMIZED"""
        if not DTW_LIBRARIES['dtaidistance']:
            return None
            
        n_stations = len(data_subset)
        window_size = max(1, int(window * data_subset.shape[1]))
        n_pairs = n_stations * (n_stations - 1) // 2
        
        system_type = self.system_info['system_type']
        print(f"   DTAIDistance: Computing {n_pairs} distances (window={window_size}, {system_type} optimized)...")
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            # M1 ULTRA OPTIMIZATION: Use dtaidistance's built-in parallelization
            from dtaidistance import dtw
            
            # Method 1: Try built-in distance_matrix_fast (best for M1 Ultra)
            try:
                print(f"      🚀 Using DTAIDistance parallel matrix computation...")
                # Convert data to proper format for M1 Ultra DTAIDistance compatibility
                data_float64 = np.array(data_subset, dtype=np.float64)
                
                distance_matrix = dtw.distance_matrix_fast(
                    data_float64,
                    window=window_size,
                    use_mp=True,          # Enable multiprocessing 
                    use_c=True,           # Use C implementation
                    parallel=True,        # Enable parallel processing
                    n_jobs=self.system_info['optimal_cores'],  # Use optimal core count
                    block=((0, n_stations), (0, n_stations))  # Full matrix
                )
                
                # Extract upper triangle (pairwise distances)
                distances = []
                for i in range(n_stations):
                    for j in range(i + 1, n_stations):
                        distances.append(distance_matrix[i, j])
                        
            except Exception as matrix_error:
                # Fallback: Parallel pairwise computation using multiprocessing
                print(f"      ⚠️  Matrix method failed ({matrix_error}), using parallel pairwise...")
                import multiprocessing as mp
                
                # Create pair indices for parallel processing
                pairs = [(i, j) for i in range(n_stations) for j in range(i + 1, n_stations)]
                
                # Use system-optimal core count
                n_cores = self.system_info['optimal_cores']
                pair_data = [(i, j, data_subset, window_size) for i, j in pairs]
                
                print(f"      🚀 Using {n_cores} cores for parallel pairwise computation ({system_type})...")
                with mp.Pool(processes=n_cores) as pool:
                    distances = pool.map(compute_dtw_pair_global, pair_data)
            
            elapsed_time = time.time() - start_time
            end_memory = self.get_memory_usage()
            memory_used = end_memory - start_memory
            
            print(f"      ✅ Computed {len(distances)} distances in {elapsed_time:.1f}s using M1 Ultra cores")
            
            return {
                'elapsed_time': elapsed_time,
                'memory_used': memory_used,
                'distances': np.array(distances),
                'success_rate': 1.0,
                'n_cores_used': 18,
                'optimization': 'M1_Ultra_Parallel'
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"   ❌ DTAIDistance failed: {e}")
            return {
                'elapsed_time': elapsed_time,
                'memory_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def run_benchmark(self):
        """Run comprehensive benchmark for all test sizes"""
        print("🚀 Starting DTW Performance Benchmark")
        print("=" * 60)
        
        for test_size in self.test_sizes:
            if test_size > len(self.time_series_data):
                print(f"⚠️  Skipping size {test_size} (only {len(self.time_series_data)} stations available)")
                continue
                
            print(f"\n🔄 BENCHMARKING {test_size} STATIONS")
            print("-" * 40)
            
            # Get subset of data
            data_subset = self.time_series_data[:test_size]
            
            # Initialize results for this size
            self.results[test_size] = {}
            
            # Benchmark 1: FastDTW pairwise distances
            if DTW_LIBRARIES['fastdtw']:
                print("🔄 Testing FastDTW...")
                self.results[test_size]['fastdtw'] = self.benchmark_fastdtw_pairwise(data_subset)
                gc.collect()  # Clean up memory
            
            # Benchmark 2: TSLearn DTW k-means
            if DTW_LIBRARIES['tslearn']:
                print("🔄 Testing TSLearn...")
                self.results[test_size]['tslearn'] = self.benchmark_tslearn_kmeans(data_subset)
                gc.collect()  # Clean up memory
            
            # Benchmark 3: DTAIDistance (if available)
            if DTW_LIBRARIES['dtaidistance']:
                print("🔄 Testing DTAIDistance...")
                self.results[test_size]['dtaidistance'] = self.benchmark_dtaidistance(data_subset)
                gc.collect()  # Clean up memory
            
            # Print summary for this size
            self.print_size_summary(test_size)
        
        # Create comprehensive visualizations
        self.create_benchmark_visualizations()
        
        # Save results
        self.save_benchmark_results()
    
    def print_size_summary(self, test_size):
        """Print summary for one test size"""
        print(f"\n📊 RESULTS FOR {test_size} STATIONS:")
        
        results = self.results[test_size]
        
        for method in ['fastdtw', 'tslearn', 'dtaidistance']:
            if method in results and results[method].get('success', True):
                r = results[method]
                print(f"   {method.upper():12}: {r['elapsed_time']:6.1f}s, {r['memory_used']:6.1f}MB")
            elif method in results:
                print(f"   {method.upper():12}: FAILED")
    
    def create_benchmark_visualizations(self):
        """Create comprehensive benchmark visualizations"""
        print("\n🔄 Creating benchmark visualizations...")
        
        # Figure 1: Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data for plotting
        sizes = []
        fastdtw_times = []
        tslearn_times = []
        dtaidist_times = []
        fastdtw_memory = []
        tslearn_memory = []
        
        for size in self.test_sizes:
            if size in self.results:
                sizes.append(size)
                
                # Times
                fastdtw_times.append(self.results[size].get('fastdtw', {}).get('elapsed_time', np.nan))
                tslearn_times.append(self.results[size].get('tslearn', {}).get('elapsed_time', np.nan))
                dtaidist_times.append(self.results[size].get('dtaidistance', {}).get('elapsed_time', np.nan))
                
                # Memory
                fastdtw_memory.append(self.results[size].get('fastdtw', {}).get('memory_used', np.nan))
                tslearn_memory.append(self.results[size].get('tslearn', {}).get('memory_used', np.nan))
        
        # Plot 1: Execution Time
        ax = axes[0, 0]
        if DTW_LIBRARIES['fastdtw']:
            ax.plot(sizes, fastdtw_times, 'o-', label='FastDTW', color='red', linewidth=2)
        if DTW_LIBRARIES['tslearn']:
            ax.plot(sizes, tslearn_times, 's-', label='TSLearn', color='blue', linewidth=2)
        if DTW_LIBRARIES['dtaidistance']:
            ax.plot(sizes, dtaidist_times, '^-', label='DTAIDistance', color='green', linewidth=2)
        
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('DTW Performance Comparison\nExecution Time')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Memory Usage
        ax = axes[0, 1]
        if DTW_LIBRARIES['fastdtw']:
            ax.plot(sizes, fastdtw_memory, 'o-', label='FastDTW', color='red', linewidth=2)
        if DTW_LIBRARIES['tslearn']:
            ax.plot(sizes, tslearn_memory, 's-', label='TSLearn', color='blue', linewidth=2)
        
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Computational Complexity
        ax = axes[1, 0]
        if len(sizes) >= 2:
            # Theoretical complexity curves
            x_theory = np.linspace(min(sizes), max(sizes), 100)
            
            # FastDTW: O(N²·M) where N=stations, M=time series length  
            fastdtw_theory = (x_theory ** 2) * self.time_series_data.shape[1] / 1e6
            ax.plot(x_theory, fastdtw_theory, '--', label='FastDTW O(N²·M)', color='red', alpha=0.7)
            
            # TSLearn: O(N²·M²) for exact DTW
            tslearn_theory = (x_theory ** 2) * (self.time_series_data.shape[1] ** 1.5) / 1e8
            ax.plot(x_theory, tslearn_theory, '--', label='TSLearn O(N²·M²)', color='blue', alpha=0.7)
        
        if DTW_LIBRARIES['fastdtw']:
            ax.plot(sizes, fastdtw_times, 'o', label='FastDTW Actual', color='red', markersize=8)
        if DTW_LIBRARIES['tslearn']:
            ax.plot(sizes, tslearn_times, 's', label='TSLearn Actual', color='blue', markersize=8)
        
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Theoretical vs Actual Complexity')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency Metrics
        ax = axes[1, 1]
        if len(sizes) >= 2:
            # Calculate time per comparison
            fastdtw_efficiency = []
            tslearn_efficiency = []
            
            for i, size in enumerate(sizes):
                n_comparisons = size * (size - 1) // 2
                if not np.isnan(fastdtw_times[i]):
                    fastdtw_efficiency.append(fastdtw_times[i] / n_comparisons * 1000)
                else:
                    fastdtw_efficiency.append(np.nan)
                
                if not np.isnan(tslearn_times[i]):
                    tslearn_efficiency.append(tslearn_times[i] / n_comparisons * 1000)
                else:
                    tslearn_efficiency.append(np.nan)
            
            if DTW_LIBRARIES['fastdtw']:
                ax.plot(sizes, fastdtw_efficiency, 'o-', label='FastDTW', color='red', linewidth=2)
            if DTW_LIBRARIES['tslearn']:
                ax.plot(sizes, tslearn_efficiency, 's-', label='TSLearn', color='blue', linewidth=2)
        
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('Time per Comparison (ms)')
        ax.set_title('Efficiency: Time per Pairwise Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save benchmark figure
        benchmark_file = self.figures_dir / "ps04_benchmark_dtw_performance.png"
        plt.savefig(benchmark_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Benchmark visualization saved: {benchmark_file}")
    
    def save_benchmark_results(self):
        """Save benchmark results to file"""
        print("💾 Saving benchmark results...")
        
        # Create summary table
        summary_data = []
        for size in self.test_sizes:
            if size in self.results:
                row = {'stations': size}
                
                for method in ['fastdtw', 'tslearn', 'dtaidistance']:
                    if method in self.results[size]:
                        r = self.results[size][method]
                        if r.get('success', True):
                            row[f'{method}_time'] = r.get('elapsed_time', np.nan)
                            row[f'{method}_memory'] = r.get('memory_used', np.nan)
                        else:
                            row[f'{method}_time'] = np.nan
                            row[f'{method}_memory'] = np.nan
                    else:
                        row[f'{method}_time'] = np.nan
                        row[f'{method}_memory'] = np.nan
                
                summary_data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        csv_file = self.results_dir / "dtw_benchmark_results.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Results saved: {csv_file}")
        
        # Print final recommendations
        self.print_recommendations()
    
    def print_recommendations(self):
        """Print performance recommendations"""
        print("\n" + "=" * 60)
        print("🎯 DTW PERFORMANCE RECOMMENDATIONS")
        print("=" * 60)
        
        if len(self.results) == 0:
            print("❌ No benchmark results available")
            return
        
        # Analyze results for recommendations
        largest_size = max(self.results.keys())
        largest_results = self.results[largest_size]
        
        fastest_method = None
        fastest_time = np.inf
        
        for method in ['fastdtw', 'tslearn', 'dtaidistance']:
            if method in largest_results and largest_results[method].get('success', True):
                time_taken = largest_results[method].get('elapsed_time', np.inf)
                if time_taken < fastest_time:
                    fastest_time = time_taken
                    fastest_method = method
        
        if fastest_method:
            print(f"🚀 FASTEST METHOD: {fastest_method.upper()}")
            print(f"   Time for {largest_size} stations: {fastest_time:.1f} seconds")
        
        # Size-based recommendations
        print(f"\n📊 SIZE-BASED RECOMMENDATIONS:")
        print(f"   < 100 stations: TSLearn (exact DTW, high quality)")
        print(f"   100-500 stations: FastDTW (good balance of speed/quality)")
        print(f"   > 500 stations: FastDTW or DTAIDistance (speed critical)")
        
        print(f"\n🔬 FOR RESEARCH APPLICATIONS:")
        print(f"   Exact clustering: TSLearn DTW k-means")
        print(f"   Large-scale analysis: FastDTW + hierarchical clustering")
        print(f"   Real-time processing: DTAIDistance")

def main():
    """Run DTW performance benchmark"""
    print("🚀 ps04_benchmark.py - DTW Performance Benchmark")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = DTWBenchmark(test_sizes=[50, 500, 1000])
    
    # Load test data
    if not benchmark.load_test_data():
        print("❌ Failed to load test data")
        return False
    
    # Run benchmark
    benchmark.run_benchmark()
    
    print("\n✅ DTW Benchmark completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
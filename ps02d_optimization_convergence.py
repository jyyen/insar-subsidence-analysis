#!/usr/bin/env python3
"""
InSAR Signal Fitting - Optimization Convergence Visualization
Tracks all simulation runs during differential evolution optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from functools import partial
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InSARParameters:
    """InSAR time series parameters"""
    trend: float = 0.0              # mm/year linear trend
    annual_amp: float = 0.0          # mm annual amplitude
    annual_phase: float = 0.0        # radians annual phase
    semiannual_amp: float = 0.0      # mm semi-annual amplitude  
    semiannual_phase: float = 0.0    # radians semi-annual phase
    quarterly_amp: float = 0.0       # mm quarterly amplitude
    quarterly_phase: float = 0.0     # radians quarterly phase
    longperiod_amp: float = 0.0      # mm long-period amplitude
    noise_std: float = 1.0           # mm noise standard deviation

@dataclass 
class OptimizationTracker:
    """Track optimization convergence"""
    iteration: List[int] = field(default_factory=list)
    rmse: List[float] = field(default_factory=list)
    correlation: List[float] = field(default_factory=list)
    parameters: List[np.ndarray] = field(default_factory=list)
    
    def add_evaluation(self, iter_num: int, rmse: float, corr: float, params: np.ndarray):
        """Add evaluation result"""
        self.iteration.append(iter_num)
        self.rmse.append(rmse)
        self.correlation.append(corr)
        self.parameters.append(params.copy())

class InSARFitterWithTracking:
    """InSAR time series fitting with optimization tracking"""
    
    def __init__(self, time_vector: np.ndarray):
        self.time_vector = time_vector
        self.time_years = time_vector / 365.25
        self.tracker = OptimizationTracker()
        self.evaluation_count = 0
        self.observed_signal = None
        
    def array_to_params(self, array: np.ndarray) -> InSARParameters:
        """Convert parameter array to InSARParameters object"""
        return InSARParameters(
            trend=array[0],
            annual_amp=array[1],
            annual_phase=array[2],
            semiannual_amp=array[3], 
            semiannual_phase=array[4],
            quarterly_amp=array[5],
            quarterly_phase=array[6],
            longperiod_amp=array[7],
            noise_std=array[8]
        )
    
    def generate_signal(self, params: InSARParameters) -> np.ndarray:
        """Generate synthetic InSAR signal"""
        t_years = self.time_years
        
        # Linear trend
        signal = params.trend * t_years
        
        # Annual component
        signal += params.annual_amp * np.sin(2*np.pi*t_years + params.annual_phase)
        
        # Semi-annual component  
        signal += params.semiannual_amp * np.sin(4*np.pi*t_years + params.semiannual_phase)
        
        # Quarterly component
        signal += params.quarterly_amp * np.sin(8*np.pi*t_years + params.quarterly_phase)
        
        # Long-period component (2-year)
        signal += params.longperiod_amp * np.sin(np.pi*t_years + 0)
        
        return signal
    
    def objective_function(self, param_array: np.ndarray, observed_signal: np.ndarray) -> float:
        """Objective function for optimization with tracking"""
        try:
            params = self.array_to_params(param_array)
            synthetic = self.generate_signal(params)
            
            # Calculate metrics
            residuals = synthetic - observed_signal
            rmse = np.sqrt(np.mean(residuals**2))
            corr, _ = pearsonr(synthetic, observed_signal)
            
            # Track this evaluation
            self.evaluation_count += 1
            self.tracker.add_evaluation(self.evaluation_count, rmse, corr, param_array)
            
            return rmse
            
        except Exception:
            return 1e6

def load_ps00_data():
    """Load ps00 preprocessed data"""
    try:
        data_file = Path("data/processed/ps00_preprocessed_data.npz")
        if data_file.exists():
            print("📂 Loading ps00 preprocessed data...")
            data = np.load(data_file)
            
            displacement = -data['displacement']  # Fix sign convention: subsidence = negative
            coordinates = data['coordinates']
            n_stations, n_times = displacement.shape
            
            print(f"   📍 {n_stations} stations available")
            print(f"   📅 {n_times} time points")
            
            # Create time vector
            time_vector = np.arange(0, n_times * 6, 6)  # 6-day intervals
            
            return displacement, coordinates, time_vector
        else:
            raise FileNotFoundError("ps00 data not found")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def fit_station_with_tracking(station_idx: int, station_data: np.ndarray, 
                             station_coords: np.ndarray, time_vector: np.ndarray) -> Dict[str, Any]:
    """Fit single station with full optimization tracking"""
    
    print(f"🔄 Fitting station {station_idx} at [{station_coords[0]:.4f}, {station_coords[1]:.4f}]")
    
    try:
        # Initialize fitter with tracking
        fitter = InSARFitterWithTracking(time_vector)
        fitter.observed_signal = station_data
        
        # Define realistic Taiwan parameter bounds
        bounds = [
            (-200, 50),     # trend: mm/year (subsidence to slight uplift)
            (0, 50),        # annual_amp: mm
            (-np.pi, np.pi), # annual_phase
            (0, 30),        # semiannual_amp: mm
            (-np.pi, np.pi), # semiannual_phase
            (0, 20),        # quarterly_amp: mm
            (-np.pi, np.pi), # quarterly_phase
            (0, 15),        # longperiod_amp: mm
            (0.1, 10),      # noise_std: mm
        ]
        
        # Run optimization with parallel workers and tracking
        import multiprocessing as mp
        n_workers = min(4, mp.cpu_count())  # Conservative for stability
        
        result = differential_evolution(
            partial(fitter.objective_function, observed_signal=station_data),
            bounds,
            maxiter=200,  # Sufficient for convergence tracking
            popsize=8,    # Reasonable population size
            tol=0.01,
            seed=42 + station_idx,  # Different seed per station
            disp=False,
            workers=n_workers,  # Enable parallel evaluation
            updating='deferred'  # Better for parallel workers
        )
        
        if not hasattr(result, 'x'):
            return None
        
        # Extract final results
        fitted_params = fitter.array_to_params(result.x)
        synthetic = fitter.generate_signal(fitted_params)
        
        # Calculate final quality metrics
        residuals = synthetic - station_data
        rmse = np.sqrt(np.mean(residuals**2))
        corr, _ = pearsonr(synthetic, station_data)
        
        print(f"   ✅ Converged: RMSE={rmse:.2f}mm, Corr={corr:.3f}, {len(fitter.tracker.rmse)} evaluations")
        
        return {
            'station_idx': station_idx,
            'coordinates': station_coords,
            'fitted_params': fitted_params,
            'final_rmse': rmse,
            'final_correlation': corr,
            'tracker': fitter.tracker,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return {
            'station_idx': station_idx,
            'coordinates': station_coords,
            'error': str(e),
            'success': False
        }

def create_convergence_plots(results: List[Dict], save_dir: str = "figures"):
    """Create convergence visualization plots"""
    
    successful_results = [r for r in results if r['success']]
    n_stations = len(successful_results)
    
    if n_stations == 0:
        print("❌ No successful fits to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    if n_stations < 8:
        # Hide unused subplots
        for i in range(n_stations, 8):
            row, col = divmod(i, 4)
            axes[row, col].set_visible(False)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_stations))
    
    for i, result in enumerate(successful_results):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        
        tracker = result['tracker']
        station_idx = result['station_idx']
        coords = result['coordinates']
        
        # Plot all evaluations
        scatter = ax.scatter(tracker.correlation, tracker.rmse, 
                           c=tracker.iteration, cmap='viridis', 
                           alpha=0.6, s=20, edgecolors='none')
        
        # Mark final solution
        final_rmse = result['final_rmse']
        final_corr = result['final_correlation']
        ax.scatter([final_corr], [final_rmse], 
                  color='red', s=100, marker='*', 
                  edgecolors='black', linewidth=1,
                  label=f'Final: RMSE={final_rmse:.1f}mm')
        
        # Mark best RMSE point
        best_rmse_idx = np.argmin(tracker.rmse)
        best_rmse = tracker.rmse[best_rmse_idx]
        best_corr = tracker.correlation[best_rmse_idx]
        ax.scatter([best_corr], [best_rmse],
                  color='orange', s=80, marker='o',
                  edgecolors='black', linewidth=1,
                  label=f'Best RMSE: {best_rmse:.1f}mm')
        
        # Formatting
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('RMSE (mm)')
        ax.set_title(f'Station {station_idx}\n[{coords[0]:.3f}, {coords[1]:.3f}]\n{len(tracker.rmse)} evaluations')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Add colorbar for iterations
        if i == 0:  # Only add colorbar to first subplot
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Iteration Number')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    plot_path = save_path / "ps02d_optimization_convergence.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Convergence plot saved: {plot_path}")
    
    plt.close()  # Close figure to free memory

def create_summary_plot(results: List[Dict], save_dir: str = "figures"):
    """Create summary convergence statistics"""
    
    successful_results = [r for r in results if r['success']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Final RMSE vs Correlation for all stations
    final_rmse = [r['final_rmse'] for r in successful_results]
    final_corr = [r['final_correlation'] for r in successful_results]
    station_indices = [r['station_idx'] for r in successful_results]
    
    scatter = ax1.scatter(final_corr, final_rmse, c=station_indices, 
                         cmap='tab10', s=100, alpha=0.8)
    ax1.set_xlabel('Final Correlation Coefficient')
    ax1.set_ylabel('Final RMSE (mm)')
    ax1.set_title('Final Optimization Results\nAll Stations')
    ax1.grid(True, alpha=0.3)
    
    # Add station labels
    for i, (corr, rmse, idx) in enumerate(zip(final_corr, final_rmse, station_indices)):
        ax1.annotate(f'S{idx}', (corr, rmse), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # Plot 2: Number of evaluations per station
    n_evaluations = [len(r['tracker'].rmse) for r in successful_results]
    ax2.bar(range(len(successful_results)), n_evaluations, 
           color=plt.cm.tab10(np.arange(len(successful_results))))
    ax2.set_xlabel('Station Index')
    ax2.set_ylabel('Number of Evaluations')
    ax2.set_title('Optimization Effort\nEvaluations per Station')
    ax2.set_xticks(range(len(successful_results)))
    ax2.set_xticklabels([f'S{r["station_idx"]}' for r in successful_results])
    
    # Plot 3: RMSE improvement distribution
    rmse_improvements = []
    for result in successful_results:
        tracker = result['tracker']
        initial_rmse = tracker.rmse[0]
        final_rmse = result['final_rmse']
        improvement = (initial_rmse - final_rmse) / initial_rmse * 100
        rmse_improvements.append(improvement)
    
    ax3.hist(rmse_improvements, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('RMSE Improvement (%)')
    ax3.set_ylabel('Number of Stations')
    ax3.set_title('RMSE Improvement Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence efficiency
    convergence_rates = []
    for result in successful_results:
        tracker = result['tracker']
        # Calculate convergence rate as improvement per 100 evaluations
        if len(tracker.rmse) > 10:
            initial_rmse = np.mean(tracker.rmse[:10])
            final_rmse = np.mean(tracker.rmse[-10:])
            rate = (initial_rmse - final_rmse) / len(tracker.rmse) * 100
            convergence_rates.append(rate)
    
    if convergence_rates:
        ax4.hist(convergence_rates, bins=8, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Convergence Rate (mm improvement per 100 eval)')
        ax4.set_ylabel('Number of Stations')
        ax4.set_title('Convergence Efficiency')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    plot_path = save_path / "ps02d_convergence_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Summary plot saved: {plot_path}")
    
    plt.close()  # Close figure to free memory

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='InSAR Optimization Convergence Analysis')
    parser.add_argument('--n-stations', type=int, default=6, 
                       help='Number of stations to analyze (6-8 recommended)')
    parser.add_argument('--save-dir', type=str, default='figures',
                       help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for station selection')
    
    args = parser.parse_args()
    
    print("🎯 InSAR Optimization Convergence Analysis")
    print("=" * 50)
    
    # Load data
    displacement, coordinates, time_vector = load_ps00_data()
    if displacement is None:
        return
    
    # Select stations
    np.random.seed(args.seed)
    n_available = displacement.shape[0]
    n_stations = min(args.n_stations, n_available, 8)  # Max 8 for subplot layout
    
    # Select diverse stations (spread across coordinate range)
    station_indices = np.random.choice(n_available, size=n_stations, replace=False)
    station_indices = sorted(station_indices)
    
    print(f"🎲 Selected {n_stations} stations: {station_indices}")
    
    # Fit stations with tracking
    results = []
    for i, station_idx in enumerate(station_indices):
        print(f"\n[{i+1}/{n_stations}] Processing station {station_idx}")
        
        station_data = displacement[station_idx, :]
        station_coords = coordinates[station_idx, :]
        
        # Remove NaN values
        valid_mask = ~np.isnan(station_data)
        if np.sum(valid_mask) < 50:  # Need minimum data points
            print(f"   ⚠️ Insufficient data ({np.sum(valid_mask)} points)")
            continue
            
        clean_data = station_data[valid_mask]
        clean_time = time_vector[valid_mask]
        
        result = fit_station_with_tracking(station_idx, clean_data, station_coords, clean_time)
        if result:
            results.append(result)
    
    print(f"\n✅ Successfully fitted {len(results)} stations")
    
    # Create visualizations
    if results:
        print("\n📊 Creating convergence visualizations...")
        create_convergence_plots(results, args.save_dir)
        create_summary_plot(results, args.save_dir)
        
        # Print summary statistics
        print("\n📈 Summary Statistics:")
        successful_results = [r for r in results if r['success']]
        for result in successful_results:
            station_idx = result['station_idx']
            final_rmse = result['final_rmse']
            final_corr = result['final_correlation']
            n_eval = len(result['tracker'].rmse)
            print(f"   Station {station_idx}: RMSE={final_rmse:.2f}mm, Corr={final_corr:.3f}, {n_eval} evaluations")
    else:
        print("❌ No successful fits to visualize")

if __name__ == "__main__":
    main()
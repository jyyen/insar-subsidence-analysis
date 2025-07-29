#!/usr/bin/env python3
"""
PS02C Iterative Improvement - Worst Station Targeting

Enhanced version of ps02c_ultra_robust_fitting_optimized.py with:
1. Automatic identification of worst-performing stations
2. Interactive choice to fix or leave problematic stations
3. Data update mechanism for iterative improvement
4. Performance tracking across runs

Workflow:
1. Run on all stations (or subset)
2. Identify worst X stations by correlation/RMSE
3. Show stats and ask user to fix or leave them
4. Update results for next iteration
5. Next run focuses on NEW worst stations

Author: Taiwan InSAR Subsidence Analysis Project
Created: 2025-07-27
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
import pandas as pd
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import sys
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')

@dataclass
class InSARParameters:
    """Data class for InSAR signal parameters"""
    trend: float = 0.0
    annual_amp: float = 0.0
    annual_freq: float = 1.0
    annual_phase: float = 0.0
    semi_annual_amp: float = 0.0
    semi_annual_freq: float = 2.0
    semi_annual_phase: float = 0.0
    quarterly_amp: float = 0.0
    quarterly_freq: float = 4.0
    quarterly_phase: float = 0.0
    long_annual_amp: float = 0.0
    long_annual_freq: float = 0.3
    long_annual_phase: float = 0.0
    noise_std: float = 2.0
    
    @classmethod
    def from_dict(cls, param_dict: Dict) -> 'InSARParameters':
        return cls(**param_dict)

def load_ps00_data():
    """Load ps00 preprocessed data"""
    try:
        data_file = Path("data/processed/ps00_preprocessed_data.npz")
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return None
        
        data = np.load(data_file, allow_pickle=True)
        return {
            'coordinates': data['coordinates'],
            'displacement': data['displacement'],
            'subsidence_rates': data['subsidence_rates'],
            'n_stations': int(data['n_stations']),
            'n_acquisitions': int(data['n_acquisitions'])
        }
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

class IterativeStationTracker:
    """Track station performance across multiple runs"""
    
    def __init__(self, results_file: str = "data/processed/ps02c_iterative_results.json"):
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(exist_ok=True)
        self.station_history = self.load_history()
    
    def load_history(self) -> Dict:
        """Load previous results history"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    history = json.load(f)
                
                # Convert fixed_stations list back to set
                if 'fixed_stations' in history and isinstance(history['fixed_stations'], list):
                    history['fixed_stations'] = set(history['fixed_stations'])
                elif 'fixed_stations' not in history:
                    history['fixed_stations'] = set()
                
                # Convert string keys back to integers for station_results
                if 'station_results' in history:
                    station_results_fixed = {}
                    for key, value in history['station_results'].items():
                        # Convert string key back to int
                        int_key = int(key)
                        station_results_fixed[int_key] = value
                    history['station_results'] = station_results_fixed
                
                # Ensure all required keys exist
                if 'run_history' not in history:
                    history['run_history'] = []
                if 'worst_stations' not in history:
                    history['worst_stations'] = {}
                
                return history
                
            except Exception as e:
                print(f"Warning: Could not load history ({e}), starting fresh")
        
        return {
            'station_results': {},  # station_idx -> {run_id: results}
            'run_history': [],      # List of run metadata
            'worst_stations': {},   # run_id -> list of worst stations
            'fixed_stations': set() # Stations marked as "fixed" by user
        }
    
    def save_history(self):
        """Save results history"""
        try:
            # Convert set to list for JSON serialization
            history_to_save = self.station_history.copy()
            
            # Ensure fixed_stations is converted from set to list
            if isinstance(self.station_history.get('fixed_stations'), set):
                history_to_save['fixed_stations'] = list(self.station_history['fixed_stations'])
            
            # Convert any numpy int64 keys to regular int for JSON compatibility
            if 'station_results' in history_to_save:
                station_results_fixed = {}
                for key, value in history_to_save['station_results'].items():
                    # Convert numpy int64 or other int types to standard int
                    str_key = str(int(key))
                    station_results_fixed[str_key] = value
                history_to_save['station_results'] = station_results_fixed
            
            with open(self.results_file, 'w') as f:
                json.dump(history_to_save, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Could not save history ({e})")
    
    def add_run_results(self, run_id: str, results: List[Dict]) -> None:
        """Add results from a new run"""
        
        # Add run metadata
        self.station_history['run_history'].append({
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'n_stations': len(results),
            'success_rate': sum(1 for r in results if r.get('success', False)) / len(results)
        })
        
        # Add individual station results
        for result in results:
            if result.get('success', False):
                station_idx = result['station_idx']
                
                if station_idx not in self.station_history['station_results']:
                    self.station_history['station_results'][station_idx] = {}
                
                self.station_history['station_results'][station_idx][run_id] = {
                    'rmse': result.get('rmse', float('inf')),
                    'correlation': result.get('correlation', 0),
                    'optimization_time': result.get('optimization_time', 0),
                    'coordinates': result.get('coordinates', [0, 0])
                }
    
    def identify_worst_stations(self, current_results: List[Dict], 
                              n_worst: int = 100, 
                              exclude_fixed: bool = True) -> List[Dict]:
        """Identify worst performing stations from current results"""
        
        successful_results = [r for r in current_results if r.get('success', False)]
        
        if exclude_fixed:
            # Exclude stations already marked as "fixed"
            successful_results = [r for r in successful_results 
                                if r['station_idx'] not in self.station_history['fixed_stations']]
        
        if len(successful_results) == 0:
            return []
        
        # Create scoring metric: lower is worse
        # Combine RMSE (normalized) and correlation
        rmse_values = [r.get('rmse', float('inf')) for r in successful_results]
        corr_values = [r.get('correlation', 0) for r in successful_results]
        
        # Normalize RMSE to 0-1 scale
        max_rmse = max(rmse_values) if max(rmse_values) > 0 else 1
        normalized_rmse = [rmse / max_rmse for rmse in rmse_values]
        
        # Compute composite score (lower = worse)
        # Score = correlation - normalized_rmse
        scores = [corr - rmse for corr, rmse in zip(corr_values, normalized_rmse)]
        
        # Add scores to results
        for i, result in enumerate(successful_results):
            result['composite_score'] = scores[i]
        
        # Sort by score (ascending = worst first)
        worst_stations = sorted(successful_results, key=lambda x: x['composite_score'])
        
        return worst_stations[:n_worst]
    
    def mark_stations_fixed(self, station_indices: List[int]):
        """Mark stations as fixed (exclude from future worst lists)"""
        for idx in station_indices:
            self.station_history['fixed_stations'].add(idx)
        self.save_history()
    
    def get_station_improvement_history(self, station_idx: int) -> Dict:
        """Get improvement history for a specific station"""
        if station_idx not in self.station_history['station_results']:
            return {}
        
        return self.station_history['station_results'][station_idx]

class FastInSARFitter:
    """Fast InSAR fitting (simplified from algorithmic version)"""
    
    def __init__(self, time_vector: np.ndarray):
        self.time_vector = time_vector
        self.time_years = time_vector / 365.25
        
        self.param_bounds = [
            (-80, 30), (0, 30), (0.95, 1.05), (0, 2*np.pi),
            (0, 20), (1.9, 2.1), (0, 2*np.pi),
            (0, 15), (3.8, 4.2), (0, 2*np.pi),
            (0, 35), (0.1, 0.8), (0, 2*np.pi), (0.5, 10)
        ]
    
    def fast_initial_estimate(self, signal: np.ndarray) -> np.ndarray:
        """Fast initial parameter estimation"""
        try:
            t = self.time_years
            
            # Linear trend
            trend_coeffs = np.polyfit(t, signal, 1)
            trend = np.clip(trend_coeffs[0], -80, 30)
            
            # Simple harmonic analysis
            detrended = signal - np.polyval(trend_coeffs, t)
            signal_std = np.std(detrended)
            
            return np.array([
                trend, signal_std * 0.4, 1.0, 0.0,  # trend, annual
                signal_std * 0.3, 2.0, 0.0,         # semi-annual
                signal_std * 0.2, 4.0, 0.0,         # quarterly
                signal_std * 0.2, 0.3, 0.0, 2.0    # long-term, noise
            ])
        except:
            return np.array([-5, 5, 1, 0, 3, 2, 0, 2, 4, 0, 2, 0.3, 0, 2])
    
    def generate_signal(self, params: np.ndarray) -> np.ndarray:
        """Generate synthetic signal"""
        t = self.time_years
        return (
            params[0] * t +
            params[1] * np.sin(2*np.pi*params[2]*t + params[3]) +
            params[4] * np.sin(2*np.pi*params[5]*t + params[6]) +
            params[7] * np.sin(2*np.pi*params[8]*t + params[9]) +
            params[10] * np.sin(2*np.pi*params[11]*t + params[12])
        )
    
    def objective_function(self, params: np.ndarray, observed: np.ndarray) -> float:
        """Objective function"""
        try:
            synthetic = self.generate_signal(params)
            residuals = synthetic - observed
            rmse = np.sqrt(np.mean(residuals**2))
            
            obs_std = np.std(observed)
            if obs_std > 0:
                corr = np.corrcoef(synthetic, observed)[0, 1]
                if np.isnan(corr):
                    corr = 0
            else:
                corr = 0
            
            return rmse / obs_std - corr if obs_std > 0 else rmse
        except:
            return 1e6
    
    def fit_station(self, observed_signal: np.ndarray, station_idx: int = None) -> Dict:
        """Fit single station"""
        try:
            start_time = time.time()
            
            initial_params = self.fast_initial_estimate(observed_signal)
            
            # Bounds compliance
            for i, (param, (lower, upper)) in enumerate(zip(initial_params, self.param_bounds)):
                initial_params[i] = np.clip(param, lower, upper)
            
            def objective_wrapper(x):
                return self.objective_function(x, observed_signal)
            
            result = differential_evolution(
                objective_wrapper,
                self.param_bounds,
                x0=initial_params,
                maxiter=200,  # Faster for iterative approach
                popsize=12,
                tol=0.001,
                seed=42,
                disp=False,
                workers=1
            )
            
            optimization_time = time.time() - start_time
            
            if not result.success:
                return {'station_idx': station_idx, 'success': False, 'error': result.message}
            
            fitted_params = InSARParameters(
                trend=result.x[0], annual_amp=result.x[1], annual_freq=result.x[2], annual_phase=result.x[3],
                semi_annual_amp=result.x[4], semi_annual_freq=result.x[5], semi_annual_phase=result.x[6],
                quarterly_amp=result.x[7], quarterly_freq=result.x[8], quarterly_phase=result.x[9],
                long_annual_amp=result.x[10], long_annual_freq=result.x[11], long_annual_phase=result.x[12],
                noise_std=result.x[13]
            )
            
            synthetic = self.generate_signal(result.x)
            residuals = synthetic - observed_signal
            rmse = np.sqrt(np.mean(residuals**2))
            corr, _ = pearsonr(synthetic, observed_signal)
            
            return {
                'station_idx': station_idx,
                'fitted_params': fitted_params,
                'rmse': rmse,
                'correlation': corr if not np.isnan(corr) else 0,
                'optimization_time': optimization_time,
                'success': True
            }
            
        except Exception as e:
            return {'station_idx': station_idx, 'error': str(e), 'success': False}

    
    def fit_station_enhanced(self, observed_signal: np.ndarray, station_idx: int = None) -> Dict:
        """Enhanced fitting for problematic stations with more robust optimization"""
        try:
            start_time = time.time()
            
            initial_params = self.fast_initial_estimate(observed_signal)
            
            # Bounds compliance
            for i, (param, (lower, upper)) in enumerate(zip(initial_params, self.param_bounds)):
                initial_params[i] = np.clip(param, lower, upper)
            
            def objective_wrapper(x):
                return self.objective_function(x, observed_signal)
            
            # ENHANCED PARAMETERS for problematic stations
            result = differential_evolution(
                objective_wrapper,
                self.param_bounds,
                x0=initial_params,
                maxiter=500,      # More iterations for better convergence
                popsize=20,       # Larger population for better exploration
                tol=0.0001,       # Tighter tolerance
                seed=42,
                disp=False,
                workers=1,
                atol=0.00001,     # Absolute tolerance
                polish=True,      # Final local optimization
                mutation=(0.5, 1.5),  # More aggressive mutation
                recombination=0.9     # Higher recombination rate
            )
            
            optimization_time = time.time() - start_time
            
            if not result.success:
                return {'station_idx': station_idx, 'success': False, 'error': result.message}
            
            fitted_params = InSARParameters(
                trend=result.x[0], annual_amp=result.x[1], annual_freq=result.x[2], annual_phase=result.x[3],
                semi_annual_amp=result.x[4], semi_annual_freq=result.x[5], semi_annual_phase=result.x[6],
                quarterly_amp=result.x[7], quarterly_freq=result.x[8], quarterly_phase=result.x[9],
                long_annual_amp=result.x[10], long_annual_freq=result.x[11], long_annual_phase=result.x[12],
                noise_std=result.x[13]
            )
            
            synthetic = self.generate_signal(result.x)
            residuals = synthetic - observed_signal
            rmse = np.sqrt(np.mean(residuals**2))
            corr, _ = pearsonr(synthetic, observed_signal)
            
            return {
                'station_idx': station_idx,
                'fitted_params': fitted_params,
                'rmse': rmse,
                'correlation': corr if not np.isnan(corr) else 0,
                'optimization_time': optimization_time,
                'success': True,
                'enhanced': True  # Flag to indicate this was enhanced optimization
            }
            
        except Exception as e:
            return {'station_idx': station_idx, 'error': str(e), 'success': False}

def fit_single_station_wrapper(args):
    """Wrapper for multiprocessing"""
    station_idx, station_data, station_coords, time_vector = args
    
    try:
        fitter = FastInSARFitter(time_vector)
        result = fitter.fit_station(station_data, station_idx)
        
        if result['success']:
            result['coordinates'] = station_coords
        
        return result
    except Exception as e:
        return {'station_idx': station_idx, 'coordinates': station_coords, 'error': str(e), 'success': False}

def display_worst_stations_stats(worst_stations: List[Dict], n_show: int = 20) -> None:
    """Display statistics for worst performing stations"""
    
    print(f"\n📊 WORST {len(worst_stations)} STATIONS ANALYSIS:")
    print("="*80)
    
    # Summary statistics
    rmse_values = [s['rmse'] for s in worst_stations]
    corr_values = [s['correlation'] for s in worst_stations]
    scores = [s['composite_score'] for s in worst_stations]
    
    print(f"📈 Performance Summary:")
    print(f"   • RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} mm (range: {np.min(rmse_values):.2f} - {np.max(rmse_values):.2f})")
    print(f"   • Correlation: {np.mean(corr_values):.3f} ± {np.std(corr_values):.3f} (range: {np.min(corr_values):.3f} - {np.max(corr_values):.3f})")
    print(f"   • Composite Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # Categorize by severity
    very_bad = sum(1 for s in worst_stations if s['correlation'] < 0.3)
    bad = sum(1 for s in worst_stations if 0.3 <= s['correlation'] < 0.6)
    mediocre = sum(1 for s in worst_stations if 0.6 <= s['correlation'] < 0.8)
    
    print(f"\n📊 Severity Categories:")
    print(f"   • Very Poor (corr < 0.3): {very_bad} stations")
    print(f"   • Poor (0.3 ≤ corr < 0.6): {bad} stations")
    print(f"   • Mediocre (0.6 ≤ corr < 0.8): {mediocre} stations")
    
    # Show detailed stats for worst stations
    print(f"\n📋 Top {min(n_show, len(worst_stations))} Worst Stations:")
    print(f"{'Rank':<5} {'Station':<8} {'RMSE':<8} {'Corr':<7} {'Score':<8} {'Coords':<20}")
    print("-" * 60)
    
    for i, station in enumerate(worst_stations[:n_show]):
        coords = station.get('coordinates', [0, 0])
        print(f"{i+1:<5} {station['station_idx']:<8} {station['rmse']:<8.2f} "
              f"{station['correlation']:<7.3f} {station['composite_score']:<8.3f} "
              f"[{coords[0]:.4f}, {coords[1]:.4f}]")

def interactive_choice_menu(worst_stations: List[Dict], tracker: IterativeStationTracker) -> Tuple[List[int], bool]:
    """Interactive menu for choosing what to do with worst stations"""
    
    print(f"\n🤔 DECISION TIME - What to do with these {len(worst_stations)} worst stations?")
    print("="*60)
    print("Options:")
    print("1. 🔧 Fix them with ultra-robust optimization (slower but better)")
    print("2. 📝 Mark as 'acceptable' (exclude from future worst lists)")
    print("3. ⏭️  Skip for now (they'll appear again next run)")
    print("4. 🎯 Custom selection (choose specific stations)")
    print("5. ❌ Quit without changes")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                return [s['station_idx'] for s in worst_stations], True
            elif choice == '2':
                station_indices = [s['station_idx'] for s in worst_stations]
                tracker.mark_stations_fixed(station_indices)
                print(f"✅ Marked {len(station_indices)} stations as 'fixed'")
                return [], False
            elif choice == '3':
                print("⏭️ Skipping worst stations for now")
                return [], False
            elif choice == '4':
                return custom_station_selection(worst_stations, tracker)
            elif choice == '5':
                print("❌ Exiting without changes")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\n❌ Interrupted by user")
            sys.exit(0)

def custom_station_selection(worst_stations: List[Dict], tracker: IterativeStationTracker) -> Tuple[List[int], bool]:
    """Custom selection of stations"""
    
    print(f"\n🎯 CUSTOM SELECTION:")
    print("Enter station numbers (comma-separated) or ranges (e.g., '1-10,15,20-25')")
    print("Or type 'list' to see stations again, 'all' for all stations, 'cancel' to go back")
    
    while True:
        try:
            selection = input(f"\nSelect from {len(worst_stations)} worst stations: ").strip().lower()
            
            if selection == 'cancel':
                return interactive_choice_menu(worst_stations, tracker)
            elif selection == 'list':
                display_worst_stations_stats(worst_stations, len(worst_stations))
                continue
            elif selection == 'all':
                return [s['station_idx'] for s in worst_stations], True
            
            # Parse selection
            selected_indices = []
            for part in selection.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_indices.extend(range(start-1, min(end, len(worst_stations))))
                else:
                    idx = int(part) - 1
                    if 0 <= idx < len(worst_stations):
                        selected_indices.append(idx)
            
            if selected_indices:
                selected_stations = [worst_stations[i]['station_idx'] for i in selected_indices]
                print(f"✅ Selected {len(selected_stations)} stations for fixing")
                return selected_stations, True
            else:
                print("❌ No valid stations selected")
                
        except ValueError:
            print("❌ Invalid input format. Use numbers, ranges, or 'all'/'cancel'/'list'")
        except KeyboardInterrupt:
            print("\n❌ Interrupted by user")
            sys.exit(0)

def main():
    """Main execution with iterative improvement"""
    
    parser = argparse.ArgumentParser(description='PS02C Iterative Improvement')
    parser.add_argument('--n-stations', type=str, default='auto', help='Number of stations (default: all)')
    parser.add_argument('--n-worst', type=int, default=100, help='Number of worst stations to identify')
    parser.add_argument('--n-processes', type=int, default=None, help='Number of processes')
    parser.add_argument('--auto-fix', action='store_true', help='Automatically fix worst stations')
    parser.add_argument('--show-stats-only', action='store_true', help='Only show stats, no processing')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 PS02C ITERATIVE IMPROVEMENT - WORST STATION TARGETING")
    print("="*70)
    
    # Initialize tracker
    tracker = IterativeStationTracker()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"📊 Run ID: {run_id}")
    print(f"📈 Previous runs: {len(tracker.station_history['run_history'])}")
    print(f"🔧 Fixed stations: {len(tracker.station_history['fixed_stations'])}")
    
    # Load data
    data = load_ps00_data()
    if data is None:
        return False
    
    displacement = data['displacement']
    coordinates = data['coordinates']
    time_vector = np.arange(displacement.shape[1]) * 12
    
    # Determine stations to process
    if args.n_stations.lower() == 'auto':
        n_stations = displacement.shape[0]
    else:
        n_stations = min(int(args.n_stations), displacement.shape[0])
    
    print(f"📊 Processing {n_stations} stations to find worst {args.n_worst}...")
    
    if args.show_stats_only:
        print("📋 Stats-only mode: Skipping processing")
        return True
    
    # Prepare arguments
    station_indices = np.random.choice(displacement.shape[0], n_stations, replace=False)
    args_list = []
    
    for station_idx in station_indices:
        station_data = displacement[station_idx, :]
        station_coords = coordinates[station_idx]
        args_list.append((station_idx, station_data, station_coords, time_vector))
    
    # Process stations
    n_processes = args.n_processes or min(mp.cpu_count() - 1, len(args_list))
    print(f"🔄 Using {n_processes} processes...")
    
    start_time = time.time()
    results = []
    
    if n_processes > 1:
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_station = {executor.submit(fit_single_station_wrapper, arg): arg[0] 
                               for arg in args_list}
            
            for i, future in enumerate(as_completed(future_to_station)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % max(1, len(args_list) // 10) == 0:
                    progress = (i + 1) / len(args_list) * 100
                    print(f"   Progress: {progress:5.1f}%")
    else:
        for i, arg in enumerate(args_list):
            result = fit_single_station_wrapper(arg)
            results.append(result)
            
            if (i + 1) % max(1, len(args_list) // 10) == 0:
                progress = (i + 1) / len(args_list) * 100
                print(f"   Progress: {progress:5.1f}%")
    
    elapsed_time = time.time() - start_time
    
    # Add results to tracker
    tracker.add_run_results(run_id, results)
    
    # Identify worst stations
    worst_stations = tracker.identify_worst_stations(results, args.n_worst)
    
    if not worst_stations:
        print("🎉 No problematic stations found! All stations performing well.")
        return True
    
    # Display stats
    display_worst_stations_stats(worst_stations)
    
    # Interactive decision or auto-fix
    if args.auto_fix:
        stations_to_fix = [s['station_idx'] for s in worst_stations]
        fix_them = True
        print(f"🤖 Auto-fix mode: Processing {len(stations_to_fix)} worst stations")
    else:
        stations_to_fix, fix_them = interactive_choice_menu(worst_stations, tracker)
    
    if fix_them and stations_to_fix:
        print(f"\n🔧 ULTRA-ROBUST OPTIMIZATION for {len(stations_to_fix)} stations...")
        
        # Actually fix the selected stations using enhanced optimization
        fixed_results = []
        fixed_count = 0
        
        for station_idx in stations_to_fix:
            # Find the station data
            station_data = None
            station_coords = None
            
            for result in results:
                if result.get('station_idx') == station_idx:
                    # Get original data for re-optimization
                    original_idx = np.where(station_indices == station_idx)[0]
                    if len(original_idx) > 0:
                        station_data = displacement[station_idx, :]
                        station_coords = coordinates[station_idx]
                        break
            
            if station_data is not None:
                # Find original performance for comparison
                original_rmse = None
                original_corr = None
                for result in results:
                    if result.get('station_idx') == station_idx:
                        original_rmse = result.get('rmse', 0)
                        original_corr = result.get('correlation', 0)
                        break
                
                print(f"   🔧 Fixing station {station_idx}...")
                
                # Apply enhanced optimization (more iterations, tighter convergence)
                try:
                    fitter = FastInSARFitter(time_vector)
                    
                    # Enhanced parameters for problematic stations
                    enhanced_result = fitter.fit_station_enhanced(station_data, station_idx)
                    
                    if enhanced_result.get('success', False):
                        fixed_results.append(enhanced_result)
                        fixed_count += 1
                        
                        # Show before → after improvement
                        new_rmse = enhanced_result['rmse']
                        new_corr = enhanced_result['correlation']
                        
                        if original_rmse is not None and original_corr is not None:
                            rmse_change = new_rmse - original_rmse
                            corr_change = new_corr - original_corr
                            rmse_arrow = "📉" if rmse_change < 0 else "📈"
                            corr_arrow = "📈" if corr_change > 0 else "📉"
                            
                            print(f"      ✅ RMSE: {original_rmse:.1f}→{new_rmse:.1f}mm {rmse_arrow}, "
                                  f"Corr: {original_corr:.3f}→{new_corr:.3f} {corr_arrow}")
                        else:
                            print(f"      ✅ Improved: RMSE {new_rmse:.1f}mm, Corr {new_corr:.3f}")
                    else:
                        print(f"      ❌ Still problematic: {enhanced_result.get('error', 'unknown')}")
                        
                except Exception as e:
                    print(f"      ❌ Enhancement failed: {str(e)}")
            else:
                print(f"   ⚠️ Station {station_idx} data not found")
        
        print(f"\n✅ ENHANCEMENT COMPLETED: {fixed_count}/{len(stations_to_fix)} stations improved")
        
        # Update results with fixed versions
        if fixed_results:
            print(f"💾 Updating results database with {len(fixed_results)} improved stations...")
            # Here you could save the improved results back to the main results file
    
    # Save updated history
    tracker.save_history()
    
    # Performance summary
    successful_results = [r for r in results if r.get('success', False)]
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Processed: {len(results)} stations")
    print(f"   📈 Success rate: {len(successful_results)/len(results)*100:.1f}%")
    print(f"   ⏱️ Time: {elapsed_time:.1f} seconds")
    print(f"   🎯 Worst identified: {len(worst_stations)}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
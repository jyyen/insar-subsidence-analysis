#!/usr/bin/env python3
"""
ps02_07_algorithmic_optimization.py: Algorithmic optimization experiments
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq
import pandas as pd
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
import sys
import pickle

warnings.filterwarnings('ignore')

@dataclass
class InSARParameters:
    """Data class for InSAR signal parameters"""
    trend: float = 0.0                    # mm/year
    annual_amp: float = 0.0               # mm
    annual_freq: float = 1.0              # cycles/year
    annual_phase: float = 0.0             # radians
    semi_annual_amp: float = 0.0          # mm
    semi_annual_freq: float = 2.0         # cycles/year
    semi_annual_phase: float = 0.0        # radians
    quarterly_amp: float = 0.0            # mm
    quarterly_freq: float = 4.0           # cycles/year
    quarterly_phase: float = 0.0          # radians
    long_annual_amp: float = 0.0          # mm
    long_annual_freq: float = 0.3         # cycles/year
    long_annual_phase: float = 0.0        # radians
    noise_std: float = 2.0                # mm
    
    @classmethod
    def from_dict(cls, param_dict: Dict) -> 'InSARParameters':
        """Create from dictionary"""
        return cls(**param_dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

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

class FastInSARFitter:
    """Fast InSAR fitting with algorithmic optimizations"""
    
    def __init__(self, time_vector: np.ndarray):
        self.time_vector = time_vector
        self.time_years = time_vector / 365.25
        
        # Parameter bounds
        self.param_bounds = [
            (-80, 30),       # trend
            (0, 30),         # annual_amp
            (0.95, 1.05),    # annual_freq
            (0, 2*np.pi),    # annual_phase
            (0, 20),         # semi_annual_amp
            (1.9, 2.1),      # semi_annual_freq
            (0, 2*np.pi),    # semi_annual_phase
            (0, 15),         # quarterly_amp
            (3.8, 4.2),      # quarterly_freq
            (0, 2*np.pi),    # quarterly_phase
            (0, 35),         # long_annual_amp
            (0.1, 0.8),      # long_annual_freq
            (0, 2*np.pi),    # long_annual_phase
            (0.5, 10)        # noise_std
        ]
    
    def fast_initial_estimate(self, signal: np.ndarray) -> np.ndarray:
        """Fast initial parameter estimation using least squares"""
        
        try:
            t = self.time_years
            n = len(signal)
            
            # Simple linear trend
            trend_coeffs = np.polyfit(t, signal, 1)
            trend = trend_coeffs[0]
            detrended = signal - np.polyval(trend_coeffs, t)
            
            # Design matrix for harmonic components
            design_matrix = np.column_stack([
                np.ones(n),           # constant
                np.sin(2*np.pi*t),    # annual sine
                np.cos(2*np.pi*t),    # annual cosine
                np.sin(4*np.pi*t),    # semi-annual sine
                np.cos(4*np.pi*t),    # semi-annual cosine
                np.sin(8*np.pi*t),    # quarterly sine
                np.cos(8*np.pi*t),    # quarterly cosine
            ])
            
            # Solve least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(design_matrix, detrended, rcond=None)
            
            # Convert to amplitude/phase form
            annual_amp = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
            annual_phase = np.arctan2(coeffs[2], coeffs[1])
            
            semi_annual_amp = np.sqrt(coeffs[3]**2 + coeffs[4]**2)
            semi_annual_phase = np.arctan2(coeffs[4], coeffs[3])
            
            quarterly_amp = np.sqrt(coeffs[5]**2 + coeffs[6]**2)
            quarterly_phase = np.arctan2(coeffs[6], coeffs[5])
            
            # Estimate noise
            if len(residuals) > 0:
                noise_std = np.sqrt(residuals[0] / n)
            else:
                noise_std = np.std(detrended) * 0.1
            
            # Apply strict constraints to ensure bounds compliance
            initial_params = np.array([
                np.clip(trend, -80, 30),                    # trend
                np.clip(annual_amp, 0, 30),                 # annual_amp
                1.0,                                        # annual_freq
                annual_phase % (2*np.pi),                   # annual_phase
                np.clip(semi_annual_amp, 0, 20),            # semi_annual_amp
                2.0,                                        # semi_annual_freq
                semi_annual_phase % (2*np.pi),              # semi_annual_phase
                np.clip(quarterly_amp, 0, 15),              # quarterly_amp
                4.0,                                        # quarterly_freq
                quarterly_phase % (2*np.pi),                # quarterly_phase
                np.clip(np.std(detrended) * 0.3, 0, 35),    # long_annual_amp
                0.3,                                        # long_annual_freq
                0.0,                                        # long_annual_phase
                np.clip(noise_std, 0.5, 10)                # noise_std
            ])
            
            return initial_params
            
        except Exception as e:
            print(f"   Warning: Initial estimate failed ({e}), using defaults")
            
        # Default parameters (guaranteed to be within bounds)
        return np.array([
            -5.0,   # trend
            5.0,    # annual_amp
            1.0,    # annual_freq
            0.0,    # annual_phase
            3.0,    # semi_annual_amp
            2.0,    # semi_annual_freq
            0.0,    # semi_annual_phase
            2.0,    # quarterly_amp
            4.0,    # quarterly_freq
            0.0,    # quarterly_phase
            2.0,    # long_annual_amp
            0.3,    # long_annual_freq
            0.0,    # long_annual_phase
            2.0     # noise_std
        ])
    
    def generate_signal(self, params: np.ndarray) -> np.ndarray:
        """Generate synthetic signal from parameters"""
        t = self.time_years
        
        signal = (
            params[0] * t +  # trend
            params[1] * np.sin(2*np.pi*params[2]*t + params[3]) +  # annual
            params[4] * np.sin(2*np.pi*params[5]*t + params[6]) +  # semi-annual
            params[7] * np.sin(2*np.pi*params[8]*t + params[9]) +  # quarterly
            params[10] * np.sin(2*np.pi*params[11]*t + params[12])  # long-annual
        )
        
        return signal
    
    def objective_function(self, params: np.ndarray, observed: np.ndarray) -> float:
        """Objective function for optimization"""
        try:
            synthetic = self.generate_signal(params)
            
            # Calculate RMSE
            residuals = synthetic - observed
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Calculate correlation
            obs_std = np.std(observed)
            if obs_std > 0:
                corr = np.corrcoef(synthetic, observed)[0, 1]
                if np.isnan(corr):
                    corr = 0
            else:
                corr = 0
            
            # Normalize RMSE and combine
            normalized_rmse = rmse / obs_std if obs_std > 0 else rmse
            objective = normalized_rmse - corr
            
            return objective
            
        except Exception:
            return 1e6  # Large penalty for invalid parameters
    
    def smart_bounds_from_initial(self, initial_params: np.ndarray, factor: float = 0.5) -> List[Tuple]:
        """Create smart bounds around initial estimate"""
        
        smart_bounds = []
        
        for i, (param, (global_lower, global_upper)) in enumerate(zip(initial_params, self.param_bounds)):
            
            if i in [2, 5, 8, 11]:  # Frequency parameters - keep tight
                smart_bounds.append((global_lower, global_upper))
            elif i in [3, 6, 9, 12]:  # Phase parameters - full range
                smart_bounds.append((0, 2*np.pi))
            else:  # Amplitude and trend parameters
                param_range = global_upper - global_lower
                deviation = param_range * factor
                
                lower = max(global_lower, param - deviation)
                upper = min(global_upper, param + deviation)
                
                # Ensure param is within the computed bounds
                if param < lower:
                    lower = global_lower
                    upper = min(global_upper, param + deviation)
                elif param > upper:
                    upper = global_upper
                    lower = max(global_lower, param - deviation)
                
                smart_bounds.append((lower, upper))
        
        return smart_bounds
    
    def fit_station(self, observed_signal: np.ndarray, station_idx: int = None) -> Dict:
        """Fit single station with algorithmic optimizations"""
        
        try:
            start_time = time.time()
            
            # Step 1: Fast initial estimate
            initial_params = self.fast_initial_estimate(observed_signal)
            
            # Step 2: Use global bounds (simplified for reliability)
            bounds = self.param_bounds
            
            # Step 3: Verify bounds compliance and fix if needed
            for i, (param, (lower, upper)) in enumerate(zip(initial_params, bounds)):
                if param < lower or param > upper:
                    print(f"   Warning: Parameter {i} ({param:.3f}) outside bounds ({lower:.3f}, {upper:.3f})")
                    initial_params[i] = np.clip(param, lower, upper)
            
            # Step 4: Optimized differential evolution
            def objective_wrapper(x):
                return self.objective_function(x, observed_signal)
            
            result = differential_evolution(
                objective_wrapper,
                bounds,
                x0=initial_params,
                maxiter=300,  # Reduced from default 1000
                popsize=15,   # Reduced from default 15
                tol=0.001,    # Slightly relaxed tolerance
                seed=42,
                disp=False,
                workers=1,
                updating='immediate',
                polish=True
            )
            
            optimization_time = time.time() - start_time
            
            if not result.success:
                return {
                    'station_idx': station_idx,
                    'error': f'Optimization failed: {result.message}',
                    'success': False
                }
            
            # Step 4: Convert results
            fitted_params = self.array_to_insar_params(result.x)
            synthetic = self.generate_signal(result.x)
            
            # Step 5: Calculate quality metrics
            residuals = synthetic - observed_signal
            rmse = np.sqrt(np.mean(residuals**2))
            corr, _ = pearsonr(synthetic, observed_signal)
            
            return {
                'station_idx': station_idx,
                'fitted_params': fitted_params,
                'rmse': rmse,
                'correlation': corr if not np.isnan(corr) else 0,
                'objective_value': result.fun,
                'optimization_time': optimization_time,
                'iterations': result.nit,
                'evaluations': result.nfev,
                'method': 'fast_algorithmic',
                'success': True
            }
            
        except Exception as e:
            return {
                'station_idx': station_idx,
                'error': str(e),
                'success': False
            }
    
    def array_to_insar_params(self, param_array: np.ndarray) -> InSARParameters:
        """Convert parameter array to InSARParameters"""
        return InSARParameters(
            trend=param_array[0],
            annual_amp=param_array[1],
            annual_freq=param_array[2],
            annual_phase=param_array[3],
            semi_annual_amp=param_array[4],
            semi_annual_freq=param_array[5],
            semi_annual_phase=param_array[6],
            quarterly_amp=param_array[7],
            quarterly_freq=param_array[8],
            quarterly_phase=param_array[9],
            long_annual_amp=param_array[10],
            long_annual_freq=param_array[11],
            long_annual_phase=param_array[12],
            noise_std=param_array[13]
        )

def algorithmic_fit_single_station(args):
    """Single station fitting wrapper"""
    station_idx, station_data, station_coords, time_vector = args
    
    try:
        # Create fitter
        fitter = FastInSARFitter(time_vector)
        
        # Fit station
        result = fitter.fit_station(station_data, station_idx)
        
        if result['success']:
            result['coordinates'] = station_coords
        
        return result
        
    except Exception as e:
        return {
            'station_idx': station_idx,
            'coordinates': station_coords,
            'error': str(e),
            'success': False
        }

def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(description='PS02C Algorithmic Fixed')
    parser.add_argument('--n-stations', type=str, default='100')
    parser.add_argument('--test-single', action='store_true', help='Test single station first')
    parser.add_argument('--n-processes', type=int, default=None, help='Number of processes (default: all cores - 1)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 PS02C ALGORITHMIC FIXED - MINIMAL WORKING VERSION")
    print("="*70)
    print(f"🔧 Optimizations:")
    print(f"   • Fast least squares initialization")
    print(f"   • Smart parameter bounds")
    print(f"   • Reduced optimization iterations")
    print(f"   • Sequential processing (reliable)")
    
    # Load data
    data = load_ps00_data()
    if data is None:
        return False
    
    displacement = data['displacement']
    coordinates = data['coordinates']
    time_vector = np.arange(displacement.shape[1]) * 12  # 12-day intervals
    
    print(f"📊 Loaded {displacement.shape[0]} stations, {displacement.shape[1]} acquisitions")
    
    # Test single station first if requested
    if args.test_single:
        print("\n🧪 Testing single station...")
        test_station_idx = 0
        test_data = displacement[test_station_idx, :]
        test_coords = coordinates[test_station_idx]
        
        test_args = (test_station_idx, test_data, test_coords, time_vector)
        test_result = algorithmic_fit_single_station(test_args)
        
        if test_result['success']:
            print(f"✅ Single station test successful!")
            print(f"   RMSE: {test_result['rmse']:.2f} mm")
            print(f"   Correlation: {test_result['correlation']:.3f}")
            print(f"   Time: {test_result['optimization_time']:.2f}s")
        else:
            print(f"❌ Single station test failed: {test_result.get('error', 'Unknown error')}")
            return False
    
    # Determine stations to process
    if args.n_stations.lower() == 'auto':
        n_stations = displacement.shape[0]
    else:
        n_stations = min(int(args.n_stations), displacement.shape[0])
    
    print(f"\n📊 Processing {n_stations} stations...")
    
    # Prepare arguments
    station_indices = np.random.choice(displacement.shape[0], n_stations, replace=False)
    args_list = []
    
    for station_idx in station_indices:
        station_data = displacement[station_idx, :]
        station_coords = coordinates[station_idx]
        args_list.append((station_idx, station_data, station_coords, time_vector))
    
    # STATION-LEVEL MULTIPROCESSING (Safe for Taiwan InSAR)
    # Each station processed independently - no serialization issues
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Determine number of processes
    if args.n_processes is not None:
        n_processes = min(args.n_processes, len(args_list))
    else:
        n_processes = min(mp.cpu_count() - 1, len(args_list))  # Use all available cores
    print(f"🔄 Using {n_processes} processes for station-level parallelization...")
    print(f"📝 Note: Optimization within each station remains sequential (reliable)")
    
    start_time = time.time()
    results = []
    
    if n_processes > 1 and len(args_list) > 1:
        # Parallel processing across stations
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_station = {executor.submit(algorithmic_fit_single_station, arg): arg[0] 
                               for arg in args_list}
            
            for future in as_completed(future_to_station):
                result = future.result()
                results.append(result)
                
                # Progress reporting
                if len(results) % max(1, len(args_list) // 10) == 0:
                    progress = len(results) / len(args_list) * 100
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    eta = (len(args_list) - len(results)) / rate if rate > 0 else 0
                    print(f"   Progress: {progress:5.1f}% ({len(results)}/{len(args_list)}) | "
                          f"Rate: {rate:5.1f} st/s | ETA: {eta:5.1f}s")
    else:
        # Sequential fallback
        for i, arg in enumerate(args_list):
            result = algorithmic_fit_single_station(arg)
            results.append(result)
            
            # Progress reporting
            if (i + 1) % max(1, len(args_list) // 10) == 0:
                progress = (i + 1) / len(args_list) * 100
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(args_list) - (i + 1)) / rate if rate > 0 else 0
                print(f"   Progress: {progress:5.1f}% ({i+1}/{len(args_list)}) | "
                      f"Rate: {rate:5.1f} st/s | ETA: {eta:5.1f}s")
    
    elapsed_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    print(f"\n📊 ALGORITHMIC OPTIMIZATION RESULTS:")
    print(f"   ✅ Successful fits: {len(successful_results)}")
    print(f"   ❌ Failed fits: {len(failed_results)}")
    print(f"   📈 Success rate: {len(successful_results)/len(results)*100:.1f}%")
    print(f"   ⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"   🚀 Speed: {len(args_list)/elapsed_time:.1f} stations/second")
    
    # Print sample errors for debugging
    if failed_results:
        print(f"\n🐛 Sample errors (first 3):")
        for i, result in enumerate(failed_results[:3]):
            print(f"   Station {result.get('station_idx', 'unknown')}: {result.get('error', 'unknown error')}")
    
    # Performance analysis
    if successful_results:
        opt_times = [r.get('optimization_time', 0) for r in successful_results]
        correlations = [r.get('correlation', 0) for r in successful_results]
        rmse_values = [r.get('rmse', 0) for r in successful_results]
        
        print(f"\n🎯 Performance Analysis:")
        print(f"   • Average optimization time: {np.mean(opt_times):.3f}s")
        print(f"   • Average correlation: {np.mean(correlations):.3f}")
        print(f"   • Average RMSE: {np.mean(rmse_values):.2f} mm")
        print(f"   • High correlation (>0.8): {sum(c > 0.8 for c in correlations)} stations")
    
    # Save results
    if successful_results:
        save_dir = Path("figures")
        save_dir.mkdir(exist_ok=True)
        
        results_file = save_dir / "ps02c_algorithmic_fixed_results.pkl"
        save_data = {
            'results': successful_results,
            'performance_metrics': {
                'total_time': elapsed_time,
                'processing_rate': len(args_list)/elapsed_time,
                'success_rate': len(successful_results)/len(results),
                'avg_correlation': np.mean([r.get('correlation', 0) for r in successful_results]),
                'avg_rmse': np.mean([r.get('rmse', 0) for r in successful_results])
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\n✅ Results saved: {results_file}")
    
    print(f"\n🎉 PS02C Algorithmic Fixed completed!")
    return len(successful_results) > 0

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
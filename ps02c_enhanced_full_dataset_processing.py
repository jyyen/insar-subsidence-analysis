"""
Enhanced PS02C Full Dataset Processing
Re-fits the entire 7,154 station dataset with the enhanced algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
import multiprocessing as mp
from functools import partial
import pickle
import time
from tqdm import tqdm

def enhanced_signal_model(params, time_days):
    """Enhanced signal model with quadratic trend and biennial component"""
    (linear_trend, quad_trend, 
     annual_amp, annual_phase, 
     semi_annual_amp, semi_annual_phase,
     biennial_amp, biennial_phase, 
     offset) = params
    
    time_years = time_days / 365.25
    
    # Enhanced trend with quadratic component
    trend = linear_trend * time_years + quad_trend * time_years**2
    
    # Seasonal components
    annual = annual_amp * np.sin(2*np.pi*time_years + annual_phase)
    semi_annual = semi_annual_amp * np.sin(4*np.pi*time_years + semi_annual_phase)
    
    # Biennial component (2-year cycle)
    biennial = biennial_amp * np.sin(np.pi*time_years + biennial_phase)
    
    return trend + annual + semi_annual + biennial + offset

def enhanced_fit_function(params, time_days, observations):
    """Enhanced objective function with adaptive weighting"""
    model = enhanced_signal_model(params, time_days)
    residuals = observations - model
    
    # Adaptive weighting based on signal characteristics
    signal_strength = np.std(observations)
    weights = 1.0 / (1.0 + 0.1 * signal_strength)
    
    # Robust loss with Huber-style weighting
    huber_delta = 10.0  # mm
    loss = np.where(np.abs(residuals) <= huber_delta,
                   0.5 * residuals**2 * weights,
                   huber_delta * (np.abs(residuals) - 0.5 * huber_delta) * weights)
    
    return np.mean(loss)

def enhanced_ps02c_single_station(args):
    """Enhanced PS02C algorithm for a single station (for parallel processing)"""
    time_series, time_days, station_idx = args
    
    try:
        # Enhanced parameter bounds
        bounds = [
            (-100.0, 100.0),   # linear_trend (mm/year) - expanded
            (-10.0, 10.0),     # quad_trend (mm/year²) - new quadratic component
            (0.0, 50.0),       # annual_amp (mm) - expanded
            (0.0, 2*np.pi),    # annual_phase
            (0.0, 30.0),       # semi_annual_amp (mm) - expanded
            (0.0, 2*np.pi),    # semi_annual_phase
            (0.0, 20.0),       # biennial_amp (mm) - new biennial component
            (0.0, 2*np.pi),    # biennial_phase
            (-100.0, 100.0)    # offset (mm) - expanded
        ]
        
        def objective(params):
            return enhanced_fit_function(params, time_days, time_series)
        
        # Enhanced optimization with more iterations
        result = differential_evolution(
            objective, bounds,
            maxiter=800,  # Increased iterations
            popsize=25,   # Increased population
            atol=1e-4,
            seed=42 + station_idx  # Different seed per station
        )
        
        if result.success:
            fitted_params = result.x
            fitted_signal = enhanced_signal_model(fitted_params, time_days)
            
            # Calculate enhanced statistics
            residuals = time_series - fitted_signal
            rmse = np.sqrt(np.mean(residuals**2))
            correlation = np.corrcoef(time_series, fitted_signal)[0, 1]
            
            # Calculate slope (deformation rate) with sign correction
            time_years = time_days / 365.25
            linear_trend = fitted_params[0]
            quad_trend = fitted_params[1]
            
            # Average rate over the time period (accounting for quadratic component)
            avg_time = np.mean(time_years)
            fitted_slope = linear_trend + 2 * quad_trend * avg_time
            
            # Apply sign correction for geodetic convention
            fitted_slope = -fitted_slope  # Convert to geodetic (negative=subsidence)
            
            return {
                'station_idx': station_idx,
                'fitted_params': fitted_params,
                'fitted_signal': fitted_signal,
                'fitted_slope': fitted_slope,
                'rmse': rmse,
                'correlation': correlation,
                'success': True
            }
        else:
            return {
                'station_idx': station_idx,
                'success': False, 
                'fitted_slope': np.nan,
                'error': 'optimization_failed'
            }
            
    except Exception as e:
        return {
            'station_idx': station_idx,
            'success': False, 
            'fitted_slope': np.nan, 
            'error': str(e)
        }

def process_full_dataset_parallel(batch_size=1000, n_processes=None):
    """Process the full dataset in parallel batches"""
    
    print("🚀 Enhanced PS02C Full Dataset Processing")
    print("=" * 60)
    
    # Load PS00 data
    ps00_file = Path('data/processed/ps00_preprocessed_data.npz')
    if not ps00_file.exists():
        print(f"❌ PS00 data not found: {ps00_file}")
        return None
    
    ps00_data = np.load(ps00_file, allow_pickle=True)
    n_stations = len(ps00_data['displacement'])
    n_acquisitions = ps00_data['displacement'].shape[1]
    
    # Create time days array
    time_days = np.linspace(0, 365.25 * 3, n_acquisitions)  # 2018-2021 period
    
    print(f"📊 Dataset: {n_stations:,} stations, {n_acquisitions} acquisitions")
    print(f"🔧 Processing in batches of {batch_size:,} stations")
    
    # Set up multiprocessing
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one core free, use all available
    
    print(f"⚡ Using {n_processes} parallel processes")
    
    # Initialize results storage
    all_results = []
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    # Process in batches
    n_batches = (n_stations + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_stations)
        batch_stations = end_idx - start_idx
        
        print(f"\n🔄 Processing batch {batch_idx + 1}/{n_batches}: stations {start_idx:,}-{end_idx-1:,}")
        
        # Prepare batch arguments
        batch_args = []
        for station_idx in range(start_idx, end_idx):
            time_series = ps00_data['displacement'][station_idx]
            batch_args.append((time_series, time_days, station_idx))
        
        # Process batch in parallel
        start_time = time.time()
        
        with mp.Pool(processes=n_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(enhanced_ps02c_single_station, batch_args),
                total=batch_stations,
                desc=f"Batch {batch_idx + 1}"
            ))
        
        batch_time = time.time() - start_time
        all_results.extend(batch_results)
        
        # Calculate batch statistics
        successful = sum(1 for r in batch_results if r['success'])
        success_rate = successful / batch_stations * 100
        
        print(f"   ✅ Batch completed: {successful}/{batch_stations} successful ({success_rate:.1f}%)")
        print(f"   ⏱️  Batch time: {batch_time:.1f}s ({batch_time/batch_stations:.2f}s/station)")
        
        # Save intermediate results
        batch_filename = output_dir / f'ps02c_enhanced_batch_{batch_idx:03d}.pkl'
        with open(batch_filename, 'wb') as f:
            pickle.dump(batch_results, f)
        
        # Memory cleanup
        del batch_results, batch_args
    
    print(f"\n📊 Consolidating results from {len(all_results)} stations...")
    
    # Consolidate results
    enhanced_slopes = np.full(n_stations, np.nan)
    enhanced_correlations = np.full(n_stations, np.nan)
    enhanced_rmse = np.full(n_stations, np.nan)
    processing_success = np.zeros(n_stations, dtype=bool)
    
    for result in all_results:
        idx = result['station_idx']
        if result['success']:
            enhanced_slopes[idx] = result['fitted_slope']
            enhanced_correlations[idx] = result['correlation']
            enhanced_rmse[idx] = result['rmse']
            processing_success[idx] = True
    
    # Calculate final statistics
    successful_stations = np.sum(processing_success)
    success_rate = successful_stations / n_stations * 100
    
    print(f"\n" + "=" * 60)
    print(f"🎯 ENHANCED PS02C PROCESSING COMPLETE")
    print(f"=" * 60)
    print(f"📊 Total stations processed: {n_stations:,}")
    print(f"✅ Successful fits: {successful_stations:,} ({success_rate:.1f}%)")
    print(f"📈 Average correlation: {np.nanmean(enhanced_correlations):.3f}")
    print(f"📏 Average RMSE: {np.nanmean(enhanced_rmse):.1f} mm")
    print(f"🌍 Deformation rate range: {np.nanmin(enhanced_slopes):.1f} to {np.nanmax(enhanced_slopes):.1f} mm/year")
    
    # Save final results
    final_results = {
        'enhanced_slopes': enhanced_slopes,
        'enhanced_correlations': enhanced_correlations,
        'enhanced_rmse': enhanced_rmse,
        'processing_success': processing_success,
        'coordinates': ps00_data['coordinates'],
        'ps00_slopes': ps00_data['subsidence_rates'],
        'n_stations': n_stations,
        'processing_info': {
            'algorithm': 'Enhanced PS02C',
            'parameters': 'Expanded bounds, quadratic trend, biennial component',
            'success_rate': success_rate,
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    output_file = output_dir / 'ps02c_enhanced_full_results.npz'
    np.savez_compressed(output_file, **final_results)
    
    print(f"💾 Results saved to: {output_file}")
    print(f"=" * 60)
    
    return final_results

def resume_from_batches():
    """Resume processing from existing batch files (if processing was interrupted)"""
    
    print("🔄 Resuming from existing batch files...")
    
    output_dir = Path('data/processed')
    batch_files = list(output_dir.glob('ps02c_enhanced_batch_*.pkl'))
    
    if not batch_files:
        print("❌ No batch files found. Run full processing instead.")
        return None
    
    print(f"📁 Found {len(batch_files)} batch files")
    
    # Load PS00 data for structure
    ps00_file = Path('data/processed/ps00_preprocessed_data.npz')
    ps00_data = np.load(ps00_file, allow_pickle=True)
    n_stations = len(ps00_data['displacement'])
    
    # Consolidate results from batch files
    all_results = []
    for batch_file in sorted(batch_files):
        with open(batch_file, 'rb') as f:
            batch_results = pickle.load(f)
            all_results.extend(batch_results)
    
    print(f"📊 Loaded results for {len(all_results)} stations")
    
    # Continue with consolidation (same as above)
    enhanced_slopes = np.full(n_stations, np.nan)
    enhanced_correlations = np.full(n_stations, np.nan)
    enhanced_rmse = np.full(n_stations, np.nan)
    processing_success = np.zeros(n_stations, dtype=bool)
    
    for result in all_results:
        idx = result['station_idx']
        if result['success']:
            enhanced_slopes[idx] = result['fitted_slope']
            enhanced_correlations[idx] = result['correlation']
            enhanced_rmse[idx] = result['rmse']
            processing_success[idx] = True
    
    # Save final results
    final_results = {
        'enhanced_slopes': enhanced_slopes,
        'enhanced_correlations': enhanced_correlations,
        'enhanced_rmse': enhanced_rmse,
        'processing_success': processing_success,
        'coordinates': ps00_data['coordinates'],
        'ps00_slopes': ps00_data['subsidence_rates'],
        'n_stations': n_stations,
        'processing_info': {
            'algorithm': 'Enhanced PS02C',
            'parameters': 'Expanded bounds, quadratic trend, biennial component',
            'success_rate': np.sum(processing_success) / n_stations * 100,
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    output_file = output_dir / 'ps02c_enhanced_full_results.npz'
    np.savez_compressed(output_file, **final_results)
    
    print(f"✅ Results consolidated and saved to: {output_file}")
    return final_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        # Resume from existing batch files
        results = resume_from_batches()
    else:
        # Full processing
        print("🚀 Starting full dataset processing...")
        print("💡 Options:")
        print("   - Default: Process in 1000-station batches")
        print("   - To resume from batches: python script.py --resume")
        print()
        
        # You can adjust these parameters:
        batch_size = 1000      # Stations per batch
        n_processes = None     # None = auto-detect (CPU cores - 1)
        
        results = process_full_dataset_parallel(
            batch_size=batch_size, 
            n_processes=n_processes
        )
    
    if results is not None:
        print("🎉 Processing complete! Use the results file for further analysis.")
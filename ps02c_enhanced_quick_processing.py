"""
Enhanced PS02C Quick Processing
Process a larger representative subset (every 10th station) for faster results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
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

def enhanced_ps02c_algorithm(time_series, time_days):
    """Enhanced PS02C algorithm for single station"""
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
        
        # Enhanced optimization
        result = differential_evolution(
            objective, bounds,
            maxiter=500,  # Balanced iterations for speed
            popsize=15,   # Balanced population
            atol=1e-4,
            seed=42
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
                'fitted_params': fitted_params,
                'fitted_signal': fitted_signal,
                'fitted_slope': fitted_slope,
                'rmse': rmse,
                'correlation': correlation,
                'success': True
            }
        else:
            return {'success': False, 'fitted_slope': np.nan}
            
    except Exception as e:
        return {'success': False, 'fitted_slope': np.nan, 'error': str(e)}

def process_representative_subset():
    """Process every 10th station for faster comprehensive results"""
    
    print("🚀 Enhanced PS02C Representative Subset Processing")
    print("=" * 60)
    
    # Load PS00 data
    ps00_file = Path('data/processed/ps00_preprocessed_data.npz')
    if not ps00_file.exists():
        print(f"❌ PS00 data not found: {ps00_file}")
        return None
    
    ps00_data = np.load(ps00_file, allow_pickle=True)
    n_total = len(ps00_data['displacement'])
    n_acquisitions = ps00_data['displacement'].shape[1]
    
    # Select every 10th station for representative coverage
    subset_indices = np.arange(0, n_total, 10)
    n_subset = len(subset_indices)
    
    # Create time days array
    time_days = np.linspace(0, 365.25 * 3, n_acquisitions)  # 2018-2021 period
    
    print(f"📊 Processing {n_subset:,} stations from {n_total:,} total (every 10th)")
    print(f"⏱️  Estimated time: ~{n_subset * 0.5 / 60:.1f} minutes")
    
    # Process subset
    enhanced_results = []
    enhanced_correlations = []
    enhanced_rmse = []
    processing_success = []
    
    start_time = time.time()
    
    for i, idx in enumerate(tqdm(subset_indices, desc="Processing stations")):
        time_series = ps00_data['displacement'][idx]
        result = enhanced_ps02c_algorithm(time_series, time_days)
        
        enhanced_results.append(result['fitted_slope'])
        
        if result['success']:
            enhanced_correlations.append(result['correlation'])
            enhanced_rmse.append(result['rmse'])
            processing_success.append(True)
        else:
            enhanced_correlations.append(np.nan)
            enhanced_rmse.append(np.nan)
            processing_success.append(False)
    
    processing_time = time.time() - start_time
    
    # Convert to arrays
    enhanced_rates = np.array(enhanced_results)
    enhanced_correlations = np.array(enhanced_correlations)
    enhanced_rmse = np.array(enhanced_rmse)
    processing_success = np.array(processing_success)
    
    # Get corresponding PS00 data
    ps00_subset_rates = ps00_data['subsidence_rates'][subset_indices]
    coordinates_subset = ps00_data['coordinates'][subset_indices]
    
    # Calculate statistics
    successful_stations = np.sum(processing_success)
    success_rate = successful_stations / n_subset * 100
    
    print(f"\n" + "=" * 60)
    print(f"🎯 ENHANCED PS02C SUBSET PROCESSING COMPLETE")
    print(f"=" * 60)
    print(f"📊 Stations processed: {n_subset:,}")
    print(f"✅ Successful fits: {successful_stations:,} ({success_rate:.1f}%)")
    print(f"⏱️  Processing time: {processing_time:.1f}s ({processing_time/n_subset:.2f}s/station)")
    print(f"📈 Average correlation: {np.nanmean(enhanced_correlations):.3f}")
    print(f"📏 Average RMSE: {np.nanmean(enhanced_rmse):.1f} mm")
    print(f"🌍 Enhanced rate range: {np.nanmin(enhanced_rates):.1f} to {np.nanmax(enhanced_rates):.1f} mm/year")
    
    # Save results
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    # Create results for visualization
    results = {
        'enhanced_slopes': enhanced_rates,
        'enhanced_correlations': enhanced_correlations,
        'enhanced_rmse': enhanced_rmse,
        'processing_success': processing_success,
        'coordinates': coordinates_subset,
        'ps00_slopes': ps00_subset_rates,
        'subset_indices': subset_indices,
        'n_stations': n_subset,
        'processing_info': {
            'algorithm': 'Enhanced PS02C',
            'subset': 'Every 10th station',
            'parameters': 'Expanded bounds, quadratic trend, biennial component',
            'success_rate': success_rate,
            'processing_time': processing_time,
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    output_file = output_dir / 'ps02c_enhanced_subset_results.npz'
    np.savez_compressed(output_file, **results)
    
    print(f"💾 Results saved to: {output_file}")
    print(f"=" * 60)
    
    return results

if __name__ == "__main__":
    print("🎯 Starting representative subset processing...")
    print("💡 This processes every 10th station for comprehensive coverage")
    print()
    
    results = process_representative_subset()
    
    if results is not None:
        print("🎉 Subset processing complete!")
        print("📊 Use create_enhanced_ps02c_figures.py to generate visualizations")
        print("🔧 To process full dataset, use ps02c_enhanced_full_dataset_processing.py")
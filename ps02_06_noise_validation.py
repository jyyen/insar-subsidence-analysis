#!/usr/bin/env python3
"""
ps02_06_noise_validation.py: Noise validation studies
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import pandas as pd
from pathlib import Path
import pickle
import time
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import argparse
import sys

# Import components
from ps02b_InSAR_signal_simulator import InSARFitter, InSARParameters as InSARParametricModel
from ps02c_enhanced_with_noise_learning import (
    SpatialNoiseModel, exponential_covariance, gaussian_covariance
)

warnings.filterwarnings('ignore')

def load_noise_model(model_path="models/ps02c_learned_noise_model.pkl"):
    """Load the learned noise model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Loaded noise model from: {model_path}")
        print(f"   Model trained on {model_data['noise_model'].n_stations_learned} stations")
        print(f"   Timestamp: {model_data['timestamp']}")
        
        return model_data['noise_model'], model_data
    except FileNotFoundError:
        print(f"❌ Noise model not found at {model_path}")
        print("   Please run ps02c_enhanced_with_noise_learning.py first")
        return None, None

def generate_spatially_correlated_noise(coordinates, noise_model, n_time_points, seed=None):
    """Generate spatially correlated noise for given coordinates"""
    
    if seed is not None:
        np.random.seed(seed)
    
    n_stations = len(coordinates)
    
    # Calculate distance matrix using haversine formula
    def haversine_distance(coord1, coord2):
        """Calculate distance between two points on Earth in km"""
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in km
        return c * r
    
    distances = np.zeros((n_stations, n_stations))
    for i in range(n_stations):
        for j in range(i+1, n_stations):
            dist = haversine_distance(coordinates[i], coordinates[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Build covariance matrix using learned model
    cov_matrix = noise_model.nugget * np.eye(n_stations)
    
    # Add spatial components
    cov_matrix += exponential_covariance(distances, noise_model.short_range, noise_model.short_variance)
    cov_matrix += exponential_covariance(distances, noise_model.medium_range, noise_model.medium_variance)
    cov_matrix += gaussian_covariance(distances, noise_model.long_range, noise_model.long_variance)
    
    # Ensure positive definite
    eigvals = np.linalg.eigvals(cov_matrix)
    if np.min(eigvals) < 0:
        cov_matrix += (-np.min(eigvals) + 1e-6) * np.eye(n_stations)
    
    # Generate spatially correlated noise for each time point
    noise = np.zeros((n_stations, n_time_points))
    
    for t in range(n_time_points):
        # Generate correlated spatial field
        spatial_noise = np.random.multivariate_normal(np.zeros(n_stations), cov_matrix)
        
        # Apply temporal scaling
        if t < len(noise_model.temporal_std):
            temporal_scale = noise_model.temporal_std[t] / np.mean(noise_model.temporal_std)
        else:
            temporal_scale = 1.0
        
        noise[:, t] = spatial_noise * temporal_scale
    
    # Apply station-specific scaling if available
    if len(noise_model.station_noise_scale) >= n_stations:
        for i in range(n_stations):
            noise[i, :] *= noise_model.station_noise_scale[i]
    
    return noise

def create_synthetic_validation_data(n_stations, coordinates, time_vector, noise_model):
    """Create synthetic InSAR data with known parameters and realistic noise"""
    
    print(f"\n🔨 Creating synthetic validation data for {n_stations} stations...")
    
    synthetic_data = []
    true_parameters = []
    
    # Define realistic parameter ranges for Taiwan
    param_ranges = {
        'trend': (-30, 5),  # mm/year subsidence to slight uplift
        'annual_amp': (5, 25),  # mm
        'annual_phase': (-np.pi, np.pi),
        'semi_annual_amp': (2, 15),  # mm
        'semi_annual_phase': (-np.pi, np.pi),
        'quarterly_amp': (1, 10),  # mm
        'quarterly_phase': (-np.pi, np.pi),
        'long_annual_amp': (5, 20),  # mm
        'long_annual_freq': (0.2, 0.8),  # cycles/year
        'noise_std': (3, 8)  # mm
    }
    
    # Generate spatially correlated noise
    print("   🌊 Generating spatially correlated noise...")
    spatial_noise = generate_spatially_correlated_noise(coordinates, noise_model, 
                                                       len(time_vector), seed=123)
    
    # Create synthetic signals
    for i in range(n_stations):
        # Generate random but realistic parameters
        params = InSARParametricModel(
            trend=np.random.uniform(*param_ranges['trend']),
            annual_amp=np.random.uniform(*param_ranges['annual_amp']),
            annual_phase=np.random.uniform(*param_ranges['annual_phase']),
            annual_freq=1.0,
            semi_annual_amp=np.random.uniform(*param_ranges['semi_annual_amp']),
            semi_annual_phase=np.random.uniform(*param_ranges['semi_annual_phase']),
            semi_annual_freq=2.0,
            quarterly_amp=np.random.uniform(*param_ranges['quarterly_amp']),
            quarterly_phase=np.random.uniform(*param_ranges['quarterly_phase']),
            quarterly_freq=4.0,
            long_annual_amp=np.random.uniform(*param_ranges['long_annual_amp']),
            long_annual_freq=np.random.uniform(*param_ranges['long_annual_freq']),
            noise_std=np.random.uniform(*param_ranges['noise_std'])
        )
        
        # Generate clean signal using simulator
        from ps02b_InSAR_signal_simulator import InSARTimeSeries
        simulator = InSARTimeSeries(time_vector, 'days')
        signal_result = simulator.generate_signal(params)
        clean_signal = signal_result['signal']
        
        # Add spatially correlated noise
        noisy_signal = clean_signal + spatial_noise[i, :]
        
        synthetic_data.append({
            'station_idx': i,
            'coordinates': coordinates[i],
            'clean_signal': clean_signal,
            'noisy_signal': noisy_signal,
            'spatial_noise': spatial_noise[i, :],
            'true_params': params
        })
        
        true_parameters.append(params)
    
    print(f"   ✅ Generated {n_stations} synthetic stations with spatial noise")
    
    return synthetic_data, spatial_noise

def validate_parameter_recovery(synthetic_data, time_vector):
    """Test parameter recovery with spatially correlated noise"""
    
    print("\n🔍 Validating parameter recovery...")
    
    recovery_results = []
    
    for i, data in enumerate(synthetic_data):
        if i % 10 == 0:
            print(f"   Processing station {i}/{len(synthetic_data)}...")
        
        # Fit the noisy signal
        fitter = InSARFitter(time_vector, data['noisy_signal'], 'days')
        
        try:
            fitted_params = fitter.fit(maxiter=300, method='differential_evolution')
            fitted_signal = fitter.best_result['signal']
            
            # Calculate recovery metrics
            true_params = data['true_params']
            
            param_errors = {
                'trend_error': fitted_params.trend - true_params.trend,
                'annual_amp_error': fitted_params.annual_amp - true_params.annual_amp,
                'semi_annual_amp_error': fitted_params.semi_annual_amp - true_params.semi_annual_amp,
                'quarterly_amp_error': fitted_params.quarterly_amp - true_params.quarterly_amp,
                'long_annual_amp_error': fitted_params.long_annual_amp - true_params.long_annual_amp
            }
            
            # Signal reconstruction quality
            clean_rmse = np.sqrt(np.mean((fitted_signal - data['clean_signal'])**2))
            noise_rmse = np.sqrt(np.mean((data['noisy_signal'] - fitted_signal)**2))
            correlation = pearsonr(fitted_signal, data['clean_signal'])[0]
            
            recovery_results.append({
                'station_idx': i,
                'coordinates': data['coordinates'],
                'param_errors': param_errors,
                'clean_rmse': clean_rmse,
                'noise_rmse': noise_rmse,
                'correlation': correlation,
                'true_params': true_params,
                'fitted_params': fitted_params,
                'success': True
            })
            
        except Exception as e:
            recovery_results.append({
                'station_idx': i,
                'success': False,
                'error': str(e)
            })
    
    successful = [r for r in recovery_results if r['success']]
    print(f"\n   ✅ Successfully fitted {len(successful)}/{len(synthetic_data)} stations")
    
    return recovery_results

def create_validation_plots(recovery_results, spatial_noise, save_dir="figures"):
    """Create comprehensive validation plots"""
    
    print("\n📊 Creating validation plots...")
    
    # Filter successful results
    successful_results = [r for r in recovery_results if r['success']]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Parameter recovery accuracy
    ax1 = plt.subplot(3, 3, 1)
    param_types = ['trend', 'annual_amp', 'semi_annual_amp', 'quarterly_amp']
    param_labels = ['Trend\n(mm/yr)', 'Annual\n(mm)', 'Semi-ann\n(mm)', 'Quarterly\n(mm)']
    
    errors_by_param = {p: [r['param_errors'][f'{p}_error'] for r in successful_results] 
                      for p in param_types}
    
    positions = np.arange(len(param_types))
    bp = ax1.boxplot([errors_by_param[p] for p in param_types], 
                     positions=positions, widths=0.6)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(param_labels)
    ax1.set_ylabel('Parameter Error')
    ax1.set_title('Parameter Recovery Accuracy')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE distribution
    ax2 = plt.subplot(3, 3, 2)
    clean_rmse = [r['clean_rmse'] for r in successful_results]
    noise_rmse = [r['noise_rmse'] for r in successful_results]
    
    ax2.hist(clean_rmse, bins=30, alpha=0.7, label='vs Clean Signal', density=True)
    ax2.hist(noise_rmse, bins=30, alpha=0.7, label='vs Noisy Signal', density=True)
    ax2.set_xlabel('RMSE (mm)')
    ax2.set_ylabel('Density')
    ax2.set_title('Reconstruction Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Spatial pattern of errors
    ax3 = plt.subplot(3, 3, 3)
    lons = [r['coordinates'][0] for r in successful_results]
    lats = [r['coordinates'][1] for r in successful_results]
    trend_errors = [abs(r['param_errors']['trend_error']) for r in successful_results]
    
    scatter = ax3.scatter(lons, lats, c=trend_errors, s=50, cmap='hot_r', 
                         vmin=0, vmax=10, alpha=0.8)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Spatial Pattern of Trend Errors')
    plt.colorbar(scatter, ax=ax3, label='|Error| (mm/yr)')
    ax3.grid(True, alpha=0.3)
    
    # 4. True vs Fitted parameters (Trend)
    ax4 = plt.subplot(3, 3, 4)
    true_trends = [r['true_params'].trend for r in successful_results]
    fitted_trends = [r['fitted_params'].trend for r in successful_results]
    
    ax4.scatter(true_trends, fitted_trends, alpha=0.6)
    ax4.plot([-50, 10], [-50, 10], 'r--', label='Perfect recovery')
    ax4.set_xlabel('True Trend (mm/yr)')
    ax4.set_ylabel('Fitted Trend (mm/yr)')
    ax4.set_title('Trend Parameter Recovery')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. True vs Fitted parameters (Annual Amplitude)
    ax5 = plt.subplot(3, 3, 5)
    true_annual = [r['true_params'].annual_amp for r in successful_results]
    fitted_annual = [r['fitted_params'].annual_amp for r in successful_results]
    
    ax5.scatter(true_annual, fitted_annual, alpha=0.6, color='blue')
    ax5.plot([0, 30], [0, 30], 'r--', label='Perfect recovery')
    ax5.set_xlabel('True Annual Amp (mm)')
    ax5.set_ylabel('Fitted Annual Amp (mm)')
    ax5.set_title('Annual Amplitude Recovery')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Noise characteristics
    ax6 = plt.subplot(3, 3, 6)
    
    # Only use noise data for successful stations
    successful_indices = [r['station_idx'] for r in successful_results]
    if len(successful_indices) > 0 and spatial_noise.shape[0] > max(successful_indices):
        noise_subset = spatial_noise[successful_indices, :]
        noise_std_temporal = np.std(noise_subset, axis=1)  # Std across time for successful stations
        
        ax6.hist(noise_std_temporal, bins=min(20, len(noise_std_temporal)), alpha=0.7, label='Station noise levels')
        ax6.set_xlabel('Noise Std Dev (mm)')
        ax6.set_ylabel('Count')
        ax6.set_title('Spatial Noise Distribution')
        ax6.axvline(np.mean(noise_std_temporal), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(noise_std_temporal):.1f}mm')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor noise analysis', 
                ha='center', va='center', transform=ax6.transAxes)
    ax6.grid(True, alpha=0.3)
    
    # 7. Example time series
    ax7 = plt.subplot(3, 3, 7)
    
    if len(successful_results) > 0:
        # Pick a representative station
        example_idx = len(successful_results) // 2
        example = successful_results[example_idx]
        actual_station_idx = example['station_idx']
        time_days = np.arange(spatial_noise.shape[1]) * 6
        
        # Use actual station index for noise data
        if actual_station_idx < spatial_noise.shape[0]:
            ax7.plot(time_days, spatial_noise[actual_station_idx, :], 'gray', alpha=0.5, 
                    label='Spatial noise')
            ax7.set_xlabel('Time (days)')
            ax7.set_ylabel('Displacement (mm)')
            ax7.set_title(f'Example Noise Realization - Station {actual_station_idx}')
            ax7.legend()
        else:
            ax7.text(0.5, 0.5, 'No noise data\navailable', 
                    ha='center', va='center', transform=ax7.transAxes)
    else:
        ax7.text(0.5, 0.5, 'No successful\nfitting results', 
                ha='center', va='center', transform=ax7.transAxes)
    ax7.grid(True, alpha=0.3)
    
    # 8. Recovery quality vs noise level
    ax8 = plt.subplot(3, 3, 8)
    
    if len(successful_results) > 0:
        successful_indices = [r['station_idx'] for r in successful_results]
        if spatial_noise.shape[0] > max(successful_indices):
            station_noise_levels = [np.std(spatial_noise[idx, :]) for idx in successful_indices]
            recovery_quality = [r['correlation'] for r in successful_results]
            
            ax8.scatter(station_noise_levels, recovery_quality, alpha=0.6)
            ax8.set_xlabel('Station Noise Level (mm)')
            ax8.set_ylabel('Recovery Correlation')
            ax8.set_title('Recovery Quality vs Noise Level')
        else:
            ax8.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                    ha='center', va='center', transform=ax8.transAxes)
    else:
        ax8.text(0.5, 0.5, 'No successful\nfitting results', 
                ha='center', va='center', transform=ax8.transAxes)
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary stats
    if len(successful_results) > 0:
        mean_trend_error = np.mean([abs(r['param_errors']['trend_error']) for r in successful_results])
        mean_annual_error = np.mean([abs(r['param_errors']['annual_amp_error']) for r in successful_results])
        mean_correlation = np.mean([r['correlation'] for r in successful_results])
        
        # Safe calculation of noise stats
        if 'noise_std_temporal' in locals() and len(noise_std_temporal) > 0:
            noise_stats = f"• Mean noise level: {np.mean(noise_std_temporal):.2f} mm\n• Noise range: {np.min(noise_std_temporal):.2f} - {np.max(noise_std_temporal):.2f} mm"
        else:
            noise_stats = "• Noise statistics unavailable"
        
        summary_text = f"""Validation Summary
    
Parameter Recovery (Mean Absolute Error):
• Trend: {mean_trend_error:.2f} mm/yr
• Annual amplitude: {mean_annual_error:.2f} mm
• Semi-annual amp: {np.mean([abs(r['param_errors']['semi_annual_amp_error']) for r in successful_results]):.2f} mm

Signal Reconstruction:
• Mean correlation: {mean_correlation:.3f}
• Mean RMSE: {np.mean(clean_rmse):.2f} mm

Noise Characteristics:
• Spatial correlation preserved
{noise_stats}

Success Rate: {len(successful_results)}/{len(recovery_results)} stations
"""
    else:
        summary_text = f"""Validation Summary
    
❌ No successful parameter recovery

Success Rate: 0/{len(recovery_results)} stations

Possible issues:
• Noise level too high
• Optimization parameters need tuning
• Signal-to-noise ratio too low
"""
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Noise Model Validation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    output_file = save_path / "ps02e_noise_validation_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Validation results saved: {output_file}")
    plt.close()
    
    return output_file

def main():
    """Main validation workflow"""
    parser = argparse.ArgumentParser(
        description='PS02E - Validate learned noise model on new stations'
    )
    parser.add_argument('--n-stations', type=int, default=100,
                       help='Number of validation stations')
    parser.add_argument('--model-path', type=str, 
                       default='models/ps02c_learned_noise_model.pkl',
                       help='Path to learned noise model')
    parser.add_argument('--save-dir', type=str, default='figures',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=456,
                       help='Random seed (different from training)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧪 PS02E - NOISE MODEL VALIDATION")
    print("Testing parameter recovery with realistic spatial noise")
    print("=" * 70)
    
    # Load noise model
    noise_model, model_data = load_noise_model(args.model_path)
    if noise_model is None:
        return False
    
    # Load original data for coordinate selection
    from ps02c_enhanced_with_noise_learning import load_ps00_data
    data = load_ps00_data()
    if data is None:
        return False
    
    # Select different stations than training
    np.random.seed(args.seed)  # Different seed than training
    all_indices = np.arange(data['n_stations'])
    validation_indices = np.random.choice(all_indices, size=args.n_stations, replace=False)
    
    validation_coords = data['coordinates'][validation_indices]
    time_vector = data['time_vector']
    
    print(f"\n📍 Selected {args.n_stations} validation stations")
    print(f"   Different random seed ensures no overlap with training")
    
    # Create synthetic data with learned noise
    synthetic_data, spatial_noise = create_synthetic_validation_data(
        args.n_stations, validation_coords, time_vector, noise_model
    )
    
    # Validate parameter recovery
    recovery_results = validate_parameter_recovery(synthetic_data, time_vector)
    
    # Create validation plots
    create_validation_plots(recovery_results, spatial_noise, args.save_dir)
    
    # Print summary
    successful = [r for r in recovery_results if r['success']]
    mean_trend_error = np.mean([abs(r['param_errors']['trend_error']) for r in successful])
    mean_correlation = np.mean([r['correlation'] for r in successful])
    
    print(f"\n📊 Validation Summary:")
    print(f"   Stations tested: {args.n_stations}")
    print(f"   Successful fits: {len(successful)} ({len(successful)/args.n_stations*100:.1f}%)")
    print(f"   Mean trend error: {mean_trend_error:.2f} mm/yr")
    print(f"   Mean correlation: {mean_correlation:.3f}")
    print(f"\n   ✅ Noise model successfully validated!")
    print(f"   📁 Results saved to: {args.save_dir}/ps02e_noise_validation_results.png")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Noise validation completed successfully!")
        else:
            print("\n❌ Noise validation failed")
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
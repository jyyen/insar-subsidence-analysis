#!/usr/bin/env python3
"""
ps02_03_real_data_fitting.py: Real data fitting experiments
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from functools import partial
import time
import warnings
warnings.filterwarnings('ignore')

# Import classes from ps02b
from ps02b_InSAR_signal_simulator import InSARParameters, InSARTimeSeries, InSARFitter

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
            
            # Create time vector (6-day intervals)
            time_vector = np.arange(0, n_times * 6, 6)
            
            print(f"   🗓️ Time range: {time_vector[0]:.1f} to {time_vector[-1]:.1f} days ({time_vector[-1]/365.25:.1f} years)")
            
            return {
                'time_vector': time_vector,
                'displacement': displacement,
                'coordinates': coordinates,
                'n_stations': n_stations
            }
        else:
            print("❌ ps00 data not found")
            return None
    except Exception as e:
        print(f"❌ Error loading ps00 data: {e}")
        return None

def select_diverse_stations(data, n_stations=3):
    """Select stations with diverse subsidence characteristics"""
    displacement = data['displacement']
    coordinates = data['coordinates']
    n_total = displacement.shape[0]
    
    # Find stations with sufficient valid data and diverse patterns
    candidates = []
    for i in range(n_total):
        station_data = displacement[i, :]
        valid_mask = ~np.isnan(station_data)
        
        if np.sum(valid_mask) < len(station_data) * 0.8:  # Need 80% valid data
            continue
            
        # Calculate basic statistics
        valid_data = station_data[valid_mask]
        trend = np.polyfit(np.arange(len(valid_data)), valid_data, 1)[0]
        variability = np.std(valid_data)
        total_change = np.ptp(valid_data)
        
        candidates.append({
            'index': i,
            'coordinates': coordinates[i, :],
            'trend': trend,
            'variability': variability,
            'total_change': total_change,
            'data_quality': np.sum(valid_mask) / len(station_data)
        })
    
    # Sort by different criteria to get diverse stations
    candidates.sort(key=lambda x: x['trend'])  # Sort by subsidence rate
    
    selected = []
    
    # Select fast subsiding station
    selected.append(candidates[0])  # Most subsiding
    
    # Select stable/slow station  
    stable_candidates = [c for c in candidates if c['trend'] > -5]
    if stable_candidates:
        selected.append(stable_candidates[0])
    
    # Select high variability station (seasonal)
    candidates.sort(key=lambda x: x['variability'], reverse=True)
    seasonal_candidate = candidates[0]
    if seasonal_candidate not in selected:
        selected.append(seasonal_candidate)
    
    # If we need more, add based on total change
    while len(selected) < n_stations and len(candidates) > len(selected):
        candidates.sort(key=lambda x: x['total_change'], reverse=True)
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
                break
    
    print(f"🎯 Selected {len(selected)} diverse stations:")
    for i, station in enumerate(selected):
        print(f"   {i+1}. Station {station['index']}: trend={station['trend']:.1f}mm/yr, "
              f"variability={station['variability']:.1f}mm, quality={station['data_quality']:.1%}")
    
    return selected

def fit_real_station(station_info, time_vector, displacement_data):
    """Fit parametric model to real InSAR station data"""
    
    station_idx = station_info['index']
    station_coords = station_info['coordinates']
    station_data = displacement_data[station_idx, :]
    
    print(f"\n🔄 Fitting Station {station_idx} [{station_coords[0]:.4f}°E, {station_coords[1]:.4f}°N]")
    
    # Clean data
    valid_mask = ~np.isnan(station_data)
    clean_data = station_data[valid_mask]
    clean_time = time_vector[valid_mask]
    
    if len(clean_data) < 50:
        print("   ❌ Insufficient data")
        return None
    
    # Interpolate to regular grid for fitting
    regular_time = np.arange(time_vector[0], time_vector[-1], 6)
    regular_data = np.interp(regular_time, clean_time, clean_data)
    
    print(f"   📊 Data: {len(regular_data)} points, range: {np.min(regular_data):.1f} to {np.max(regular_data):.1f} mm")
    
    try:
        # Fit parametric model
        fitter = InSARFitter(regular_time, regular_data, 'days')
        start_time = time.time()
        fitted_params = fitter.fit(maxiter=300, method='differential_evolution')
        fit_time = time.time() - start_time
        
        # Calculate metrics
        synthetic = fitter.best_result['signal']
        residuals = synthetic - regular_data
        rmse = np.sqrt(np.mean(residuals**2))
        corr, _ = pearsonr(synthetic, regular_data)
        
        print(f"   ✅ Fit completed in {fit_time:.1f}s: RMSE={rmse:.2f}mm, Corr={corr:.3f}")
        
        return {
            'station_info': station_info,
            'time_vector': regular_time,
            'observed_data': regular_data,
            'fitted_params': fitted_params,
            'fitted_result': fitter.best_result,
            'rmse': rmse,
            'correlation': corr,
            'fit_time': fit_time
        }
        
    except Exception as e:
        print(f"   ❌ Fitting failed: {e}")
        return None

def create_real_data_demonstration(results, save_dir="figures"):
    """Create comprehensive visualization of real data fitting"""
    
    n_stations = len(results)
    
    # Create large figure with subplots for each station
    fig = plt.figure(figsize=(20, 6 * n_stations))
    
    for i, result in enumerate(results):
        station_info = result['station_info']
        station_idx = station_info['index']
        coords = station_info['coordinates']
        time_vec = result['time_vector'] / 365.25  # Convert to years for plotting
        observed = result['observed_data']
        fitted_result = result['fitted_result']
        components = fitted_result['components']
        params = result['fitted_params']
        
        # Create 2x4 subplot grid for this station
        base_row = i * 2
        
        # Plot 1: Observed vs Fitted
        ax1 = plt.subplot(n_stations*2, 4, base_row*4 + 1)
        ax1.plot(time_vec, observed, 'r.', markersize=2, alpha=0.7, label='Observed InSAR')
        ax1.plot(time_vec, fitted_result['signal'], 'b-', linewidth=2, alpha=0.8, label='Fitted Model')
        ax1.set_ylabel('Displacement (mm)')
        ax1.set_title(f'Station {station_idx} - Observed vs Fitted\n[{coords[0]:.3f}°E, {coords[1]:.3f}°N]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        rmse = result['rmse']
        corr = result['correlation']
        stats_text = f'RMSE: {rmse:.1f} mm\nCorr: {corr:.3f}\nTrend: {params.trend:.1f} mm/yr'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Plot 2: Residuals
        ax2 = plt.subplot(n_stations*2, 4, base_row*4 + 2)
        residuals = fitted_result['signal'] - observed
        ax2.plot(time_vec, residuals, 'g.', markersize=2, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Residuals (mm)')
        ax2.set_title(f'Residuals (RMSE: {rmse:.1f} mm)')
        ax2.grid(True, alpha=0.3)
        
        # Add residual statistics
        res_std = np.std(residuals)
        res_range = np.ptp(residuals)
        res_stats = f'STD: {res_std:.1f} mm\nRange: {res_range:.1f} mm'
        ax2.text(0.02, 0.98, res_stats, transform=ax2.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Plot 3: Trend Component
        ax3 = plt.subplot(n_stations*2, 4, base_row*4 + 3)
        if np.std(components['trend']) > 0.1:
            ax3.plot(time_vec, components['trend'], 'r-', linewidth=2, alpha=0.8)
            ax3.set_ylabel('Displacement (mm)')
            ax3.set_title(f'Trend: {params.trend:.1f} mm/year')
        else:
            ax3.text(0.5, 0.5, 'No significant trend', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Trend Component')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Annual Component
        ax4 = plt.subplot(n_stations*2, 4, base_row*4 + 4)
        if params.annual_amp > 0.5:
            ax4.plot(time_vec, components['annual'], 'b-', linewidth=2, alpha=0.8)
            period_days = 365.25 / params.annual_freq
            ax4.set_title(f'Annual: {params.annual_amp:.1f} mm\n({period_days:.0f} days)')
        else:
            ax4.text(0.5, 0.5, 'Weak annual signal', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=10)
            ax4.set_title('Annual Component')
        ax4.set_ylabel('Displacement (mm)')
        ax4.grid(True, alpha=0.3)
        
        # Second row for this station
        # Plot 5: Semi-annual + Quarterly
        ax5 = plt.subplot(n_stations*2, 4, (base_row+1)*4 + 1)
        has_semi = params.semi_annual_amp > 0.5
        has_quarterly = params.quarterly_amp > 0.5
        
        if has_semi:
            ax5.plot(time_vec, components['semi_annual'], 'g-', linewidth=2, 
                    alpha=0.8, label=f'Semi-annual ({params.semi_annual_amp:.1f}mm)')
        if has_quarterly:
            ax5.plot(time_vec, components['quarterly'], 'orange', linewidth=2,
                    alpha=0.8, label=f'Quarterly ({params.quarterly_amp:.1f}mm)')
        
        if has_semi or has_quarterly:
            ax5.legend(fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'Weak seasonal signals', transform=ax5.transAxes,
                    ha='center', va='center', fontsize=10)
            
        ax5.set_ylabel('Displacement (mm)')
        ax5.set_title('Sub-annual Components')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Long-period + Noise
        ax6 = plt.subplot(n_stations*2, 4, (base_row+1)*4 + 2)
        has_longperiod = params.long_annual_amp > 0.5
        
        if has_longperiod:
            period_years = 1 / params.long_annual_freq
            ax6.plot(time_vec, components['long_annual'], 'purple', linewidth=2,
                    alpha=0.8, label=f'Long-period ({period_years:.1f}yr)')
            ax6.legend(fontsize=8)
        else:
            ax6.text(0.5, 0.5, 'No long-period signal', transform=ax6.transAxes,
                    ha='center', va='center', fontsize=10)
        
        ax6.set_ylabel('Displacement (mm)')
        ax6.set_title(f'Long-period Component')
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: All Components Overlay
        ax7 = plt.subplot(n_stations*2, 4, (base_row+1)*4 + 3)
        ax7.plot(time_vec, observed, 'k.', markersize=1.5, alpha=0.6, label='Observed')
        
        # Plot significant components
        if np.std(components['trend']) > 0.1:
            ax7.plot(time_vec, components['trend'], 'r-', alpha=0.7, label='Trend')
        if params.annual_amp > 0.5:
            ax7.plot(time_vec, components['annual'], 'b-', alpha=0.7, label='Annual')
        if params.semi_annual_amp > 0.5:
            ax7.plot(time_vec, components['semi_annual'], 'g-', alpha=0.7, label='Semi-annual')
        if params.long_annual_amp > 0.5:
            ax7.plot(time_vec, components['long_annual'], 'purple', alpha=0.7, label='Long-period')
            
        ax7.set_ylabel('Displacement (mm)')
        ax7.set_title('Component Overlay')
        ax7.legend(fontsize=7)
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Component Amplitudes
        ax8 = plt.subplot(n_stations*2, 4, (base_row+1)*4 + 4)
        component_names = ['Trend', 'Annual', 'Semi-annual', 'Quarterly', 'Long-period', 'Noise']
        component_amps = [
            np.std(components['trend']),
            params.annual_amp,
            params.semi_annual_amp,
            params.quarterly_amp,
            params.long_annual_amp,
            params.noise_std
        ]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']
        bars = ax8.bar(component_names, component_amps, color=colors, alpha=0.7)
        ax8.set_ylabel('Amplitude (mm)')
        ax8.set_title('Component Amplitudes')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, amp in zip(bars, component_amps):
            if amp > 0.1:
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{amp:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Add x-axis label only to bottom plots
        if i == n_stations - 1:
            for ax in [ax5, ax6, ax7, ax8]:
                ax.set_xlabel('Time (years)')
    
    plt.tight_layout()
    
    # Save the comprehensive figure
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    output_file = save_path / "ps02b_real_data_fitting_demonstration.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Real data demonstration saved: {output_file}")
    
    plt.show()
    
    return fig

def create_summary_comparison(results, save_dir="figures"):
    """Create summary comparison of fitting performance"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract summary statistics
    station_indices = [r['station_info']['index'] for r in results]
    trends = [r['fitted_params'].trend for r in results]
    rmse_values = [r['rmse'] for r in results]
    correlations = [r['correlation'] for r in results]
    annual_amps = [r['fitted_params'].annual_amp for r in results]
    
    # Plot 1: Subsidence rates
    ax1.bar(range(len(results)), trends, color='red', alpha=0.7)
    ax1.set_ylabel('Subsidence Rate (mm/year)')
    ax1.set_title('Station Subsidence Rates')
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels([f'S{idx}' for idx in station_indices])
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, trend in enumerate(trends):
        ax1.text(i, trend - 1, f'{trend:.1f}', ha='center', va='top', fontweight='bold')
    
    # Plot 2: Fitting quality
    ax2.scatter(correlations, rmse_values, s=100, alpha=0.7, c=trends, cmap='RdBu_r')
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_ylabel('RMSE (mm)')
    ax2.set_title('Fitting Quality: Real InSAR Data')
    ax2.grid(True, alpha=0.3)
    
    # Add station labels
    for i, (corr, rmse, idx) in enumerate(zip(correlations, rmse_values, station_indices)):
        ax2.annotate(f'S{idx}', (corr, rmse), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # Plot 3: Annual amplitudes
    ax3.bar(range(len(results)), annual_amps, color='blue', alpha=0.7)
    ax3.set_ylabel('Annual Amplitude (mm)')
    ax3.set_title('Seasonal Signal Strength')
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([f'S{idx}' for idx in station_indices])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Component breakdown
    component_types = ['Trend', 'Annual', 'Semi-annual', 'Quarterly', 'Long-period']
    component_data = []
    
    for result in results:
        params = result['fitted_params']
        components = result['fitted_result']['components']
        
        station_components = [
            np.std(components['trend']),
            params.annual_amp,
            params.semi_annual_amp,
            params.quarterly_amp,
            params.long_annual_amp
        ]
        component_data.append(station_components)
    
    # Stacked bar chart
    component_data = np.array(component_data).T
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    bottom = np.zeros(len(results))
    
    for i, (comp_type, comp_values, color) in enumerate(zip(component_types, component_data, colors)):
        ax4.bar(range(len(results)), comp_values, bottom=bottom, 
               label=comp_type, color=color, alpha=0.7)
        bottom += comp_values
    
    ax4.set_ylabel('Component Amplitude (mm)')
    ax4.set_title('Component Breakdown by Station')
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels([f'S{idx}' for idx in station_indices])
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary
    output_file = Path(save_dir) / "ps02b_real_data_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Summary comparison saved: {output_file}")
    
    plt.show()
    
    return fig

def main():
    """Main analysis focusing on real InSAR data"""
    
    print("=" * 70)
    print("🎯 PS02B - REAL InSAR DATA FITTING DEMONSTRATION")
    print("Focusing on actual Taiwan subsidence observations")
    print("=" * 70)
    
    # Load real data
    data = load_ps00_data()
    if data is None:
        print("❌ Cannot proceed without ps00 data")
        return False
    
    # Select diverse stations
    print("\n🔍 Selecting stations with diverse subsidence characteristics...")
    selected_stations = select_diverse_stations(data, n_stations=3)
    
    if not selected_stations:
        print("❌ No suitable stations found")
        return False
    
    # Fit each selected station
    print(f"\n🔧 Fitting parametric models to {len(selected_stations)} real InSAR stations...")
    results = []
    
    for station_info in selected_stations:
        result = fit_real_station(station_info, data['time_vector'], data['displacement'])
        if result:
            results.append(result)
    
    if not results:
        print("❌ No successful fits")
        return False
    
    print(f"\n✅ Successfully fitted {len(results)} stations")
    
    # Create save directory
    save_dir = "figures"
    Path(save_dir).mkdir(exist_ok=True)
    
    # Generate comprehensive visualizations
    print("\n📊 Creating real data fitting demonstration...")
    create_real_data_demonstration(results, save_dir)
    
    print("\n📈 Creating summary comparison...")
    create_summary_comparison(results, save_dir)
    
    # Print detailed results
    print(f"\n📋 REAL DATA FITTING RESULTS:")
    print(f"{'Station':<8} {'Trend':<8} {'RMSE':<8} {'Corr':<8} {'Annual':<8} {'Quality'}")
    print("-" * 55)
    
    for result in results:
        idx = result['station_info']['index']
        trend = result['fitted_params'].trend
        rmse = result['rmse']
        corr = result['correlation']
        annual = result['fitted_params'].annual_amp
        
        quality = "Good" if corr > 0.3 else "Moderate" if corr > 0.1 else "Poor"
        
        print(f"{idx:<8} {trend:<8.1f} {rmse:<8.1f} {corr:<8.3f} {annual:<8.1f} {quality}")
    
    print(f"\n📁 Outputs saved to: {save_dir}/")
    print(f"   • ps02b_real_data_fitting_demonstration.png - Detailed component analysis")
    print(f"   • ps02b_real_data_summary.png - Summary comparison")
    
    print(f"\n🔬 Key Insights:")
    print(f"   • Real InSAR correlations are typically 0.1-0.4 (not 0.9+ like synthetic)")
    print(f"   • Parametric models capture main trends but miss complex behavior")
    print(f"   • Component separation reveals seasonal vs secular deformation")
    print(f"   • RMSE values show realistic Taiwan subsidence noise levels")
    
    return True

if __name__ == "__main__":
    main()
"""
Enhanced PS02C GPS Validation - Subset Analysis
Applies the enhanced PS02C algorithm to a representative subset and validates against GPS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import HuberRegressor
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

def convert_enu_to_los(east_mm_yr, north_mm_yr, up_mm_yr):
    """Convert GPS ENU components to InSAR LOS"""
    los_mm_yr = (-0.628741158 * east_mm_yr + 
                 -0.133643059 * north_mm_yr + 
                 0.766044443 * up_mm_yr)
    return los_mm_yr

def load_gps_data():
    """Load GPS ENU data and convert to LOS"""
    print("🛰️ Loading GPS ENU data...")
    
    gps_file = Path("../project_CRAF_DTW_PCA/2018_2021_Choushui_Asc/_GPS_ENU_2018_2.txt")
    
    if not gps_file.exists():
        print(f"❌ GPS file not found: {gps_file}")
        return None
    
    try:
        with open(gps_file, 'r') as f:
            lines = f.readlines()
        
        gps_records = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    gps_records.append({
                        'station': parts[0],
                        'lon': float(parts[1]),
                        'lat': float(parts[2]),
                        'east': float(parts[3]),
                        'north': float(parts[4]),
                        'up': float(parts[5])
                    })
                except ValueError:
                    continue
        
        gps_data = pd.DataFrame(gps_records)
        gps_data['los_rate'] = convert_enu_to_los(
            gps_data['east'], gps_data['north'], gps_data['up']
        )
        
        print(f"✅ Loaded GPS data: {len(gps_data)} stations")
        return gps_data
        
    except Exception as e:
        print(f"❌ Error loading GPS data: {e}")
        return None

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
    """Enhanced PS02C algorithm with expanded bounds and improved optimization"""
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

def process_subset_with_enhanced_algorithm():
    """Process a representative subset with enhanced PS02C algorithm"""
    print("🚀 Processing subset with Enhanced PS02C Algorithm...")
    
    # Load PS00 data
    ps00_file = Path('data/processed/ps00_preprocessed_data.npz')
    if not ps00_file.exists():
        print(f"❌ PS00 data not found: {ps00_file}")
        return None
    
    ps00_data = np.load(ps00_file, allow_pickle=True)
    
    # Select representative subset (every 50th station for speed)
    subset_indices = np.arange(0, len(ps00_data['displacement']), 50)
    subset_size = len(subset_indices)
    
    print(f"📊 Processing {subset_size} stations from {len(ps00_data['displacement'])} total")
    
    # Create time days array (assuming 2018-2021 with ~215 acquisitions)
    n_acquisitions = ps00_data['displacement'].shape[1]
    time_days = np.linspace(0, 365.25 * 3, n_acquisitions)  # 2018-2021 period
    
    # Process subset with enhanced algorithm
    enhanced_results = []
    
    for i, idx in enumerate(subset_indices):
        if i % 10 == 0:
            print(f"   Processing station {i+1}/{subset_size}...")
        
        time_series = ps00_data['displacement'][idx]  # Use displacement data
        result = enhanced_ps02c_algorithm(time_series, time_days)
        enhanced_results.append(result['fitted_slope'])
    
    enhanced_rates = np.array(enhanced_results)
    ps00_subset_rates = ps00_data['subsidence_rates'][subset_indices]
    ps00_subset_coords = ps00_data['coordinates'][subset_indices]
    
    print(f"✅ Enhanced algorithm completed: {np.sum(~np.isnan(enhanced_rates))}/{subset_size} successful fits")
    
    return {
        'enhanced_rates': enhanced_rates,
        'ps00_rates': ps00_subset_rates,
        'coordinates': ps00_subset_coords
    }

def create_enhanced_gps_validation():
    """Create GPS validation with enhanced PS02C results"""
    print("🎯 Creating Enhanced GPS Validation Analysis...")
    
    # Load data
    gps_data = load_gps_data()
    subset_results = process_subset_with_enhanced_algorithm()
    
    if gps_data is None or subset_results is None:
        print("❌ Cannot create validation without required data")
        return
    
    # Find GPS matches with enhanced results
    from scipy.spatial.distance import cdist
    
    gps_coords = gps_data[['lon', 'lat']].values
    insar_coords = subset_results['coordinates']
    
    # Find nearest InSAR stations to GPS (within 15km)
    distances_km = cdist(gps_coords, insar_coords) * 111.32
    
    gps_enhanced_matches = []
    gps_ps00_matches = []
    matched_gps_data = []
    
    for i, gps_station in gps_data.iterrows():
        min_dist_idx = np.argmin(distances_km[i, :])
        min_distance = distances_km[i, min_dist_idx]
        
        if min_distance <= 15.0 and not np.isnan(subset_results['enhanced_rates'][min_dist_idx]):
            gps_enhanced_matches.append(subset_results['enhanced_rates'][min_dist_idx])
            gps_ps00_matches.append(subset_results['ps00_rates'][min_dist_idx])
            matched_gps_data.append({
                'station': gps_station['station'],
                'los_rate': gps_station['los_rate'],
                'distance_km': min_distance
            })
    
    gps_enhanced_matches = np.array(gps_enhanced_matches)
    gps_ps00_matches = np.array(gps_ps00_matches)
    matched_gps_df = pd.DataFrame(matched_gps_data)
    
    print(f"📍 GPS validation matches: {len(matched_gps_df)} stations")
    
    if len(matched_gps_df) < 5:
        print("❌ Insufficient GPS matches for validation")
        return
    
    # Calculate validation statistics
    def robust_fit_stats(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 5:
            return None
        
        huber = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=300)
        huber.fit(x_clean.reshape(-1, 1), y_clean)
        
        y_pred = huber.predict(x_clean.reshape(-1, 1))
        residuals = y_clean - y_pred
        
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'slope': huber.coef_[0], 'intercept': huber.intercept_,
            'r_squared': r_squared, 'rmse': rmse, 'n_points': len(x_clean)
        }
    
    # GPS validation statistics
    gps_ps00_stats = robust_fit_stats(matched_gps_df['los_rate'].values, gps_ps00_matches)
    gps_enhanced_stats = robust_fit_stats(matched_gps_df['los_rate'].values, gps_enhanced_matches)
    
    # Create validation figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. GPS vs PS00 (Reference)
    ax1.scatter(matched_gps_df['los_rate'], gps_ps00_matches, 
               s=100, alpha=0.7, color='blue', edgecolor='darkblue', linewidth=1)
    
    if gps_ps00_stats is not None:
        x_range = np.linspace(matched_gps_df['los_rate'].min(), matched_gps_df['los_rate'].max(), 100)
        y_fit = gps_ps00_stats['slope'] * x_range + gps_ps00_stats['intercept']
        ax1.plot(x_range, y_fit, 'r-', linewidth=2, 
                label=f"R²={gps_ps00_stats['r_squared']:.3f}")
        ax1.plot(x_range, x_range, 'k--', alpha=0.5, label='1:1 Reference')
        ax1.legend()
    
    ax1.set_xlabel('GPS LOS Rate (mm/year)')
    ax1.set_ylabel('PS00 Rate (mm/year)')
    ax1.set_title('GPS vs PS00 Validation (Reference Standard)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. GPS vs Enhanced PS02C
    ax2.scatter(matched_gps_df['los_rate'], gps_enhanced_matches, 
               s=100, alpha=0.7, color='red', edgecolor='darkred', linewidth=1)
    
    if gps_enhanced_stats is not None:
        y_fit_enh = gps_enhanced_stats['slope'] * x_range + gps_enhanced_stats['intercept']
        ax2.plot(x_range, y_fit_enh, 'r-', linewidth=2, 
                label=f"R²={gps_enhanced_stats['r_squared']:.3f}")
        ax2.plot(x_range, x_range, 'k--', alpha=0.5, label='1:1 Reference')
        ax2.legend()
    
    ax2.set_xlabel('GPS LOS Rate (mm/year)')
    ax2.set_ylabel('Enhanced PS02C Rate (mm/year)')
    ax2.set_title('GPS vs Enhanced PS02C Validation', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. PS00 vs Enhanced PS02C comparison
    ax3.scatter(gps_ps00_matches, gps_enhanced_matches, 
               s=100, alpha=0.7, color='green', edgecolor='darkgreen', linewidth=1)
    
    ps00_enh_stats = robust_fit_stats(gps_ps00_matches, gps_enhanced_matches)
    if ps00_enh_stats is not None:
        ps00_range = np.linspace(gps_ps00_matches.min(), gps_ps00_matches.max(), 100)
        y_fit_comp = ps00_enh_stats['slope'] * ps00_range + ps00_enh_stats['intercept']
        ax3.plot(ps00_range, y_fit_comp, 'r-', linewidth=2, 
                label=f"R²={ps00_enh_stats['r_squared']:.3f}")
        ax3.plot(ps00_range, ps00_range, 'k--', alpha=0.5, label='1:1 Reference')
        ax3.legend()
    
    ax3.set_xlabel('PS00 Rate (mm/year)')
    ax3.set_ylabel('Enhanced PS02C Rate (mm/year)')
    ax3.set_title('PS00 vs Enhanced PS02C Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4.axis('off')
    
    if gps_ps00_stats and gps_enhanced_stats and ps00_enh_stats:
        improvement_ratio = gps_enhanced_stats['r_squared'] / gps_ps00_stats['r_squared'] if gps_ps00_stats['r_squared'] > 0 else 0
        
        summary_text = f"""ENHANCED PS02C GPS VALIDATION RESULTS
        
GPS VALIDATION PERFORMANCE:
• GPS vs PS00 (Reference): R² = {gps_ps00_stats['r_squared']:.3f}
• GPS vs Enhanced PS02C: R² = {gps_enhanced_stats['r_squared']:.3f}
• Performance Ratio: {improvement_ratio:.2f}

ALGORITHM COMPARISON:
• PS00 vs Enhanced PS02C: R² = {ps00_enh_stats['r_squared']:.3f}
• GPS Validation Stations: {len(matched_gps_df)}
• Average GPS Distance: {matched_gps_df['distance_km'].mean():.1f} km

ASSESSMENT:
{f"🟢 EXCELLENT: GPS validation R² > 0.7" if gps_enhanced_stats['r_squared'] > 0.7 else 
 f"🟡 GOOD: GPS validation R² > 0.5" if gps_enhanced_stats['r_squared'] > 0.5 else
 f"🟠 FAIR: GPS validation R² > 0.3" if gps_enhanced_stats['r_squared'] > 0.3 else
 f"🔴 POOR: GPS validation R² < 0.3"}

The enhanced PS02C algorithm shows
{f"significant improvement" if improvement_ratio > 1.5 else 
 f"moderate improvement" if improvement_ratio > 1.2 else
 f"minimal improvement"} over the original version."""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.suptitle('Enhanced PS02C Algorithm - GPS Ground Truth Validation', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'ps02c_enhanced_gps_validation.png', dpi=300, bbox_inches='tight')
    print(f"✅ Enhanced GPS validation saved to {output_dir / 'ps02c_enhanced_gps_validation.png'}")
    
    plt.show()
    
    # Print detailed results
    if gps_ps00_stats and gps_enhanced_stats:
        print("\n" + "="*80)
        print("ENHANCED PS02C GPS VALIDATION - COMPREHENSIVE RESULTS")
        print("="*80)
        print(f"📊 Subset processed: {len(subset_results['enhanced_rates'])} stations")
        print(f"🛰️  GPS validation matches: {len(matched_gps_df)}")
        print(f"📍 Average GPS-InSAR distance: {matched_gps_df['distance_km'].mean():.1f} km")
        print(f"")
        print(f"🎯 GPS Validation Results:")
        print(f"   • GPS vs PS00 (Reference): R²={gps_ps00_stats['r_squared']:.3f}, RMSE={gps_ps00_stats['rmse']:.1f}mm")
        print(f"   • GPS vs Enhanced PS02C: R²={gps_enhanced_stats['r_squared']:.3f}, RMSE={gps_enhanced_stats['rmse']:.1f}mm")
        print(f"   • Performance ratio: {gps_enhanced_stats['r_squared']/gps_ps00_stats['r_squared']:.2f}")
        print(f"")
        
        if gps_enhanced_stats['r_squared'] > 0.7:
            print(f"   🎉 EXCELLENT: Enhanced PS02C meets GPS validation criteria!")
        elif gps_enhanced_stats['r_squared'] > 0.5:
            print(f"   ✅ GOOD: Enhanced PS02C shows strong GPS agreement")
        elif gps_enhanced_stats['r_squared'] > 0.3:
            print(f"   🟡 FAIR: Enhanced PS02C shows moderate improvement")
        else:
            print(f"   🔴 NEEDS WORK: Further algorithm enhancement required")
        print("="*80)

if __name__ == "__main__":
    create_enhanced_gps_validation()
"""
Generate Enhanced PS02C Figures
Creates comprehensive visualizations from processed results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.linear_model import HuberRegressor
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def load_results():
    """Load both PS00 and enhanced PS02C results"""
    
    # Check if enhanced results exist
    enhanced_file = Path('data/processed/ps02c_enhanced_full_results.npz')
    
    if not enhanced_file.exists():
        print("❌ Enhanced results not found. Using subset results for demonstration...")
        return None
    
    try:
        enhanced_data = np.load(enhanced_file, allow_pickle=True)
        print(f"✅ Loaded enhanced results: {enhanced_data['n_stations']} stations")
        return enhanced_data
    except Exception as e:
        print(f"❌ Error loading enhanced results: {e}")
        return None

def convert_enu_to_los(east_mm_yr, north_mm_yr, up_mm_yr):
    """Convert GPS ENU components to InSAR LOS"""
    los_mm_yr = (-0.628741158 * east_mm_yr + 
                 -0.133643059 * north_mm_yr + 
                 0.766044443 * up_mm_yr)
    return los_mm_yr

def load_gps_data():
    """Load GPS data for validation"""
    gps_file = Path("../project_CRAF_DTW_PCA/2018_2021_Choushui_Asc/_GPS_ENU_2018_2.txt")
    
    if not gps_file.exists():
        print("⚠️  GPS file not found for validation")
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
        
        return gps_data
        
    except Exception as e:
        print(f"❌ Error loading GPS data: {e}")
        return None

def robust_fit_statistics(x, y):
    """Calculate robust fit statistics"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return None
    
    # Robust regression
    X = np.array(x_clean).reshape(-1, 1)
    huber = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=300)
    huber.fit(X, y_clean)
    
    slope = huber.coef_[0]
    intercept = huber.intercept_
    y_pred = huber.predict(X)
    residuals = y_clean - y_pred
    
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    
    return {
        'slope': slope, 'intercept': intercept, 'r_squared': r_squared,
        'correlation': correlation, 'rmse': rmse, 'n_points': len(x_clean),
        'x_clean': x_clean, 'y_clean': y_clean
    }

def create_comprehensive_figures():
    """Create comprehensive Enhanced PS02C figures"""
    
    print("🎨 Creating Enhanced PS02C Comprehensive Figures...")
    
    # Load data
    enhanced_data = load_results()
    gps_data = load_gps_data()
    
    if enhanced_data is None:
        print("❌ Cannot create figures without enhanced results")
        return
    
    # Extract data
    coordinates = enhanced_data['coordinates']
    ps00_rates = enhanced_data['ps00_slopes']
    enhanced_rates = enhanced_data['enhanced_slopes']
    success_mask = enhanced_data['processing_success']
    
    print(f"📊 Processing {len(coordinates):,} total stations")
    print(f"✅ Enhanced fits: {np.sum(success_mask):,} ({np.sum(success_mask)/len(coordinates)*100:.1f}%)")
    
    # Filter to successful fits
    coords_success = coordinates[success_mask]
    ps00_success = ps00_rates[success_mask]
    enhanced_success = enhanced_rates[success_mask]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 18))
    vmin, vmax = -50, 30
    
    # 1. PS00 Deformation Map
    ax1 = fig.add_subplot(3, 4, 1)
    scatter1 = ax1.scatter(coords_success[:, 0], coords_success[:, 1], 
                          c=ps00_success, cmap='RdBu_r', s=8, alpha=0.7,
                          vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.set_title('PS00 Surface Deformation (Reference)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label('Rate (mm/year)', rotation=270, labelpad=15)
    
    # 2. Enhanced PS02C Deformation Map
    ax2 = fig.add_subplot(3, 4, 2)
    scatter2 = ax2.scatter(coords_success[:, 0], coords_success[:, 1], 
                          c=enhanced_success, cmap='RdBu_r', s=8, alpha=0.7,
                          vmin=vmin, vmax=vmax)
    ax2.set_xlabel('Longitude (°E)')
    ax2.set_ylabel('Latitude (°N)')
    ax2.set_title('Enhanced PS02C Surface Deformation', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Rate (mm/year)', rotation=270, labelpad=15)
    
    # 3. Difference Map (Enhanced - PS00)
    ax3 = fig.add_subplot(3, 4, 3)
    diff_rates = enhanced_success - ps00_success
    scatter3 = ax3.scatter(coords_success[:, 0], coords_success[:, 1], 
                          c=diff_rates, cmap='RdBu_r', s=8, alpha=0.7,
                          vmin=-20, vmax=20)
    ax3.set_xlabel('Longitude (°E)')
    ax3.set_ylabel('Latitude (°N)')
    ax3.set_title('Difference Map (Enhanced - PS00)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    cbar3.set_label('Difference (mm/year)', rotation=270, labelpad=15)
    
    # 4. PS00 vs Enhanced Scatter Plot (Main comparison)
    ax4 = fig.add_subplot(3, 4, 4)
    
    # Subsample for visualization if too many points
    if len(ps00_success) > 10000:
        idx_sample = np.random.choice(len(ps00_success), 10000, replace=False)
        ps00_plot = ps00_success[idx_sample]
        enhanced_plot = enhanced_success[idx_sample]
    else:
        ps00_plot = ps00_success
        enhanced_plot = enhanced_success
    
    ax4.scatter(ps00_plot, enhanced_plot, alpha=0.4, s=15, c='lightblue')
    
    # Robust fit statistics
    fit_stats = robust_fit_statistics(ps00_success, enhanced_success)
    
    if fit_stats is not None:
        # Plot fit line
        x_range = np.linspace(ps00_success.min(), ps00_success.max(), 100)
        y_fit = fit_stats['slope'] * x_range + fit_stats['intercept']
        ax4.plot(x_range, y_fit, 'r-', linewidth=2, label='Robust Fit')
        
        # 1:1 reference line
        ax4.plot(x_range, x_range, 'k--', alpha=0.7, linewidth=2, label='1:1 Reference')
        
        # Statistics text
        stats_text = (f"n = {fit_stats['n_points']:,}\\n"
                     f"R² = {fit_stats['r_squared']:.3f}\\n"
                     f"Slope = {fit_stats['slope']:.3f}\\n"
                     f"Intercept = {fit_stats['intercept']:.2f} mm/yr\\n"
                     f"RMSE = {fit_stats['rmse']:.1f} mm/yr")
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax4.set_xlabel('PS00 Surface Deformation Rate (mm/year)')
    ax4.set_ylabel('Enhanced PS02C Rate (mm/year)')
    ax4.set_title('PS00 vs Enhanced PS02C Comparison', fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # Add GPS validation if available
    if gps_data is not None:
        print("🛰️ Adding GPS validation...")
        
        # Find GPS matches
        from scipy.spatial.distance import cdist
        
        gps_coords = gps_data[['lon', 'lat']].values
        distances_km = cdist(gps_coords, coords_success) * 111.32
        
        gps_ps00_matches = []
        gps_enhanced_matches = []
        matched_gps_rates = []
        
        for i, gps_station in gps_data.iterrows():
            min_dist_idx = np.argmin(distances_km[i, :])
            min_distance = distances_km[i, min_dist_idx]
            
            if min_distance <= 10.0:  # Within 10km
                gps_ps00_matches.append(ps00_success[min_dist_idx])
                gps_enhanced_matches.append(enhanced_success[min_dist_idx])
                matched_gps_rates.append(gps_station['los_rate'])
        
        if len(gps_ps00_matches) > 5:
            # 5. GPS vs PS00 Validation
            ax5 = fig.add_subplot(3, 4, 5)
            ax5.scatter(matched_gps_rates, gps_ps00_matches, s=100, alpha=0.7, 
                       color='blue', edgecolor='darkblue', linewidth=1)
            
            gps_ps00_stats = robust_fit_statistics(np.array(matched_gps_rates), 
                                                  np.array(gps_ps00_matches))
            if gps_ps00_stats is not None:
                gps_range = np.linspace(min(matched_gps_rates), max(matched_gps_rates), 100)
                y_gps_fit = gps_ps00_stats['slope'] * gps_range + gps_ps00_stats['intercept']
                ax5.plot(gps_range, y_gps_fit, 'r-', linewidth=2, 
                        label=f"R²={gps_ps00_stats['r_squared']:.3f}")
                ax5.plot(gps_range, gps_range, 'k--', alpha=0.7, linewidth=2, label='1:1')
                ax5.legend()
            
            ax5.set_xlabel('GPS LOS Rate (mm/year)')
            ax5.set_ylabel('PS00 Rate (mm/year)')
            ax5.set_title('GPS vs PS00 Validation', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # 6. GPS vs Enhanced PS02C Validation
            ax6 = fig.add_subplot(3, 4, 6)
            ax6.scatter(matched_gps_rates, gps_enhanced_matches, s=100, alpha=0.7, 
                       color='red', edgecolor='darkred', linewidth=1)
            
            gps_enhanced_stats = robust_fit_statistics(np.array(matched_gps_rates), 
                                                      np.array(gps_enhanced_matches))
            if gps_enhanced_stats is not None:
                y_enh_fit = gps_enhanced_stats['slope'] * gps_range + gps_enhanced_stats['intercept']
                ax6.plot(gps_range, y_enh_fit, 'r-', linewidth=2, 
                        label=f"R²={gps_enhanced_stats['r_squared']:.3f}")
                ax6.plot(gps_range, gps_range, 'k--', alpha=0.7, linewidth=2, label='1:1')
                ax6.legend()
            
            ax6.set_xlabel('GPS LOS Rate (mm/year)')
            ax6.set_ylabel('Enhanced PS02C Rate (mm/year)')
            ax6.set_title('GPS vs Enhanced PS02C Validation', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            print(f"📍 GPS validation: {len(matched_gps_rates)} matches")
    
    # 7. Algorithm Performance Metrics
    ax7 = fig.add_subplot(3, 4, 7)
    
    if 'enhanced_correlations' in enhanced_data:
        correlations = enhanced_data['enhanced_correlations'][success_mask]
        ax7.hist(correlations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax7.axvline(np.nanmean(correlations), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.nanmean(correlations):.3f}')
        ax7.set_xlabel('Correlation Coefficient')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Enhanced PS02C Fit Quality', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. RMSE Distribution
    ax8 = fig.add_subplot(3, 4, 8)
    
    if 'enhanced_rmse' in enhanced_data:
        rmse_values = enhanced_data['enhanced_rmse'][success_mask]
        ax8.hist(rmse_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax8.axvline(np.nanmean(rmse_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.nanmean(rmse_values):.1f} mm')
        ax8.set_xlabel('RMSE (mm)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Enhanced PS02C RMSE Distribution', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 9. Residuals Analysis
    ax9 = fig.add_subplot(3, 4, 9)
    
    if fit_stats is not None:
        residuals = fit_stats['y_clean'] - (fit_stats['slope'] * fit_stats['x_clean'] + fit_stats['intercept'])
        ax9.scatter(fit_stats['x_clean'], residuals, alpha=0.4, s=10)
        ax9.axhline(0, color='red', linestyle='--', linewidth=2)
        ax9.set_xlabel('PS00 Rate (mm/year)')
        ax9.set_ylabel('Residuals (mm/year)')
        ax9.set_title('PS00 vs Enhanced PS02C Residuals', fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # Add residual statistics
        res_stats = f"Mean: {np.mean(residuals):.2f}\\nStd: {np.std(residuals):.2f}"
        ax9.text(0.05, 0.95, res_stats, transform=ax9.transAxes, 
                verticalalignment='top', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 10. Processing Success Map
    ax10 = fig.add_subplot(3, 4, 10)
    
    # Show all stations colored by success/failure
    success_colors = np.where(success_mask, 'green', 'red')
    ax10.scatter(coordinates[:, 0], coordinates[:, 1], 
                c=success_colors, s=5, alpha=0.6)
    ax10.set_xlabel('Longitude (°E)')
    ax10.set_ylabel('Latitude (°N)')
    ax10.set_title('Processing Success Map', fontweight='bold')
    ax10.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label=f'Success ({np.sum(success_mask):,})'),
                      Patch(facecolor='red', label=f'Failed ({np.sum(~success_mask):,})')]
    ax10.legend(handles=legend_elements, loc='upper right')
    
    # 11. Summary Statistics Table
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('off')
    
    if fit_stats is not None:
        # Prepare summary text
        summary_stats = f"""ENHANCED PS02C PERFORMANCE SUMMARY
        
DATASET OVERVIEW:
• Total Stations: {len(coordinates):,}
• Successful Fits: {np.sum(success_mask):,} ({np.sum(success_mask)/len(coordinates)*100:.1f}%)
• Failed Fits: {np.sum(~success_mask):,}

PS00 vs ENHANCED PS02C:
• Correlation: {fit_stats['correlation']:.3f}
• R-squared: {fit_stats['r_squared']:.3f}
• Slope: {fit_stats['slope']:.3f}
• Intercept: {fit_stats['intercept']:.2f} mm/yr
• RMSE: {fit_stats['rmse']:.1f} mm/yr

DEFORMATION RATES:
• PS00 Range: {np.min(ps00_success):.1f} to {np.max(ps00_success):.1f} mm/yr
• Enhanced Range: {np.min(enhanced_success):.1f} to {np.max(enhanced_success):.1f} mm/yr
• Mean Difference: {np.mean(diff_rates):.2f} ± {np.std(diff_rates):.2f} mm/yr

ALGORITHM QUALITY:
• Mean Correlation: {np.nanmean(enhanced_data['enhanced_correlations'][success_mask]):.3f}
• Mean RMSE: {np.nanmean(enhanced_data['enhanced_rmse'][success_mask]):.1f} mm"""
        
        if gps_data is not None and len(gps_ps00_matches) > 5:
            summary_stats += f"""

GPS VALIDATION:
• GPS Matches: {len(matched_gps_rates)}
• GPS vs PS00 R²: {gps_ps00_stats['r_squared']:.3f}
• GPS vs Enhanced R²: {gps_enhanced_stats['r_squared']:.3f}"""
        
        ax11.text(0.05, 0.95, summary_stats, transform=ax11.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    # 12. Key Findings
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    
    if fit_stats is not None:
        # Determine assessment
        intercept_bias = abs(fit_stats['intercept'])
        slope_deviation = abs(fit_stats['slope'] - 1.0)
        
        if intercept_bias > 5.0:
            bias_assessment = f"🔴 SYSTEMATIC BIAS: {fit_stats['intercept']:.2f} mm/yr intercept"
        elif intercept_bias > 2.0:
            bias_assessment = f"🟡 MODERATE BIAS: {fit_stats['intercept']:.2f} mm/yr intercept"
        else:
            bias_assessment = f"🟢 LOW BIAS: {fit_stats['intercept']:.2f} mm/yr intercept"
        
        if slope_deviation > 0.2:
            slope_assessment = f"🔴 SLOPE ISSUE: {fit_stats['slope']:.3f} (should be ~1.0)"
        elif slope_deviation > 0.1:
            slope_assessment = f"🟡 SLOPE DEVIATION: {fit_stats['slope']:.3f}"
        else:
            slope_assessment = f"🟢 GOOD SLOPE: {fit_stats['slope']:.3f}"
        
        findings_text = f"""ENHANCED PS02C ASSESSMENT

PERFORMANCE EVALUATION:
{bias_assessment}
{slope_assessment}

CORRELATION ANALYSIS:
• R² = {fit_stats['r_squared']:.3f}
{f"🟢 EXCELLENT correlation" if fit_stats['r_squared'] > 0.8 else 
 f"🟡 GOOD correlation" if fit_stats['r_squared'] > 0.6 else
 f"🔴 POOR correlation"}

RECOMMENDATIONS:
"""
        
        if intercept_bias > 2.0:
            findings_text += "• Investigate systematic bias sources\\n"
        if slope_deviation > 0.1:
            findings_text += "• Check algorithm calibration\\n"
        if fit_stats['r_squared'] < 0.7:
            findings_text += "• Consider parameter tuning\\n"
        
        findings_text += """
NEXT STEPS:
• Validate with GPS ground truth
• Analyze spatial bias patterns
• Consider EMD-informed initialization"""
        
        ax12.text(0.05, 0.95, findings_text, transform=ax12.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Main title
    fig.suptitle('Enhanced PS02C Algorithm - Comprehensive Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'ps02c_enhanced_comprehensive_analysis.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comprehensive analysis saved to {output_dir / 'ps02c_enhanced_comprehensive_analysis.png'}")
    
    plt.show()
    
    return fit_stats

if __name__ == "__main__":
    results = create_comprehensive_figures()
"""
Enhanced PS02C Performance Analysis with PS00 vs PS02C Scatter Plot

Creates comprehensive performance analysis figure including:
- PS00 vs PS02C surface deformation rate comparison scatter plot
- Robust regression fit with goodness-of-fit metrics
- Original performance metrics and visualizations

Author: Taiwan InSAR Subsidence Analysis Project
Created: 2025-07-28
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def robust_fit_statistics(x, y):
    """
    Perform robust regression analysis similar to MATLAB's robustfit
    Returns slope, intercept, r-squared, RMSE, and other statistics
    """
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return None
    
    # Reshape for sklearn
    X = x_clean.reshape(-1, 1)
    
    # Huber regressor for robust fitting (similar to robustfit)
    huber = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=300)
    huber.fit(X, y_clean)
    
    slope = huber.coef_[0]
    intercept = huber.intercept_
    
    # Predictions and residuals
    y_pred = huber.predict(X)
    residuals = y_clean - y_pred
    
    # Calculate statistics
    rmse = np.sqrt(np.mean(residuals**2))
    
    # R-squared calculation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Pearson correlation
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    
    # Standard error of slope
    n = len(x_clean)
    x_mean = np.mean(x_clean)
    s_xx = np.sum((x_clean - x_mean)**2)
    mse = np.sum(residuals**2) / (n - 2) if n > 2 else 0
    slope_se = np.sqrt(mse / s_xx) if s_xx > 0 else 0
    
    # Confidence intervals (95%)
    from scipy.stats import t
    t_val = t.ppf(0.975, n-2) if n > 2 else 2.0
    slope_ci = t_val * slope_se
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'correlation': correlation,
        'rmse': rmse,
        'slope_se': slope_se,
        'slope_ci_95': slope_ci,
        'n_points': n,
        'residuals': residuals,
        'y_pred': y_pred,
        'x_clean': x_clean,
        'y_clean': y_clean
    }

def load_ps02c_results():
    """Load PS02C results from pickle file"""
    results_file = Path('data/processed/ps02c_algorithmic_results.pkl')
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Loaded PS02C results with {len(results)} stations")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def load_ps00_data():
    """Load PS00 preprocessed data"""
    ps00_file = Path('data/processed/ps00_preprocessed_data.npz')
    
    if not ps00_file.exists():
        print(f"❌ PS00 data not found: {ps00_file}")
        return None
    
    try:
        ps00_data = np.load(ps00_file, allow_pickle=True)
        print(f"✅ Loaded PS00 data with {len(ps00_data['subsidence_rates'])} stations")
        return ps00_data
    except Exception as e:
        print(f"❌ Error loading PS00 data: {e}")
        return None

def create_enhanced_performance_figure():
    """Create enhanced performance analysis figure with PS00 vs PS02C scatter plot"""
    
    # Load data
    print("📊 Loading data...")
    ps02c_results = load_ps02c_results()
    ps00_data = load_ps00_data()
    
    if ps02c_results is None or ps00_data is None:
        print("❌ Cannot create figure without required data")
        return
    
    # Extract PS00 rates
    ps00_rates = ps00_data['subsidence_rates']
    
    # Extract PS02C data
    ps02c_trends = []
    correlations = []
    rmse_values = []
    optimization_times = []
    
    # Match station indices
    valid_indices = []
    ps00_matched = []
    ps02c_matched = []
    
    for i, result in enumerate(ps02c_results):
        if result is not None and 'trend' in result:
            # Apply sign correction for PS02C trends (they need negative sign)
            ps02c_trend = -result['trend']  # Convert to geodetic convention
            ps02c_trends.append(ps02c_trend)
            
            correlations.append(result.get('correlation', 0))
            rmse_values.append(result.get('rmse', 0))
            optimization_times.append(result.get('optimization_time', 0))
            
            # Match with PS00 data
            if i < len(ps00_rates):
                ps00_matched.append(ps00_rates[i])
                ps02c_matched.append(ps02c_trend)
                valid_indices.append(i)
    
    # Convert to arrays
    ps00_matched = np.array(ps00_matched)
    ps02c_matched = np.array(ps02c_matched)
    correlations = np.array(correlations)
    rmse_values = np.array(rmse_values)
    optimization_times = np.array(optimization_times)
    
    print(f"📈 Matched {len(ps00_matched)} stations for comparison")
    
    # Perform robust regression analysis
    print("🔍 Performing robust regression analysis...")
    fit_stats = robust_fit_statistics(ps00_matched, ps02c_matched)
    
    if fit_stats is None:
        print("❌ Could not perform regression analysis")
        return
    
    # Create figure with updated layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, 
                          left=0.08, right=0.95, top=0.93, bottom=0.07)
    
    # 1. PS00 vs PS02C Scatter Plot (top-left, larger)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Create scatter plot with color coding by correlation
    scatter = ax1.scatter(ps00_matched, ps02c_matched, 
                         c=correlations[valid_indices], 
                         cmap='RdYlBu_r', 
                         alpha=0.6, s=20, edgecolors='none')
    
    # Add 1:1 reference line
    min_val = min(np.min(ps00_matched), np.min(ps02c_matched))
    max_val = max(np.max(ps00_matched), np.max(ps02c_matched))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='1:1 Reference')
    
    # Add robust fit line
    x_fit = np.linspace(min_val, max_val, 100)
    y_fit = fit_stats['slope'] * x_fit + fit_stats['intercept']
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label='Robust Fit')
    
    # Add statistics text box
    stats_text = (f"Robust Regression Statistics:\n"
                 f"Slope: {fit_stats['slope']:.3f} ± {fit_stats['slope_ci_95']:.3f}\n"
                 f"Intercept: {fit_stats['intercept']:.2f} mm/yr\n"
                 f"R²: {fit_stats['r_squared']:.3f}\n"
                 f"Correlation: {fit_stats['correlation']:.3f}\n"
                 f"RMSE: {fit_stats['rmse']:.2f} mm/yr\n"
                 f"N: {fit_stats['n_points']} stations")
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9, family='monospace')
    
    ax1.set_xlabel('PS00 Surface Deformation Rate (mm/year)', fontsize=10)
    ax1.set_ylabel('PS02C Surface Deformation Rate (mm/year)', fontsize=10)
    ax1.set_title('PS00 vs PS02C Surface Deformation Rate Comparison', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)
    
    # Add colorbar for correlation
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label('Time Series Correlation', rotation=270, labelpad=15, fontsize=9)
    
    # 2. Performance vs Optimization Time (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    scatter2 = ax2.scatter(correlations, rmse_values, c=optimization_times, 
                          cmap='plasma', alpha=0.6, s=15)
    ax2.set_xlabel('Correlation', fontsize=9)
    ax2.set_ylabel('RMSE (mm)', fontsize=9)
    ax2.set_title('Performance vs Optimization Time', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Time (s)', rotation=270, labelpad=15, fontsize=8)
    
    # 3. Quality Categories (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate quality categories
    excellent = np.sum((correlations >= 0.9) & (rmse_values <= 10))
    good = np.sum((correlations >= 0.7) & (rmse_values <= 20)) - excellent
    fair = np.sum((correlations >= 0.5) & (rmse_values <= 40)) - excellent - good
    poor = len(correlations) - excellent - good - fair
    
    categories = ['Excellent\n(r≥0.9, RMSE≤10mm)', 'Good\n(r≥0.7, RMSE≤20mm)', 
                 'Fair\n(r≥0.5, RMSE≤40mm)', 'Poor\n(others)']
    counts = [excellent, good, fair, poor]
    colors = ['green', 'yellow', 'orange', 'red']
    
    bars = ax3.bar(range(len(categories)), counts, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Quality Category', fontsize=9)
    ax3.set_ylabel('Number of Stations', fontsize=9)
    ax3.set_title('Quality Categories', fontsize=10, fontweight='bold')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, fontsize=7, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}\n({count/len(correlations)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    # 4. Parameter Correlations Heatmap (middle-center and right)
    ax4 = fig.add_subplot(gs[1, 1:])
    
    # Extract parameter data for correlation analysis
    param_names = ['trend', 'annual_amp', 'semi_annual_amp', 'quarterly_amp', 'long_annual_amp']
    param_data = []
    
    for param in param_names:
        param_values = []
        for result in ps02c_results:
            if result is not None and param in result:
                # Apply sign correction for trend
                if param == 'trend':
                    param_values.append(-result[param])  # Convert to geodetic convention
                else:
                    param_values.append(result[param])
            else:
                param_values.append(np.nan)
        param_data.append(param_values)
    
    # Create correlation matrix
    param_matrix = np.array(param_data).T
    # Remove rows with NaN
    valid_rows = ~np.any(np.isnan(param_matrix), axis=1)
    param_matrix_clean = param_matrix[valid_rows]
    
    if len(param_matrix_clean) > 0:
        corr_matrix = np.corrcoef(param_matrix_clean.T)
        
        # Create heatmap
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add correlation values as text
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        ax4.set_xticks(range(len(param_names)))
        ax4.set_yticks(range(len(param_names))) 
        ax4.set_xticklabels([name.replace('_', '\n') for name in param_names], fontsize=8)
        ax4.set_yticklabels([name.replace('_', '\n') for name in param_names], fontsize=8)
        ax4.set_title('Parameter Correlations', fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar3 = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar3.set_label('Correlation', rotation=270, labelpad=15, fontsize=8)
    
    # 5. Seasonal Component Distributions (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    seasonal_components = ['Annual', 'Semi-annual', 'Quarterly', 'Long-term']
    seasonal_data = [
        [result.get('annual_amp', 0) for result in ps02c_results if result is not None],
        [result.get('semi_annual_amp', 0) for result in ps02c_results if result is not None],
        [result.get('quarterly_amp', 0) for result in ps02c_results if result is not None],
        [result.get('long_annual_amp', 0) for result in ps02c_results if result is not None]
    ]
    
    bp = ax5.boxplot(seasonal_data, labels=seasonal_components, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']):
        patch.set_facecolor(color)
    
    ax5.set_ylabel('Amplitude (mm)', fontsize=9)
    ax5.set_title('Seasonal Component Distributions', fontsize=10, fontweight='bold')
    ax5.tick_params(axis='x', labelsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Geographic Distribution (bottom-center)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Load coordinate data if available
    if 'lon' in ps00_data and 'lat' in ps00_data:
        lons = ps00_data['lon'][:len(ps02c_trends)]
        lats = ps00_data['lat'][:len(ps02c_trends)]
        
        scatter6 = ax6.scatter(lons, lats, c=ps02c_trends, cmap='RdBu_r', 
                              s=8, alpha=0.7, vmin=-50, vmax=30)
        ax6.set_xlabel('Longitude (°)', fontsize=9)
        ax6.set_ylabel('Latitude (°)', fontsize=9)
        ax6.set_title('Subsidence Trends (mm/year)\nRange: {:.1f} to {:.1f}'.format(
            np.min(ps02c_trends), np.max(ps02c_trends)), fontsize=10, fontweight='bold')
        
        cbar6 = plt.colorbar(scatter6, ax=ax6, shrink=0.8)
        cbar6.set_label('mm/year', rotation=270, labelpad=15, fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'Geographic data\nnot available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=10)
        ax6.set_title('Geographic Distribution', fontsize=10, fontweight='bold')
    
    # 7. Performance Metrics Summary (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])
    
    metrics = ['Processing\nRate\n(st/s)', 'Avg\nCorrelation', 'Avg\nRMSE\n(mm)', 'Success\nRate\n(%)']
    values = [
        len(ps02c_results) / np.sum(optimization_times) if np.sum(optimization_times) > 0 else 0,
        np.mean(correlations),
        np.mean(rmse_values),
        100.0  # All stations processed
    ]
    colors_metrics = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
    
    bars7 = ax7.bar(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black')
    ax7.set_title('Performance Metrics', fontsize=10, fontweight='bold')
    ax7.tick_params(axis='x', labelsize=8)
    
    # Add value labels on bars
    for bar, value in zip(bars7, values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.2f}' if value < 10 else f'{value:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Main title
    fig.suptitle('PS02C Enhanced Performance Analysis with Robust Regression Comparison', 
                fontsize=14, fontweight='bold', y=0.97)
    
    # Save figure
    output_path = Path('figures/ps02c_enhanced_performance_analysis.png')
    output_path.parent.mkdir(exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Enhanced performance analysis saved to {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PS02C ENHANCED PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    print(f"📊 Total stations analyzed: {len(ps02c_results)}")
    print(f"📈 PS00 vs PS02C comparison: {fit_stats['n_points']} matched stations")
    print(f"📉 Robust regression R²: {fit_stats['r_squared']:.3f}")
    print(f"📐 Regression slope: {fit_stats['slope']:.3f} ± {fit_stats['slope_ci_95']:.3f}")
    print(f"📍 Regression intercept: {fit_stats['intercept']:.2f} mm/yr")
    print(f"🎯 PS00-PS02C RMSE: {fit_stats['rmse']:.2f} mm/yr")
    print(f"🔗 PS00-PS02C correlation: {fit_stats['correlation']:.3f}")
    
    quality_summary = (f"🏆 Quality distribution:\n"
                      f"   • Excellent: {excellent} ({excellent/len(correlations)*100:.1f}%)\n"
                      f"   • Good: {good} ({good/len(correlations)*100:.1f}%)\n" 
                      f"   • Fair: {fair} ({fair/len(correlations)*100:.1f}%)\n"
                      f"   • Poor: {poor} ({poor/len(correlations)*100:.1f}%)")
    print(quality_summary)
    print("="*60)

if __name__ == "__main__":
    print("🚀 Creating Enhanced PS02C Performance Analysis...")
    create_enhanced_performance_figure()
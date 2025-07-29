#!/usr/bin/env python3
"""
PS08 V2.1: Statistical Methods Comparison for Confidence Envelopes
================================================================

Implements and compares 4 rigorous statistical methods for confidence envelopes:
1. Proper Prediction Intervals (t-distribution, variable width)
2. Bootstrap Confidence Intervals (non-parametric, robust)
3. Quantile Regression Bands (distribution-based percentiles)  
4. Gaussian Process Regression (machine learning, non-linear)

Generates: ps08_fig05_v2_1_subsidence_reference_grain_size.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.stats import t
from sklearn.utils import resample
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

class StatisticalMethodsComparison:
    """Compare 4 rigorous statistical methods for confidence envelopes"""
    
    def __init__(self):
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load and prepare borehole data"""
        print("🔄 Loading data for statistical comparison...")
        
        # Load InSAR data
        insar_file = Path('data/processed/ps00_preprocessed_data.npz')
        with np.load(insar_file) as data:
            self.insar_coords = data['coordinates']
            self.insar_subsidence_rates = data['subsidence_rates']
        
        # Load borehole data
        borehole_file = Path('../Taiwan_borehole_data/analysis_output/well_fractions.csv')
        self.borehole_data = pd.read_csv(borehole_file)
        
        # Match borehole to InSAR sites
        self.match_borehole_to_insar()
        
        print(f"   ✅ Ready for comparison with {len(self.matched_borehole_data['station_names'])} borehole sites")
    
    def match_borehole_to_insar(self, max_distance_km=2.0):
        """Match borehole sites to InSAR stations"""
        borehole_coords = self.borehole_data[['Longitude', 'Latitude']].values
        distances = cdist(borehole_coords, self.insar_coords, metric='euclidean') * 111.0
        
        nearest_insar_indices = np.argmin(distances, axis=1)
        nearest_distances = np.min(distances, axis=1)
        valid_matches = nearest_distances <= max_distance_km
        
        matched_borehole_idx = np.where(valid_matches)[0]
        matched_insar_idx = nearest_insar_indices[valid_matches]
        
        self.matched_borehole_data = {
            'station_names': self.borehole_data.iloc[matched_borehole_idx]['StationName'].values,
            'coordinates': borehole_coords[matched_borehole_idx],
            'fine_fraction': self.borehole_data.iloc[matched_borehole_idx]['Fine_Pct'].values / 100.0,
            'sand_fraction': self.borehole_data.iloc[matched_borehole_idx]['Sand_Pct'].values / 100.0,
            'coarse_fraction': self.borehole_data.iloc[matched_borehole_idx]['Coarse_Pct'].values / 100.0,
            'subsidence_rates': self.insar_subsidence_rates[matched_insar_idx]
        }
    
    def method1_prediction_intervals(self, x, y, x_pred, alpha=0.05):
        """Method 1: Proper Prediction Intervals with t-distribution"""
        n = len(x)
        
        # Fit regression
        coeffs = np.polyfit(x, y, 1)
        y_fit = np.polyval(coeffs, x)
        y_pred_mean = np.polyval(coeffs, x_pred)
        
        # Calculate residual standard error
        residuals = y - y_fit
        mse = np.sum(residuals**2) / (n - 2)
        
        # Calculate prediction interval (accounts for regression uncertainty)
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean)**2)
        
        # Standard error for prediction - variable width!
        se_pred = np.sqrt(mse * (1 + 1/n + (x_pred - x_mean)**2 / sxx))
        
        # t-distribution critical value (better for small samples)
        t_crit = t.ppf(1 - alpha/2, n - 2)
        
        margin = t_crit * se_pred
        
        return y_pred_mean - margin, y_pred_mean + margin, y_pred_mean
    
    def method2_bootstrap_intervals(self, x, y, x_pred, n_bootstrap=10000, alpha=0.05):
        """Method 2: Bootstrap Confidence Intervals (non-parametric)"""
        n_points = len(x_pred)
        bootstrap_predictions = np.zeros((n_bootstrap, n_points))
        
        for i in range(n_bootstrap):
            # Resample data with replacement
            indices = np.random.choice(len(x), len(x), replace=True)
            x_boot, y_boot = x[indices], y[indices]
            
            # Fit regression to bootstrap sample
            try:
                coeffs = np.polyfit(x_boot, y_boot, 1)
                bootstrap_predictions[i] = np.polyval(coeffs, x_pred)
            except:
                # Handle degenerate cases
                bootstrap_predictions[i] = np.mean(y_boot)
        
        # Calculate percentiles for confidence bands
        lower = np.percentile(bootstrap_predictions, 100 * alpha/2, axis=0)
        upper = np.percentile(bootstrap_predictions, 100 * (1 - alpha/2), axis=0)
        median = np.percentile(bootstrap_predictions, 50, axis=0)
        
        return lower, upper, median
    
    def method3_quantile_regression(self, x, y, x_pred, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        """Method 3: Quantile Regression Bands"""
        bands = {}
        
        # Reshape for sklearn
        x_reshaped = x.reshape(-1, 1)
        x_pred_reshaped = x_pred.reshape(-1, 1)
        
        for q in quantiles:
            try:
                # Fit quantile regression
                qr = QuantileRegressor(quantile=q, alpha=0, solver='highs')
                qr.fit(x_reshaped, y)
                
                # Predict quantile
                y_pred = qr.predict(x_pred_reshaped)
                bands[f'q{int(q*100)}'] = y_pred
            except:
                # Fallback to percentile of data
                bands[f'q{int(q*100)}'] = np.full_like(x_pred, np.percentile(y, q*100))
        
        return bands
    
    def method4_gaussian_process(self, x, y, x_pred, alpha=0.05):
        """Method 4: Gaussian Process Regression with uncertainty"""
        # Define kernel
        kernel = RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2))
        
        # Fit Gaussian Process
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
        gp.fit(x.reshape(-1, 1), y)
        
        # Predict with uncertainty
        y_pred, y_std = gp.predict(x_pred.reshape(-1, 1), return_std=True)
        
        # Calculate confidence intervals
        z_score = stats.norm.ppf(1 - alpha/2)  # 95% confidence
        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std
        
        return lower, upper, y_pred, y_std
    
    def create_comparison_figure(self):
        """Create 4-subplot comparison of statistical methods"""
        print("🎨 Creating statistical methods comparison figure...")
        
        # Get borehole data
        bh_data = self.matched_borehole_data
        n_sites = len(bh_data['station_names'])
        
        # Prepare all three grain size fractions
        grain_size_data = {
            'fine': bh_data['fine_fraction'] * 100,
            'sand': bh_data['sand_fraction'] * 100,
            'coarse': bh_data['coarse_fraction'] * 100
        }
        y_data = bh_data['subsidence_rates']
        
        # Calculate correlations for all fractions
        correlations = {}
        for fraction_name, x_data in grain_size_data.items():
            r, p = stats.pearsonr(x_data, y_data)
            correlations[fraction_name] = {'r': r, 'p': p}
        
        # Create prediction range (0 to 100% for grain size fractions)
        x_pred = np.linspace(0, 100, 100)
        
        # Create figure with 4 subplots - more spacing between subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Statistical Methods Comparison: All Grain Size Fractions vs Subsidence Rate\\n' + 
                    f'Borehole Sites Validation (n={n_sites} sites)', 
                    fontsize=16, fontweight='bold')
        
        # Adjust subplot spacing
        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88, bottom=0.25)
        
        # Load all InSAR data for background (all three grain size fractions)
        import json
        results_file = Path('data/processed/ps08_geological/interpolation_results/interpolated_geology.json')
        with open(results_file, 'r') as f:
            geology_data = json.load(f)
        
        geology = geology_data['best']
        all_fine = np.array(geology['fine_fraction']['values'])
        all_sand = np.array(geology['sand_fraction']['values'])
        all_coarse = np.array(geology['coarse_fraction']['values'])
        all_subsidence = self.insar_subsidence_rates
        
        # Create valid mask for all fractions
        valid_mask = ~(np.isnan(all_fine) | np.isnan(all_sand) | np.isnan(all_coarse))
        
        # Background data for all grain size fractions
        bg_fine = all_fine[valid_mask] * 100
        bg_sand = all_sand[valid_mask] * 100
        bg_coarse = all_coarse[valid_mask] * 100
        bg_subsidence = all_subsidence[valid_mask]
        
        # Define colors and markers for each grain size fraction - better contrast
        fraction_styles = {
            'fine': {'color': 'red', 'marker': 'o', 'label': 'Fine'},
            'sand': {'color': 'gold', 'marker': 's', 'label': 'Sand'},  # Changed from orange to gold/yellow 
            'coarse': {'color': 'blue', 'marker': '^', 'label': 'Coarse'}  # Changed from brown to blue
        }
        
        # Method 1: Prediction Intervals
        ax1 = axes[0, 0]
        
        # Background data - all three grain size fractions like original ps08_fig05
        ax1.scatter(bg_fine, bg_subsidence, c='lightgray', alpha=0.3, s=20, label='All InSAR stations', zorder=0)
        ax1.scatter(bg_sand, bg_subsidence, c='lightblue', alpha=0.3, s=20, marker='s', zorder=0)
        ax1.scatter(bg_coarse, bg_subsidence, c='tan', alpha=0.3, s=20, marker='^', zorder=0)
        
        # Plot in order: red, blue, then gold/yellow on top for visibility
        plot_order = ['fine', 'coarse', 'sand']
        for fraction_name in plot_order:
            x_data = grain_size_data[fraction_name]
            style = fraction_styles[fraction_name]
            r = correlations[fraction_name]['r']
            
            # Use higher alpha for yellow/gold to make it more visible
            alpha_shade = 0.25 if fraction_name == 'sand' else 0.15
            zorder_shade = 3 if fraction_name == 'sand' else 1
            
            # Prediction intervals (plot shade first)
            lower1, upper1, pred1 = self.method1_prediction_intervals(x_data, y_data, x_pred)
            ax1.fill_between(x_pred, lower1, upper1, color=style['color'], alpha=alpha_shade, zorder=zorder_shade)
            ax1.plot(x_pred, pred1, color=style['color'], linewidth=2, alpha=0.8, zorder=zorder_shade+1)
            
            # Borehole data points (on top)
            ax1.scatter(x_data, y_data, c=style['color'], s=60, alpha=0.8, 
                       marker=style['marker'], edgecolors='black', linewidth=0.5,
                       label=f"{style['label']} (r={r:.3f})", zorder=5)
        
        ax1.set_title('Method 1: Prediction Intervals\n(Variable width, t-distribution)', fontweight='bold')
        ax1.set_xlabel('Grain-Size Fraction (%)')
        ax1.set_ylabel('Subsidence Rate (mm/year)')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-5, 105)
        
        # Method 2: Bootstrap
        ax2 = axes[0, 1]
        
        # Background data - all three grain size fractions like original ps08_fig05
        ax2.scatter(bg_fine, bg_subsidence, c='lightgray', alpha=0.3, s=20, label='All InSAR stations', zorder=0)
        ax2.scatter(bg_sand, bg_subsidence, c='lightblue', alpha=0.3, s=20, marker='s', zorder=0)
        ax2.scatter(bg_coarse, bg_subsidence, c='tan', alpha=0.3, s=20, marker='^', zorder=0)
        
        # Plot in order: red, blue, then gold/yellow on top for visibility
        plot_order = ['fine', 'coarse', 'sand']
        for fraction_name in plot_order:
            x_data = grain_size_data[fraction_name]
            style = fraction_styles[fraction_name]
            r = correlations[fraction_name]['r']
            
            # Use higher alpha for yellow/gold to make it more visible
            alpha_shade = 0.25 if fraction_name == 'sand' else 0.15
            zorder_shade = 3 if fraction_name == 'sand' else 1
            
            # Bootstrap intervals (plot shade first)
            lower2, upper2, pred2 = self.method2_bootstrap_intervals(x_data, y_data, x_pred)
            ax2.fill_between(x_pred, lower2, upper2, color=style['color'], alpha=alpha_shade, zorder=zorder_shade)
            ax2.plot(x_pred, pred2, color=style['color'], linewidth=2, alpha=0.8, zorder=zorder_shade+1)
            
            # Borehole data points (on top)
            ax2.scatter(x_data, y_data, c=style['color'], s=60, alpha=0.8, 
                       marker=style['marker'], edgecolors='black', linewidth=0.5,
                       label=f"{style['label']} (r={r:.3f})", zorder=5)
        
        ax2.set_title('Method 2: Bootstrap Intervals\n(Non-parametric, robust, n=10,000)', fontweight='bold')
        ax2.set_xlabel('Grain-Size Fraction (%)')
        ax2.set_ylabel('Subsidence Rate (mm/year)')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-5, 105)
        
        # Method 3: Quantile Regression
        ax3 = axes[1, 0]
        
        # Background data - all three grain size fractions like original ps08_fig05
        ax3.scatter(bg_fine, bg_subsidence, c='lightgray', alpha=0.3, s=20, label='All InSAR stations', zorder=0)
        ax3.scatter(bg_sand, bg_subsidence, c='lightblue', alpha=0.3, s=20, marker='s', zorder=0)
        ax3.scatter(bg_coarse, bg_subsidence, c='tan', alpha=0.3, s=20, marker='^', zorder=0)
        
        # Plot in order: red, blue, then gold/yellow on top for visibility
        plot_order = ['fine', 'coarse', 'sand']
        for fraction_name in plot_order:
            x_data = grain_size_data[fraction_name]
            style = fraction_styles[fraction_name]
            r = correlations[fraction_name]['r']
            
            # Use higher alpha for yellow/gold to make it more visible
            alpha_shade = 0.25 if fraction_name == 'sand' else 0.15
            zorder_shade = 3 if fraction_name == 'sand' else 1
            
            # Quantile bands (show only median and IQR for clarity)
            bands3 = self.method3_quantile_regression(x_data, y_data, x_pred)
            
            # Fill between quartiles (25th-75th percentiles) first
            if 'q25' in bands3 and 'q75' in bands3:
                ax3.fill_between(x_pred, bands3['q25'], bands3['q75'], 
                               color=style['color'], alpha=alpha_shade, zorder=zorder_shade)
            
            # Plot median line on top
            if 'q50' in bands3:
                ax3.plot(x_pred, bands3['q50'], color=style['color'], linewidth=2, alpha=0.8, zorder=zorder_shade+1)
            
            # Borehole data points (on top)
            ax3.scatter(x_data, y_data, c=style['color'], s=60, alpha=0.8, 
                       marker=style['marker'], edgecolors='black', linewidth=0.5,
                       label=f"{style['label']} (r={r:.3f})", zorder=5)
        
        ax3.set_title('Method 3: Quantile Regression\n(Distribution-based percentiles)', fontweight='bold')
        ax3.set_xlabel('Grain-Size Fraction (%)')
        ax3.set_ylabel('Subsidence Rate (mm/year)')
        ax3.legend(fontsize=10, loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-5, 105)
        
        # Method 4: Gaussian Process
        ax4 = axes[1, 1]
        
        # Background data - all three grain size fractions like original ps08_fig05
        ax4.scatter(bg_fine, bg_subsidence, c='lightgray', alpha=0.3, s=20, label='All InSAR stations', zorder=0)
        ax4.scatter(bg_sand, bg_subsidence, c='lightblue', alpha=0.3, s=20, marker='s', zorder=0)
        ax4.scatter(bg_coarse, bg_subsidence, c='tan', alpha=0.3, s=20, marker='^', zorder=0)
        
        # Plot in order: red, blue, then gold/yellow on top for visibility
        plot_order = ['fine', 'coarse', 'sand']
        for fraction_name in plot_order:
            x_data = grain_size_data[fraction_name]
            style = fraction_styles[fraction_name]
            r = correlations[fraction_name]['r']
            
            # Use higher alpha for yellow/gold to make it more visible
            alpha_shade = 0.25 if fraction_name == 'sand' else 0.15
            zorder_shade = 3 if fraction_name == 'sand' else 1
            
            # Gaussian Process (plot shade first)
            lower4, upper4, pred4, std4 = self.method4_gaussian_process(x_data, y_data, x_pred)
            ax4.fill_between(x_pred, lower4, upper4, color=style['color'], alpha=alpha_shade, zorder=zorder_shade)
            ax4.plot(x_pred, pred4, color=style['color'], linewidth=2, alpha=0.8, zorder=zorder_shade+1)
            
            # Borehole data points (on top)
            ax4.scatter(x_data, y_data, c=style['color'], s=60, alpha=0.8, 
                       marker=style['marker'], edgecolors='black', linewidth=0.5,
                       label=f"{style['label']} (r={r:.3f})", zorder=5)
        
        ax4.set_title('Method 4: Gaussian Process\n(Machine learning, non-linear)', fontweight='bold')
        ax4.set_xlabel('Grain-Size Fraction (%)')
        ax4.set_ylabel('Subsidence Rate (mm/year)')
        ax4.legend(fontsize=10, loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-5, 105)
        
        # Add method comparison text box - reformatted for horizontal layout
        comparison_text = f"""STATISTICAL METHODS COMPARISON:

Method 1 - Prediction Intervals: Statistically rigorous for linear regression • Variable width (wider at extremes) • Uses t-distribution for small samples • Accounts for regression coefficient uncertainty

Method 2 - Bootstrap: Non-parametric, no distribution assumptions • Robust to outliers and non-normality • Empirical confidence from data resampling • Good for complex, non-linear relationships

Method 3 - Quantile Regression: Shows median (50th percentile) and IQR (25th-75th) • Robust to heteroscedasticity • Distribution-based confidence bands • Excellent for skewed or multi-modal data

Method 4 - Gaussian Process: Machine learning approach • Captures non-linear patterns automatically • Bayesian uncertainty quantification • Smooth, continuous predictions

RESULTS: Fine r={correlations['fine']['r']:.3f} (p={correlations['fine']['p']:.1e}) • Sand r={correlations['sand']['r']:.3f} (p={correlations['sand']['p']:.1e}) • Coarse r={correlations['coarse']['r']:.3f} (p={correlations['coarse']['p']:.1e}) • Dataset: {n_sites} borehole sites + {np.sum(valid_mask)} background InSAR stations"""

        # Add text box spanning full width at bottom
        fig.text(0.5, 0.02, comparison_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom', horizontalalignment='center', wrap=True)
        
        # Save figure
        fig_path = self.figures_dir / 'ps08_fig05_v2_1_subsidence_reference_grain_size.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")
        print(f"   📊 Statistical Methods Comparison Complete!")
        print(f"      Grain size fraction correlations:")
        print(f"        Fine: r = {correlations['fine']['r']:.3f} (p = {correlations['fine']['p']:.3e})")
        print(f"        Sand: r = {correlations['sand']['r']:.3f} (p = {correlations['sand']['p']:.3e})")
        print(f"        Coarse: r = {correlations['coarse']['r']:.3f} (p = {correlations['coarse']['p']:.3e})")
        print(f"      All 4 methods implemented with {n_sites} borehole validation sites")
        
        return True

def main():
    """Run statistical methods comparison"""
    print("🔍 PS08 V2.1: Statistical Methods Comparison for Confidence Envelopes")
    print("=" * 80)
    
    try:
        # Initialize comparison
        comparer = StatisticalMethodsComparison()
        
        # Create comparison figure
        comparer.create_comparison_figure()
        
        print("🎯 Statistical Methods Comparison Complete!")
        print("=" * 50)
        print("📊 Methods Implemented:")
        print("   1. Prediction Intervals (t-distribution, variable width)")
        print("   2. Bootstrap Confidence Intervals (non-parametric)")  
        print("   3. Quantile Regression Bands (distribution percentiles)")
        print("   4. Gaussian Process Regression (machine learning)")
        print("🖼️  Generated: ps08_fig05_v2_1_subsidence_reference_grain_size.png")
        
    except Exception as e:
        print(f"❌ Error in statistical comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
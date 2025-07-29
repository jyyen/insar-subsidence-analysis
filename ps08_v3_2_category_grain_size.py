#!/usr/bin/env python3
"""
PS08 V3: 2-Category Grain Size Analysis with Phi-Scale Classification
====================================================================

Implements new phi-based grain size categorization:
- Coarse: phi < 1 (coarse sand and coarser) 
- Fine: phi > 1 (medium sand and finer)

Generates: ps08_fig05_v3_2_types_subsidence_reference_grain_size.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from scipy import stats
from scipy.stats import t
from sklearn.utils import resample
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

class PhiBasedGrainSizeAnalysis:
    """2-category grain size analysis using phi-scale classification"""
    
    def __init__(self):
        self.base_dir = Path('.')
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load and process data
        self.load_data()
        self.process_phi_data()
        self.match_borehole_to_insar()
        self.interpolate_grain_size()
        
    def load_data(self):
        """Load InSAR and phi-scale borehole data"""
        print("🔄 Loading data for phi-based 2-category analysis...")
        
        # Load InSAR data
        insar_file = Path('data/processed/ps00_preprocessed_data.npz')
        with np.load(insar_file) as data:
            self.insar_coords = data['coordinates']
            self.insar_subsidence_rates = data['subsidence_rates']
        
        print(f"   ✅ InSAR data: {len(self.insar_coords):,} stations")
        
        # Load phi-scale grain size data from both regions
        changhua_file = Path('../Taiwan_borehole_data/彰化平原/litho/彰化平原_合併Litho_粒徑_phi.csv')
        yunlin_file = Path('../Taiwan_borehole_data/雲林平原/litho/雲林平原_合併Litho_粒徑_phi.csv')
        
        # Read phi data
        self.changhua_phi = pd.read_csv(changhua_file, encoding='utf-8')
        self.yunlin_phi = pd.read_csv(yunlin_file, encoding='utf-8')
        
        # Combine phi data
        self.phi_data = pd.concat([self.changhua_phi, self.yunlin_phi], ignore_index=True)
        
        # Load borehole locations and metadata
        borehole_file = Path('../Taiwan_borehole_data/analysis_output/well_fractions.csv')
        self.borehole_locations = pd.read_csv(borehole_file)
        
        print(f"   ✅ Phi data: {len(self.phi_data):,} depth intervals from {len(self.phi_data['Bore'].unique())} boreholes")
        print(f"   ✅ Borehole locations: {len(self.borehole_locations)} stations")
        
    def process_phi_data(self):
        """Process phi-scale data to calculate percentages of coarse and fine sediments"""
        print("🔄 Processing phi-scale data to calculate coarse/fine percentages...")
        
        # Group by borehole and calculate percentage of coarse vs fine
        borehole_fractions = []
        
        for bore_name in self.phi_data['Bore'].unique():
            bore_data = self.phi_data[self.phi_data['Bore'] == bore_name]
            
            # Get thickness for each layer
            depths = bore_data['Depth2'].values - bore_data['Depth1'].values
            phi_values = bore_data['phi'].values
            
            # Calculate thickness for coarse (phi < 1) and fine (phi >= 1)
            coarse_mask = phi_values < 1
            fine_mask = phi_values >= 1
            
            coarse_thickness = np.sum(depths[coarse_mask])
            fine_thickness = np.sum(depths[fine_mask])
            total_thickness = np.sum(depths)
            
            # Calculate percentages
            if total_thickness > 0:
                coarse_pct = (coarse_thickness / total_thickness) * 100
                fine_pct = (fine_thickness / total_thickness) * 100
            else:
                coarse_pct = 0.0
                fine_pct = 0.0
                
            borehole_fractions.append({
                'Bore': bore_name,
                'coarse_pct': coarse_pct,
                'fine_pct': fine_pct,
                'total_thickness': total_thickness,
                'n_layers': len(bore_data)
            })
        
        self.phi_fractions = pd.DataFrame(borehole_fractions)
        
        print(f"   ✅ Calculated fractions for {len(self.phi_fractions)} boreholes:")
        print(f"      • Mean coarse percentage: {np.mean(self.phi_fractions['coarse_pct']):.1f}%")
        print(f"      • Mean fine percentage: {np.mean(self.phi_fractions['fine_pct']):.1f}%")
        print(f"      • Total layers processed: {np.sum(self.phi_fractions['n_layers']):,}")
        
    def match_borehole_to_insar(self, max_distance_km=2.0):
        """Match borehole percentage data to InSAR stations"""
        print("🔄 Matching borehole percentage data to InSAR stations...")
        
        # Merge phi fractions with borehole locations
        merged_data = self.borehole_locations.merge(
            self.phi_fractions, 
            left_on='StationName', 
            right_on='Bore', 
            how='inner'
        )
        
        if len(merged_data) == 0:
            print("❌ No matches found between borehole locations and phi data")
            return
            
        print(f"   ✅ Found {len(merged_data)} boreholes with both location and phi fraction data")
        
        # Match to nearest InSAR stations
        borehole_coords = merged_data[['Longitude', 'Latitude']].values
        distances = cdist(borehole_coords, self.insar_coords, metric='euclidean') * 111.0
        
        nearest_insar_indices = np.argmin(distances, axis=1)
        nearest_distances = np.min(distances, axis=1)
        valid_matches = nearest_distances <= max_distance_km
        
        matched_borehole_idx = np.where(valid_matches)[0]
        matched_insar_idx = nearest_insar_indices[valid_matches]
        
        self.matched_data = {
            'station_names': merged_data.iloc[matched_borehole_idx]['StationName'].values,
            'coordinates': borehole_coords[matched_borehole_idx],
            'coarse_pct': merged_data.iloc[matched_borehole_idx]['coarse_pct'].values,
            'fine_pct': merged_data.iloc[matched_borehole_idx]['fine_pct'].values,
            'subsidence_rates': self.insar_subsidence_rates[matched_insar_idx],
            'distances': nearest_distances[valid_matches]
        }
        
        print(f"   ✅ Matched {len(self.matched_data['station_names'])} boreholes to InSAR (within {max_distance_km}km)")
        print(f"      • Coarse percentage range: {np.min(self.matched_data['coarse_pct']):.1f}% - {np.max(self.matched_data['coarse_pct']):.1f}%")
        print(f"      • Fine percentage range: {np.min(self.matched_data['fine_pct']):.1f}% - {np.max(self.matched_data['fine_pct']):.1f}%")
        
    def interpolate_grain_size(self, method='idw', power=2.0, max_distance_km=15.0):
        """Interpolate grain size percentages using IDW with distance constraints"""
        print(f"🔄 Interpolating grain size percentages using {method} method (power={power}, max_dist={max_distance_km}km)...")
        
        # Use borehole coordinates and percentage data for interpolation
        borehole_coords = self.matched_data['coordinates']
        coarse_percentages = self.matched_data['coarse_pct']
        fine_percentages = self.matched_data['fine_pct']
        
        if method == 'idw':
            # Inverse Distance Weighting with power parameter for both coarse and fine percentages
            interpolated_coarse_pct = self._idw_interpolation(
                borehole_coords, coarse_percentages, self.insar_coords, 
                power=power, max_distance_km=max_distance_km
            )
            interpolated_fine_pct = self._idw_interpolation(
                borehole_coords, fine_percentages, self.insar_coords, 
                power=power, max_distance_km=max_distance_km
            )
        else:
            # Fallback to griddata for other methods
            interpolated_coarse_pct = griddata(
                borehole_coords, coarse_percentages, self.insar_coords,
                method=method, fill_value=np.nan
            )
            interpolated_fine_pct = griddata(
                borehole_coords, fine_percentages, self.insar_coords,
                method=method, fill_value=np.nan
            )
        
        # Handle NaN values (areas too far from boreholes)
        valid_mask = ~(np.isnan(interpolated_coarse_pct) | np.isnan(interpolated_fine_pct))
        
        # Store interpolated percentages
        self.insar_coarse_pct = np.full(len(self.insar_coords), np.nan)
        self.insar_fine_pct = np.full(len(self.insar_coords), np.nan)
        
        self.insar_coarse_pct[valid_mask] = interpolated_coarse_pct[valid_mask]
        self.insar_fine_pct[valid_mask] = interpolated_fine_pct[valid_mask]
        
        # Normalize percentages to sum to 100% for valid stations
        valid_coarse = self.insar_coarse_pct[valid_mask]
        valid_fine = self.insar_fine_pct[valid_mask]
        total_pct = valid_coarse + valid_fine
        
        # Avoid division by zero
        nonzero_mask = total_pct > 0
        if np.sum(nonzero_mask) > 0:
            valid_coarse[nonzero_mask] = (valid_coarse[nonzero_mask] / total_pct[nonzero_mask]) * 100
            valid_fine[nonzero_mask] = (valid_fine[nonzero_mask] / total_pct[nonzero_mask]) * 100
            
            self.insar_coarse_pct[valid_mask] = valid_coarse
            self.insar_fine_pct[valid_mask] = valid_fine
        
        # Statistics for valid interpolations only
        valid_stations = np.sum(valid_mask)
        
        print(f"   ✅ Interpolated percentages to {len(self.insar_coords):,} InSAR stations:")
        print(f"      • Valid interpolations: {valid_stations:,} stations ({valid_stations/len(self.insar_coords)*100:.1f}%)")
        if valid_stations > 0:
            print(f"      • Coarse percentage: {np.nanmean(self.insar_coarse_pct):.1f}% ± {np.nanstd(self.insar_coarse_pct):.1f}%")
            print(f"      • Fine percentage: {np.nanmean(self.insar_fine_pct):.1f}% ± {np.nanstd(self.insar_fine_pct):.1f}%")
        print(f"      • No data: {len(self.insar_coords) - valid_stations:,} stations ({(len(self.insar_coords) - valid_stations)/len(self.insar_coords)*100:.1f}%)")
        
    def _idw_interpolation(self, borehole_coords, values, target_coords, power=2.0, max_distance_km=15.0):
        """Perform Inverse Distance Weighting interpolation with distance constraint"""
        
        # Calculate distances between all target points and borehole points
        distances = cdist(target_coords, borehole_coords, metric='euclidean') * 111.0  # Convert to km
        
        # Initialize output array
        interpolated_values = np.full(len(target_coords), np.nan)
        
        # Process each target point
        for i in range(len(target_coords)):
            point_distances = distances[i, :]
            
            # Only use boreholes within max_distance_km
            valid_mask = point_distances <= max_distance_km
            
            if np.sum(valid_mask) == 0:
                # No boreholes within maximum distance
                continue
                
            valid_distances = point_distances[valid_mask]
            valid_values = values[valid_mask]
            
            # Handle case where target point is very close to a borehole
            if np.min(valid_distances) < 0.001:  # Very close (< 1 meter)
                closest_idx = np.argmin(valid_distances)
                interpolated_values[i] = valid_values[closest_idx]
                continue
            
            # IDW calculation: weight = 1/distance^power
            weights = 1.0 / (valid_distances ** power)
            interpolated_values[i] = np.sum(weights * valid_values) / np.sum(weights)
        
        return interpolated_values
        
    def create_comparison_figure(self):
        """Create figure matching ps08_fig05_v2 exactly with percentage-based data"""
        print("🎨 Creating ps08_fig05_v3 2-category grain size analysis figure...")
        
        # Get borehole and InSAR data
        bh_data = self.matched_data
        n_borehole_sites = len(bh_data['station_names'])
        
        # Create single figure matching ps08_fig05_v2 layout
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Subsidence Rate vs Grain-Size Fractions\\n' + 
                    f'Borehole Sites Validation (n={n_borehole_sites} sites)', 
                    fontsize=14, fontweight='bold')
        
        # Plot InSAR stations as background with both fine and coarse percentages
        valid_insar_mask = ~np.isnan(self.insar_fine_pct)
        if np.sum(valid_insar_mask) > 0:
            insar_fine_pct = self.insar_fine_pct[valid_insar_mask]
            insar_coarse_pct = self.insar_coarse_pct[valid_insar_mask]
            insar_y = self.insar_subsidence_rates[valid_insar_mask]
            
            # Plot InSAR stations: Fine fraction (red, transparent)
            ax.scatter(insar_fine_pct, insar_y, c='red', alpha=0.15, s=4, zorder=0, 
                      label=f'All InSAR stations')
            
            # Plot InSAR stations: Coarse fraction (blue, transparent) 
            ax.scatter(insar_coarse_pct, insar_y, c='blue', alpha=0.15, s=4, zorder=0)
            
            print(f"   📊 Plotted {np.sum(valid_insar_mask):,} InSAR stations as background")
        
        # Get borehole data for plotting
        borehole_fine_pct = bh_data['fine_pct']
        borehole_coarse_pct = bh_data['coarse_pct']  
        borehole_subsidence = bh_data['subsidence_rates']
        
        # Plot borehole sites: Fine fraction (red circles)
        scatter_fine = ax.scatter(borehole_fine_pct, borehole_subsidence, 
                                c='red', s=80, alpha=0.8, marker='o',
                                edgecolors='black', linewidth=0.5, zorder=10,
                                label=f'Borehole Fine ({n_borehole_sites} sites)')
        
        # Plot borehole sites: Coarse fraction (blue triangles)
        scatter_coarse = ax.scatter(borehole_coarse_pct, borehole_subsidence,
                                  c='blue', s=80, alpha=0.8, marker='^', 
                                  edgecolors='black', linewidth=0.5, zorder=10,
                                  label=f'Borehole Coarse ({n_borehole_sites} sites)')
        
        # Calculate and plot trend lines for both fine and coarse
        if len(borehole_fine_pct) > 1:
            # Fit linear regression for fine fraction
            coeffs_fine = np.polyfit(borehole_fine_pct, borehole_subsidence, 1)
            x_trend = np.linspace(0, 100, 100)
            y_trend_fine = np.polyval(coeffs_fine, x_trend)
            
            # Fit linear regression for coarse fraction  
            coeffs_coarse = np.polyfit(borehole_coarse_pct, borehole_subsidence, 1)
            y_trend_coarse = np.polyval(coeffs_coarse, x_trend)
            
            # Calculate bootstrap confidence intervals (n=10000)
            print("   🔄 Calculating bootstrap intervals (n=10,000)...")
            n_bootstrap = 10000
            
            # Bootstrap for fine fraction
            fine_bootstrap_preds = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(len(borehole_fine_pct), len(borehole_fine_pct), replace=True)
                boot_x_fine = borehole_fine_pct[indices]
                boot_y = borehole_subsidence[indices]
                
                try:
                    coeffs = np.polyfit(boot_x_fine, boot_y, 1)
                    y_pred = np.polyval(coeffs, x_trend)
                    fine_bootstrap_preds.append(y_pred)
                except:
                    continue
            
            # Bootstrap for coarse fraction
            coarse_bootstrap_preds = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(len(borehole_coarse_pct), len(borehole_coarse_pct), replace=True)
                boot_x_coarse = borehole_coarse_pct[indices]
                boot_y = borehole_subsidence[indices]
                
                try:
                    coeffs = np.polyfit(boot_x_coarse, boot_y, 1)
                    y_pred = np.polyval(coeffs, x_trend)
                    coarse_bootstrap_preds.append(y_pred)
                except:
                    continue
            
            # Calculate 99% confidence intervals
            if len(fine_bootstrap_preds) > 0:
                fine_bootstrap_preds = np.array(fine_bootstrap_preds)
                fine_lower = np.percentile(fine_bootstrap_preds, 0.5, axis=0)  # 99% CI: 0.5th percentile
                fine_upper = np.percentile(fine_bootstrap_preds, 99.5, axis=0)  # 99% CI: 99.5th percentile
                fine_median = np.percentile(fine_bootstrap_preds, 50, axis=0)
                
                # Plot bootstrap confidence interval for fine
                ax.fill_between(x_trend, fine_lower, fine_upper, color='red', alpha=0.15, 
                               label='Fine 99% Bootstrap CI', zorder=2)
            
            if len(coarse_bootstrap_preds) > 0:
                coarse_bootstrap_preds = np.array(coarse_bootstrap_preds)
                coarse_lower = np.percentile(coarse_bootstrap_preds, 0.5, axis=0)  # 99% CI: 0.5th percentile
                coarse_upper = np.percentile(coarse_bootstrap_preds, 99.5, axis=0)  # 99% CI: 99.5th percentile
                coarse_median = np.percentile(coarse_bootstrap_preds, 50, axis=0)
                
                # Plot bootstrap confidence interval for coarse
                ax.fill_between(x_trend, coarse_lower, coarse_upper, color='blue', alpha=0.15,
                               label='Coarse 99% Bootstrap CI', zorder=2)
            
            # Plot trend lines on top of bootstrap intervals
            ax.plot(x_trend, y_trend_fine, 'r-', linewidth=2, alpha=0.8,
                   label=f'Fine trend', zorder=6)
            ax.plot(x_trend, y_trend_coarse, 'b--', linewidth=2, alpha=0.8,  
                   label=f'Coarse trend', zorder=6)
            
            # Calculate correlations
            r_fine, p_fine = stats.pearsonr(borehole_fine_pct, borehole_subsidence)
            r_coarse, p_coarse = stats.pearsonr(borehole_coarse_pct, borehole_subsidence)
            
            print(f"   ✅ Bootstrap intervals calculated: {len(fine_bootstrap_preds)} fine, {len(coarse_bootstrap_preds)} coarse iterations")
            
            # Add correlation and statistics text box for 2-category system
            corr_text = f"""BOREHOLE SITE CORRELATIONS:
Fine Fraction vs Subsidence: r = {r_fine:.3f} (p = {p_fine:.2e})
Coarse Fraction vs Subsidence: r = {r_coarse:.3f} (p = {p_coarse:.2e})

2-Category Statistics (φ-scale):
Fine (\u03c6≥1): Mean = {np.mean(borehole_subsidence[borehole_fine_pct > 50]):.1f} mm/year (n={np.sum(borehole_fine_pct > 50)})
Coarse (\u03c6<1): Mean = {np.mean(borehole_subsidence[borehole_coarse_pct > 50]):.1f} mm/year (n={np.sum(borehole_coarse_pct > 50)})
Bootstrap CI: n=10,000 iterations"""
            
            ax.text(0.98, 0.02, corr_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Set axes properties to match reference figure exactly
        ax.set_xlabel('Grain-Size Fraction (%)', fontsize=12)
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=12)
        ax.set_xlim(0, 100)  # 0-100% fine fraction
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper left')
        
        # Add explanatory note
        ax.text(0.02, 0.98, 'X-axis: Fine Fraction (φ>1) %', 
               transform=ax.transAxes, fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps08_fig05_v3_2_types_subsidence_reference_grain_size.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")
        print(f"   📊 2-Category Grain Size Analysis Complete!")
        print(f"      Borehole validation sites: {n_borehole_sites}")
        print(f"      InSAR background stations: {np.sum(valid_insar_mask):,}")
        if len(borehole_fine_pct) > 1:
            print(f"      Fine fraction range: {np.min(borehole_fine_pct):.1f}% - {np.max(borehole_fine_pct):.1f}%")
            print(f"      Coarse fraction range: {np.min(borehole_coarse_pct):.1f}% - {np.max(borehole_coarse_pct):.1f}%")
            print(f"      Fine correlation: r = {r_fine:.3f}, p = {p_fine:.3e}")
            print(f"      Coarse correlation: r = {r_coarse:.3f}, p = {p_coarse:.3e}")
            fine_sig = "significant" if p_fine < 0.05 else "not significant"
            coarse_sig = "significant" if p_coarse < 0.05 else "not significant"
            print(f"      Fine significance: {fine_sig}")
            print(f"      Coarse significance: {coarse_sig}")
    
    def create_geographic_subsidence_map(self):
        """Create professional geographic map using Cartopy with GPS-corrected subsidence rate contours and quantile-based highlighting"""
        print("🗺️ Creating redesigned geographic map with subsidence rate contours and quantile highlighting...")
        
        # Load GPS-corrected subsidence rates from ps00
        print("   🔄 Loading GPS-corrected subsidence rates from ps00...")
        ps00_data = np.load(self.base_dir / 'data/processed/ps00_preprocessed_data.npz', allow_pickle=True)
        insar_coords = ps00_data['coordinates']  # [N, 2] - lon, lat
        insar_subsidence_rates = ps00_data['subsidence_rates']  # [N] - mm/year
        print(f"      ✅ Loaded {len(insar_coords):,} InSAR stations with GPS-corrected rates")
        print(f"      📊 Subsidence rate range: {np.min(insar_subsidence_rates):.1f} to {np.max(insar_subsidence_rates):.1f} mm/year")
        
        # Get borehole data
        bh_data = self.matched_data
        coordinates = bh_data['coordinates']
        fine_pct = bh_data['fine_pct']
        coarse_pct = bh_data['coarse_pct']
        subsidence_rates = bh_data['subsidence_rates']
        
        # Identify sites with greater than 25% fine fraction
        print("   🔄 Identifying sites with >25% fine fraction and calculating subsidence rate quantiles...")
        fine_threshold_mask = fine_pct > 25  # Changed from 50% to 25%
        fine_sites = fine_threshold_mask
        
        # Get subsidence rates for sites with >25% fine fraction
        fine_subsidence_rates = subsidence_rates[fine_sites]
        fine_coords = coordinates[fine_sites]
        fine_fractions = fine_pct[fine_sites]
        
        # Define least subsidence criteria: 0 to -12.5 mm/yr AND uplift region (>0)
        if len(fine_subsidence_rates) > 0:
            # Create highlight mask for sites with least subsidence (0 to -12.5 mm/yr OR uplift >0)
            highlight_least_subsidence = ((fine_subsidence_rates >= -12.5) & (fine_subsidence_rates <= 0)) | (fine_subsidence_rates > 0)
            normal_fine_sites = ~highlight_least_subsidence
            
            print(f"      📊 Sites with >25% fine fraction: {np.sum(fine_sites)} total")
            print(f"      📊 Least subsidence criteria: 0 to -12.5 mm/yr + uplift (>0)")
            print(f"      🔴 Red highlighted sites (least subsidence): {np.sum(highlight_least_subsidence)}")
            print(f"      ⚪ Normal sites (>25% fine): {np.sum(normal_fine_sites)}")
        else:
            print("      ⚠️ No sites with >25% fine fraction found!")
            return
        
        # Create figure with 2 subplots - Fine fraction (left) and Coarse fraction (right)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
        def setup_map_background(ax, title):
            """Setup common map background for both subplots"""
            # Define map bounds based on borehole coverage area
            borehole_lons = coordinates[:, 0]
            borehole_lats = coordinates[:, 1]
            lon_margin = (np.max(borehole_lons) - np.min(borehole_lons)) * 0.1  # 10% margin
            lat_margin = (np.max(borehole_lats) - np.min(borehole_lats)) * 0.1  # 10% margin
            extent = [np.min(borehole_lons) - lon_margin, np.max(borehole_lons) + lon_margin,
                     np.min(borehole_lats) - lat_margin, np.max(borehole_lats) + lat_margin]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, color='black')
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.8)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.8)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            
            # Create subsidence rate contours as background
            from scipy.interpolate import griddata
            
            # Filter InSAR data within map bounds for contouring
            extent_mask = ((insar_coords[:, 0] >= extent[0]) & (insar_coords[:, 0] <= extent[1]) & 
                          (insar_coords[:, 1] >= extent[2]) & (insar_coords[:, 1] <= extent[3]))
            contour_coords = insar_coords[extent_mask]
            contour_rates = insar_subsidence_rates[extent_mask]
            
            # Create interpolation grid
            lon_grid = np.linspace(extent[0], extent[1], 100)
            lat_grid = np.linspace(extent[2], extent[3], 80)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Interpolate subsidence rates to grid
            grid_rates = griddata(contour_coords, contour_rates, (lon_mesh, lat_mesh), method='linear')
            
            # Create coastline mask to exclude ocean areas
            print(f"      🔄 Applying coastline masking to exclude ocean areas...")
            
            try:
                import cartopy.io.shapereader as shpreader
                from shapely.geometry import Point
                from shapely.ops import unary_union
                import shapely.vectorized
                
                # Get Natural Earth land polygons for Taiwan region
                land_shp = shpreader.natural_earth(resolution='10m', category='physical', name='land')
                land_geoms = []
                
                # Read land polygons and filter for Taiwan region
                for record in shpreader.Reader(land_shp).records():
                    geom = record.geometry
                    # Check if geometry intersects with our map bounds
                    geom_bounds = geom.bounds
                    if (geom_bounds[0] < extent[1] and geom_bounds[2] > extent[0] and 
                        geom_bounds[1] < extent[3] and geom_bounds[3] > extent[2]):
                        land_geoms.append(geom)
                
                if land_geoms:
                    # Combine all land geometries
                    taiwan_land = unary_union(land_geoms)
                    
                    # Create mask using vectorized point-in-polygon test
                    land_mask = shapely.vectorized.contains(taiwan_land, lon_mesh, lat_mesh)
                    
                    print(f"         ✅ Created coastline mask: {np.sum(land_mask):,}/{land_mask.size:,} points on land")
                else:
                    print(f"         ⚠️ No land polygons found, using data availability mask")
                    # Fallback: use data availability as mask
                    land_mask = ~np.isnan(grid_rates)
                    
            except Exception as e:
                print(f"         ⚠️ Coastline masking failed ({str(e)}), using data availability mask")
                # Fallback: use data availability as mask  
                land_mask = ~np.isnan(grid_rates)
            
            # Apply coastline masking - set ocean points to NaN
            grid_rates_masked = np.where(land_mask, grid_rates, np.nan)
            
            # Create contour levels focused on subsidence range (0 to -50 mm/yr)
            contour_levels = np.arange(-50, 5, 5)  # Every 5 mm/year from -50 to 0
            
            # Plot contours (only where data is valid/on land) - lines only, no fill
            contour_lines = ax.contour(lon_mesh, lat_mesh, grid_rates_masked, levels=contour_levels,
                                      colors='black', alpha=0.6, linewidths=0.8, 
                                      transform=ccrs.PlateCarree())
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%d', colors='black')
            
            # Set title
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            return contour_lines, len(contour_coords)
        
        print("   🔄 Creating subsidence rate contours...")
        
        # Setup both subplots with background
        contour_lines1, n_contour_coords = setup_map_background(ax1, 
            'Fine Fraction Sites (>25% Fine)\nLeast Subsidence Quantile')
        contour_lines2, _ = setup_map_background(ax2, 
            'Coarse Fraction Sites (>25% Coarse)\nMost Subsidence Quantile')
        
        print(f"      ✅ Created contours for {n_contour_coords:,} InSAR stations")
        
        # Prepare coarse fraction analysis for second subplot
        print("   🔄 Preparing coarse fraction analysis for second subplot...")
        coarse_threshold_mask = coarse_pct > 25  # Sites with >25% coarse fraction
        coarse_sites = coarse_threshold_mask
        
        # Get subsidence rates for sites with >25% coarse fraction
        coarse_subsidence_rates = subsidence_rates[coarse_sites]
        coarse_coords = coordinates[coarse_sites]
        coarse_fractions = coarse_pct[coarse_sites]
        
        # Define most subsidence criteria: smaller than -20 mm/yr (updated from -25)
        if len(coarse_subsidence_rates) > 0:
            # Create highlight mask for coarse sites with most subsidence (<-20 mm/yr)
            highlight_most_subsidence = (coarse_subsidence_rates < -20)
            normal_coarse_sites = ~highlight_most_subsidence
            
            print(f"      📊 Sites with >25% coarse fraction: {np.sum(coarse_sites)} total")
            print(f"      📊 Most subsidence criteria: < -20 mm/yr")
            print(f"      🔵 Blue highlighted sites (most subsidence < -20 mm/yr): {np.sum(highlight_most_subsidence)}")
            print(f"      ⚪ Normal coarse sites: {np.sum(normal_coarse_sites)}")
        
        # Plot all 102 sites in both subplots with conditional highlighting and labeling
        print("   🔄 Plotting LEFT subplot - All sites with fine fraction highlighting...")
        print("   🔄 Plotting RIGHT subplot - All sites with coarse fraction highlighting...")
        
        # LEFT SUBPLOT (ax1): Plot all sites, highlight those meeting fine fraction + subsidence criteria
        for i, (lon, lat) in enumerate(coordinates):
            fine_pct_val = fine_pct[i]
            coarse_pct_val = coarse_pct[i]
            subsidence_val = subsidence_rates[i]
            
            # Check criteria
            meets_fine_fraction = fine_pct_val > 25
            meets_least_subsidence = ((subsidence_val >= -12.5) & (subsidence_val <= 0)) | (subsidence_val > 0)
            
            # Determine color and size
            if meets_fine_fraction and meets_least_subsidence:
                # RED: Both criteria met
                color, size, alpha, edge_color, edge_width = 'red', 100, 0.9, 'darkred', 2.0
                label_text = f'F:{fine_pct_val:.0f}%\n{subsidence_val:.1f}mm/yr'
                label_color, label_weight = 'darkred', 'bold'
                bbox_style = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red')
            elif meets_fine_fraction:
                # GRAY: Only fine fraction criteria met
                color, size, alpha, edge_color, edge_width = 'gray', 60, 0.7, 'black', 1.0
                label_text = f'F:{fine_pct_val:.0f}%'
                label_color, label_weight = 'black', 'normal'
                bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
            else:
                # LIGHT GRAY: No criteria met
                color, size, alpha, edge_color, edge_width = 'lightgray', 40, 0.5, 'gray', 0.5
                label_text = None  # No label
                
            # Plot site
            ax1.scatter(lon, lat, s=size, c=color, marker='o', alpha=alpha,
                       edgecolors=edge_color, linewidth=edge_width, zorder=8,
                       transform=ccrs.PlateCarree())
            
            # Add label if criteria met
            if label_text:
                ax1.annotate(label_text, (lon, lat), xytext=(4, 4), textcoords='offset points',
                           fontsize=6 if not meets_least_subsidence else 7, alpha=0.8 if not meets_least_subsidence else 1.0, 
                           zorder=9, color=label_color, weight=label_weight,
                           transform=ccrs.PlateCarree(), bbox=bbox_style)
        
        # RIGHT SUBPLOT (ax2): Plot all sites, highlight those meeting coarse fraction + subsidence criteria
        for i, (lon, lat) in enumerate(coordinates):
            fine_pct_val = fine_pct[i]
            coarse_pct_val = coarse_pct[i]
            subsidence_val = subsidence_rates[i]
            
            # Check criteria
            meets_coarse_fraction = coarse_pct_val > 25
            meets_most_subsidence = subsidence_val < -20
            
            # Determine color and size
            if meets_coarse_fraction and meets_most_subsidence:
                # BLUE: Both criteria met
                color, size, alpha, edge_color, edge_width = 'blue', 100, 0.9, 'darkblue', 2.0
                label_text = f'C:{coarse_pct_val:.0f}%\n{subsidence_val:.1f}mm/yr'
                label_color, label_weight = 'darkblue', 'bold'
                bbox_style = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='blue')
            elif meets_coarse_fraction:
                # GRAY: Only coarse fraction criteria met
                color, size, alpha, edge_color, edge_width = 'gray', 60, 0.7, 'black', 1.0
                label_text = f'C:{coarse_pct_val:.0f}%'
                label_color, label_weight = 'black', 'normal'
                bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
            else:
                # LIGHT GRAY: No criteria met
                color, size, alpha, edge_color, edge_width = 'lightgray', 40, 0.5, 'gray', 0.5
                label_text = None  # No label
                
            # Plot site
            ax2.scatter(lon, lat, s=size, c=color, marker='s', alpha=alpha,
                       edgecolors=edge_color, linewidth=edge_width, zorder=8,
                       transform=ccrs.PlateCarree())
            
            # Add label if criteria met
            if label_text:
                ax2.annotate(label_text, (lon, lat), xytext=(4, 4), textcoords='offset points',
                           fontsize=6 if not meets_most_subsidence else 7, alpha=0.8 if not meets_most_subsidence else 1.0, 
                           zorder=9, color=label_color, weight=label_weight,
                           transform=ccrs.PlateCarree(), bbox=bbox_style)
        
        # Count sites for legends
        fine_both_criteria = np.sum((fine_pct > 25) & (((subsidence_rates >= -12.5) & (subsidence_rates <= 0)) | (subsidence_rates > 0)))
        fine_fraction_only = np.sum((fine_pct > 25) & ~(((subsidence_rates >= -12.5) & (subsidence_rates <= 0)) | (subsidence_rates > 0)))
        coarse_both_criteria = np.sum((coarse_pct > 25) & (subsidence_rates < -20))
        coarse_fraction_only = np.sum((coarse_pct > 25) & ~(subsidence_rates < -20))
        
        # Add manual legends for both subplots
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements1 = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                   markeredgecolor='darkred', markeredgewidth=2, label=f'Fine >25% + Least subsidence (n={fine_both_criteria})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
                   markeredgecolor='black', markeredgewidth=1, label=f'Fine >25% only (n={fine_fraction_only})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=6, 
                   markeredgecolor='gray', markeredgewidth=0.5, label=f'Other sites (n={102 - fine_both_criteria - fine_fraction_only})')
        ]
        
        legend_elements2 = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, 
                   markeredgecolor='darkblue', markeredgewidth=2, label=f'Coarse >25% + Most subsidence (n={coarse_both_criteria})'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, 
                   markeredgecolor='black', markeredgewidth=1, label=f'Coarse >25% only (n={coarse_fraction_only})'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markersize=6, 
                   markeredgecolor='gray', markeredgewidth=0.5, label=f'Other sites (n={102 - coarse_both_criteria - coarse_fraction_only})')
        ]
        
        ax1.legend(handles=legend_elements1, fontsize=9, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax2.legend(handles=legend_elements2, fontsize=9, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Add compact statistics text boxes for both subplots
        stats_text1 = f"""FINE FRACTION ANALYSIS:
🔴 Fine >25% + Least subsidence: {fine_both_criteria} sites
⚪ Fine >25% only: {fine_fraction_only} sites
⚫ Other sites: {102 - fine_both_criteria - fine_fraction_only} sites
Total: 102 borehole sites"""
        
        stats_text2 = f"""COARSE FRACTION ANALYSIS:
🔵 Coarse >25% + Subsidence < -20mm/yr: {coarse_both_criteria} sites  
⚪ Coarse >25% only: {coarse_fraction_only} sites
⚫ Other sites: {102 - coarse_both_criteria - coarse_fraction_only} sites
Total: 102 borehole sites"""
        
        ax1.text(0.99, 0.01, stats_text1, transform=ax1.transAxes, fontsize=7,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
        
        ax2.text(0.99, 0.01, stats_text2, transform=ax2.transAxes, fontsize=7,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.9))
        
        # Add main title for the entire figure
        fig.suptitle('GPS-Corrected Subsidence Rates with Borehole Grain Size Analysis\nLeft: Fine Fraction (Least Subsidence) | Right: Coarse Fraction (Most Subsidence)', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust for suptitle
        plt.subplots_adjust(wspace=0.15)  # Reduce white space between subplots
        
        # Save figure
        fig_path = self.figures_dir / "ps08_fig05_v3_2_types_subsidence_reference_geographic.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")
        print(f"   📊 Final updated map with all sites plotted and conditional highlighting!")
        print(f"      🗺️ Map extent: Borehole coverage area + 10% margin")
        print(f"      LEFT SUBPLOT - Fine fraction analysis:")
        print(f"         🔴 Red (Fine >25% + Least subsidence): {fine_both_criteria} sites")
        print(f"         ⚪ Gray (Fine >25% only): {fine_fraction_only} sites") 
        print(f"         ⚫ Light gray (Other): {102 - fine_both_criteria - fine_fraction_only} sites")
        print(f"      RIGHT SUBPLOT - Coarse fraction analysis:")
        print(f"         🔵 Blue (Coarse >25% + Subsidence < -20mm/yr): {coarse_both_criteria} sites")
        print(f"         ⚪ Gray (Coarse >25% only): {coarse_fraction_only} sites")
        print(f"         ⚫ Light gray (Other): {102 - coarse_both_criteria - coarse_fraction_only} sites")
        print(f"      📊 Total sites: 102 boreholes (all plotted in both subplots)")
        print(f"      🎨 Background: Clean contour lines only (no color fill)")
        
    def _plot_method1_prediction_intervals(self, ax):
        """Method 1: Prediction Intervals (variable width, t-distribution)"""
        self._plot_base_data(ax)
        
        # Calculate prediction intervals
        x_pred = np.array([0, 1])  # Coarse and Fine categories
        
        # Get borehole data for fitting
        borehole_categories = np.array([0 if cat == 'coarse' else 1 for cat in self.matched_data['category']])
        borehole_rates = self.matched_data['subsidence_rates']
        
        # Fit linear regression
        coeffs = np.polyfit(borehole_categories, borehole_rates, 1)
        y_pred = np.polyval(coeffs, x_pred)
        
        # Calculate prediction intervals
        n = len(borehole_categories)
        y_fit = np.polyval(coeffs, borehole_categories)
        residuals = borehole_rates - y_fit
        mse = np.sum(residuals**2) / (n - 2)
        
        x_mean = np.mean(borehole_categories)
        sxx = np.sum((borehole_categories - x_mean)**2)
        
        # Variable width prediction intervals
        se_pred = []
        for x in x_pred:
            se = np.sqrt(mse * (1 + 1/n + (x - x_mean)**2 / sxx))
            se_pred.append(se)
        
        from scipy.stats import t
        t_crit = t.ppf(0.975, n - 2)  # 95% confidence
        
        y_lower = y_pred - t_crit * np.array(se_pred)
        y_upper = y_pred + t_crit * np.array(se_pred)
        
        # Plot prediction intervals
        ax.fill_between(x_pred, y_lower, y_upper, alpha=0.3, color='lightblue', 
                       label='95% Prediction Interval')
        ax.plot(x_pred, y_pred, 'b-', linewidth=2, label='Linear fit')
        
        ax.set_title('Method 1: Prediction Intervals\\n(Variable width, t-distribution)', fontsize=11)
        ax.set_xlabel('2-Category Grain Size', fontsize=10)
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_method2_bootstrap_intervals(self, ax):
        """Method 2: Bootstrap Intervals (non-parametric, robust, n=10,000)"""
        self._plot_base_data(ax)
        
        # Bootstrap resampling
        n_bootstrap = 10000
        borehole_categories = np.array([0 if cat == 'coarse' else 1 for cat in self.matched_data['category']])
        borehole_rates = self.matched_data['subsidence_rates']
        
        # Bootstrap predictions for each category
        bootstrap_preds = {0: [], 1: []}
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(borehole_rates), len(borehole_rates), replace=True)
            boot_x = borehole_categories[indices]
            boot_y = borehole_rates[indices]
            
            # Fit and predict
            try:
                coeffs = np.polyfit(boot_x, boot_y, 1)
                for cat in [0, 1]:
                    pred = np.polyval(coeffs, cat)
                    bootstrap_preds[cat].append(pred)
            except:
                continue
        
        # Calculate confidence intervals
        x_pred = np.array([0, 1])
        y_lower = []
        y_upper = []
        y_mean = []
        
        for cat in [0, 1]:
            preds = bootstrap_preds[cat]
            if len(preds) > 0:
                y_mean.append(np.mean(preds))
                y_lower.append(np.percentile(preds, 2.5))
                y_upper.append(np.percentile(preds, 97.5))
            else:
                y_mean.append(0)
                y_lower.append(0)
                y_upper.append(0)
        
        # Plot bootstrap intervals
        ax.fill_between(x_pred, y_lower, y_upper, alpha=0.3, color='lightcoral',
                       label='95% Bootstrap Interval')
        ax.plot(x_pred, y_mean, 'r-', linewidth=2, label='Bootstrap mean')
        
        ax.set_title('Method 2: Bootstrap Intervals\\n(Non-parametric, robust, n=10,000)', fontsize=11)
        ax.set_xlabel('2-Category Grain Size', fontsize=10)
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_method3_quantile_regression(self, ax):
        """Method 3: Quantile Regression (distribution-based percentiles)"""
        self._plot_base_data(ax)
        
        borehole_categories = np.array([0 if cat == 'coarse' else 1 for cat in self.matched_data['category']])
        borehole_rates = self.matched_data['subsidence_rates']
        
        try:
            from sklearn.linear_model import QuantileRegressor
            
            x_pred = np.array([0, 1])
            X_borehole = borehole_categories.reshape(-1, 1)
            X_pred = x_pred.reshape(-1, 1)
            
            # Fit quantile regressions
            q10_model = QuantileRegressor(quantile=0.1, alpha=0)
            q50_model = QuantileRegressor(quantile=0.5, alpha=0)
            q90_model = QuantileRegressor(quantile=0.9, alpha=0)
            
            q10_model.fit(X_borehole, borehole_rates)
            q50_model.fit(X_borehole, borehole_rates)
            q90_model.fit(X_borehole, borehole_rates)
            
            y_q10 = q10_model.predict(X_pred)
            y_q50 = q50_model.predict(X_pred)
            y_q90 = q90_model.predict(X_pred)
            
            # Plot quantile bands
            ax.fill_between(x_pred, y_q10, y_q90, alpha=0.3, color='lightgreen',
                           label='10th-90th percentile')
            ax.plot(x_pred, y_q50, 'g-', linewidth=2, label='50th percentile (median)')
            
        except ImportError:
            # Fallback to simple percentile calculation
            coarse_rates = [rate for cat, rate in zip(self.matched_data['category'], borehole_rates) if cat == 'coarse']
            fine_rates = [rate for cat, rate in zip(self.matched_data['category'], borehole_rates) if cat == 'fine']
            
            x_pred = [0, 1]
            y_q10 = [np.percentile(coarse_rates, 10) if coarse_rates else 0, 
                     np.percentile(fine_rates, 10) if fine_rates else 0]
            y_q50 = [np.percentile(coarse_rates, 50) if coarse_rates else 0,
                     np.percentile(fine_rates, 50) if fine_rates else 0]
            y_q90 = [np.percentile(coarse_rates, 90) if coarse_rates else 0,
                     np.percentile(fine_rates, 90) if fine_rates else 0]
            
            ax.fill_between(x_pred, y_q10, y_q90, alpha=0.3, color='lightgreen',
                           label='10th-90th percentile')
            ax.plot(x_pred, y_q50, 'g-', linewidth=2, label='50th percentile (median)')
        
        ax.set_title('Method 3: Quantile Regression\\n(Distribution-based percentiles)', fontsize=11)
        ax.set_xlabel('2-Category Grain Size', fontsize=10)
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_method4_gaussian_process(self, ax):
        """Method 4: Gaussian Process (Machine learning, non-linear)"""
        self._plot_base_data(ax)
        
        borehole_categories = np.array([0 if cat == 'coarse' else 1 for cat in self.matched_data['category']])
        borehole_rates = self.matched_data['subsidence_rates']
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            
            # Gaussian Process with RBF kernel
            kernel = RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-10, 1e+1))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
            
            X_borehole = borehole_categories.reshape(-1, 1)
            X_pred = np.array([0, 1]).reshape(-1, 1)
            
            gp.fit(X_borehole, borehole_rates)
            y_pred, sigma = gp.predict(X_pred, return_std=True)
            
            # Plot GP predictions with uncertainty
            x_pred = np.array([0, 1])
            y_lower = y_pred - 1.96 * sigma  # 95% confidence
            y_upper = y_pred + 1.96 * sigma
            
            ax.fill_between(x_pred, y_lower, y_upper, alpha=0.3, color='lightyellow',
                           label='95% GP Uncertainty')
            ax.plot(x_pred, y_pred, 'orange', linewidth=2, label='GP Prediction')
            
        except ImportError:
            # Fallback to simple mean calculation
            coarse_rates = [rate for cat, rate in zip(self.matched_data['category'], borehole_rates) if cat == 'coarse']
            fine_rates = [rate for cat, rate in zip(self.matched_data['category'], borehole_rates) if cat == 'fine']
            
            x_pred = [0, 1]
            y_pred = [np.mean(coarse_rates) if coarse_rates else 0,
                      np.mean(fine_rates) if fine_rates else 0]
            y_std = [np.std(coarse_rates) if coarse_rates else 0,
                     np.std(fine_rates) if fine_rates else 0]
            
            y_lower = np.array(y_pred) - 1.96 * np.array(y_std)
            y_upper = np.array(y_pred) + 1.96 * np.array(y_std)
            
            ax.fill_between(x_pred, y_lower, y_upper, alpha=0.3, color='lightyellow',
                           label='95% Confidence')
            ax.plot(x_pred, y_pred, 'orange', linewidth=2, label='Mean')
        
        ax.set_title('Method 4: Gaussian Process\\n(Machine learning, non-linear)', fontsize=11)
        ax.set_xlabel('2-Category Grain Size', fontsize=10)
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_base_data(self, ax):
        """Plot base data (InSAR background + borehole sites) for all methods"""
        
        # Plot all InSAR stations as background (only valid interpolations)
        valid_mask = self.insar_categories >= 0  # Exclude -1 (no data)
        if np.sum(valid_mask) > 0:
            # Create x-coordinates for valid InSAR stations
            insar_x = self.insar_categories_continuous[valid_mask]
            insar_y = self.insar_subsidence_rates[valid_mask]
            ax.scatter(insar_x, insar_y, c='lightgray', s=1, alpha=0.3, zorder=1, label='All InSAR stations')
        
        # Plot borehole sites with different symbols for each category
        coarse_rates = []
        fine_rates = []
        coarse_x = []
        fine_x = []
        
        for category, rate in zip(self.matched_data['category'], self.matched_data['subsidence_rates']):
            if category == 'coarse':
                coarse_rates.append(rate)
                coarse_x.append(0)
            else:
                fine_rates.append(rate)
                fine_x.append(1)
        
        # Plot borehole sites
        if coarse_rates:
            ax.scatter(coarse_x, coarse_rates, c='red', s=40, marker='s', 
                      edgecolor='white', linewidth=1, alpha=0.8, zorder=10, 
                      label=f'Coarse (n={len(coarse_rates)})')
        
        if fine_rates:
            ax.scatter(fine_x, fine_rates, c='blue', s=40, marker='^',
                      edgecolor='white', linewidth=1, alpha=0.8, zorder=10,
                      label=f'Fine (n={len(fine_rates)})')
        
        # Set common formatting
        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Coarse\\n(φ<1)', 'Fine\\n(φ>1)'])
        ax.grid(True, alpha=0.3)
        
    def _plot_statistical_comparison(self, ax):
        """Statistical comparison of the two categories"""
        
        coarse_rates = self.insar_subsidence_rates[self.insar_categories == 0]
        fine_rates = self.insar_subsidence_rates[self.insar_categories == 1]
        
        # Box plot comparison
        data_to_plot = [coarse_rates, fine_rates]
        colors = ['sandybrown', 'steelblue']
        
        box_plot = ax.boxplot(data_to_plot, labels=['Coarse\\n(φ<1)', 'Fine\\n(φ>1)'], 
                             patch_artist=True, showmeans=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add statistical annotations
        from scipy.stats import ttest_ind, mannwhitneyu
        
        # T-test
        t_stat, t_p = ttest_ind(coarse_rates, fine_rates)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = mannwhitneyu(coarse_rates, fine_rates, alternative='two-sided')
        
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=11)
        ax.set_title('Statistical Comparison\\nof 2 Categories', fontsize=12, fontweight='bold')
        
        # Add statistical test results
        ax.text(0.5, 0.95, f't-test: p={t_p:.4f}\\nMann-Whitney: p={u_p:.4f}',
                transform=ax.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        
    def _plot_geographic_distribution(self, ax):
        """Geographic distribution of grain size categories"""
        
        # Plot InSAR stations colored by category
        coarse_mask = self.insar_categories == 0
        fine_mask = self.insar_categories == 1
        
        ax.scatter(self.insar_coords[coarse_mask, 0], self.insar_coords[coarse_mask, 1],
                  c='sandybrown', s=2, alpha=0.6, label='Coarse')
        
        ax.scatter(self.insar_coords[fine_mask, 0], self.insar_coords[fine_mask, 1],
                  c='steelblue', s=2, alpha=0.6, label='Fine')
        
        # Overlay borehole sites
        borehole_coords = self.matched_data['coordinates']
        categories = self.matched_data['category']
        
        for coord, category in zip(borehole_coords, categories):
            color = 'darkred' if category == 'coarse' else 'darkblue'
            marker = 's' if category == 'coarse' else 'D'
            ax.scatter(coord[0], coord[1], c=color, s=40, marker=marker,
                      edgecolor='white', linewidth=1, alpha=0.9, zorder=10)
        
        ax.set_xlabel('Longitude (°E)', fontsize=11)
        ax.set_ylabel('Latitude (°N)', fontsize=11)
        ax.set_title('Geographic Distribution\\nof Grain Size Categories', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio for geographic accuracy
        ax.set_aspect('equal', adjustable='box')
        
    def _plot_category_distribution(self, ax):
        """Distribution of categories and phi values"""
        
        # Histogram of phi values at borehole sites
        coarse_phi = self.matched_data['weighted_phi'][self.matched_data['category'] == 'coarse']
        fine_phi = self.matched_data['weighted_phi'][self.matched_data['category'] == 'fine']
        
        bins = np.linspace(-2, 4, 20)
        
        ax.hist(coarse_phi, bins=bins, alpha=0.6, color='sandybrown', label='Coarse', density=True)
        ax.hist(fine_phi, bins=bins, alpha=0.6, color='steelblue', label='Fine', density=True)
        
        # Add classification boundary
        ax.axvline(1, color='red', linestyle='--', linewidth=2, alpha=0.8, label='φ=1 boundary')
        
        ax.set_xlabel('Weighted Phi Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11) 
        ax.set_title('Distribution of Phi Values\\nat Borehole Sites', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
    def _plot_borehole_validation(self, ax):
        """Validation using borehole data"""
        
        # Plot subsidence rates at borehole sites vs phi values
        phi_values = self.matched_data['weighted_phi']
        subsidence_rates = self.matched_data['subsidence_rates']
        categories = self.matched_data['category']
        
        # Separate by category
        coarse_mask = categories == 'coarse'
        fine_mask = categories == 'fine'
        
        ax.scatter(phi_values[coarse_mask], subsidence_rates[coarse_mask],
                  c='sandybrown', s=60, alpha=0.8, label='Coarse', marker='s', edgecolor='black')
        
        ax.scatter(phi_values[fine_mask], subsidence_rates[fine_mask], 
                  c='steelblue', s=60, alpha=0.8, label='Fine', marker='D', edgecolor='black')
        
        # Add trend line
        if len(phi_values) > 2:
            z = np.polyfit(phi_values, subsidence_rates, 1)
            p = np.poly1d(z)
            phi_trend = np.linspace(min(phi_values), max(phi_values), 100)
            ax.plot(phi_trend, p(phi_trend), "r--", alpha=0.8, linewidth=2)
            
            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(subsidence_rates, p(phi_values))
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add classification boundary
        ax.axvline(1, color='red', linestyle=':', alpha=0.8, label='φ=1')
        
        ax.set_xlabel('Weighted Phi Value', fontsize=11)
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=11)
        ax.set_title(f'Borehole Validation\\n({len(self.matched_data["station_names"])} sites)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

def main():
    """Main execution function"""
    print("🔍 PS08 V3: 2-Category Phi-Scale Grain Size Analysis")
    print("=" * 60)
    
    try:
        # Create analysis instance
        analysis = PhiBasedGrainSizeAnalysis()
        
        # Create comparison figure
        analysis.create_comparison_figure()
        
        # Create geographic subsidence rate map
        analysis.create_geographic_subsidence_map()
        
        # Print summary statistics for borehole analysis only
        print("\\n📊 Borehole Analysis Summary:")
        print("=" * 40)
        
        bh_data = analysis.matched_data
        coarse_bh_mask = bh_data['coarse_pct'] > 50
        fine_bh_mask = bh_data['fine_pct'] > 50
        
        coarse_bh_rates = bh_data['subsidence_rates'][coarse_bh_mask]
        fine_bh_rates = bh_data['subsidence_rates'][fine_bh_mask]
        
        print(f"Coarse-dominated boreholes (>50% coarse): {len(coarse_bh_rates)} sites")
        if len(coarse_bh_rates) > 0:
            print(f"  Mean subsidence: {np.mean(coarse_bh_rates):.2f} ± {np.std(coarse_bh_rates):.2f} mm/year")
            print(f"  Range: {np.min(coarse_bh_rates):.1f} to {np.max(coarse_bh_rates):.1f} mm/year")
        
        print(f"\\nFine-dominated boreholes (>50% fine): {len(fine_bh_rates)} sites")
        if len(fine_bh_rates) > 0:
            print(f"  Mean subsidence: {np.mean(fine_bh_rates):.2f} ± {np.std(fine_bh_rates):.2f} mm/year")
            print(f"  Range: {np.min(fine_bh_rates):.1f} to {np.max(fine_bh_rates):.1f} mm/year")
        
        # Statistical significance for borehole sites
        if len(coarse_bh_rates) > 0 and len(fine_bh_rates) > 0:
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(coarse_bh_rates, fine_bh_rates)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"\\nStatistical difference (boreholes): {significance} (p={p_value:.6f})")
        
        print("\\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
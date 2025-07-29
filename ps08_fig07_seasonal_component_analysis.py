#!/usr/bin/env python3
"""
PS08 Fig07: Seasonal Component Strength vs Log(Fine/Coarse) Analysis
===================================================================

Creates analysis similar to ps08_fig06 but with seasonal component strengths:
- Y-axis: Quarterly strength, Semi-annual strength, Annual strength
- X-axis: log₁₀(Fine/Coarse Ratio) 
- Uses EMD decomposition results from ps02
- Applies frequency band classification to identify seasonal components
- Enhanced with kriging-interpolated grain-size data and distance-based coloring

Based on EMD Intrinsic Mode Functions (IMFs) with frequency analysis.

Author: Claude Code Assistant
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class SeasonalComponentAnalysis:
    """Seasonal component strength analysis vs grain-size relationships"""
    
    def __init__(self):
        self.base_dir = Path('.')
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_decomposition_data()
        self.load_borehole_data()
        self.match_borehole_to_insar()
        self.calculate_seasonal_strengths()
        self.load_kriging_results()
        
    def load_decomposition_data(self):
        """Load EMD decomposition results"""
        print("🔄 Loading EMD decomposition data...")
        
        decomp_file = Path('data/processed/ps02_emd_decomposition.npz')
        if not decomp_file.exists():
            raise FileNotFoundError(f"EMD decomposition file not found: {decomp_file}")
        
        with np.load(decomp_file) as data:
            self.imfs = data['imfs']  # Shape: (stations, imfs, time)
            self.coordinates = data['coordinates']
            self.time_vector = data['time_vector']
            self.n_imfs_per_station = data['n_imfs_per_station']
            self.subsidence_rates = data['subsidence_rates']
        
        print(f"   ✅ Loaded EMD data: {len(self.coordinates):,} stations")
        print(f"   📊 IMFs shape: {self.imfs.shape}")
        print(f"   📊 Time points: {len(self.time_vector)}")
        
    def load_borehole_data(self):
        """Load borehole grain-size data"""
        print("🗻 Loading borehole data...")
        
        borehole_file = Path('../Taiwan_borehole_data/analysis_output/well_fractions.csv')
        if not borehole_file.exists():
            raise FileNotFoundError(f"Borehole file not found: {borehole_file}")
        
        self.borehole_data = pd.read_csv(borehole_file)
        
        # Remove duplicates
        coords = self.borehole_data[['Longitude', 'Latitude']].values
        unique_indices = []
        seen = set()
        for i, coord in enumerate(coords):
            coord_tuple = (round(coord[0], 6), round(coord[1], 6))
            if coord_tuple not in seen:
                seen.add(coord_tuple)
                unique_indices.append(i)
        
        self.borehole_data = self.borehole_data.iloc[unique_indices].reset_index(drop=True)
        
        # Process grain-size data
        self.borehole_data['coarse_pct'] = self.borehole_data['Coarse_Pct']
        self.borehole_data['fine_pct'] = self.borehole_data['Sand_Pct'] + self.borehole_data['Fine_Pct']
        
        # Normalize
        total_pct = self.borehole_data['coarse_pct'] + self.borehole_data['fine_pct']
        self.borehole_data['coarse_pct'] = (self.borehole_data['coarse_pct'] / total_pct) * 100
        self.borehole_data['fine_pct'] = (self.borehole_data['fine_pct'] / total_pct) * 100
        
        print(f"   ✅ Processed {len(self.borehole_data)} borehole stations")
        
    def match_borehole_to_insar(self):
        """Match borehole locations to InSAR stations"""
        print("🔄 Matching borehole locations to InSAR stations...")
        
        matched_data = []
        for _, row in self.borehole_data.iterrows():
            bh_coord = np.array([row['Longitude'], row['Latitude']])
            
            # Find nearest InSAR station
            distances = cdist([bh_coord], self.coordinates)[0]
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            
            if nearest_distance <= 0.018:  # 2 km threshold
                matched_data.append({
                    'station_name': row['StationName'],
                    'longitude': row['Longitude'],
                    'latitude': row['Latitude'],
                    'coarse_pct': row['coarse_pct'],
                    'fine_pct': row['fine_pct'],
                    'insar_idx': nearest_idx,
                    'distance_to_insar': nearest_distance * 111.0
                })
        
        self.matched_data = pd.DataFrame(matched_data)
        print(f"   ✅ Matched {len(self.matched_data)} boreholes to InSAR stations")
        
    def calculate_seasonal_strengths(self):
        """Calculate seasonal component strengths using ps02 recategorization (same as ps08_fig05)"""
        print("📊 Loading seasonal component strengths from ps02 recategorization...")
        
        try:
            import json
            recat_file = "data/processed/ps02_emd_recategorization.json"
            with open(recat_file, 'r') as f:
                recategorization = json.load(f)
            print(f"   ✅ Loaded EMD recategorization data")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Could not load ps02 recategorization: {e}")
            print("   This is required to match ps08_fig05 seasonal strength data")
            return
        
        n_stations = len(self.coordinates)
        
        # Initialize strength arrays
        self.quarterly_strength = np.zeros(n_stations)
        self.semi_annual_strength = np.zeros(n_stations)
        self.annual_strength = np.zeros(n_stations)
        
        valid_stations = 0
        
        # Process each station using the same logic as ps08_geological_integration.py
        for station_idx in range(n_stations):
            try:
                n_imfs = int(self.n_imfs_per_station[station_idx])
                if n_imfs < 1:
                    continue
                
                station_imfs = self.imfs[station_idx, :n_imfs, :]
                
                # Get recategorization for this station
                station_recat = recategorization.get(str(station_idx), {})
                
                # Initialize band strengths for this station
                quarterly_strength = 0
                semi_annual_strength = 0
                annual_strength = 0
                
                # Sum strength of IMFs that belong to each frequency band
                for imf_idx in range(n_imfs):
                    if imf_idx >= station_imfs.shape[0]:
                        continue
                        
                    imf = station_imfs[imf_idx, :]
                    if len(imf) <= 10 or np.all(np.isnan(imf)):
                        continue
                    
                    imf_key = f"imf_{imf_idx}"
                    if imf_key in station_recat:
                        imf_category = station_recat[imf_key].get('final_category', '')
                        imf_strength = np.std(imf)  # Use standard deviation as strength measure (same as original)
                        
                        # Assign to appropriate seasonal band
                        if imf_category == 'quarterly':
                            quarterly_strength += imf_strength
                        elif imf_category == 'semi_annual':
                            semi_annual_strength += imf_strength
                        elif imf_category == 'annual':
                            annual_strength += imf_strength
                
                # Store results
                self.quarterly_strength[station_idx] = quarterly_strength
                self.semi_annual_strength[station_idx] = semi_annual_strength
                self.annual_strength[station_idx] = annual_strength
                
                valid_stations += 1
                
            except Exception as e:
                continue
        
        print(f"   ✅ Calculated seasonal strengths for {valid_stations:,} stations using ps02 recategorization")
        
        # Show statistics (should now match ps08_fig05 data)
        quarterly_nonzero = np.sum(self.quarterly_strength > 0)
        semi_annual_nonzero = np.sum(self.semi_annual_strength > 0)
        annual_nonzero = np.sum(self.annual_strength > 0)
        
        print(f"   📊 Quarterly: {quarterly_nonzero}/{n_stations} stations have components (max: {np.max(self.quarterly_strength):.2f} mm)")
        print(f"   📊 Semi-annual: {semi_annual_nonzero}/{n_stations} stations have components (max: {np.max(self.semi_annual_strength):.2f} mm)")
        print(f"   📊 Annual: {annual_nonzero}/{n_stations} stations have components (max: {np.max(self.annual_strength):.2f} mm)")
        
    def load_kriging_results(self):
        """Load kriging interpolation results for grain-size data"""
        print("📥 Loading kriging interpolation results...")
        
        kriging_file = Path('data/processed/ps08_kriging_interpolation_results.npz')
        if kriging_file.exists():
            with np.load(kriging_file) as data:
                self.insar_coarse_kriged = data['coarse_interpolated']
                self.insar_fine_kriged = data['fine_interpolated']
            print(f"   ✅ Loaded kriging results")
        else:
            print(f"   ⚠️  Kriging results not found, using simple interpolation")
            # Fallback to simple interpolation
            self.simple_interpolation()
    
    def simple_interpolation(self):
        """Simple IDW interpolation as fallback"""
        print("🔄 Performing simple IDW interpolation...")
        
        bh_coords = self.matched_data[['longitude', 'latitude']].values
        bh_coarse = self.matched_data['coarse_pct'].values
        bh_fine = self.matched_data['fine_pct'].values
        
        self.insar_coarse_kriged = np.full(len(self.coordinates), np.nan)
        self.insar_fine_kriged = np.full(len(self.coordinates), np.nan)
        
        for i, coord in enumerate(self.coordinates):
            distances = cdist([coord], bh_coords)[0]
            
            # Use points within 15 km
            valid_mask = distances <= 15.0/111.0
            if np.sum(valid_mask) > 0:
                valid_distances = distances[valid_mask]
                valid_coarse = bh_coarse[valid_mask]
                valid_fine = bh_fine[valid_mask]
                
                # IDW with power = 2
                weights = 1.0 / (valid_distances**2 + 1e-10)
                self.insar_coarse_kriged[i] = np.sum(weights * valid_coarse) / np.sum(weights)
                self.insar_fine_kriged[i] = np.sum(weights * valid_fine) / np.sum(weights)
    
    def get_lnjs_coordinates(self):
        """Get LNJS reference station coordinates"""
        self.lnjs_coord = np.array([120.5921603, 23.7574494])  # [lon, lat]
        
        print(f"📍 LNJS Reference Station:")
        print(f"   📍 Lon: {self.lnjs_coord[0]:.6f}°, Lat: {self.lnjs_coord[1]:.6f}°")
        print(f"   📋 Purpose: GPS reference station for InSAR analysis")
        
    def calculate_distances_to_lnjs(self):
        """Calculate distances from boreholes to LNJS reference station"""
        distances = []
        for _, row in self.matched_data.iterrows():
            bh_coord = np.array([row['longitude'], row['latitude']])
            distance = np.sqrt((bh_coord[0] - self.lnjs_coord[0])**2 + 
                             (bh_coord[1] - self.lnjs_coord[1])**2)
            distance_km = distance * 111.0
            distances.append(distance_km)
        
        self.matched_data['distance_to_lnjs_km'] = distances
        
        print(f"   📏 Borehole distances to LNJS: {np.min(distances):.1f} - {np.max(distances):.1f} km")
        print(f"   📊 Mean distance to LNJS: {np.mean(distances):.1f} km")
        
    def create_seasonal_component_plots(self):
        """Create seasonal component strength vs log(fine/coarse) plots"""
        print("🎨 Creating seasonal component analysis plots...")
        
        # Get LNJS coordinates and calculate distances
        self.get_lnjs_coordinates()
        self.calculate_distances_to_lnjs()
        
        # Create three separate plots for quarterly, semi-annual, and annual
        seasonal_components = [
            ('quarterly', self.quarterly_strength, 'Quarterly Strength (mm)'),
            ('semi_annual', self.semi_annual_strength, 'Semi-Annual Strength (mm)'),
            ('annual', self.annual_strength, 'Annual Strength (mm)')
        ]
        
        for comp_name, comp_strength, ylabel in seasonal_components:
            self.create_single_component_plot(comp_name, comp_strength, ylabel)
        
        # Create combined plot
        self.create_combined_seasonal_plot()
    
    def create_single_component_plot(self, component_name, component_strength, ylabel):
        """Create individual seasonal component plot"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get valid interpolated grain-size data
        valid_insar_mask = ~(np.isnan(self.insar_coarse_kriged) | np.isnan(self.insar_fine_kriged) | 
                            np.isnan(component_strength))
        
        if np.sum(valid_insar_mask) == 0:
            print(f"   ⚠️  No valid data for {component_name} component")
            plt.close(fig)
            return
        
        valid_coarse = self.insar_coarse_kriged[valid_insar_mask]
        valid_fine = self.insar_fine_kriged[valid_insar_mask]
        valid_strength = component_strength[valid_insar_mask]
        
        # Calculate log(fine/coarse) ratio for InSAR data
        insar_ratio = np.zeros_like(valid_fine)
        nonzero_coarse = valid_coarse > 0.1
        insar_ratio[nonzero_coarse] = valid_fine[nonzero_coarse] / valid_coarse[nonzero_coarse]
        
        valid_ratio_mask = (insar_ratio > 0) & (insar_ratio < 1000)
        log_insar_ratio = np.log10(insar_ratio[valid_ratio_mask])
        insar_strength_filtered = valid_strength[valid_ratio_mask]
        
        # Plot InSAR stations
        ax.scatter(log_insar_ratio, insar_strength_filtered, 
                  c='lightblue', alpha=0.3, s=15, 
                  label=f'InSAR Stations (n={len(log_insar_ratio):,})')
        
        # Get borehole data with seasonal strengths
        bh_data_with_strength = []
        for _, row in self.matched_data.iterrows():
            insar_idx = row['insar_idx']
            strength_val = component_strength[insar_idx]
            if not np.isnan(strength_val):
                bh_data_with_strength.append({
                    'longitude': row['longitude'],
                    'latitude': row['latitude'],
                    'coarse_pct': row['coarse_pct'],
                    'fine_pct': row['fine_pct'],
                    'strength': strength_val,
                    'distance_to_lnjs_km': row['distance_to_lnjs_km']
                })
        
        if len(bh_data_with_strength) == 0:
            print(f"   ⚠️  No borehole data with {component_name} strength")
            plt.close(fig)
            return
        
        bh_df = pd.DataFrame(bh_data_with_strength)
        
        # Calculate log(fine/coarse) for boreholes
        bh_coarse = bh_df['coarse_pct'].values
        bh_fine = bh_df['fine_pct'].values
        bh_strength = bh_df['strength'].values
        bh_distances = bh_df['distance_to_lnjs_km'].values
        
        bh_ratio = np.zeros_like(bh_fine)
        bh_nonzero_coarse = bh_coarse > 0.1
        bh_ratio[bh_nonzero_coarse] = bh_fine[bh_nonzero_coarse] / bh_coarse[bh_nonzero_coarse]
        
        bh_valid_ratio = (bh_ratio > 0) & (bh_ratio < 1000) & bh_nonzero_coarse
        log_bh_ratio = np.log10(bh_ratio[bh_valid_ratio])
        bh_strength_filtered = bh_strength[bh_valid_ratio]
        bh_distances_filtered = bh_distances[bh_valid_ratio]
        
        # Plot borehole stations colored by distance
        if len(log_bh_ratio) > 0:
            scatter_bh = ax.scatter(log_bh_ratio, bh_strength_filtered, 
                                  c=bh_distances_filtered, cmap='viridis', s=80, alpha=0.9,
                                  edgecolors='black', linewidth=1.0, 
                                  label=f'Borehole Stations (n={len(log_bh_ratio)})')
            
            # Add colorbar
            cbar = plt.colorbar(scatter_bh, ax=ax, shrink=0.8)
            cbar.set_label('Distance to LNJS Reference Station (km)', fontsize=12)
        
        # Add vertical lines for coarse fraction thresholds
        coarse_percentages = [1, 25, 50, 75, 99]
        for coarse_pct in coarse_percentages:
            fine_pct = 100 - coarse_pct
            ratio = fine_pct / coarse_pct
            log_ratio = np.log10(ratio)
            
            ax.axvline(x=log_ratio, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(log_ratio, ax.get_ylim()[1] * 0.95, f'{coarse_pct}% coarse', 
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=10, color='gray')
        
        # Calculate correlation if sufficient data
        if len(log_bh_ratio) > 3:
            r, p = stats.pearsonr(log_bh_ratio, bh_strength_filtered)
            correlation_text = (f'Borehole correlation:\nr = {r:.3f}, p = {p:.3f}\n\n'
                              f'LNJS Reference Station:\n'
                              f'Lon: {self.lnjs_coord[0]:.3f}°\n'
                              f'Lat: {self.lnjs_coord[1]:.3f}°\n'
                              f'GPS reference for InSAR')
            ax.text(0.95, 0.95, correlation_text, 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('log₁₀(Fine/Coarse Ratio)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'{component_name.replace("_", "-").title()} Component Strength vs log₁₀(Fine/Coarse Ratio)\n'
                    f'Borehole Points Colored by Distance to LNJS Reference Station', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Add explanatory text
        explanation = ("Vertical lines: Coarse fraction thresholds\n"
                      "Blue points: InSAR interpolated data\n"
                      "Colored points: Distance to LNJS reference")
        ax.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / f"ps08_fig07_{component_name}_log_fine_coarse.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")
    
    def create_combined_seasonal_plot(self):
        """Create combined plot with all three seasonal components"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        seasonal_components = [
            ('Quarterly', self.quarterly_strength, 'Quarterly Strength (mm)', 'lightcoral'),
            ('Semi-Annual', self.semi_annual_strength, 'Semi-Annual Strength (mm)', 'lightgreen'),
            ('Annual', self.annual_strength, 'Annual Strength (mm)', 'lightblue')
        ]
        
        for idx, (comp_name, comp_strength, ylabel, color) in enumerate(seasonal_components):
            ax = axes[idx]
            
            # Get valid data
            valid_insar_mask = ~(np.isnan(self.insar_coarse_kriged) | np.isnan(self.insar_fine_kriged) | 
                                np.isnan(comp_strength))
            
            if np.sum(valid_insar_mask) == 0:
                ax.text(0.5, 0.5, f'No valid {comp_name.lower()} data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{comp_name} Component')
                continue
            
            valid_coarse = self.insar_coarse_kriged[valid_insar_mask]
            valid_fine = self.insar_fine_kriged[valid_insar_mask]
            valid_strength = comp_strength[valid_insar_mask]
            
            # Calculate ratios
            insar_ratio = np.zeros_like(valid_fine)
            nonzero_coarse = valid_coarse > 0.1
            insar_ratio[nonzero_coarse] = valid_fine[nonzero_coarse] / valid_coarse[nonzero_coarse]
            
            valid_ratio_mask = (insar_ratio > 0) & (insar_ratio < 1000)
            log_insar_ratio = np.log10(insar_ratio[valid_ratio_mask])
            insar_strength_filtered = valid_strength[valid_ratio_mask]
            
            # Plot InSAR data
            ax.scatter(log_insar_ratio, insar_strength_filtered, 
                      c=color, alpha=0.4, s=8, label=f'InSAR (n={len(log_insar_ratio):,})')
            
            # Add borehole data if available
            bh_data_with_strength = []
            for _, row in self.matched_data.iterrows():
                insar_idx = row['insar_idx']
                strength_val = comp_strength[insar_idx]
                if not np.isnan(strength_val):
                    bh_data_with_strength.append({
                        'coarse_pct': row['coarse_pct'],
                        'fine_pct': row['fine_pct'],
                        'strength': strength_val,
                        'distance_to_lnjs_km': row['distance_to_lnjs_km']
                    })
            
            if len(bh_data_with_strength) > 0:
                bh_df = pd.DataFrame(bh_data_with_strength)
                bh_coarse = bh_df['coarse_pct'].values
                bh_fine = bh_df['fine_pct'].values
                bh_strength = bh_df['strength'].values
                bh_distances = bh_df['distance_to_lnjs_km'].values
                
                bh_ratio = np.zeros_like(bh_fine)
                bh_nonzero_coarse = bh_coarse > 0.1
                bh_ratio[bh_nonzero_coarse] = bh_fine[bh_nonzero_coarse] / bh_coarse[bh_nonzero_coarse]
                
                bh_valid_ratio = (bh_ratio > 0) & (bh_ratio < 1000) & bh_nonzero_coarse
                log_bh_ratio = np.log10(bh_ratio[bh_valid_ratio])
                bh_strength_filtered = bh_strength[bh_valid_ratio]
                bh_distances_filtered = bh_distances[bh_valid_ratio]
                
                if len(log_bh_ratio) > 0:
                    scatter_bh = ax.scatter(log_bh_ratio, bh_strength_filtered, 
                                          c=bh_distances_filtered, cmap='viridis', s=60, alpha=0.9,
                                          edgecolors='black', linewidth=0.8, 
                                          label=f'Boreholes (n={len(log_bh_ratio)})')
                    
                    if idx == 2:  # Add colorbar only to last subplot
                        cbar = plt.colorbar(scatter_bh, ax=ax, shrink=0.8)
                        cbar.set_label('Distance to LNJS Reference (km)', fontsize=10)
            
            # Add vertical lines
            coarse_percentages = [25, 50, 75]
            for coarse_pct in coarse_percentages:
                fine_pct = 100 - coarse_pct
                ratio = fine_pct / coarse_pct
                log_ratio = np.log10(ratio)
                ax.axvline(x=log_ratio, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Formatting
            ax.set_xlabel('log₁₀(Fine/Coarse Ratio)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{comp_name} Component', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # Save combined figure
        fig_path = self.figures_dir / "ps08_fig07_combined_seasonal_components.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")

def main():
    """Main execution function"""
    print("🚀 PS08 Fig07: Seasonal Component Strength vs Log(Fine/Coarse) Analysis")
    print("=" * 80)
    
    try:
        analysis = SeasonalComponentAnalysis()
        analysis.create_seasonal_component_plots()
        
        print("\n✅ PS08 Fig07 seasonal component analysis completed successfully!")
        print("\n📝 Summary:")
        print("   • Analyzed EMD decomposition results for seasonal component identification")
        print("   • Calculated quarterly, semi-annual, and annual component strengths")
        print("   • Created grain-size vs seasonal strength correlation plots")
        print("   • Applied distance-based coloring to fastest subsidence location")
        print("   • Generated individual and combined seasonal component visualizations")
        
    except Exception as e:
        print(f"❌ Error in seasonal component analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
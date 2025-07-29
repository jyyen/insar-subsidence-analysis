#!/usr/bin/env python3
"""
PS08 Fig06: Log(Fine/Coarse) vs Subsidence Rate Analysis
========================================================

Creates a new figure series showing:
- All InSAR stations: subsidence rates vs log(fine/coarse) in transparent light color
- Borehole stations: real grain-size data with subsidence rates in non-transparent color
- Vertical dashed lines at coarse fractions of 25%, 50%, 75%

Based on phi-scale classification:
- Coarse: phi < 1 (coarse sand and coarser)
- Fine: phi ≥ 1 (medium sand and finer)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LogFineCoarseAnalysis:
    """Log(Fine/Coarse) vs Subsidence Rate Analysis"""
    
    def __init__(self):
        self.base_dir = Path('.')
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_insar_data()
        self.find_fastest_subsidence_point()
        self.load_borehole_data()
        self.process_phi_data()
        self.match_borehole_to_insar()
        self.calculate_distances_to_fastest_point()
        self.interpolate_grain_size()
        
    def load_insar_data(self):
        """Load InSAR data from ps00 preprocessing"""
        print("📡 Loading InSAR data...")
        
        insar_file = Path('data/processed/ps00_preprocessed_data.npz')
        with np.load(insar_file) as data:
            self.insar_coords = data['coordinates']
            self.insar_subsidence_rates = data['subsidence_rates']
        
        print(f"   ✅ InSAR data: {len(self.insar_coords):,} stations")
        print(f"   📊 Subsidence range: {np.min(self.insar_subsidence_rates):.1f} to {np.max(self.insar_subsidence_rates):.1f} mm/year")
        
    def find_fastest_subsidence_point(self):
        """Find the location with fastest subsidence rate"""
        print("🎯 Finding fastest subsidence point...")
        
        min_idx = np.argmin(self.insar_subsidence_rates)
        self.fastest_subsidence_rate = self.insar_subsidence_rates[min_idx]
        self.fastest_subsidence_coord = self.insar_coords[min_idx]
        
        print(f"   ✅ Fastest subsidence point:")
        print(f"      📍 Longitude: {self.fastest_subsidence_coord[0]:.6f}°")
        print(f"      📍 Latitude: {self.fastest_subsidence_coord[1]:.6f}°")
        print(f"      📉 Subsidence rate: {self.fastest_subsidence_rate:.2f} mm/year")
    
    def load_borehole_data(self):
        """Load borehole data using same approach as ps08_geological_integration.py"""
        print("🗻 Loading borehole data...")
        
        # Load borehole fractions data (same as ps08_geological_integration.py)
        borehole_file = Path('../Taiwan_borehole_data/analysis_output/well_fractions.csv')
        if not borehole_file.exists():
            print(f"❌ Borehole file not found: {borehole_file}")
            return False
        
        self.borehole_data = pd.read_csv(borehole_file)
        print(f"   ✅ Loaded {len(self.borehole_data)} borehole stations")
        print(f"   📋 Available columns: {list(self.borehole_data.columns)}")
        
        # Check coordinate bounds
        lon_min, lon_max = self.borehole_data['Longitude'].min(), self.borehole_data['Longitude'].max()
        lat_min, lat_max = self.borehole_data['Latitude'].min(), self.borehole_data['Latitude'].max()
        print(f"   📍 Coordinate bounds: Lon {lon_min:.3f} to {lon_max:.3f}, Lat {lat_min:.3f} to {lat_max:.3f}")
        
        return True
    
    def process_phi_data(self):
        """Use existing grain-size fractions from well_fractions.csv"""
        print("🔄 Processing grain-size fractions from borehole data...")
        
        # The well_fractions.csv already contains Coarse_Pct, Sand_Pct, Fine_Pct
        # We need to convert to 2-category system: coarse vs fine
        # Based on ps08_v3_2: phi < 1 = coarse, phi >= 1 = fine
        # Since the data already gives us fractions, we'll use:
        # Coarse = Coarse_Pct, Fine = Sand_Pct + Fine_Pct
        
        self.borehole_data['coarse_pct'] = self.borehole_data['Coarse_Pct']
        self.borehole_data['fine_pct'] = self.borehole_data['Sand_Pct'] + self.borehole_data['Fine_Pct']
        
        # Ensure percentages sum to 100%
        total_pct = self.borehole_data['coarse_pct'] + self.borehole_data['fine_pct']
        self.borehole_data['coarse_pct'] = (self.borehole_data['coarse_pct'] / total_pct) * 100
        self.borehole_data['fine_pct'] = (self.borehole_data['fine_pct'] / total_pct) * 100
        
        print(f"   ✅ Processed fractions for {len(self.borehole_data)} boreholes:")
        print(f"      • Mean coarse percentage: {np.mean(self.borehole_data['coarse_pct']):.1f}%")
        print(f"      • Mean fine percentage: {np.mean(self.borehole_data['fine_pct']):.1f}%")
        print(f"      • Coarse range: {np.min(self.borehole_data['coarse_pct']):.1f}% - {np.max(self.borehole_data['coarse_pct']):.1f}%")
        print(f"      • Fine range: {np.min(self.borehole_data['fine_pct']):.1f}% - {np.max(self.borehole_data['fine_pct']):.1f}%")
    
    def match_borehole_to_insar(self):
        """Match borehole data to nearest InSAR stations"""
        print("🔄 Matching borehole data to InSAR stations...")
        
        matched_data = []
        for _, row in self.borehole_data.iterrows():
            bh_coord = np.array([row['Longitude'], row['Latitude']])
            
            # Find nearest InSAR station
            distances = cdist([bh_coord], self.insar_coords)[0]
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            
            if nearest_distance <= 2.0:  # Within 2km
                matched_data.append({
                    'station_name': row['StationName'],
                    'longitude': row['Longitude'],
                    'latitude': row['Latitude'],
                    'coarse_pct': row['coarse_pct'],
                    'fine_pct': row['fine_pct'],
                    'subsidence_rate': self.insar_subsidence_rates[nearest_idx],
                    'insar_idx': nearest_idx,
                    'distance_km': nearest_distance
                })
        
        self.matched_data = pd.DataFrame(matched_data)
        print(f"   ✅ Matched {len(self.matched_data)} boreholes to InSAR stations")
        print(f"   📊 Coarse percentage range: {np.min(self.matched_data['coarse_pct']):.1f}% - {np.max(self.matched_data['coarse_pct']):.1f}%")
        print(f"   📊 Fine percentage range: {np.min(self.matched_data['fine_pct']):.1f}% - {np.max(self.matched_data['fine_pct']):.1f}%")
        
    def calculate_distances_to_fastest_point(self):
        """Calculate distances from each borehole to the fastest subsidence point"""
        print("📏 Calculating distances to fastest subsidence point...")
        
        distances = []
        for _, row in self.matched_data.iterrows():
            bh_coord = np.array([row['longitude'], row['latitude']])
            # Calculate Euclidean distance (approximate for small geographic areas)
            distance = np.sqrt((bh_coord[0] - self.fastest_subsidence_coord[0])**2 + 
                             (bh_coord[1] - self.fastest_subsidence_coord[1])**2)
            # Convert to kilometers (approximate: 1 degree ≈ 111 km)
            distance_km = distance * 111.0
            distances.append(distance_km)
        
        self.matched_data['distance_to_fastest_km'] = distances
        
        print(f"   ✅ Calculated distances for {len(self.matched_data)} boreholes")
        print(f"   📊 Distance range: {np.min(distances):.1f} - {np.max(distances):.1f} km")
        print(f"   📊 Mean distance: {np.mean(distances):.1f} km")
    
    def interpolate_grain_size(self):
        """Interpolate grain size to all InSAR stations using IDW"""
        print("🔄 Interpolating grain size to InSAR stations...")
        
        bh_coords = self.matched_data[['longitude', 'latitude']].values
        coarse_pct = self.matched_data['coarse_pct'].values
        fine_pct = self.matched_data['fine_pct'].values
        
        # IDW interpolation
        interpolated_coarse = self._idw_interpolation(bh_coords, coarse_pct, self.insar_coords)
        interpolated_fine = self._idw_interpolation(bh_coords, fine_pct, self.insar_coords)
        
        # Store results
        self.insar_coarse_pct = interpolated_coarse
        self.insar_fine_pct = interpolated_fine
        
        # Calculate coverage
        valid_mask = ~(np.isnan(interpolated_coarse) | np.isnan(interpolated_fine))
        coverage = np.sum(valid_mask) / len(self.insar_coords) * 100
        
        print(f"   ✅ Interpolated to {len(self.insar_coords):,} InSAR stations")
        print(f"   📊 Coverage: {coverage:.1f}% of stations")
    
    def _idw_interpolation(self, known_coords, known_values, target_coords, power=2.0, max_distance_km=15.0):
        """Inverse Distance Weighting interpolation"""
        interpolated = np.full(len(target_coords), np.nan)
        
        for i, target_coord in enumerate(target_coords):
            distances = cdist([target_coord], known_coords)[0]
            
            # Only use points within max distance
            valid_mask = distances <= max_distance_km
            if np.sum(valid_mask) == 0:
                continue
            
            valid_distances = distances[valid_mask]
            valid_values = known_values[valid_mask]
            
            # Avoid division by zero
            valid_distances[valid_distances == 0] = 1e-10
            
            # IDW calculation
            weights = 1.0 / (valid_distances ** power)
            interpolated[i] = np.sum(weights * valid_values) / np.sum(weights)
        
        return interpolated
    
    def create_log_fine_coarse_plot(self):
        """Create log(fine/coarse) vs subsidence rate plot"""
        print("🎨 Creating log(fine/coarse) vs subsidence rate plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate log(fine/coarse) for InSAR stations (interpolated data)
        valid_insar_mask = ~(np.isnan(self.insar_coarse_pct) | np.isnan(self.insar_fine_pct))
        valid_insar_coarse = self.insar_coarse_pct[valid_insar_mask]
        valid_insar_fine = self.insar_fine_pct[valid_insar_mask]
        valid_insar_subsidence = self.insar_subsidence_rates[valid_insar_mask]
        
        # Calculate fine/coarse ratio and log for InSAR
        insar_ratio = np.zeros_like(valid_insar_fine)
        nonzero_coarse = valid_insar_coarse > 0.1  # Avoid division by very small numbers
        insar_ratio[nonzero_coarse] = valid_insar_fine[nonzero_coarse] / valid_insar_coarse[nonzero_coarse]
        
        # Calculate log(fine/coarse) for valid ratios
        valid_ratio_mask = (insar_ratio > 0) & (insar_ratio < 1000)  # Remove extreme ratios
        log_insar_ratio = np.log10(insar_ratio[valid_ratio_mask])
        insar_subsidence_filtered = valid_insar_subsidence[valid_ratio_mask]
        
        # Plot InSAR stations (transparent light color)
        ax.scatter(log_insar_ratio, insar_subsidence_filtered, 
                  c='lightblue', alpha=0.3, s=15, label=f'InSAR Stations (n={len(log_insar_ratio):,})')
        
        # Calculate log(fine/coarse) for borehole stations (real data)
        bh_coarse = self.matched_data['coarse_pct'].values
        bh_fine = self.matched_data['fine_pct'].values
        bh_subsidence = self.matched_data['subsidence_rate'].values
        bh_distances = self.matched_data['distance_to_fastest_km'].values
        
        # Calculate fine/coarse ratio for boreholes
        bh_ratio = np.zeros_like(bh_fine)
        bh_nonzero_coarse = bh_coarse > 0.1
        bh_ratio[bh_nonzero_coarse] = bh_fine[bh_nonzero_coarse] / bh_coarse[bh_nonzero_coarse]
        
        # Calculate log(fine/coarse) for boreholes
        bh_valid_ratio = (bh_ratio > 0) & (bh_ratio < 1000) & bh_nonzero_coarse
        log_bh_ratio = np.log10(bh_ratio[bh_valid_ratio])
        bh_subsidence_filtered = bh_subsidence[bh_valid_ratio]
        bh_distances_filtered = bh_distances[bh_valid_ratio]
        
        # Plot borehole stations (non-transparent) - colored by distance to fastest subsidence point
        scatter_bh = ax.scatter(log_bh_ratio, bh_subsidence_filtered, 
                              c=bh_distances_filtered, cmap='viridis', s=80, alpha=0.9,
                              edgecolors='black', linewidth=1.0, 
                              label=f'Borehole Stations (n={len(log_bh_ratio)})')
        
        # Add colorbar for borehole points
        cbar = plt.colorbar(scatter_bh, ax=ax, shrink=0.8)
        cbar.set_label('Distance to Fastest Subsidence Point (km)', fontsize=12)
        
        # Add vertical dashed lines for coarse fractions of 1%, 25%, 50%, 75%, 99%
        # Convert coarse percentages to fine/coarse ratios and then to log scale
        coarse_percentages = [1, 25, 50, 75, 99]
        for coarse_pct in coarse_percentages:
            fine_pct = 100 - coarse_pct
            ratio = fine_pct / coarse_pct
            log_ratio = np.log10(ratio)
            
            ax.axvline(x=log_ratio, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(log_ratio, ax.get_ylim()[1] * 0.95, f'{coarse_pct}% coarse', 
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=10, color='gray')
        
        # Calculate and show correlation for borehole data (move to top-right)
        if len(log_bh_ratio) > 3:
            r, p = stats.pearsonr(log_bh_ratio, bh_subsidence_filtered)
            correlation_text = (f'Borehole correlation:\nr = {r:.3f}, p = {p:.3f}\n\n'
                              f'Fastest subsidence point:\n'
                              f'Lon: {self.fastest_subsidence_coord[0]:.3f}°\n'
                              f'Lat: {self.fastest_subsidence_coord[1]:.3f}°\n'
                              f'Rate: {self.fastest_subsidence_rate:.1f} mm/yr')
            ax.text(0.95, 0.95, correlation_text, 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('log₁₀(Fine/Coarse Ratio)', fontsize=14)
        ax.set_ylabel('Subsidence Rate (mm/year)', fontsize=14)
        ax.set_title('Subsidence Rate vs log₁₀(Fine/Coarse Ratio)\nBorehole Points Colored by Distance to Fastest Subsidence Location', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Add explanatory text (move to lower-left)
        explanation = ("Vertical lines: Coarse fraction thresholds\n"
                      "Blue points: InSAR interpolated data\n"
                      "Colored points: Distance to fastest subsidence")
        ax.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps08_fig06_log_fine_coarse_subsidence.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")
        print(f"   📊 InSAR stations plotted: {len(log_insar_ratio):,}")
        print(f"   📊 Borehole stations plotted: {len(log_bh_ratio)}")
        print(f"   📊 Correlation (boreholes): r = {r:.3f}, p = {p:.3f}" if len(log_bh_ratio) > 3 else "")

def main():
    """Main execution function"""
    print("🚀 PS08 Fig06: Log(Fine/Coarse) vs Subsidence Rate Analysis")
    print("=" * 60)
    
    try:
        analysis = LogFineCoarseAnalysis()
        analysis.create_log_fine_coarse_plot()
        
        print("\\n✅ PS08 Fig06 analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in PS08 Fig06 analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Enhanced Geographic Plotting for PS04C Motif Discovery
=====================================================

Replaces basic matplotlib scatter plots with professional cartographic visualization
using Cartopy, automatic data coverage bounds, and proper coordinate system handling.

Features:
- Cartopy-based geographic plotting with proper projections
- Automatic data coverage area calculation with margins
- Professional coastline and terrain features
- Taiwan-specific geographic context
- High-resolution output for publication

Author: Claude Code
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

class EnhancedGeographicPlotter:
    """Professional geographic plotting with automatic data coverage"""
    
    def __init__(self, coordinates, margin_degrees=0.05):
        """
        Initialize with station coordinates and coverage margin
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Station coordinates [longitude, latitude]
        margin_degrees : float
            Margin around data coverage in degrees (default: 0.05)
        """
        self.coordinates = coordinates
        self.margin = margin_degrees
        
        # Calculate data coverage bounds
        self.lon_min = np.min(coordinates[:, 0]) - margin_degrees
        self.lon_max = np.max(coordinates[:, 0]) + margin_degrees
        self.lat_min = np.min(coordinates[:, 1]) - margin_degrees
        self.lat_max = np.max(coordinates[:, 1]) + margin_degrees
        
        print(f"📍 Data coverage bounds:")
        print(f"   Longitude: {self.lon_min:.3f}°E to {self.lon_max:.3f}°E")
        print(f"   Latitude: {self.lat_min:.3f}°N to {self.lat_max:.3f}°N")
        print(f"   Coverage area: {(self.lon_max-self.lon_min)*111:.1f} × {(self.lat_max-self.lat_min)*111:.1f} km")
    
    def setup_taiwan_map(self, ax, add_features=True):
        """
        Setup Taiwan map with proper projection and features
        
        Parameters:
        -----------
        ax : cartopy.mpl.geoaxes.GeoAxes
            Cartopy axis object
        add_features : bool
            Whether to add geographic features
        """
        # Set extent to data coverage
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max], 
                     crs=ccrs.PlateCarree())
        
        if add_features:
            # Add coastlines and borders
            ax.coastlines(resolution='50m', color='black', linewidth=1.0, alpha=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, alpha=0.8)
            
            # Add physical features
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2)
            ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.4, linewidth=0.5)
            ax.add_feature(cfeature.LAKES, color='blue', alpha=0.4)
            
            # Add grid lines
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
    
    def plot_spatial_pattern_map_enhanced(self, discovered_motifs, detected_discords, 
                                        output_path='figures/ps04c_fig04_spatial_pattern_map_enhanced.png'):
        """
        Create enhanced spatial pattern map with professional cartography
        """
        print("🗺️  Creating enhanced spatial pattern map with Cartopy...")
        
        # Create figure with cartopy subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define projection (Taiwan TM2 would be ideal, but PlateCarree works for this scale)
        proj = ccrs.PlateCarree()
        
        # Create subplots with cartopy projection
        ax1 = fig.add_subplot(2, 2, 1, projection=proj)
        ax2 = fig.add_subplot(2, 2, 2, projection=proj) 
        ax3 = fig.add_subplot(2, 2, 3, projection=proj)
        ax4 = fig.add_subplot(2, 2, 4)  # Non-geographic subplot for statistics
        
        fig.suptitle('Enhanced Spatial Pattern Distribution - Taiwan Subsidence Analysis', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Plot 1: Motif spatial distribution
        self.setup_taiwan_map(ax1)
        
        if discovered_motifs:
            # Enhanced color scheme for patterns
            pattern_colors = {
                'irrigation_season': '#2E8B57',     # Sea Green
                'monsoon_pattern': '#4169E1',       # Royal Blue
                'dry_season': '#FF8C00',           # Dark Orange  
                'annual_cycle': '#9932CC',         # Dark Orchid
                'winter_pattern': '#00CED1',       # Dark Turquoise
                'short_term_fluctuation': '#FFD700' # Gold
            }
            
            # Plot all stations as background with proper transform
            ax1.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                       c='lightgray', s=2, alpha=0.4, label='All stations (2000)',
                       transform=ccrs.PlateCarree(), zorder=1)
            
            # Plot motif locations with enhanced symbols
            legend_added = set()
            for window_size, motifs in discovered_motifs.items():
                for motif in motifs:
                    for station_idx in motif.station_indices:
                        coord = self.coordinates[station_idx]
                        color = pattern_colors.get(motif.temporal_context, '#FF1493')  # Deep Pink fallback
                        
                        # Size based on significance score
                        size = 60 + motif.significance_score * 15
                        
                        # Add to legend only once per pattern type
                        label = None
                        if motif.temporal_context not in legend_added:
                            label = f"{motif.temporal_context.replace('_', ' ').title()} ({window_size}d)"
                            legend_added.add(motif.temporal_context)
                        
                        ax1.scatter(coord[0], coord[1], c=color, s=size,
                                   alpha=0.8, edgecolor='black', linewidth=0.8,
                                   label=label, transform=ccrs.PlateCarree(), zorder=3)
            
            ax1.set_title('Motif Locations by Temporal Pattern', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9, 
                      fancybox=True, shadow=True)
        
        # Plot 2: Discord spatial distribution
        self.setup_taiwan_map(ax2)
        
        if detected_discords:
            # Enhanced color scheme for anomalies
            anomaly_colors = {
                'extreme_subsidence': '#8B0000',    # Dark Red
                'extreme_uplift': '#006400',        # Dark Green
                'rapid_subsidence': '#DC143C',      # Crimson
                'rapid_uplift': '#32CD32',         # Lime Green
                'high_variability': '#FF4500',     # Orange Red
                'unusual_pattern': '#4B0082'       # Indigo
            }
            
            # Plot all stations as background
            ax2.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                       c='lightgray', s=2, alpha=0.4, label='All stations (2000)',
                       transform=ccrs.PlateCarree(), zorder=1)
            
            # Plot discord locations with enhanced symbols
            legend_added = set()
            for window_size, discords in detected_discords.items():
                for discord in discords:
                    for station_idx in discord.station_indices:
                        coord = self.coordinates[station_idx]
                        color = anomaly_colors.get(discord.anomaly_type, '#000000')  # Black fallback
                        
                        # Size based on anomaly score  
                        size = 60 + discord.anomaly_score * 15
                        
                        # Add to legend only once per anomaly type
                        label = None
                        if discord.anomaly_type not in legend_added:
                            label = f"{discord.anomaly_type.replace('_', ' ').title()} ({window_size}d)"
                            legend_added.add(discord.anomaly_type)
                        
                        ax2.scatter(coord[0], coord[1], c=color, s=size,
                                   alpha=0.9, edgecolor='black', linewidth=0.8, marker='^',
                                   label=label, transform=ccrs.PlateCarree(), zorder=3)
            
            ax2.set_title('Anomaly Locations by Type', fontsize=14, fontweight='bold')  
            ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9,
                      fancybox=True, shadow=True)
        
        # Plot 3: Pattern density heatmap
        self.setup_taiwan_map(ax3)
        
        if discovered_motifs:
            # Collect all pattern locations
            all_lons, all_lats = [], []
            for window_size, motifs in discovered_motifs.items():
                for motif in motifs:
                    for station_idx in motif.station_indices:
                        coord = self.coordinates[station_idx]
                        all_lons.append(coord[0])
                        all_lats.append(coord[1])
            
            if all_lons:
                # Create high-resolution density map
                from scipy.stats import gaussian_kde
                
                # Create grid for density estimation
                lon_grid = np.linspace(self.lon_min, self.lon_max, 100)
                lat_grid = np.linspace(self.lat_min, self.lat_max, 100)
                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                
                # Calculate density using Gaussian KDE
                if len(all_lons) > 1:
                    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
                    values = np.vstack([all_lons, all_lats])
                    kernel = gaussian_kde(values)
                    density = np.reshape(kernel(positions).T, lon_mesh.shape)
                    
                    # Plot density contours
                    contour = ax3.contourf(lon_mesh, lat_mesh, density, levels=15,
                                          cmap='YlOrRd', alpha=0.7, 
                                          transform=ccrs.PlateCarree(), zorder=2)
                    
                    # Add colorbar
                    cbar = plt.colorbar(contour, ax=ax3, shrink=0.8, aspect=20)
                    cbar.set_label('Pattern Density', fontsize=11)
                
                # Overlay all stations
                ax3.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                           c='blue', s=1, alpha=0.3, transform=ccrs.PlateCarree(), zorder=1)
                
                ax3.set_title('Pattern Density Distribution', fontsize=14, fontweight='bold')
        
        # Plot 4: Statistical summary (non-geographic)
        if discovered_motifs and detected_discords:
            # Count patterns by type
            pattern_counts = {}
            anomaly_counts = {}
            
            for window_size, motifs in discovered_motifs.items():
                for motif in motifs:
                    key = f"{motif.temporal_context} ({window_size}d)"
                    pattern_counts[key] = pattern_counts.get(key, 0) + len(motif.station_indices)
            
            for window_size, discords in detected_discords.items():
                for discord in discords:
                    key = f"{discord.anomaly_type} ({window_size}d)"
                    anomaly_counts[key] = anomaly_counts.get(key, 0) + len(discord.station_indices)
            
            # Create combined pie chart
            if pattern_counts or anomaly_counts:
                all_counts = {**pattern_counts, **anomaly_counts}
                labels = list(all_counts.keys())
                sizes = list(all_counts.values())
                
                # Enhanced color scheme
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                  colors=colors, startangle=90,
                                                  textprops={'fontsize': 9})
                
                ax4.set_title('Pattern Distribution Summary\\nTotal Stations with Patterns: {:,}'.format(
                             sum(sizes)), fontsize=14, fontweight='bold')
                
                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.15)
        
        # Save with high resolution
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   💾 Enhanced spatial pattern map saved: {output_path}")
        return output_path

def apply_enhanced_geographic_plotting(coordinates_file, motifs_file, discords_file):
    """
    Apply enhanced geographic plotting to existing ps04c results
    """
    print("🗺️  Applying enhanced geographic plotting to PS04C results...")
    
    # This would be called from the main ps04c analysis
    # For now, we'll create a standalone version that can be integrated
    
    print("✅ Enhanced geographic plotting functions ready for integration")
    return True

if __name__ == "__main__":
    # Example usage - would be integrated into ps04c_motif_discovery.py
    print("📍 Enhanced Geographic Plotting for PS04C Motif Discovery")
    print("=" * 60)
    print("Ready for integration into ps04c_motif_discovery.py")
    print("Features:")
    print("  • Cartopy-based professional cartography")
    print("  • Automatic data coverage bounds")
    print("  • Taiwan-specific geographic context")
    print("  • High-resolution publication-ready output")
    print("  • Enhanced color schemes and symbols")
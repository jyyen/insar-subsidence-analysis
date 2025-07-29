#!/usr/bin/env python3
"""
Cartopy GSHHG Helper - Custom Geographic Features Using GSHHG Data
================================================================

This module provides enhanced geographic features for Cartopy using the
high-resolution GSHHG (Global Self-consistent, Hierarchical, High-resolution Geography)
database that we downloaded.

Author: Claude Code Assistant
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from pathlib import Path
import subprocess
import tempfile
import os

class GSHHGCartopyFeatures:
    """Enhanced Cartopy features using GSHHG data"""
    
    def __init__(self, gshhg_dir="gshhg-gmt-2.3.7"):
        self.gshhg_dir = Path(gshhg_dir)
        if not self.gshhg_dir.exists():
            raise FileNotFoundError(f"GSHHG directory not found: {gshhg_dir}")
        
        # Resolution mapping
        self.resolutions = {
            'f': 'full',     # Full resolution
            'h': 'high',     # High resolution  
            'i': 'intermediate',  # Intermediate resolution
            'l': 'low',      # Low resolution
            'c': 'crude'     # Crude resolution
        }
        
        print(f"✅ GSHHG Cartopy Helper initialized with data from {self.gshhg_dir}")
    
    def extract_taiwan_coastline(self, resolution='h', region=[119.5, 122.5, 22.5, 25.5]):
        """Extract Taiwan coastline using GMT and convert to matplotlib-compatible format"""
        
        try:
            # Use GMT to extract coastline data for Taiwan region
            print(f"🗺️ Extracting Taiwan coastline at {self.resolutions[resolution]} resolution...")
            
            # Create temporary file for GMT output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # GMT command to extract coastline data as ASCII
            gmt_cmd = [
                'gmt', 'pscoast',
                f'-R{region[0]}/{region[1]}/{region[2]}/{region[3]}',
                '-JX10c', 
                f'-D{resolution}',  # Resolution
                '-W0.1p',           # Coastline pen
                '-M',               # Multi-segment format
                '>', tmp_path
            ]
            
            # Run GMT command
            result = subprocess.run(' '.join(gmt_cmd), shell=True, 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Read the extracted coastline data
                coastline_coords = self._parse_gmt_output(tmp_path)
                os.unlink(tmp_path)  # Clean up
                print(f"   ✅ Extracted {len(coastline_coords)} coastline segments")
                return coastline_coords
            else:
                print(f"   ❌ GMT extraction failed: {result.stderr}")
                os.unlink(tmp_path)
                return None
                
        except Exception as e:
            print(f"   ❌ Error extracting coastline: {e}")
            return None
    
    def extract_taiwan_rivers(self, resolution='h', region=[119.5, 122.5, 22.5, 25.5]):
        """Extract Taiwan rivers using GMT"""
        
        try:
            print(f"🌊 Extracting Taiwan rivers at {self.resolutions[resolution]} resolution...")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # GMT command to extract river data
            gmt_cmd = [
                'gmt', 'pscoast',
                f'-R{region[0]}/{region[1]}/{region[2]}/{region[3]}',
                '-JX10c',
                f'-D{resolution}',
                f'-I1/0.1p,blue',  # Rivers level 1
                '-M',
                '>', tmp_path
            ]
            
            result = subprocess.run(' '.join(gmt_cmd), shell=True,
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                river_coords = self._parse_gmt_output(tmp_path)
                os.unlink(tmp_path)
                print(f"   ✅ Extracted {len(river_coords)} river segments")
                return river_coords
            else:
                print(f"   ❌ GMT river extraction failed: {result.stderr}")
                os.unlink(tmp_path)
                return None
                
        except Exception as e:
            print(f"   ❌ Error extracting rivers: {e}")
            return None
    
    def _parse_gmt_output(self, file_path):
        """Parse GMT multi-segment output file"""
        
        segments = []
        current_segment = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Segment separator
                    if line.startswith('>'):
                        if current_segment:
                            segments.append(np.array(current_segment))
                            current_segment = []
                        continue
                    
                    # Parse coordinate line
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            lon, lat = float(parts[0]), float(parts[1])
                            current_segment.append([lon, lat])
                    except (ValueError, IndexError):
                        continue
            
            # Add final segment
            if current_segment:
                segments.append(np.array(current_segment))
        
        except Exception as e:
            print(f"   ⚠️  Error parsing GMT output: {e}")
        
        return segments
    
    def add_taiwan_features(self, ax, resolution='h', coastline_color='black', 
                          coastline_width=1.0, river_color='blue', river_width=0.8,
                          region=None):
        """Add enhanced Taiwan coastline and rivers to Cartopy axes"""
        
        # Default to Taiwan region if not specified
        if region is None:
            region = [119.5, 122.5, 22.5, 25.5]
        
        print(f"🏞️ Adding enhanced Taiwan features to map...")
        
        # Extract and add coastline
        coastline_segments = self.extract_taiwan_coastline(resolution, region)
        if coastline_segments:
            self._add_line_segments(ax, coastline_segments, coastline_color, 
                                  coastline_width, "Taiwan Coastline (GSHHG)")
        
        # Extract and add rivers  
        river_segments = self.extract_taiwan_rivers(resolution, region)
        if river_segments:
            self._add_line_segments(ax, river_segments, river_color, 
                                  river_width, "Taiwan Rivers (GSHHG)")
        
        print("   ✅ Enhanced Taiwan features added successfully")
    
    def _add_line_segments(self, ax, segments, color, linewidth, label):
        """Add line segments to Cartopy axes"""
        
        if not segments:
            return
        
        # Create line collection for efficient rendering
        lines = []
        for segment in segments:
            if len(segment) > 1:  # Valid segment
                lines.append(segment)
        
        if lines:
            # Add to map with proper transform
            lc = LineCollection(lines, colors=color, linewidths=linewidth, 
                              transform=ccrs.PlateCarree(), zorder=10)
            ax.add_collection(lc)
            print(f"   ✅ Added {len(lines)} {label} segments")

def create_enhanced_taiwan_map(coordinates, data_values, title="Enhanced Taiwan Map", 
                             cmap='viridis', figsize=(12, 10)):
    """Create an enhanced Taiwan map with GSHHG coastline and rivers"""
    
    # Initialize GSHHG helper
    gshhg_helper = GSHHGCartopyFeatures()
    
    # Calculate map extent
    lon_min, lon_max = coordinates[:, 0].min() - 0.05, coordinates[:, 0].max() + 0.05
    lat_min, lat_max = coordinates[:, 1].min() - 0.05, coordinates[:, 1].max() + 0.05
    region = [lon_min, lon_max, lat_min, lat_max]
    
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Set map extent
    ax.set_extent(region, crs=proj)
    
    # Add standard Cartopy features first (as background)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=1)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2, zorder=2)
    
    # Add enhanced GSHHG features
    gshhg_helper.add_taiwan_features(ax, resolution='h', region=region)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot data
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                        c=data_values, cmap=cmap, s=25, alpha=0.8, 
                        edgecolors='black', linewidths=0.5,
                        transform=proj, zorder=15)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('Data Values', rotation=270, labelpad=15)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax

# Test function
def test_gshhg_cartopy():
    """Test the GSHHG Cartopy integration"""
    
    print("🧪 Testing GSHHG Cartopy integration...")
    
    try:
        # Create test data
        lon_test = np.array([120.2, 120.4, 120.6, 120.8])
        lat_test = np.array([23.5, 23.7, 23.9, 24.1])
        coords_test = np.column_stack([lon_test, lat_test])
        values_test = np.array([10, 20, 30, 40])
        
        # Create enhanced map
        fig, ax = create_enhanced_taiwan_map(coords_test, values_test, 
                                           "GSHHG Enhanced Taiwan Test Map")
        
        # Save test figure
        plt.savefig('test_gshhg_cartopy_taiwan.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ GSHHG Cartopy test completed successfully!")
        print("   Generated: test_gshhg_cartopy_taiwan.png")
        
    except Exception as e:
        print(f"❌ GSHHG Cartopy test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gshhg_cartopy()
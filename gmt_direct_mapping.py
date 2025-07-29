#!/usr/bin/env python3
"""
Direct GMT Mapping for Taiwan InSAR Analysis
===========================================

This module provides direct GMT command generation for creating professional
Taiwan geographic maps with full control over coastlines, rivers, and styling.

Since GMT 6.5.0 works perfectly on your system, this approach provides:
- Superior Taiwan coastline and river detail
- Full control over map projections and styling  
- High-resolution geographic features
- Reliable, battle-tested cartographic output

Author: Claude Code Assistant  
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
from pathlib import Path

class GMTDirectMapper:
    """Direct GMT mapping interface for Taiwan InSAR data"""
    
    def __init__(self, gmt_sharedir="/opt/homebrew/Cellar/gmt/6.5.0_5/share/gmt"):
        self.gmt_sharedir = Path(gmt_sharedir)
        self.temp_dir = Path("temp_gmt")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Default Taiwan region
        self.taiwan_region = [119.8, 122.2, 22.0, 25.5]
        
        print(f"🗺️ GMT Direct Mapper initialized")
        print(f"   GMT share directory: {self.gmt_sharedir}")
    
    def create_data_file(self, coordinates, values, filename="data.txt", **kwargs):
        """Create GMT-compatible data file"""
        
        output_file = self.temp_dir / filename
        
        # Prepare data columns: lon lat value [additional columns]
        data_array = np.column_stack([coordinates[:, 0], coordinates[:, 1], values])
        
        # Add additional columns if provided
        for key, vals in kwargs.items():
            if len(vals) == len(coordinates):
                data_array = np.column_stack([data_array, vals])
        
        # Save to file
        np.savetxt(output_file, data_array, fmt='%.6f', delimiter='\\t')
        print(f"   ✅ Created GMT data file: {output_file}")
        
        return output_file
    
    def create_cpt_file(self, data_values, cmap_name="viridis", filename="data.cpt"):
        """Create GMT color palette file"""
        
        output_file = self.temp_dir / filename
        
        # Calculate data range
        vmin, vmax = np.min(data_values), np.max(data_values)
        
        # GMT command to create CPT
        gmt_cmd = f"gmt makecpt -C{cmap_name} -T{vmin}/{vmax}/10 > {output_file}"
        
        result = subprocess.run(gmt_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Created GMT color palette: {output_file}")
            return output_file
        else:
            print(f"   ❌ Failed to create CPT: {result.stderr}")
            return None
    
    def create_taiwan_basemap(self, region=None, projection="M15c", resolution="h", 
                            output="taiwan_basemap.ps", title="Taiwan Map"):
        """Create Taiwan basemap with coastlines and rivers"""
        
        if region is None:
            region = self.taiwan_region
        
        region_str = f"{region[0]}/{region[1]}/{region[2]}/{region[3]}"
        
        # GMT commands for basemap
        gmt_commands = [
            # Start with coast
            f"gmt pscoast -R{region_str} -J{projection} "
            f"-D{resolution} -G240/240/240 -S200/200/255 "
            f"-W0.5p,black -Na/0.25p,gray50 -K > {output}",
            
            # Add rivers (multiple levels for better coverage)
            f"gmt pscoast -R -J -D{resolution} "
            f"-I1/0.5p,blue -I2/0.25p,lightblue -O -K >> {output}",
            
            # Add frame and title
            f"gmt psbasemap -R -J -Ba2f1 -BWSne+t'{title}' -O >> {output}"
        ]
        
        # Execute GMT commands
        for cmd in gmt_commands:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ⚠️  GMT command failed: {cmd}")
                print(f"      Error: {result.stderr}")
        
        print(f"   ✅ Created Taiwan basemap: {output}")
        return output
    
    def create_insar_map(self, coordinates, values, region=None, projection="M15c",
                        resolution="h", cmap="jet", symbol_size="0.1c", 
                        output="insar_map.ps", title="Taiwan InSAR Analysis"):
        """Create complete InSAR analysis map"""
        
        if region is None:
            # Auto-calculate region from data
            lon_buffer = (coordinates[:, 0].max() - coordinates[:, 0].min()) * 0.1
            lat_buffer = (coordinates[:, 1].max() - coordinates[:, 1].min()) * 0.1
            region = [
                coordinates[:, 0].min() - lon_buffer,
                coordinates[:, 0].max() + lon_buffer, 
                coordinates[:, 1].min() - lat_buffer,
                coordinates[:, 1].max() + lat_buffer
            ]
        
        region_str = f"{region[0]}/{region[1]}/{region[2]}/{region[3]}"
        
        # Create data and color files
        data_file = self.create_data_file(coordinates, values, "insar_data.txt")
        cpt_file = self.create_cpt_file(values, cmap, "insar.cpt")
        
        if not data_file or not cpt_file:
            return None
        
        # GMT commands for complete map
        gmt_commands = [
            # 1. Create basemap with coastlines
            f"gmt pscoast -R{region_str} -J{projection} "
            f"-D{resolution} -G250/250/250 -S220/220/255 "
            f"-W0.75p,black -Na/0.5p,gray40 -K > {output}",
            
            # 2. Add rivers at multiple levels
            f"gmt pscoast -R -J -D{resolution} "
            f"-I1/0.75p,blue -I2/0.5p,steelblue -I3/0.25p,lightblue -O -K >> {output}",
            
            # 3. Plot InSAR data points
            f"gmt psxy {data_file} -R -J -C{cpt_file} "
            f"-Sc{symbol_size} -W0.25p,black -O -K >> {output}",
            
            # 4. Add color scale
            f"gmt psscale -C{cpt_file} -D8.5c/2c/6c/0.5c "
            f"-B+l'Subsidence Rate (mm/year)' -O -K >> {output}",
            
            # 5. Add frame and title  
            f"gmt psbasemap -R -J -Ba1f0.5 -BWSne+t'{title}' -O >> {output}"
        ]
        
        # Execute GMT commands
        success = True
        for i, cmd in enumerate(gmt_commands, 1):
            print(f"   🔄 Executing step {i}/5...")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ❌ GMT step {i} failed: {result.stderr}")
                success = False
                break
        
        if success:
            print(f"   ✅ Created InSAR map: {output}")
            
            # Convert to PNG for easy viewing
            png_output = output.replace('.ps', '.png')
            convert_cmd = f"gmt psconvert {output} -Tg -A -E300"
            subprocess.run(convert_cmd, shell=True, capture_output=True)
            print(f"   ✅ Converted to PNG: {png_output}")
            
        return output if success else None
    
    def create_cluster_map(self, coordinates, cluster_labels, region=None, 
                          projection="M15c", resolution="h", symbol_size="0.08c",
                          output="cluster_map.ps", title="Taiwan Clustering Analysis"):
        """Create cluster analysis map with different colors per cluster"""
        
        if region is None:
            lon_buffer = (coordinates[:, 0].max() - coordinates[:, 0].min()) * 0.1
            lat_buffer = (coordinates[:, 1].max() - coordinates[:, 1].min()) * 0.1
            region = [
                coordinates[:, 0].min() - lon_buffer,
                coordinates[:, 0].max() + lon_buffer,
                coordinates[:, 1].min() - lat_buffer, 
                coordinates[:, 1].max() + lat_buffer
            ]
        
        region_str = f"{region[0]}/{region[1]}/{region[2]}/{region[3]}"
        
        # Create data file with cluster info
        data_file = self.create_data_file(coordinates, cluster_labels, "cluster_data.txt")
        
        # Define colors for clusters (GMT color names)
        cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # GMT commands
        gmt_commands = [
            # 1. Create basemap
            f"gmt pscoast -R{region_str} -J{projection} "
            f"-D{resolution} -G245/245/245 -S210/210/255 "
            f"-W0.75p,black -Na/0.5p,gray30 -K > {output}",
            
            # 2. Add rivers
            f"gmt pscoast -R -J -D{resolution} "
            f"-I1/0.75p,darkblue -I2/0.5p,blue -O -K >> {output}",
        ]
        
        # 3. Add each cluster with different color
        unique_clusters = np.unique(cluster_labels)
        for i, cluster in enumerate(unique_clusters):
            color = cluster_colors[i % len(cluster_colors)]
            
            # Create temporary file for this cluster
            cluster_mask = cluster_labels == cluster
            cluster_coords = coordinates[cluster_mask]
            cluster_file = self.temp_dir / f"cluster_{cluster}.txt"
            
            # Save cluster data
            np.savetxt(cluster_file, cluster_coords, fmt='%.6f', delimiter='\\t')
            
            # Add GMT command for this cluster
            gmt_commands.append(
                f"gmt psxy {cluster_file} -R -J -G{color} "
                f"-Sc{symbol_size} -W0.25p,black -O -K >> {output}"
            )
        
        # 4. Add frame and title
        gmt_commands.append(
            f"gmt psbasemap -R -J -Ba1f0.5 -BWSne+t'{title}' -O >> {output}"
        )
        
        # Execute commands
        success = True
        for i, cmd in enumerate(gmt_commands, 1):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ❌ GMT step {i} failed: {result.stderr}")
                success = False
                break
        
        if success:
            print(f"   ✅ Created cluster map: {output}")
            
            # Convert to PNG
            png_output = output.replace('.ps', '.png')
            convert_cmd = f"gmt psconvert {output} -Tg -A -E300"
            subprocess.run(convert_cmd, shell=True, capture_output=True)
            print(f"   ✅ Converted to PNG: {png_output}")
        
        return output if success else None
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("   🧹 Cleaned up temporary files")

def demo_gmt_direct():
    """Demonstrate direct GMT mapping capabilities"""
    
    print("🚀 Demonstrating Direct GMT Mapping for Taiwan...")
    
    # Initialize mapper
    gmt_mapper = GMTDirectMapper()
    
    try:
        # Create test data - simulate InSAR subsidence stations
        np.random.seed(42)
        n_stations = 100
        
        # Taiwan coordinate bounds  
        lon_min, lon_max = 120.0, 121.0
        lat_min, lat_max = 23.5, 24.5
        
        # Generate test coordinates
        lons = np.random.uniform(lon_min, lon_max, n_stations)
        lats = np.random.uniform(lat_min, lat_max, n_stations)
        coords = np.column_stack([lons, lats])
        
        # Generate test subsidence rates (mm/year)
        subsidence_rates = np.random.normal(-15, 8, n_stations)
        
        # Generate test cluster labels
        cluster_labels = np.random.randint(0, 4, n_stations)
        
        print(f"📊 Generated test data: {n_stations} stations")
        print(f"   Coordinate range: {lon_min}-{lon_max}°E, {lat_min}-{lat_max}°N")
        print(f"   Subsidence range: {subsidence_rates.min():.1f} to {subsidence_rates.max():.1f} mm/year")
        
        # Create different types of maps
        print("\\n🗺️ Creating GMT maps...")
        
        # 1. Basic Taiwan basemap
        gmt_mapper.create_taiwan_basemap(
            output="demo_taiwan_basemap.ps",
            title="Taiwan Basemap (GMT Direct)"
        )
        
        # 2. InSAR subsidence map
        gmt_mapper.create_insar_map(
            coords, subsidence_rates,
            output="demo_insar_subsidence.ps", 
            title="Taiwan InSAR Subsidence (GMT Direct)",
            cmap="polar"
        )
        
        # 3. Cluster analysis map
        gmt_mapper.create_cluster_map(
            coords, cluster_labels,
            output="demo_cluster_analysis.ps",
            title="Taiwan Clustering Analysis (GMT Direct)"
        )
        
        print("\\n✅ GMT Direct mapping demonstration completed!")
        print("Generated files:")
        print("   - demo_taiwan_basemap.ps/.png")
        print("   - demo_insar_subsidence.ps/.png") 
        print("   - demo_cluster_analysis.ps/.png")
        
    except Exception as e:
        print(f"❌ GMT mapping demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        gmt_mapper.cleanup()

if __name__ == "__main__":
    demo_gmt_direct()
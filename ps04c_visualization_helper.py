#!/usr/bin/env python3
"""
Improved Motif Visualization Approaches for PS04C
=================================================

Multiple enhanced visualization strategies for better motif location plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd

class ImprovedMotifVisualizer:
    """Enhanced motif visualization strategies"""
    
    def __init__(self, coordinates, discovered_motifs):
        self.coordinates = coordinates
        self.discovered_motifs = discovered_motifs
        self.fig_size = (24, 18)
        
    def approach1_spatial_clustering(self, ax, clustering_distance_km=5.0):
        """
        Approach 1: Spatial Clustering of Nearby Motifs
        Groups nearby motifs and shows cluster centroids with size proportional to count
        """
        print("📍 Approach 1: Spatial Clustering Visualization")
        
        # Collect all motif locations with metadata
        motif_data = []
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                for station_idx in motif.station_indices:
                    coord = self.coordinates[station_idx]
                    motif_data.append({
                        'lon': coord[0], 
                        'lat': coord[1],
                        'window_size': window_size,
                        'pattern_type': motif.temporal_context,
                        'significance': motif.significance_score,
                        'station_idx': station_idx
                    })
        
        if not motif_data:
            return
            
        df = pd.DataFrame(motif_data)
        
        # Convert distance to degrees (approximate for Taiwan)
        clustering_distance_deg = clustering_distance_km / 111.0
        
        # Spatial clustering for each pattern type
        pattern_colors = {
            'irrigation_season': '#2E8B57',
            'monsoon_pattern': '#4169E1', 
            'dry_season': '#FF8C00',
            'annual_cycle': '#9932CC',
            'winter_pattern': '#00CED1',
            'short_term_fluctuation': '#FFD700'
        }
        
        for pattern_type in df['pattern_type'].unique():
            pattern_df = df[df['pattern_type'] == pattern_type]
            
            if len(pattern_df) < 2:
                # Single point, plot directly
                row = pattern_df.iloc[0]
                ax.scatter(row['lon'], row['lat'], 
                          c=pattern_colors.get(pattern_type, 'red'),
                          s=80 + row['significance']*20,
                          alpha=0.8, edgecolor='black', linewidth=1,
                          label=f"{pattern_type.replace('_', ' ').title()}")
                continue
            
            # Hierarchical clustering based on spatial distance
            coords = pattern_df[['lon', 'lat']].values
            distances = pdist(coords)
            
            if len(distances) > 0:
                linkage_matrix = linkage(distances, method='ward')
                clusters = fcluster(linkage_matrix, clustering_distance_deg, criterion='distance')
                pattern_df_copy = pattern_df.copy()
                pattern_df_copy['cluster'] = clusters
                
                # Plot cluster centroids
                for cluster_id in np.unique(clusters):
                    cluster_points = pattern_df_copy[pattern_df_copy['cluster'] == cluster_id]
                    
                    # Centroid location
                    centroid_lon = cluster_points['lon'].mean()
                    centroid_lat = cluster_points['lat'].mean()
                    
                    # Aggregate properties
                    avg_significance = cluster_points['significance'].mean()
                    count = len(cluster_points)
                    
                    # Symbol size based on count and significance
                    symbol_size = 60 + count*15 + avg_significance*10
                    
                    # Plot centroid with count annotation
                    scatter = ax.scatter(centroid_lon, centroid_lat,
                                       c=pattern_colors.get(pattern_type, 'red'),
                                       s=symbol_size, alpha=0.7,
                                       edgecolor='black', linewidth=1.5,
                                       label=f"{pattern_type.replace('_', ' ').title()}" if cluster_id == np.unique(clusters)[0] else "")
                    
                    # Add count annotation
                    if count > 1:
                        ax.annotate(f'{count}', (centroid_lon, centroid_lat), 
                                   xytext=(0, 0), textcoords='offset points',
                                   ha='center', va='center', fontsize=8, fontweight='bold',
                                   color='white')
        
        ax.set_title('Spatial Clustering of Motifs\n(Numbers show motif count per cluster)', 
                    fontsize=14, fontweight='bold')
        
    def approach2_hexbin_density(self, ax):
        """
        Approach 2: Hexagonal Binning for Pattern Density
        Shows spatial density using hexagonal bins with pattern-specific coloring
        """
        print("📍 Approach 2: Hexagonal Density Visualization")
        
        # Collect all motif locations
        all_lons, all_lats, all_significance = [], [], []
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                for station_idx in motif.station_indices:
                    coord = self.coordinates[station_idx]
                    all_lons.append(coord[0])
                    all_lats.append(coord[1])
                    all_significance.append(motif.significance_score)
        
        if not all_lons:
            return
            
        # Create hexbin plot with significance weighting
        hexbin = ax.hexbin(all_lons, all_lats, C=all_significance, 
                          gridsize=15, cmap='YlOrRd', alpha=0.8,
                          reduce_C_function=np.mean)
        
        # Add colorbar
        cbar = plt.colorbar(hexbin, ax=ax, shrink=0.8)
        cbar.set_label('Average Pattern Significance', fontsize=10)
        
        # Overlay individual high-significance motifs
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                if motif.significance_score > 4.0:  # High significance threshold
                    for station_idx in motif.station_indices:
                        coord = self.coordinates[station_idx]
                        ax.scatter(coord[0], coord[1], 
                                  c='red', s=50, alpha=0.9,
                                  edgecolor='white', linewidth=1,
                                  marker='*')
        
        ax.set_title('Hexagonal Density Map\n(Stars = High significance patterns)', 
                    fontsize=14, fontweight='bold')
    
    def approach3_multi_symbol_encoding(self, ax):
        """
        Approach 3: Multi-dimensional Symbol Encoding
        Uses shape, size, color, and transparency to encode multiple dimensions
        """
        print("📍 Approach 3: Multi-dimensional Symbol Encoding")
        
        # Symbol mapping
        pattern_colors = {
            'irrigation_season': '#2E8B57',
            'monsoon_pattern': '#4169E1',
            'dry_season': '#FF8C00', 
            'annual_cycle': '#9932CC',
            'winter_pattern': '#00CED1',
            'short_term_fluctuation': '#FFD700'
        }
        
        window_markers = {30: 'o', 90: 's', 365: '^'}  # Circle, square, triangle
        
        # Create legend elements
        legend_elements = []
        
        for window_size, motifs in self.discovered_motifs.items():
            marker = window_markers.get(window_size, 'o')
            
            for motif in motifs:
                pattern_type = motif.temporal_context
                color = pattern_colors.get(pattern_type, 'red')
                
                for station_idx in motif.station_indices:
                    coord = self.coordinates[station_idx]
                    
                    # Encode multiple dimensions:
                    # - Color: Pattern type
                    # - Shape: Window size  
                    # - Size: Significance score
                    # - Alpha: Relative importance
                    
                    size = 40 + motif.significance_score * 20
                    alpha = 0.5 + (motif.significance_score / 5.0) * 0.4  # 0.5 to 0.9
                    
                    ax.scatter(coord[0], coord[1],
                              c=color, marker=marker, s=size, alpha=alpha,
                              edgecolor='black', linewidth=0.8)
        
        # Create custom legend
        import matplotlib.lines as mlines
        
        # Pattern type legend
        for pattern_type, color in pattern_colors.items():
            if any(motif.temporal_context == pattern_type 
                   for motifs in self.discovered_motifs.values() 
                   for motif in motifs):
                legend_elements.append(
                    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                 markersize=8, label=pattern_type.replace('_', ' ').title())
                )
        
        # Window size legend  
        legend_elements.append(mlines.Line2D([0], [0], marker='', color='w', label='Window Size:'))
        for window_size, marker in window_markers.items():
            legend_elements.append(
                mlines.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                             markersize=8, label=f'  {window_size} days')
            )
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_title('Multi-dimensional Encoding\n(Shape=Window, Color=Pattern, Size=Significance)', 
                    fontsize=14, fontweight='bold')
    
    def approach4_network_connectivity(self, ax, similarity_threshold=0.8):
        """
        Approach 4: Network Visualization
        Shows connections between stations with similar patterns
        """
        print("📍 Approach 4: Pattern Similarity Network")
        
        # Collect motif data with patterns
        motif_info = []
        for window_size, motifs in self.discovered_motifs.items():
            for motif in motifs:
                for station_idx in motif.station_indices:
                    coord = self.coordinates[station_idx]
                    motif_info.append({
                        'coord': coord,
                        'pattern': motif.pattern,
                        'type': motif.temporal_context,
                        'window_size': window_size,
                        'significance': motif.significance_score
                    })
        
        if len(motif_info) < 2:
            return
            
        # Calculate pattern similarities
        from scipy.spatial.distance import correlation
        
        pattern_colors = {
            'irrigation_season': '#2E8B57',
            'monsoon_pattern': '#4169E1',
            'dry_season': '#FF8C00',
            'annual_cycle': '#9932CC', 
            'winter_pattern': '#00CED1',
            'short_term_fluctuation': '#FFD700'
        }
        
        # Plot nodes (motifs)
        for i, info in enumerate(motif_info):
            color = pattern_colors.get(info['type'], 'red')
            size = 50 + info['significance'] * 15
            
            ax.scatter(info['coord'][0], info['coord'][1],
                      c=color, s=size, alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        # Draw connections between similar patterns
        for i in range(len(motif_info)):
            for j in range(i+1, len(motif_info)):
                info1, info2 = motif_info[i], motif_info[j]
                
                # Only connect same window size
                if info1['window_size'] != info2['window_size']:
                    continue
                    
                # Calculate pattern similarity
                try:
                    similarity = 1 - correlation(info1['pattern'], info2['pattern'])
                    if similarity > similarity_threshold:
                        # Draw connection line
                        coord1, coord2 = info1['coord'], info2['coord']
                        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]],
                               'k-', alpha=0.3, linewidth=1)
                except:
                    continue
        
        ax.set_title(f'Pattern Similarity Network\n(Lines connect similar patterns, threshold={similarity_threshold})', 
                    fontsize=14, fontweight='bold')
    
    def create_comparison_figure(self):
        """Create figure comparing all visualization approaches"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle('Improved Motif Location Visualization Approaches', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Apply each approach
        self.approach1_spatial_clustering(axes[0, 0])
        self.approach2_hexbin_density(axes[0, 1])  
        self.approach3_multi_symbol_encoding(axes[1, 0])
        self.approach4_network_connectivity(axes[1, 1])
        
        # Add background stations to all plots
        for ax in axes.flat:
            ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1],
                      c='lightgray', s=1, alpha=0.3, zorder=0)
            ax.set_xlabel('Longitude (°E)', fontsize=12)
            ax.set_ylabel('Latitude (°N)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        return fig

def demonstrate_improvements():
    """Demonstrate all improved visualization approaches"""
    
    print("🎨 Improved Motif Visualization Approaches")
    print("=" * 60)
    print()
    
    approaches = {
        1: {
            'name': 'Spatial Clustering',
            'description': 'Groups nearby motifs into clusters with centroid markers',
            'advantages': [
                '✅ Eliminates overlapping symbols',
                '✅ Shows spatial density clearly', 
                '✅ Aggregates counts for better interpretation',
                '✅ Handles multiple motifs per location'
            ],
            'best_for': 'Dense motif distributions with spatial clustering'
        },
        
        2: {
            'name': 'Hexagonal Density',
            'description': 'Uses hexagonal binning to show pattern density',
            'advantages': [
                '✅ Smooth density visualization',
                '✅ Color-encoded significance levels',
                '✅ Statistical aggregation over space',
                '✅ Highlights hotspots clearly'
            ],
            'best_for': 'Understanding spatial distribution patterns'
        },
        
        3: {
            'name': 'Multi-dimensional Encoding',
            'description': 'Uses shape, size, color, alpha to encode multiple variables',
            'advantages': [
                '✅ Encodes 4+ dimensions simultaneously',
                '✅ Clear visual separation of window sizes',
                '✅ Intuitive size-significance mapping',
                '✅ Comprehensive legend system'
            ],
            'best_for': 'Detailed analysis requiring multiple data dimensions'
        },
        
        4: {
            'name': 'Network Connectivity', 
            'description': 'Shows connections between stations with similar patterns',
            'advantages': [
                '✅ Reveals pattern correlations',
                '✅ Identifies connected regions',
                '✅ Shows geological process continuity',
                '✅ Network-based clustering'
            ],
            'best_for': 'Understanding spatial pattern relationships'
        }
    }
    
    for num, approach in approaches.items():
        print(f"**Approach {num}: {approach['name']}**")
        print(f"   {approach['description']}")
        print(f"   Best for: {approach['best_for']}")
        print()
        for advantage in approach['advantages']:
            print(f"   {advantage}")
        print()
        print("-" * 50)
        print()

if __name__ == "__main__":
    demonstrate_improvements()
#!/usr/bin/env python3
"""
PS08 V2: Pure Grain Size Susceptibility Index Analysis
======================================================

Creates a grain-size susceptibility index WITHOUT including observed subsidence rates,
allowing for true correlation analysis between geological properties and deformation.

Key Changes in V2:
- Removes observed subsidence rates from susceptibility calculation
- Creates pure grain-size based compressibility index
- Enables unbiased correlation analysis
- Generates ps08_fig05_v2_subsidence_vs_grain_size.png
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats
import seaborn as sns

class GrainSizeSusceptibilityV2:
    """Pure grain-size susceptibility index without subsidence rate bias"""
    
    def __init__(self, results_dir='data/processed/ps08_geological'):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load existing PS08 results
        self.interpolated_geology = None
        self.insar_data = None
        self.load_ps08_results()
    
    def load_ps08_results(self):
        """Load existing PS08 interpolated geology and InSAR data"""
        print("🔄 Loading PS08 geological integration results...")
        
        # Load interpolated geology
        interp_file = self.results_dir / 'interpolation_results' / 'interpolated_geology.json'
        if interp_file.exists():
            with open(interp_file, 'r') as f:
                self.interpolated_geology = json.load(f)
            print(f"   ✅ Loaded interpolated geology data")
        else:
            raise FileNotFoundError(f"PS08 results not found at {interp_file}")
        
        # Load InSAR data from PS00 preprocessing
        insar_file = Path('data/processed/ps00_preprocessed_data.npz')
        if insar_file.exists():
            with np.load(insar_file) as data:
                self.insar_data = {
                    'coordinates': data['coordinates'],
                    'subsidence_rates': data['subsidence_rates']
                }
            print(f"   ✅ Loaded InSAR subsidence rates for {len(self.insar_data['subsidence_rates'])} stations")
        else:
            raise FileNotFoundError(f"InSAR data not found at {insar_file}")
    
    def calculate_pure_grain_size_index(self):
        """Calculate grain-size susceptibility index WITHOUT subsidence rates"""
        print("🔄 Calculating pure grain-size susceptibility index v2...")
        
        # Get geological data (use best interpolation method)
        geology = self.interpolated_geology['best']
        
        fine_fraction = np.array(geology['fine_fraction']['values'])
        sand_fraction = np.array(geology['sand_fraction']['values'])  
        coarse_fraction = np.array(geology['coarse_fraction']['values'])
        
        # Filter valid geological data
        valid_mask = ~(np.isnan(fine_fraction) | np.isnan(sand_fraction) | np.isnan(coarse_fraction))
        n_valid = np.sum(valid_mask)
        
        if n_valid < 10:
            raise ValueError(f"Insufficient valid geological data: {n_valid} stations")
        
        print(f"   📊 Processing {n_valid} stations with valid geological data")
        
        # Extract valid data
        valid_fine = fine_fraction[valid_mask]
        valid_sand = sand_fraction[valid_mask]
        valid_coarse = coarse_fraction[valid_mask]
        valid_coords = self.insar_data['coordinates'][valid_mask]
        valid_subsidence_rates = self.insar_data['subsidence_rates'][valid_mask]
        
        # === V2 PURE GRAIN-SIZE INDEX ===
        
        # Component 1: Compressibility Index (grain-size weighted)
        # Values are already in fraction format (0-1), not percentages
        compressibility_index = (
            valid_coarse * 0.1 +     # Gravel/cobbles: very low compressibility
            valid_sand * 0.3 +       # Sand: moderate compressibility  
            valid_fine * 0.8         # Silt/clay: high compressibility
        )
        
        # Component 2: Clay Content (already normalized 0-1)
        clay_content = valid_fine  # Already in fraction format
        
        # Component 3: Grain-size variability (geological heterogeneity)
        # Lower variability = more uniform = potentially more problematic
        grain_size_variance = np.var([valid_coarse, valid_sand, valid_fine], axis=0) / 10000.0  # Scaled variance
        uniformity_index = 1.0 / (1.0 + grain_size_variance)  # Higher = more uniform
        
        # === V2 PURE GRAIN-SIZE SUSCEPTIBILITY INDEX ===
        # (No subsidence rate component!)
        grain_size_susceptibility_v2 = (
            0.60 * clay_content +           # Clay content (60% - primary factor)
            0.40 * compressibility_index   # Compressibility (40% - secondary factor)
        )
        
        # Classification
        susceptibility_classes = []
        for index in grain_size_susceptibility_v2:
            if index > 0.7:
                sclass = 'very_high'
            elif index > 0.55:
                sclass = 'high'
            elif index > 0.4:
                sclass = 'moderate'
            elif index > 0.25:
                sclass = 'low'
            else:
                sclass = 'very_low'
            susceptibility_classes.append(sclass)
        
        # Summary statistics
        class_counts = {sclass: susceptibility_classes.count(sclass) 
                       for sclass in set(susceptibility_classes)}
        
        print(f"   📊 V2 Index Statistics:")
        print(f"      Mean: {np.mean(grain_size_susceptibility_v2):.3f}")
        print(f"      Std:  {np.std(grain_size_susceptibility_v2):.3f}")
        print(f"      Range: [{np.min(grain_size_susceptibility_v2):.3f}, {np.max(grain_size_susceptibility_v2):.3f}]")
        
        print(f"   📊 V2 Classification Distribution:")
        for sclass, count in class_counts.items():
            pct = (count / len(susceptibility_classes)) * 100
            print(f"      {sclass}: {count} stations ({pct:.1f}%)")
        
        # Store results
        self.v2_results = {
            'coordinates': valid_coords,
            'subsidence_rates': valid_subsidence_rates,
            'geological_data': {
                'fine_fraction': valid_fine,
                'sand_fraction': valid_sand, 
                'coarse_fraction': valid_coarse
            },
            'compressibility_index': compressibility_index,
            'clay_content': clay_content,
            'grain_size_susceptibility_v2': grain_size_susceptibility_v2,
            'susceptibility_classes': susceptibility_classes,
            'class_statistics': class_counts,
            'n_stations': n_valid
        }
        
        print("   ✅ Pure grain-size susceptibility index v2 calculated")
        return True
    
    def create_comparison_visualization(self):
        """Create comprehensive comparison visualization"""
        print("🎨 Creating grain-size susceptibility v2 visualization...")
        
        if not hasattr(self, 'v2_results'):
            raise ValueError("Must calculate v2 index first")
        
        results = self.v2_results
        coords = results['coordinates']
        subsidence_rates = results['subsidence_rates']
        grain_size_index_v2 = results['grain_size_susceptibility_v2']
        compressibility_index = results['compressibility_index']
        clay_content = results['clay_content']
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # === Panel 1: Scatter - Grain Size Index v2 vs Subsidence Rate ===
        ax1 = plt.subplot(3, 3, 1)
        
        scatter1 = ax1.scatter(grain_size_index_v2, np.abs(subsidence_rates), 
                              c=clay_content*100, cmap='YlOrRd', alpha=0.7, s=40)
        
        # Correlation analysis
        r_v2, p_v2 = stats.pearsonr(grain_size_index_v2, np.abs(subsidence_rates))
        
        # Add regression line
        z = np.polyfit(grain_size_index_v2, np.abs(subsidence_rates), 1)
        p = np.poly1d(z)
        x_reg = np.linspace(grain_size_index_v2.min(), grain_size_index_v2.max(), 100)
        ax1.plot(x_reg, p(x_reg), "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Grain Size Susceptibility Index v2')
        ax1.set_ylabel('Subsidence Rate (mm/year)')
        ax1.set_title(f'Grain-Size Susceptibility Index v2\nvs Subsidence Rate\nr = {r_v2:.3f}, p = {p_v2:.3e}')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter1, ax=ax1, label='Clay Content (%)')
        
        # === Panel 2: Scatter - Compressibility Index vs Subsidence Rate ===
        ax2 = plt.subplot(3, 3, 2)
        
        scatter2 = ax2.scatter(compressibility_index, np.abs(subsidence_rates),
                              c=grain_size_index_v2, cmap='viridis', alpha=0.7, s=40)
        
        r_comp, p_comp = stats.pearsonr(compressibility_index, np.abs(subsidence_rates))
        
        # Add regression line
        z2 = np.polyfit(compressibility_index, np.abs(subsidence_rates), 1)
        p2 = np.poly1d(z2)
        x_reg2 = np.linspace(compressibility_index.min(), compressibility_index.max(), 100)
        ax2.plot(x_reg2, p2(x_reg2), "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Compressibility Index')
        ax2.set_ylabel('Subsidence Rate (mm/year)')
        ax2.set_title(f'Compressibility vs Subsidence\nr = {r_comp:.3f}, p = {p_comp:.3e}')
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(scatter2, ax=ax2, label='Grain Size Index v2')
        
        # === Panel 3: Scatter - Clay Content vs Subsidence Rate ===
        ax3 = plt.subplot(3, 3, 3)
        
        scatter3 = ax3.scatter(clay_content*100, np.abs(subsidence_rates),
                              c=compressibility_index, cmap='plasma', alpha=0.7, s=40)
        
        r_clay, p_clay = stats.pearsonr(clay_content*100, np.abs(subsidence_rates))
        
        # Add regression line
        z3 = np.polyfit(clay_content*100, np.abs(subsidence_rates), 1)
        p3 = np.poly1d(z3)
        x_reg3 = np.linspace((clay_content*100).min(), (clay_content*100).max(), 100)
        ax3.plot(x_reg3, p3(x_reg3), "r--", alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Clay Content (%)')
        ax3.set_ylabel('Subsidence Rate (mm/year)')
        ax3.set_title(f'Clay Content vs Subsidence\nr = {r_clay:.3f}, p = {p_clay:.3e}')
        ax3.grid(True, alpha=0.3)
        
        plt.colorbar(scatter3, ax=ax3, label='Compressibility Index')
        
        # === Panel 4: Geographic Distribution - Grain Size Index v2 ===
        ax4 = plt.subplot(3, 3, 4)
        
        scatter4 = ax4.scatter(coords[:, 0], coords[:, 1],
                              c=grain_size_index_v2, cmap='Reds', s=30, alpha=0.8)
        ax4.set_xlabel('Longitude (°E)')
        ax4.set_ylabel('Latitude (°N)')
        ax4.set_title('V2: Grain Size Susceptibility Index')
        
        plt.colorbar(scatter4, ax=ax4, label='Susceptibility Index v2')
        
        # === Panel 5: Geographic Distribution - Subsidence Rates ===
        ax5 = plt.subplot(3, 3, 5)
        
        scatter5 = ax5.scatter(coords[:, 0], coords[:, 1],
                              c=np.abs(subsidence_rates), cmap='RdBu_r', s=30, alpha=0.8)
        ax5.set_xlabel('Longitude (°E)')
        ax5.set_ylabel('Latitude (°N)')
        ax5.set_title('Observed Subsidence Rates')
        
        plt.colorbar(scatter5, ax=ax5, label='Subsidence Rate (mm/year)')
        
        # === Panel 6: Histogram - Index Distribution ===
        ax6 = plt.subplot(3, 3, 6)
        
        ax6.hist(grain_size_index_v2, bins=30, alpha=0.7, color='red', label='V2 Index')
        ax6.axvline(np.mean(grain_size_index_v2), color='black', linestyle='--', 
                   label=f'Mean = {np.mean(grain_size_index_v2):.3f}')
        ax6.set_xlabel('Grain Size Susceptibility Index v2')
        ax6.set_ylabel('Frequency')
        ax6.set_title('V2 Index Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # === Panel 7: Box Plot - Susceptibility by Class ===
        ax7 = plt.subplot(3, 3, 7)
        
        classes = results['susceptibility_classes']
        class_names = ['very_low', 'low', 'moderate', 'high', 'very_high']
        class_data = []
        class_labels = []
        
        for class_name in class_names:
            class_indices = [i for i, c in enumerate(classes) if c == class_name]
            if len(class_indices) > 0:
                class_subsidence = np.abs(subsidence_rates)[class_indices]
                class_data.append(class_subsidence)
                class_labels.append(f'{class_name}\n(n={len(class_indices)})')
        
        if class_data:
            bp = ax7.boxplot(class_data, labels=class_labels, patch_artist=True)
            
            # Color boxes
            colors = ['green', 'yellow', 'orange', 'red', 'darkred']
            for patch, color in zip(bp['boxes'], colors[:len(class_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax7.set_ylabel('Subsidence Rate (mm/year)')
        ax7.set_title('Subsidence Rate by V2 Susceptibility Class')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # === Panel 8: Correlation Matrix ===
        ax8 = plt.subplot(3, 3, 8)
        
        # Create correlation matrix
        correlation_data = {
            'Grain Size Index V2': grain_size_index_v2,
            'Compressibility Index': compressibility_index,
            'Clay Content (%)': clay_content * 100,
            'Fine Fraction (%)': results['geological_data']['fine_fraction'],
            'Sand Fraction (%)': results['geological_data']['sand_fraction'],
            'Coarse Fraction (%)': results['geological_data']['coarse_fraction'],
            'Subsidence Rate': np.abs(subsidence_rates)
        }
        
        import pandas as pd
        correlation_df = pd.DataFrame(correlation_data)
        correlation_matrix = correlation_df.corr()
        
        # Plot correlation heatmap
        im = ax8.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add text annotations
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix.columns)):
                text = ax8.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        ax8.set_xticks(range(len(correlation_matrix.columns)))
        ax8.set_yticks(range(len(correlation_matrix)))
        ax8.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax8.set_yticklabels(correlation_matrix.index)
        ax8.set_title('Correlation Matrix')
        
        plt.colorbar(im, ax=ax8, label='Correlation Coefficient')
        
        # === Panel 9: Summary Statistics ===
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary text
        summary_text = f"""
GRAIN SIZE SUSCEPTIBILITY V2 SUMMARY

Index Composition:
• Clay Content: 60% weight
• Compressibility: 40% weight
• NO subsidence rate bias

Key Statistics:
• Stations analyzed: {results['n_stations']}
• Mean index: {np.mean(grain_size_index_v2):.3f}
• Standard deviation: {np.std(grain_size_index_v2):.3f}
• Range: [{np.min(grain_size_index_v2):.3f}, {np.max(grain_size_index_v2):.3f}]

Correlations with Subsidence:
• V2 Grain Size Index: r = {r_v2:.3f}
• Compressibility Index: r = {r_comp:.3f}
• Clay Content: r = {r_clay:.3f}

Classification Distribution:"""
        
        # Add class distribution
        for sclass, count in results['class_statistics'].items():
            pct = (count / results['n_stations']) * 100
            summary_text += f"\n• {sclass}: {count} ({pct:.1f}%)"
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / 'ps08_fig02_v2_grain_size_susceptibility_index.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")
        print(f"   📊 Key Results:")
        print(f"      V2 Index vs Subsidence correlation: r = {r_v2:.3f} (p = {p_v2:.3e})")
        print(f"      Compressibility vs Subsidence: r = {r_comp:.3f} (p = {p_comp:.3e})")
        print(f"      Clay Content vs Subsidence: r = {r_clay:.3f} (p = {p_clay:.3e})")
        
        return True
    
    def save_results(self):
        """Save V2 results to JSON"""
        if not hasattr(self, 'v2_results'):
            raise ValueError("Must calculate v2 index first")
        
        # Prepare data for JSON serialization
        results_json = {
            'method': 'grain_size_susceptibility_v2',
            'description': 'Pure grain-size susceptibility without subsidence rate bias',
            'formula': {
                'components': {
                    'clay_content': 0.60,
                    'compressibility_index': 0.40
                },
                'compressibility_formula': {
                    'coarse_weight': 0.1,
                    'sand_weight': 0.3, 
                    'fine_weight': 0.8
                }
            },
            'statistics': {
                'n_stations': int(self.v2_results['n_stations']),
                'mean_index': float(np.mean(self.v2_results['grain_size_susceptibility_v2'])),
                'std_index': float(np.std(self.v2_results['grain_size_susceptibility_v2'])),
                'min_index': float(np.min(self.v2_results['grain_size_susceptibility_v2'])),
                'max_index': float(np.max(self.v2_results['grain_size_susceptibility_v2']))
            },
            'classification_distribution': self.v2_results['class_statistics']
        }
        
        # Save to file
        results_file = self.results_dir / 'grain_size_susceptibility_v2.json'
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"   💾 Saved V2 results: {results_file}")
        return True

def main():
    """Run grain-size susceptibility v2 analysis"""
    print("🔍 PS08 V2: Pure Grain Size Susceptibility Index Analysis")
    print("=" * 80)
    
    try:
        # Initialize analysis
        analyzer = GrainSizeSusceptibilityV2()
        
        # Calculate pure grain-size index
        analyzer.calculate_pure_grain_size_index()
        
        # Create visualization
        analyzer.create_comparison_visualization()
        
        # Save results
        analyzer.save_results()
        
        print("🎯 V2 Analysis Complete!")
        print("=" * 50)
        print("📊 Key Improvements:")
        print("   • Removed subsidence rate bias from index")
        print("   • Enabled true correlation analysis")
        print("   • Pure grain-size based susceptibility")
        print("🖼️  Generated: ps08_fig02_v2_grain_size_susceptibility_index.png")
        
    except Exception as e:
        print(f"❌ Error in V2 analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
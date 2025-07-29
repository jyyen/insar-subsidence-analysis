#!/usr/bin/env python3
"""
ps02c_08_pytorch_geographic_analysis.py: Geographic analysis and visualization
Chronological order: PS02C development timeline
Taiwan InSAR Subsidence Analysis Project
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import warnings
import time
from ps02c_pytorch_optimal import OptimalTaiwanInSARFitter, OptimalInSARSignalModel

warnings.filterwarnings('ignore')

def create_comprehensive_geographic_analysis(fitter, results, loss_history, correlation_history):
    """Create comprehensive geographic and temporal analysis"""
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # Extract data
    time_years = fitter.time_years.cpu().numpy()
    coordinates = fitter.coordinates.cpu().numpy()
    
    # 1. Geographic distribution of PS00 subsidence rates
    ax1 = plt.subplot(4, 6, 1)
    scatter1 = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c=results['original_rates'], s=60, 
                          cmap='RdYlBu_r', vmin=-30, vmax=15, alpha=0.8)
    plt.colorbar(scatter1, label='Rate (mm/year)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('PS00 Subsidence Rates\n(GPS-Corrected)')
    plt.grid(True, alpha=0.3)
    
    # 2. Geographic distribution of PyTorch fitted rates
    ax2 = plt.subplot(4, 6, 2)
    scatter2 = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c=results['fitted_trends'], s=60, 
                          cmap='RdYlBu_r', vmin=-30, vmax=15, alpha=0.8)
    plt.colorbar(scatter2, label='Rate (mm/year)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')  
    plt.title('PyTorch Fitted Rates\n(Should Match PS00)')
    plt.grid(True, alpha=0.3)
    
    # 3. Geographic distribution of rate differences
    ax3 = plt.subplot(4, 6, 3)
    rate_diff = results['fitted_trends'] - results['original_rates']
    scatter3 = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c=rate_diff, s=60, 
                          cmap='RdBu', vmin=-2, vmax=2, alpha=0.8)
    plt.colorbar(scatter3, label='Difference (mm/year)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Rate Differences\n(PyTorch - PS00)')
    plt.grid(True, alpha=0.3)
    
    # 4. Geographic distribution of signal correlation
    ax4 = plt.subplot(4, 6, 4)
    scatter4 = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c=results['correlations'], s=60, 
                          cmap='viridis', vmin=0, vmax=0.3, alpha=0.8)
    plt.colorbar(scatter4, label='Correlation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Signal Correlations\n(Time Series Fit Quality)')
    plt.grid(True, alpha=0.3)
    
    # 5. Geographic distribution of RMSE
    ax5 = plt.subplot(4, 6, 5)
    scatter5 = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                          c=results['rmse'], s=60, 
                          cmap='plasma_r', vmin=20, vmax=35, alpha=0.8)
    plt.colorbar(scatter5, label='RMSE (mm)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Fitting RMSE\n(Signal Residuals)')
    plt.grid(True, alpha=0.3)
    
    # 6. Training progress
    ax6 = plt.subplot(4, 6, 6)
    epochs = range(len(loss_history))
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(epochs, loss_history, 'b-', linewidth=2, label='Loss')
    line2 = ax6_twin.plot(epochs, correlation_history, 'r-', linewidth=2, label='Correlation')
    
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss', color='b')
    ax6_twin.set_ylabel('Correlation', color='r')
    ax6.set_title('Training Progress')
    ax6.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='center right')
    
    # 7-18. Time series with trend analysis (12 stations)
    # Select diverse stations for comprehensive analysis
    corr_sorted = np.argsort(results['correlations'])
    rmse_sorted = np.argsort(results['rmse'])
    rate_sorted = np.argsort(results['original_rates'])
    
    selected_stations = [
        corr_sorted[-1],      # Best correlation
        corr_sorted[-2],      # 2nd best correlation  
        rmse_sorted[0],       # Best RMSE
        rmse_sorted[1],       # 2nd best RMSE
        rate_sorted[0],       # Fastest subsiding
        rate_sorted[1],       # 2nd fastest subsiding
        rate_sorted[-1],      # Fastest uplifting
        rate_sorted[-2],      # 2nd fastest uplifting
        len(results['correlations'])//4,     # Lower quartile
        len(results['correlations'])//2,     # Median
        3*len(results['correlations'])//4,   # Upper quartile
        corr_sorted[0]        # Worst correlation
    ]
    
    for i, station_idx in enumerate(selected_stations):
        row = 1 + i // 6
        col = i % 6
        ax = plt.subplot(4, 6, 7 + i)
        
        # Extract data
        observed = results['displacement'][station_idx]
        predicted = results['predictions'][station_idx]
        
        # Calculate trend lines
        ps00_rate = results['original_rates'][station_idx]
        fitted_rate = results['fitted_trends'][station_idx]
        station_offset = results['fitted_offsets'][station_idx]
        
        # Generate trend lines
        ps00_trend = station_offset + ps00_rate * time_years
        fitted_trend = station_offset + fitted_rate * time_years
        
        # Plot time series
        plt.plot(time_years, observed, 'b-', linewidth=2.5, label='Observed', alpha=0.8)
        plt.plot(time_years, predicted, 'r--', linewidth=2, label='PyTorch Fit', alpha=0.9)
        
        # Plot trend lines
        plt.plot(time_years, ps00_trend, 'g:', linewidth=2, label=f'PS00 Trend ({ps00_rate:.1f})', alpha=0.8)
        plt.plot(time_years, fitted_trend, 'm:', linewidth=2, label=f'Fitted Trend ({fitted_rate:.1f})', alpha=0.8)
        
        # Metrics
        corr = results['correlations'][station_idx]
        rmse = results['rmse'][station_idx]
        coords = coordinates[station_idx]
        
        plt.xlabel('Time (years)')
        plt.ylabel('Displacement (mm)')
        plt.title(f'Station {station_idx} [{coords[0]:.3f}, {coords[1]:.3f}]\n'
                 f'R={corr:.3f}, RMSE={rmse:.1f}mm')
        plt.legend(fontsize=6, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Highlight special characteristics
        if station_idx == corr_sorted[-1]:
            ax.patch.set_facecolor('lightgreen')
            ax.patch.set_alpha(0.1)
        elif station_idx == corr_sorted[0]:
            ax.patch.set_facecolor('lightcoral')
            ax.patch.set_alpha(0.1)
        elif station_idx == rate_sorted[0]:
            ax.patch.set_facecolor('lightblue')
            ax.patch.set_alpha(0.1)
    
    plt.tight_layout()
    
    # Save comprehensive analysis
    output_file = Path("figures/ps02c_pytorch_comprehensive_geographic_analysis.png")
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comprehensive geographic analysis: {output_file}")
    plt.show()

def create_focused_subsidence_analysis(fitter, results):
    """Create focused analysis of subsidence patterns"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    coordinates = fitter.coordinates.cpu().numpy()
    time_years = fitter.time_years.cpu().numpy()
    
    # 1. PS00 vs PyTorch Rate Correlation with Perfect Fit Line
    axes[0,0].scatter(results['original_rates'], results['fitted_trends'], 
                     c=results['correlations'], s=60, cmap='viridis', alpha=0.7)
    
    # Perfect correlation line
    min_rate, max_rate = np.min(results['original_rates']), np.max(results['original_rates'])
    axes[0,0].plot([min_rate, max_rate], [min_rate, max_rate], 'r--', linewidth=2, alpha=0.8)
    
    rate_corr = np.corrcoef(results['original_rates'], results['fitted_trends'])[0, 1]
    axes[0,0].set_xlabel('PS00 Subsidence Rate (mm/year)')
    axes[0,0].set_ylabel('PyTorch Fitted Rate (mm/year)')
    axes[0,0].set_title(f'Rate Correlation: R={rate_corr:.6f}\n(Colors = Signal Correlation)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add colorbar
    scatter = axes[0,0].collections[0]
    plt.colorbar(scatter, ax=axes[0,0], label='Signal Correlation')
    
    # 2. Subsidence Rate Histogram
    axes[0,1].hist(results['original_rates'], bins=20, alpha=0.7, color='blue', 
                   edgecolor='black', label='PS00 Rates')
    axes[0,1].hist(results['fitted_trends'], bins=20, alpha=0.5, color='red', 
                   edgecolor='black', label='PyTorch Rates')
    axes[0,1].set_xlabel('Subsidence Rate (mm/year)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Rate Distribution Comparison')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Geographic Rate Distribution (Large View)
    scatter = axes[0,2].scatter(coordinates[:, 0], coordinates[:, 1], 
                               c=results['original_rates'], s=80, 
                               cmap='RdYlBu_r', vmin=-30, vmax=15, alpha=0.8)
    
    # Add contour lines for rate zones
    from scipy.interpolate import griddata
    
    # Create grid for interpolation
    lon_grid = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 50)
    lat_grid = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 50)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Interpolate rates
    rate_interp = griddata(coordinates, results['original_rates'], 
                          (lon_mesh, lat_mesh), method='cubic', fill_value=0)
    
    # Add contour lines
    contours = axes[0,2].contour(lon_mesh, lat_mesh, rate_interp, 
                                levels=[-40, -30, -20, -10, 0, 10], 
                                colors='black', alpha=0.5, linewidths=1)
    axes[0,2].clabel(contours, inline=True, fontsize=8)
    
    plt.colorbar(scatter, ax=axes[0,2], label='Subsidence Rate (mm/year)')
    axes[0,2].set_xlabel('Longitude')
    axes[0,2].set_ylabel('Latitude')
    axes[0,2].set_title('Geographic Subsidence Rates\n(PS00 with Contours)')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Signal Quality vs Subsidence Rate
    axes[1,0].scatter(np.abs(results['original_rates']), results['correlations'], 
                     c=results['rmse'], s=60, cmap='plasma', alpha=0.7)
    axes[1,0].set_xlabel('|Subsidence Rate| (mm/year)')
    axes[1,0].set_ylabel('Signal Correlation')
    axes[1,0].set_title('Fit Quality vs Rate Magnitude\n(Colors = RMSE)')
    axes[1,0].grid(True, alpha=0.3)
    
    scatter = axes[1,0].collections[0]
    plt.colorbar(scatter, ax=axes[1,0], label='RMSE (mm)')
    
    # 5. Extreme Subsiding Sites Analysis
    # Find 5 most extreme subsiding sites
    fastest_subsiding_idx = np.argsort(results['original_rates'])[:5]
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, station_idx in enumerate(fastest_subsiding_idx):
        observed = results['displacement'][station_idx]
        predicted = results['predictions'][station_idx]
        rate = results['original_rates'][station_idx]
        corr = results['correlations'][station_idx]
        
        # Offset signals for clarity
        offset = i * 15
        axes[1,1].plot(time_years, observed + offset, '-', color=colors[i], linewidth=2, 
                      label=f'Obs {station_idx} ({rate:.1f}mm/yr)', alpha=0.8)
        axes[1,1].plot(time_years, predicted + offset, '--', color=colors[i], linewidth=2, 
                      label=f'Fit {station_idx} (R={corr:.3f})', alpha=0.8)
    
    axes[1,1].set_xlabel('Time (years)')
    axes[1,1].set_ylabel('Displacement (mm) + Offset')
    axes[1,1].set_title('5 Fastest Subsiding Sites\n(Observed vs Fitted)')
    axes[1,1].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Performance Summary Statistics
    axes[1,2].axis('off')
    
    # Calculate comprehensive statistics
    stats_text = f"""PYTORCH FRAMEWORK PERFORMANCE SUMMARY
    
📊 Dataset Information:
   • Stations: {len(results['correlations'])}
   • Time span: {time_years[-1]:.2f} years
   • Observations: {len(time_years)} per station
   
📈 Rate Fitting Performance:
   • Correlation: {np.corrcoef(results['original_rates'], results['fitted_trends'])[0,1]:.6f}
   • RMSE: {np.sqrt(np.mean((results['fitted_trends'] - results['original_rates'])**2)):.4f} mm/year
   • Mean absolute error: {np.mean(np.abs(results['fitted_trends'] - results['original_rates'])):.4f} mm/year
   
📊 Signal Fitting Performance:
   • Mean correlation: {np.mean(results['correlations']):.4f} ± {np.std(results['correlations']):.3f}
   • Mean RMSE: {np.mean(results['rmse']):.2f} ± {np.std(results['rmse']):.2f} mm
   • Best correlation: {np.max(results['correlations']):.4f}
   • Worst correlation: {np.min(results['correlations']):.4f}
   
🎯 Quality Distribution:
   • Excellent (R>0.15): {np.sum(results['correlations'] > 0.15)}/{len(results['correlations'])} ({np.sum(results['correlations'] > 0.15)/len(results['correlations'])*100:.1f}%)
   • Good (R>0.10): {np.sum(results['correlations'] > 0.10)}/{len(results['correlations'])} ({np.sum(results['correlations'] > 0.10)/len(results['correlations'])*100:.1f}%)
   • Fair (R>0.05): {np.sum(results['correlations'] > 0.05)}/{len(results['correlations'])} ({np.sum(results['correlations'] > 0.05)/len(results['correlations'])*100:.1f}%)
   
🌍 Geographic Range:
   • Longitude: {coordinates[:, 0].min():.3f} to {coordinates[:, 0].max():.3f}
   • Latitude: {coordinates[:, 1].min():.3f} to {coordinates[:, 1].max():.3f}
   • Rate range: {results['original_rates'].min():.1f} to {results['original_rates'].max():.1f} mm/year
    """
    
    axes[1,2].text(0.05, 0.95, stats_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save focused analysis
    output_file = Path("figures/ps02c_pytorch_focused_subsidence_analysis.png")
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved focused subsidence analysis: {output_file}")
    plt.show()

def demonstrate_geographic_framework():
    """Demonstrate framework with comprehensive geographic analysis"""
    
    print("🚀 PS02C-PYTORCH GEOGRAPHIC ANALYSIS")
    print("🎯 Enhanced with: Geographic rates + Trend lines + Spatial analysis")
    print("="*70)
    
    # Initialize optimal fitter
    fitter = OptimalTaiwanInSARFitter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print(f"\n1️⃣ Loading Taiwan InSAR Data...")
    fitter.load_data()
    
    # Use larger subset for better geographic coverage
    subset_size = min(100, fitter.n_stations)  # Increased for better geographic analysis
    
    # Select stations with better geographic distribution
    total_stations = fitter.n_stations
    step = total_stations // subset_size
    selected_indices = np.arange(0, total_stations, step)[:subset_size]
    
    fitter.displacement = fitter.displacement[selected_indices]
    fitter.coordinates = fitter.coordinates[selected_indices]
    fitter.subsidence_rates = fitter.subsidence_rates[selected_indices]
    fitter.n_stations = subset_size
    
    print(f"📊 Using geographic subset: {subset_size} stations")
    print(f"📍 Geographic range: Lon {fitter.coordinates[:, 0].min():.3f}-{fitter.coordinates[:, 0].max():.3f}, "
          f"Lat {fitter.coordinates[:, 1].min():.3f}-{fitter.coordinates[:, 1].max():.3f}")
    print(f"📈 Rate range: {fitter.subsidence_rates.min():.1f} to {fitter.subsidence_rates.max():.1f} mm/year")
    
    # Initialize model
    print(f"\n2️⃣ Initializing Optimal Model...")
    fitter.initialize_model()
    
    # Optimal training with more epochs for better geographic dataset
    print(f"\n3️⃣ Training with Geographic Considerations...")
    start_time = time.time()
    loss_history, correlation_history = fitter.train_optimal(max_epochs=800, target_correlation=0.15)
    training_time = time.time() - start_time
    
    # Evaluate results
    print(f"\n4️⃣ Comprehensive Evaluation...")
    results = fitter.evaluate_optimal()
    
    # Create comprehensive geographic visualization
    print(f"\n5️⃣ Creating Comprehensive Geographic Analysis...")
    create_comprehensive_geographic_analysis(fitter, results, loss_history, correlation_history)
    
    # Create focused subsidence analysis
    print(f"\n6️⃣ Creating Focused Subsidence Analysis...")
    create_focused_subsidence_analysis(fitter, results)
    
    print(f"\n✅ GEOGRAPHIC ANALYSIS COMPLETED!")
    print(f"⏱️ Training time: {training_time:.1f} seconds")
    
    # Final geographic summary
    coords = fitter.coordinates.cpu().numpy()
    rates = results['original_rates']
    
    print(f"\n🌍 GEOGRAPHIC SUMMARY:")
    print(f"   📍 Coverage: {coords[:, 0].max() - coords[:, 0].min():.3f}° × {coords[:, 1].max() - coords[:, 1].min():.3f}°")
    print(f"   📈 Subsidence hotspots: {np.sum(rates < -20)} stations < -20 mm/year")
    print(f"   📈 Uplift areas: {np.sum(rates > 5)} stations > 5 mm/year")
    print(f"   🎯 Rate correlation: {np.corrcoef(results['original_rates'], results['fitted_trends'])[0,1]:.6f}")
    print(f"   📊 Signal correlation: {np.mean(results['correlations']):.4f} ± {np.std(results['correlations']):.3f}")

if __name__ == "__main__":
    try:
        demonstrate_geographic_framework()
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
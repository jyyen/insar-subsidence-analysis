#!/usr/bin/env python3
"""
ps02_10_pytorch_demo.py: PyTorch framework demonstration
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from ps02c_pytorch_framework import TaiwanInSARFitter
import warnings

warnings.filterwarnings('ignore')

def load_existing_ps02c_results():
    """Load existing PS02C results for comparison"""
    try:
        # Try to load algorithmic results first
        results_file = Path("data/processed/ps02c_algorithmic_results.pkl")
        if results_file.exists():
            import pickle
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            print(f"✅ Loaded existing PS02C results: {len(results)} stations")
            return results
        else:
            print("⚠️ No existing PS02C results found")
            return None
    except Exception as e:
        print(f"⚠️ Could not load PS02C results: {e}")
        return None

def find_extreme_sites(subsidence_rates, coordinates, n_sites=10):
    """Find most extreme subsiding and uplifting sites"""
    
    # Find fastest subsiding (most negative) and uplifting (most positive)
    sorted_indices = np.argsort(subsidence_rates)
    
    fastest_subsiding_idx = sorted_indices[:n_sites]  # Most negative
    fastest_uplifting_idx = sorted_indices[-n_sites:]  # Most positive
    
    extreme_sites = {
        'subsiding': {
            'indices': fastest_subsiding_idx,
            'rates': subsidence_rates[fastest_subsiding_idx],
            'coordinates': coordinates[fastest_subsiding_idx]
        },
        'uplifting': {
            'indices': fastest_uplifting_idx, 
            'rates': subsidence_rates[fastest_uplifting_idx],
            'coordinates': coordinates[fastest_uplifting_idx]
        }
    }
    
    print(f"🔍 Extreme Sites Analysis:")
    print(f"   Fastest subsiding: {np.min(subsidence_rates):.1f} mm/year at [{coordinates[sorted_indices[0], 0]:.4f}, {coordinates[sorted_indices[0], 1]:.4f}]")
    print(f"   Fastest uplifting: {np.max(subsidence_rates):.1f} mm/year at [{coordinates[sorted_indices[-1], 0]:.4f}, {coordinates[sorted_indices[-1], 1]:.4f}]")
    
    return extreme_sites

def create_comparison_visualization(fitter, results, extreme_sites, existing_results=None):
    """Create comprehensive comparison visualization"""
    
    # Extract data for plotting
    time_years = fitter.time_years.cpu().numpy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance comparison scatter plot
    ax1 = plt.subplot(3, 4, 1)
    fitted_trends = results['fitted_trends']
    original_rates = results['original_rates']
    
    plt.scatter(original_rates, fitted_trends, alpha=0.6, s=20, c='blue')
    
    # Perfect correlation line
    min_rate, max_rate = np.min(original_rates), np.max(original_rates)
    plt.plot([min_rate, max_rate], [min_rate, max_rate], 'r--', alpha=0.8, label='Perfect fit')
    
    plt.xlabel('PS00 Subsidence Rate (mm/year)')
    plt.ylabel('PyTorch Fitted Rate (mm/year)')
    plt.title('Rate Correlation: PS00 vs PyTorch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Correlation and RMSE metrics
    rate_correlation = np.corrcoef(original_rates, fitted_trends)[0, 1]
    rate_rmse = np.sqrt(np.mean((fitted_trends - original_rates)**2))
    plt.text(0.05, 0.95, f'R={rate_correlation:.3f}\nRMSE={rate_rmse:.1f}mm/yr', 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. RMSE distribution
    ax2 = plt.subplot(3, 4, 2)
    plt.hist(results['rmse'], bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('RMSE (mm)')
    plt.ylabel('Number of Stations')
    plt.title('RMSE Distribution')
    plt.axvline(np.mean(results['rmse']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(results["rmse"]):.1f}mm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Correlation distribution
    ax3 = plt.subplot(3, 4, 3)
    plt.hist(results['correlations'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Correlation')
    plt.ylabel('Number of Stations')
    plt.title('Correlation Distribution')
    plt.axvline(np.mean(results['correlations']), color='red', linestyle='--',
                label=f'Mean: {np.mean(results["correlations"]):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Geographic distribution of performance
    ax4 = plt.subplot(3, 4, 4)
    coords = fitter.coordinates.cpu().numpy()
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=results['correlations'], 
                         s=15, cmap='RdYlBu', vmin=0, vmax=1)
    plt.colorbar(scatter, label='Correlation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Performance Distribution')
    plt.grid(True, alpha=0.3)
    
    # 5-8. Extreme subsiding sites (original vs fitted)
    for i, site_idx in enumerate(extreme_sites['subsiding']['indices'][:4]):
        ax = plt.subplot(3, 4, 5 + i)
        
        # Original signal
        original_signal = fitter.displacement[site_idx].cpu().numpy()
        fitted_signal = results['predictions'][site_idx]
        
        plt.plot(time_years, original_signal, 'b-', alpha=0.8, linewidth=2, label='Observed')
        plt.plot(time_years, fitted_signal, 'r--', alpha=0.8, linewidth=2, label='PyTorch Fit')
        
        # Correlation for this station
        station_corr = results['correlations'][site_idx]
        station_rmse = results['rmse'][site_idx]
        
        plt.xlabel('Time (years)')
        plt.ylabel('Displacement (mm)')
        plt.title(f'Station {site_idx} (Subsiding)\nR={station_corr:.3f}, RMSE={station_rmse:.1f}mm')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 9-12. Extreme uplifting sites (original vs fitted)
    for i, site_idx in enumerate(extreme_sites['uplifting']['indices'][:4]):
        ax = plt.subplot(3, 4, 9 + i)
        
        # Original signal  
        original_signal = fitter.displacement[site_idx].cpu().numpy()
        fitted_signal = results['predictions'][site_idx]
        
        plt.plot(time_years, original_signal, 'b-', alpha=0.8, linewidth=2, label='Observed')
        plt.plot(time_years, fitted_signal, 'r--', alpha=0.8, linewidth=2, label='PyTorch Fit')
        
        # Correlation for this station
        station_corr = results['correlations'][site_idx]
        station_rmse = results['rmse'][site_idx]
        
        plt.xlabel('Time (years)')
        plt.ylabel('Displacement (mm)')
        plt.title(f'Station {site_idx} (Uplifting)\nR={station_corr:.3f}, RMSE={station_rmse:.1f}mm')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path("figures/ps02c_pytorch_demonstration.png")
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization: {output_file}")
    plt.show()

def create_performance_comparison_table(pytorch_results, existing_results=None):
    """Create performance comparison table"""
    
    print(f"\n📊 PYTORCH FRAMEWORK PERFORMANCE SUMMARY:")
    print("="*60)
    
    # PyTorch results
    pytorch_mean_rmse = np.mean(pytorch_results['rmse'])
    pytorch_std_rmse = np.std(pytorch_results['rmse'])
    pytorch_mean_corr = np.mean(pytorch_results['correlations'])
    pytorch_std_corr = np.std(pytorch_results['correlations'])
    
    # Rate fitting accuracy
    rate_correlation = np.corrcoef(pytorch_results['original_rates'], pytorch_results['fitted_trends'])[0, 1]
    rate_rmse = np.sqrt(np.mean((pytorch_results['fitted_trends'] - pytorch_results['original_rates'])**2))
    
    print(f"PyTorch Framework:")
    print(f"   • Signal Fitting RMSE: {pytorch_mean_rmse:.2f} ± {pytorch_std_rmse:.2f} mm")
    print(f"   • Signal Correlation: {pytorch_mean_corr:.3f} ± {pytorch_std_corr:.3f}")
    print(f"   • Rate Fitting R: {rate_correlation:.3f}")
    print(f"   • Rate Fitting RMSE: {rate_rmse:.2f} mm/year")
    
    # Performance categories
    excellent = np.sum(pytorch_results['correlations'] > 0.9)
    good = np.sum((pytorch_results['correlations'] > 0.7) & (pytorch_results['correlations'] <= 0.9))
    fair = np.sum((pytorch_results['correlations'] > 0.5) & (pytorch_results['correlations'] <= 0.7))
    poor = np.sum(pytorch_results['correlations'] <= 0.5)
    
    total = len(pytorch_results['correlations'])
    
    print(f"\nPerformance Categories:")
    print(f"   • Excellent (R > 0.9): {excellent} ({excellent/total*100:.1f}%)")
    print(f"   • Good (0.7 < R ≤ 0.9): {good} ({good/total*100:.1f}%)")
    print(f"   • Fair (0.5 < R ≤ 0.7): {fair} ({fair/total*100:.1f}%)")
    print(f"   • Poor (R ≤ 0.5): {poor} ({poor/total*100:.1f}%)")
    
    if existing_results:
        print(f"\nComparison with Existing PS02C:")
        # Extract existing results statistics if available
        # This would depend on the structure of existing results
        print(f"   (To be implemented based on existing results format)")

def main():
    """Main demonstration function"""
    
    print("🚀 PS02C-PYTORCH FRAMEWORK DEMONSTRATION")
    print("="*50)
    
    # Initialize fitter
    print(f"\n1️⃣ Initializing PyTorch Framework...")
    fitter = TaiwanInSARFitter(device='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    
    # Load data
    print(f"\n2️⃣ Loading Taiwan InSAR Data...")
    try:
        fitter.load_data()
    except Exception as e:
        print(f"❌ Could not load data: {e}")
        print(f"ℹ️ Please ensure ps00_preprocessed_data.npz exists in data/processed/")
        return False
    
    # Find extreme sites for focused analysis
    print(f"\n3️⃣ Identifying Extreme Sites...")
    coords = fitter.coordinates.cpu().numpy()
    rates = fitter.subsidence_rates.cpu().numpy()
    extreme_sites = find_extreme_sites(rates, coords, n_sites=10)
    
    # Initialize model with physics-based parameters
    print(f"\n4️⃣ Initializing Model with Physics-Based Parameters...")
    fitter.initialize_model(physics_based_init=True)
    
    # Setup optimization
    print(f"\n5️⃣ Setting up Optimization...")
    fitter.setup_optimization(learning_rate=0.01)
    
    # Subset training for demonstration (use first 1000 stations)
    print(f"\n6️⃣ Training on Subset for Demonstration...")
    print(f"   (Using first 1000 stations for faster demonstration)")
    
    # Temporarily modify data for subset training
    original_displacement = fitter.displacement.clone()
    original_coordinates = fitter.coordinates.clone()
    original_subsidence_rates = fitter.subsidence_rates.clone()
    
    # Use subset
    subset_size = min(100, fitter.n_stations)  # Even smaller for demo
    fitter.displacement = fitter.displacement[:subset_size]
    fitter.coordinates = fitter.coordinates[:subset_size]
    fitter.subsidence_rates = fitter.subsidence_rates[:subset_size]
    fitter.n_stations = subset_size
    
    # Re-initialize model for subset
    fitter.initialize_model(physics_based_init=True)
    fitter.setup_optimization(learning_rate=0.01)
    
    # Train model
    start_time = time.time()
    loss_history = fitter.train(max_epochs=100, patience=20)  # Much faster for demo
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Evaluate results
    print(f"\n7️⃣ Evaluating Results...")
    results = fitter.evaluate()
    
    # Load existing results for comparison
    print(f"\n8️⃣ Loading Existing PS02C Results for Comparison...")
    existing_results = load_existing_ps02c_results()
    
    # Create performance comparison
    print(f"\n9️⃣ Performance Analysis...")
    create_performance_comparison_table(results, existing_results)
    
    # Find extreme sites in subset
    subset_coords = coords[:subset_size]
    subset_rates = rates[:subset_size] 
    subset_extreme_sites = find_extreme_sites(subset_rates, subset_coords, n_sites=4)
    
    # Create visualization
    print(f"\n🔟 Creating Comprehensive Visualization...")
    create_comparison_visualization(fitter, results, subset_extreme_sites, existing_results)
    
    # Training progress plot
    print(f"\n📈 Creating Training Progress Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PyTorch Training Progress')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Annotate final loss
    final_loss = loss_history[-1]
    plt.annotate(f'Final Loss: {final_loss:.6f}', 
                xy=(len(loss_history)-1, final_loss),
                xytext=(len(loss_history)*0.7, final_loss*2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    plt.tight_layout()
    
    progress_file = Path("figures/ps02c_pytorch_training_progress.png")
    progress_file.parent.mkdir(exist_ok=True)
    plt.savefig(progress_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved training progress: {progress_file}")
    plt.show()
    
    print(f"\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print(f"📊 Processed {subset_size} stations in {training_time:.1f} seconds")
    print(f"📈 Training converged in {len(loss_history)} epochs")
    print(f"🎯 Final performance: RMSE={np.mean(results['rmse']):.1f}mm, R={np.mean(results['correlations']):.3f}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Demonstration interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
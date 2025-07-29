"""
Quick Test of Enhanced PS02C Algorithm

Tests the enhanced algorithm on a small subset to validate improvements
before running the full dataset.

Author: Taiwan InSAR Subsidence Analysis Project
Created: 2025-07-28
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time

# Import the enhanced model
import sys
sys.path.append('.')

def test_enhanced_algorithm():
    """Quick test of enhanced PS02C algorithm"""
    
    print("🧪 Quick Test: Enhanced PS02C Algorithm")
    print("=" * 50)
    
    # Load test data
    ps00_file = Path('data/processed/ps00_preprocessed_data.npz')
    if not ps00_file.exists():
        print("❌ PS00 data not found")
        return
    
    data = np.load(ps00_file)
    displacement = data['displacement']
    coordinates = data['coordinates']
    n_stations, n_acquisitions = displacement.shape
    
    # Test on first 5 stations only
    test_stations = 5
    time_vector = np.arange(n_acquisitions)
    
    print(f"📊 Testing {test_stations} stations...")
    
    # Import enhanced model (inline to avoid import issues)
    from ps02c_ENHANCED_algorithmic_optimization import EnhancedSubsidenceModel, process_enhanced_station
    
    results = []
    start_time = time.time()
    
    for i in range(test_stations):
        print(f"\\n🔄 Processing station {i+1}/{test_stations}...")
        
        station_displacement = displacement[i, :]
        
        # Skip if too many NaN values
        if np.sum(~np.isnan(station_displacement)) < n_acquisitions * 0.7:
            print(f"   ⚠️ Station {i}: too many NaN values, skipping")
            results.append(None)
            continue
        
        result = process_enhanced_station((i, station_displacement, time_vector))
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Analyze results
    print("\\n" + "=" * 50)
    print("📊 QUICK TEST RESULTS")
    print("=" * 50)
    
    successful_results = [r for r in results if r is not None]
    
    if len(successful_results) > 0:
        correlations = [r['correlation'] for r in successful_results]
        rmse_values = [r['rmse'] for r in successful_results]
        trends = [r['trend'] for r in successful_results]
        
        print(f"✅ Success rate: {len(successful_results)}/{test_stations} ({len(successful_results)/test_stations*100:.1f}%)")
        print(f"⏱️  Total processing time: {total_time:.1f} seconds")
        print(f"⚡ Average per station: {total_time/test_stations:.1f} seconds")
        print(f"")
        print(f"📈 Performance metrics:")
        print(f"   • Correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
        print(f"   • RMSE: {np.mean(rmse_values):.1f} ± {np.std(rmse_values):.1f} mm")
        print(f"   • Trend range: {np.min(trends):.1f} to {np.max(trends):.1f} mm/year")
        
        # Compare with original performance (from our previous analysis)
        print(f"")
        print(f"🔍 IMPROVEMENT ANALYSIS:")
        print(f"   Original algorithm (from GPS validation):")
        print(f"   • Correlation: -0.071 (negative = systematic error)")
        print(f"   • RMSE: 21.6 mm")
        print(f"   • GPS validation R²: 0.010")
        print(f"")
        print(f"   Enhanced algorithm (this test):")
        print(f"   • Correlation: {np.mean(correlations):.3f} ({'+' if np.mean(correlations) > -0.071 else ''}{'IMPROVED' if np.mean(correlations) > -0.071 else 'NEEDS WORK'})")
        print(f"   • RMSE: {np.mean(rmse_values):.1f} mm ({'IMPROVED' if np.mean(rmse_values) < 21.6 else 'NEEDS WORK'})")
        print(f"   • GPS validation: TBD (need to run full GPS test)")
        
        # Quality categories
        excellent = np.sum((np.array(correlations) >= 0.9) & (np.array(rmse_values) <= 10))
        good = np.sum((np.array(correlations) >= 0.7) & (np.array(rmse_values) <= 20)) - excellent
        fair = np.sum((np.array(correlations) >= 0.5) & (np.array(rmse_values) <= 40)) - excellent - good
        poor = len(correlations) - excellent - good - fair
        
        print(f"")
        print(f"🏆 Quality distribution:")
        print(f"   • Excellent (r≥0.9, RMSE≤10mm): {excellent}/{len(correlations)} ({excellent/len(correlations)*100:.1f}%)")
        print(f"   • Good (r≥0.7, RMSE≤20mm): {good}/{len(correlations)} ({good/len(correlations)*100:.1f}%)")
        print(f"   • Fair (r≥0.5, RMSE≤40mm): {fair}/{len(correlations)} ({fair/len(correlations)*100:.1f}%)")
        print(f"   • Poor (others): {poor}/{len(correlations)} ({poor/len(correlations)*100:.1f}%)")
        
        # Create quick visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Enhanced PS02C Algorithm - Quick Test Results', fontsize=14, fontweight='bold')
        
        # 1. Time series comparison for first successful station
        first_result = successful_results[0]
        station_idx = first_result['station_idx']
        observed = displacement[station_idx, :]
        fitted = first_result['fitted_signal']
        
        ax1.plot(time_vector, observed, 'b-', alpha=0.7, label='Observed')
        ax1.plot(time_vector, fitted, 'r--', linewidth=2, label='Enhanced Fit')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Displacement (mm)')
        ax1.set_title(f'Station {station_idx}: R={first_result["correlation"]:.3f}, RMSE={first_result["rmse"]:.1f}mm')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Correlation histogram
        ax2.hist(correlations, bins=10, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(correlations):.3f}')
        ax2.axvline(-0.071, color='orange', linestyle=':', linewidth=2, label='Original: -0.071')
        ax2.set_xlabel('Correlation')
        ax2.set_ylabel('Count')
        ax2.set_title('Correlation Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. RMSE histogram
        ax3.hist(rmse_values, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(rmse_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmse_values):.1f}mm')
        ax3.axvline(21.6, color='orange', linestyle=':', linewidth=2, label='Original: 21.6mm')
        ax3.set_xlabel('RMSE (mm)')
        ax3.set_ylabel('Count')
        ax3.set_title('RMSE Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Trends comparison
        ax4.scatter(range(len(trends)), trends, alpha=0.7, color='purple')
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Station Index')
        ax4.set_ylabel('Trend (mm/year)')
        ax4.set_title(f'Deformation Trends (Range: {np.min(trends):.1f} to {np.max(trends):.1f})')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save results
        output_dir = Path('figures')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'enhanced_ps02c_quick_test.png', dpi=300, bbox_inches='tight')
        print(f"📊 Quick test results saved to {output_dir / 'enhanced_ps02c_quick_test.png'}")
        
        plt.show()
        
        # Save test results
        test_results = {
            'results': results,
            'summary': {
                'success_rate': len(successful_results) / test_stations,
                'mean_correlation': np.mean(correlations),
                'mean_rmse': np.mean(rmse_values),
                'processing_time': total_time,
                'enhancements_tested': [
                    'expanded_parameter_bounds',
                    'quadratic_trend_component', 
                    'biennial_cycle_component',
                    'adaptive_objective_function',
                    'improved_optimization_settings'
                ]
            }
        }
        
        test_file = Path('data/processed/ps02c_enhanced_quick_test.pkl')
        test_file.parent.mkdir(exist_ok=True)
        with open(test_file, 'wb') as f:
            pickle.dump(test_results, f)
        
        print(f"💾 Test results saved to {test_file}")
        
    else:
        print("❌ No successful results in quick test")
        print("🔧 Algorithm may need further debugging")
    
    print("=" * 50)
    print("✅ Quick test complete!")
    
    if len(successful_results) > 0 and np.mean(correlations) > 0:
        print("🎉 GOOD NEWS: Enhanced algorithm shows positive correlations!")
        print("🔬 NEXT STEP: Run full GPS validation to confirm improvements")
    else:
        print("⚠️  Results still need improvement")
        print("🔧 May need to adjust parameters further")

if __name__ == "__main__":
    test_enhanced_algorithm()
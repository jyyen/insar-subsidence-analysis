"""
Simple Test of PS02C Improvements

Tests key algorithmic improvements on synthetic data to validate fixes
before applying to real dataset.

Author: Taiwan InSAR Subsidence Analysis Project
Created: 2025-07-28
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
import time

def generate_realistic_insar_signal(time_days, params):
    """Generate realistic InSAR signal for testing"""
    t_years = time_days / 365.25
    
    # Realistic Taiwan subsidence signal
    linear_trend = params['linear_trend']
    quadratic_trend = params.get('quadratic_trend', 0)
    annual_amp = params['annual_amp']  
    semi_annual_amp = params['semi_annual_amp']
    quarterly_amp = params['quarterly_amp']
    noise_std = params['noise_std']
    
    # Signal components
    trend = linear_trend * t_years + quadratic_trend * t_years**2
    annual = annual_amp * np.sin(2*np.pi*1.0*t_years + 0.5)
    semi_annual = semi_annual_amp * np.sin(2*np.pi*2.0*t_years + 1.2)
    quarterly = quarterly_amp * np.sin(2*np.pi*4.0*t_years + 0.8)
    noise = np.random.normal(0, noise_std, len(t_years))
    
    signal = trend + annual + semi_annual + quarterly + noise
    
    return signal

def original_ps02c_model(params, time_days):
    """Original PS02C model (with restrictive bounds)"""
    t_years = time_days / 365.25
    
    # Original restrictive model
    signal = (params[0] * t_years +  # linear trend only
              params[1] * np.sin(2*np.pi*params[2]*t_years + params[3]) +  # annual
              params[4] * np.sin(2*np.pi*params[5]*t_years + params[6]) +  # semi-annual
              params[7] * np.sin(2*np.pi*params[8]*t_years + params[9]))    # quarterly
    
    return signal

def enhanced_ps02c_model(params, time_days):
    """Enhanced PS02C model (with improvements)"""
    t_years = time_days / 365.25
    
    # Enhanced model with quadratic trend and more flexibility
    linear_trend = params[0] * t_years
    quadratic_trend = params[1] * t_years**2
    annual = params[2] * np.sin(2*np.pi*params[3]*t_years + params[4])
    semi_annual = params[5] * np.sin(2*np.pi*params[6]*t_years + params[7])
    quarterly = params[8] * np.sin(2*np.pi*params[9]*t_years + params[10])
    biennial = params[11] * np.sin(2*np.pi*params[12]*t_years + params[13])  # New component
    
    signal = linear_trend + quadratic_trend + annual + semi_annual + quarterly + biennial
    
    return signal

def objective_function(params, observed, time_days, model_func):
    """Objective function for optimization"""
    try:
        predicted = model_func(params, time_days)
        residuals = predicted - observed
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Calculate correlation
        obs_std = np.std(observed)
        if obs_std > 0:
            corr = np.corrcoef(predicted, observed)[0, 1]
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0
        
        # Combined objective (minimize RMSE, maximize correlation)
        normalized_rmse = rmse / obs_std if obs_std > 0 else rmse
        objective = normalized_rmse - corr
        
        return objective
        
    except Exception:
        return 1e6

def test_algorithm_improvements():
    """Test original vs enhanced PS02C algorithms"""
    
    print("🧪 Testing PS02C Algorithm Improvements")
    print("=" * 60)
    
    # Create synthetic test data
    time_days = np.linspace(0, 4*365, 200)  # 4 years of data
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Moderate Subsidence',
            'params': {
                'linear_trend': -15,      # mm/year
                'quadratic_trend': -0.5,  # mm/year²
                'annual_amp': 12,         # mm
                'semi_annual_amp': 8,     # mm
                'quarterly_amp': 4,       # mm
                'noise_std': 2            # mm
            }
        },
        {
            'name': 'Extreme Subsidence',
            'params': {
                'linear_trend': -45,      # mm/year (extreme)
                'quadratic_trend': -1.2,  # mm/year² (accelerating)
                'annual_amp': 25,         # mm (large seasonal)
                'semi_annual_amp': 15,    # mm
                'quarterly_amp': 8,       # mm
                'noise_std': 3            # mm
            }
        },
        {
            'name': 'Slow Uplift',
            'params': {
                'linear_trend': 8,        # mm/year (uplift)
                'quadratic_trend': 0.2,   # mm/year² (accelerating)
                'annual_amp': 18,         # mm
                'semi_annual_amp': 12,    # mm
                'quarterly_amp': 6,       # mm
                'noise_std': 2.5          # mm
            }
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\\n🔬 Testing scenario: {scenario['name']}")
        print("-" * 40)
        
        # Generate synthetic signal
        np.random.seed(42)  # Reproducible results
        true_signal = generate_realistic_insar_signal(time_days, scenario['params'])
        
        scenario_results = {'name': scenario['name'], 'true_params': scenario['params']}
        
        # Test 1: Original PS02C algorithm (restrictive bounds)
        print("   Testing original algorithm...")
        
        original_bounds = [
            (-80, 30),       # trend (restrictive)
            (0, 30),         # annual_amp (restrictive) 
            (0.95, 1.05),    # annual_freq (very restrictive)
            (0, 2*np.pi),    # annual_phase
            (0, 20),         # semi_annual_amp (restrictive)
            (1.9, 2.1),      # semi_annual_freq (very restrictive)
            (0, 2*np.pi),    # semi_annual_phase
            (0, 15),         # quarterly_amp (restrictive)
            (3.8, 4.2),      # quarterly_freq (very restrictive)
            (0, 2*np.pi)     # quarterly_phase
        ]
        
        start_time = time.time()
        original_result = differential_evolution(
            func=lambda p: objective_function(p, true_signal, time_days, original_ps02c_model),
            bounds=original_bounds,
            maxiter=200,  # Limited iterations
            popsize=10,   # Small population
            seed=42
        )
        original_time = time.time() - start_time
        
        if original_result.success:
            original_fitted = original_ps02c_model(original_result.x, time_days)
            original_corr = np.corrcoef(true_signal, original_fitted)[0, 1]
            original_rmse = np.sqrt(np.mean((true_signal - original_fitted)**2))
        else:
            original_corr = 0
            original_rmse = 999
            original_fitted = np.zeros_like(true_signal)
        
        print(f"     Original: R={original_corr:.3f}, RMSE={original_rmse:.1f}mm, time={original_time:.1f}s")
        
        # Test 2: Enhanced PS02C algorithm (expanded bounds + improvements)
        print("   Testing enhanced algorithm...")
        
        enhanced_bounds = [
            (-120, 50),      # linear_trend (expanded)
            (-5, 5),         # quadratic_trend (NEW)
            (0, 50),         # annual_amp (expanded)
            (0.7, 1.3),      # annual_freq (flexible ±30%)
            (0, 2*np.pi),    # annual_phase
            (0, 40),         # semi_annual_amp (expanded)
            (1.7, 2.3),      # semi_annual_freq (flexible ±15%)
            (0, 2*np.pi),    # semi_annual_phase
            (0, 30),         # quarterly_amp (expanded)
            (3.5, 4.5),      # quarterly_freq (flexible ±12%)
            (0, 2*np.pi),    # quarterly_phase
            (0, 25),         # biennial_amp (NEW)
            (0.4, 0.6),      # biennial_freq (NEW)
            (0, 2*np.pi)     # biennial_phase (NEW)
        ]
        
        start_time = time.time()
        enhanced_result = differential_evolution(
            func=lambda p: objective_function(p, true_signal, time_days, enhanced_ps02c_model),
            bounds=enhanced_bounds,
            maxiter=400,  # More iterations
            popsize=20,   # Larger population
            seed=42
        )
        enhanced_time = time.time() - start_time
        
        if enhanced_result.success:
            enhanced_fitted = enhanced_ps02c_model(enhanced_result.x, time_days)
            enhanced_corr = np.corrcoef(true_signal, enhanced_fitted)[0, 1]
            enhanced_rmse = np.sqrt(np.mean((true_signal - enhanced_fitted)**2))
        else:
            enhanced_corr = 0
            enhanced_rmse = 999
            enhanced_fitted = np.zeros_like(true_signal)
        
        print(f"     Enhanced: R={enhanced_corr:.3f}, RMSE={enhanced_rmse:.1f}mm, time={enhanced_time:.1f}s")
        
        # Calculate improvement
        corr_improvement = enhanced_corr - original_corr
        rmse_improvement = (original_rmse - enhanced_rmse) / original_rmse * 100  # % improvement
        
        print(f"     IMPROVEMENT: ΔR={corr_improvement:+.3f}, ΔRMSE={rmse_improvement:+.1f}%")
        
        # Store results
        scenario_results.update({
            'original': {
                'correlation': original_corr,
                'rmse': original_rmse,
                'time': original_time,
                'fitted': original_fitted,
                'success': original_result.success
            },
            'enhanced': {
                'correlation': enhanced_corr,
                'rmse': enhanced_rmse,
                'time': enhanced_time,
                'fitted': enhanced_fitted,
                'success': enhanced_result.success
            },
            'improvement': {
                'correlation': corr_improvement,
                'rmse_percent': rmse_improvement
            },
            'true_signal': true_signal
        })
        
        results.append(scenario_results)
    
    # Create comprehensive comparison figure
    print("\\n📊 Creating comparison visualization...")
    
    fig, axes = plt.subplots(len(test_scenarios), 3, figsize=(18, 6*len(test_scenarios)))
    if len(test_scenarios) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('PS02C Algorithm Improvements Test Results', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results):
        # Time series comparison
        ax1 = axes[i, 0]
        ax1.plot(time_days/365.25, result['true_signal'], 'k-', linewidth=2, label='True Signal', alpha=0.8)
        ax1.plot(time_days/365.25, result['original']['fitted'], 'r--', linewidth=2, 
                label=f'Original (R={result["original"]["correlation"]:.3f})', alpha=0.7)
        ax1.plot(time_days/365.25, result['enhanced']['fitted'], 'b--', linewidth=2,
                label=f'Enhanced (R={result["enhanced"]["correlation"]:.3f})', alpha=0.7)
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Displacement (mm)')
        ax1.set_title(f'{result["name"]} - Time Series Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals comparison
        ax2 = axes[i, 1]
        original_residuals = result['true_signal'] - result['original']['fitted']
        enhanced_residuals = result['true_signal'] - result['enhanced']['fitted']
        
        ax2.plot(time_days/365.25, original_residuals, 'r-', alpha=0.7, 
                label=f'Original (RMSE={result["original"]["rmse"]:.1f}mm)')
        ax2.plot(time_days/365.25, enhanced_residuals, 'b-', alpha=0.7,
                label=f'Enhanced (RMSE={result["enhanced"]["rmse"]:.1f}mm)')
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Residuals (mm)')
        ax2.set_title(f'{result["name"]} - Residuals Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance metrics
        ax3 = axes[i, 2]
        metrics = ['Correlation', 'RMSE (mm)', 'Time (s)']
        original_values = [result['original']['correlation'], result['original']['rmse'], result['original']['time']]
        enhanced_values = [result['enhanced']['correlation'], result['enhanced']['rmse'], result['enhanced']['time']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Normalize values for visualization (except correlation which is already 0-1)
        norm_original = [original_values[0], original_values[1]/50, original_values[2]/10]
        norm_enhanced = [enhanced_values[0], enhanced_values[1]/50, enhanced_values[2]/10]
        
        bars1 = ax3.bar(x - width/2, norm_original, width, label='Original', alpha=0.7, color='red')
        bars2 = ax3.bar(x + width/2, norm_enhanced, width, label='Enhanced', alpha=0.7, color='blue')
        
        # Add value labels on bars
        for j, (bar1, bar2, orig, enh) in enumerate(zip(bars1, bars2, original_values, enhanced_values)):
            ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                    f'{orig:.2f}' if j == 0 else f'{orig:.1f}',
                    ha='center', va='bottom', fontsize=8)
            ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                    f'{enh:.2f}' if j == 0 else f'{enh:.1f}',
                    ha='center', va='bottom', fontsize=8)
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Normalized Values')
        ax3.set_title(f'{result["name"]} - Performance Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'ps02c_algorithm_improvements_test.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comparison figure saved to {output_dir / 'ps02c_algorithm_improvements_test.png'}")
    
    plt.show()
    
    # Summary analysis
    print("\\n" + "=" * 60)
    print("📊 ALGORITHM IMPROVEMENTS SUMMARY")
    print("=" * 60)
    
    avg_corr_improvement = np.mean([r['improvement']['correlation'] for r in results])
    avg_rmse_improvement = np.mean([r['improvement']['rmse_percent'] for r in results])
    
    successful_original = sum(1 for r in results if r['original']['success'])
    successful_enhanced = sum(1 for r in results if r['enhanced']['success'])
    
    print(f"✅ Success rates:")
    print(f"   • Original algorithm: {successful_original}/{len(results)} scenarios")
    print(f"   • Enhanced algorithm: {successful_enhanced}/{len(results)} scenarios")
    print(f"")
    print(f"📈 Average improvements:")
    print(f"   • Correlation: {avg_corr_improvement:+.3f} ({'BETTER' if avg_corr_improvement > 0 else 'WORSE'})")
    print(f"   • RMSE: {avg_rmse_improvement:+.1f}% ({'BETTER' if avg_rmse_improvement > 0 else 'WORSE'})")
    print(f"")
    print(f"🔍 Key findings:")
    for result in results:
        print(f"   • {result['name']}:")
        print(f"     - Correlation: {result['original']['correlation']:.3f} → {result['enhanced']['correlation']:.3f} ({result['improvement']['correlation']:+.3f})")
        print(f"     - RMSE: {result['original']['rmse']:.1f} → {result['enhanced']['rmse']:.1f}mm ({result['improvement']['rmse_percent']:+.1f}%)")
    
    if avg_corr_improvement > 0.1 and avg_rmse_improvement > 10:
        print(f"\\n🎉 EXCELLENT: Enhanced algorithm shows significant improvements!")
        print(f"🔬 READY: Can proceed with GPS validation testing")
    elif avg_corr_improvement > 0 and avg_rmse_improvement > 0:
        print(f"\\n✅ GOOD: Enhanced algorithm shows improvements")  
        print(f"🔬 RECOMMENDED: Test with GPS validation")
    else:
        print(f"\\n⚠️  WARNING: Limited improvement detected")
        print(f"🔧 NEEDED: Further algorithm refinement")
    
    print("=" * 60)
    print("✅ Algorithm improvement test complete!")
    
    return results

if __name__ == "__main__":
    test_algorithm_improvements()
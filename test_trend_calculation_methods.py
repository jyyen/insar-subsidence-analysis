#!/usr/bin/env python3
"""
Test Trend Calculation Methods
Simple test to compare fitted_params.trend vs calculated slopes from time series
without loading the complex pickle files.

Created: 2025-07-28
Purpose: Safe investigation approach
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_trend_calculation_concept():
    """Test the concept with synthetic data to understand the issue"""
    print("🧪 Testing trend calculation concept with synthetic data...")
    
    # Create synthetic time series similar to InSAR data
    time_years = np.linspace(0, 3.5, 215)  # 3.5 years, 215 points like real data
    
    # Create test cases
    test_cases = [
        {"name": "Subsiding with seasonal", "trend": 10, "seasonal_amp": 20, "noise_std": 5},
        {"name": "Uplifting with seasonal", "trend": -15, "seasonal_amp": 15, "noise_std": 3},
        {"name": "Strong subsiding", "trend": 25, "seasonal_amp": 10, "noise_std": 8},
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Trend Calculation Method Comparison\\n'
                 'Testing why fitted_params.trend might differ from actual slopes', 
                 fontsize=14, fontweight='bold')
    
    for i, case in enumerate(test_cases):
        ax = axes[i]
        
        # Generate synthetic "original" time series
        np.random.seed(42 + i)  # Reproducible
        trend_component = case["trend"] * time_years
        seasonal_component = case["seasonal_amp"] * np.sin(2 * np.pi * time_years)
        noise_component = np.random.normal(0, case["noise_std"], len(time_years))
        
        original_ts = trend_component + seasonal_component + noise_component
        
        # Simulate a "fitted" time series (what PS02C might produce)
        # This would have less noise and slightly different trend due to fitting
        fitted_trend_coeff = case["trend"] * 0.85  # Slightly different due to fitting process
        fitted_seasonal = case["seasonal_amp"] * 0.9 * np.sin(2 * np.pi * time_years + 0.1)
        fitted_ts = fitted_trend_coeff * time_years + fitted_seasonal
        
        # Method 1: Use the "fitted parameter" (like fitted_params.trend)
        param_trend_line = fitted_trend_coeff * time_years
        
        # Method 2: Calculate slope directly from fitted time series
        actual_slope = np.polyfit(time_years, fitted_ts, 1)[0]
        actual_trend_line = actual_slope * time_years
        
        # Plot everything
        ax.plot(time_years, original_ts, 'b-', linewidth=2, alpha=0.7, label='Original TS')
        ax.plot(time_years, fitted_ts, 'r-', linewidth=2, alpha=0.7, label='Fitted TS')
        
        # Show both trend line methods
        ax.plot(time_years, param_trend_line, 'r--', linewidth=2, alpha=0.8,
               label=f'Param Method: {fitted_trend_coeff:.1f}')
        ax.plot(time_years, actual_trend_line, 'g--', linewidth=2, alpha=0.8,
               label=f'Slope Method: {actual_slope:.1f}')
        
        # Check which method matches the fitted time series better
        fitted_ts_direction = "Up" if fitted_ts[-1] > fitted_ts[0] else "Down"
        param_direction = "Up" if fitted_trend_coeff > 0 else "Down"
        slope_direction = "Up" if actual_slope > 0 else "Down"
        
        param_match = fitted_ts_direction == param_direction
        slope_match = fitted_ts_direction == slope_direction
        
        ax.set_title(f'{case["name"]}\\n'
                    f'TS: {fitted_ts_direction}, Param: {param_direction} ({"✅" if param_match else "❌"}), '
                    f'Slope: {slope_direction} ({"✅" if slope_match else "❌"})',
                    fontsize=10)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Displacement (mm)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        print(f"\n{case['name']}:")
        print(f"  Fitted TS direction: {fitted_ts_direction}")
        print(f"  Param trend coefficient: {fitted_trend_coeff:.2f} ({'✅' if param_match else '❌'})")
        print(f"  Calculated slope: {actual_slope:.2f} ({'✅' if slope_match else '❌'})")
        print(f"  Difference: {abs(fitted_trend_coeff - actual_slope):.2f}")
    
    plt.tight_layout()
    
    # Save test figure
    output_file = Path("figures/trend_calculation_method_test.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✅ Saved test figure: {output_file}")
    return str(output_file)

def analyze_real_data_approach():
    """Suggest a practical approach for fixing the real data"""
    print("\n" + "="*80)
    print("💡 PRACTICAL APPROACH FOR REAL DATA FIX")
    print("="*80)
    
    print("""
Based on the synthetic testing, here's what likely happens:

1. **fitted_params.trend**: This is a MODEL PARAMETER, not necessarily the actual
   slope of the generated time series. It's what the PyTorch model learned as the
   linear trend component during training.

2. **Actual slope**: This is the real slope of the fitted time series as plotted,
   which includes the effect of all components (trend + seasonal).

LIKELY ISSUE:
- When seasonal components are present, the actual slope of the fitted time series
  can differ from the pure trend parameter.
- This explains why trend lines "betray" their time series.

PROPOSED CAUTIOUS FIX:
Instead of:  trend_fitted = -fitted_rate * time_years
Use:         calculated_slope = np.polyfit(time_years, fitted_ts, 1)[0]
             trend_fitted = calculated_slope * time_years

This ensures the trend line ALWAYS matches the actual slope of the plotted time series.

VALIDATION NEEDED:
- Test this approach on a few stations from the actual figure
- Verify it fixes both subsiding and uplifting site issues
- Ensure it doesn't break anything else
    """)

def main():
    """Main testing workflow"""
    print("=" * 80)
    print("🧪 TREND CALCULATION METHOD TESTING")
    print("=" * 80)
    
    try:
        # Test with synthetic data
        test_file = test_trend_calculation_concept()
        
        # Provide analysis
        analyze_real_data_approach()
        
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   The synthetic test suggests calculating slopes directly from time series")
        print(f"   would be more accurate than using model parameters.")
        print(f"\n🚨 NEXT STEP:")
        print(f"   Implement a CAUTIOUS test on the real visualization code")
        print(f"   Compare both methods side-by-side before committing to the change")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
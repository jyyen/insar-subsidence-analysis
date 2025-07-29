#!/usr/bin/env python3
"""
Investigate PS02C Trend Parameters vs Time Series Slopes
Carefully examine the relationship between fitted_params.trend and actual slopes
from fitted time series to understand the discrepancy in trend line plotting.

Created: 2025-07-28
Purpose: Cautious investigation before implementing trend fix
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load PS00 and PS02C data for investigation"""
    print("📊 Loading data for investigation...")
    
    # Load PS00 data
    ps00_file = Path("data/processed/ps00_preprocessed_data.npz")
    ps00_data = np.load(ps00_file, allow_pickle=True)
    
    # Load PS02C results
    ps02c_file = Path("data/processed/ps02c_algorithmic_results.pkl")
    with open(ps02c_file, 'rb') as f:
        ps02c_data = pickle.load(f)
    
    return ps00_data, ps02c_data

def generate_fitted_signal(params, time_years):
    """Generate fitted signal from PS02C parameters"""
    t = time_years
    
    signal = (
        params.trend * t +
        params.annual_amp * np.sin(2*np.pi*params.annual_freq*t + params.annual_phase) +
        params.semi_annual_amp * np.sin(2*np.pi*params.semi_annual_freq*t + params.semi_annual_phase) +
        params.quarterly_amp * np.sin(2*np.pi*params.quarterly_freq*t + params.quarterly_phase) +
        params.long_annual_amp * np.sin(2*np.pi*params.long_annual_freq*t + params.long_annual_phase)
    )
    
    return signal

def investigate_specific_stations():
    """Investigate the problematic stations mentioned by user"""
    print("🔍 Investigating specific problematic stations...")
    
    ps00_data, ps02c_data = load_data()
    
    # Extract data
    displacement = ps00_data['displacement']
    ps00_rates = ps00_data['subsidence_rates']
    coordinates = ps00_data['coordinates']
    
    # Create time vector
    n_times = displacement.shape[1]
    time_days = np.arange(n_times) * 6
    time_years = time_days / 365.25
    
    # Get PS02C results
    ps02c_results = ps02c_data.get('results', [])
    successful_results = [r for r in ps02c_results if r.get('success', False)]
    
    print(f"📊 Available data:")
    print(f"   PS00 stations: {len(ps00_rates)}")
    print(f"   PS02C successful results: {len(successful_results)}")
    
    # Find extreme stations for investigation
    subsiding_indices = np.argsort(ps00_rates)[-10:]  # Most subsiding
    uplifting_indices = np.argsort(ps00_rates)[:10]   # Most uplifting
    
    # Specifically look for station 7153 (mentioned by user)
    station_7153_found = False
    
    investigation_results = []
    
    # Investigate a few representative stations
    test_stations = list(subsiding_indices[-3:]) + list(uplifting_indices[:3])  # 6 stations
    
    for station_idx in test_stations:
        # Find PS02C result for this station
        fitted_params = None
        for result in successful_results:
            if result.get('station_idx') == station_idx:
                fitted_params = result.get('fitted_params')
                break
        
        if fitted_params is None:
            continue
            
        # Get PS00 data
        ps00_ts = displacement[station_idx, :]
        ps00_rate = ps00_rates[station_idx]
        coord = coordinates[station_idx]
        
        # Generate PS02C fitted signal
        ps02c_ts = generate_fitted_signal(fitted_params, time_years)
        
        # Calculate actual slopes from time series
        ps00_actual_slope = np.polyfit(time_years, ps00_ts, 1)[0]
        ps02c_actual_slope = np.polyfit(time_years, ps02c_ts, 1)[0]
        
        # Get fitted parameter trend
        ps02c_param_trend = fitted_params.trend
        
        result = {
            'station_idx': station_idx,
            'coordinates': coord,
            'ps00_rate': ps00_rate,
            'ps00_actual_slope': ps00_actual_slope,
            'ps02c_actual_slope': ps02c_actual_slope,
            'ps02c_param_trend': ps02c_param_trend,
            'ps00_ts': ps00_ts,
            'ps02c_ts': ps02c_ts
        }
        
        investigation_results.append(result)
        
        if station_idx == 7153:
            station_7153_found = True
        
        print(f"\n📍 Station {station_idx} [{coord[0]:.3f}, {coord[1]:.3f}]:")
        print(f"   PS00 rate: {ps00_rate:.2f} mm/year")
        print(f"   PS00 actual slope: {ps00_actual_slope:.2f} mm/year")
        print(f"   PS02C actual slope: {ps02c_actual_slope:.2f} mm/year") 
        print(f"   PS02C param trend: {ps02c_param_trend:.2f} mm/year")
        print(f"   Diff (actual vs param): {abs(ps02c_actual_slope - ps02c_param_trend):.3f} mm/year")
    
    if not station_7153_found:
        print(f"\n⚠️  Station 7153 not found in PS02C results or not in selected range")
        # Try to find it manually
        for result in successful_results[:20]:  # Check first 20
            if result.get('station_idx') == 7153:
                print(f"✅ Found Station 7153 in PS02C results!")
                break
    
    return investigation_results, time_years

def create_investigation_visualization(investigation_results, time_years):
    """Create visualization comparing different trend calculation methods"""
    print("📊 Creating investigation visualization...")
    
    n_stations = len(investigation_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fig.suptitle('Investigation: PS02C Trend Parameters vs Actual Time Series Slopes\\n'
                 'Examining why trend lines betray their time series', fontsize=14, fontweight='bold')
    
    for i, result in enumerate(investigation_results):
        if i >= 6:  # Only plot first 6
            break
            
        ax = axes[i]
        
        station_idx = result['station_idx']
        ps00_ts = result['ps00_ts']
        ps02c_ts = result['ps02c_ts']
        ps00_rate = result['ps00_rate']
        ps00_actual_slope = result['ps00_actual_slope']
        ps02c_actual_slope = result['ps02c_actual_slope']
        ps02c_param_trend = result['ps02c_param_trend']
        coord = result['coordinates']
        
        # Plot time series
        ax.plot(time_years, ps00_ts, 'b-', linewidth=2, alpha=0.8, label='PS00 Time Series')
        ax.plot(time_years, ps02c_ts, 'r-', linewidth=2, alpha=0.8, label='PS02C Fitted')
        
        # Plot trend lines using different methods
        # Method 1: PS00 with sign flip (current approach)
        trend_ps00_current = -ps00_rate * time_years
        ax.plot(time_years, trend_ps00_current, 'b--', linewidth=2, alpha=0.7, 
               label=f'PS00 Current: {ps00_rate:.1f}')
        
        # Method 2: PS02C with param trend + sign flip (current problematic approach)
        trend_ps02c_current = -ps02c_param_trend * time_years
        ax.plot(time_years, trend_ps02c_current, 'r--', linewidth=2, alpha=0.7,
               label=f'PS02C Current: {ps02c_param_trend:.1f}')
        
        # Method 3: PS02C with actual calculated slope (proposed fix)
        trend_ps02c_proposed = ps02c_actual_slope * time_years
        ax.plot(time_years, trend_ps02c_proposed, 'g--', linewidth=2, alpha=0.7,
               label=f'PS02C Proposed: {ps02c_actual_slope:.1f}')
        
        # Determine if subsiding or uplifting
        status = "Subsiding" if ps00_rate > 0 else "Uplifting"
        
        ax.set_title(f'{status} Station {station_idx}\\n'
                    f'Coord: [{coord[0]:.3f}, {coord[1]:.3f}]\\n'
                    f'Param vs Actual diff: {abs(ps02c_actual_slope - ps02c_param_trend):.2f}',
                    fontsize=10)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Displacement (mm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Check if proposed fix would work
        ps02c_ts_trend = "Up" if ps02c_ts[-1] > ps02c_ts[0] else "Down"
        proposed_trend = "Up" if ps02c_actual_slope > 0 else "Down"
        fix_works = ps02c_ts_trend == proposed_trend
        
        ax.text(0.02, 0.98, f'Fix works: {"✅" if fix_works else "❌"}', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen' if fix_works else 'lightcoral', alpha=0.7))
    
    plt.tight_layout()
    
    # Save investigation figure
    output_file = Path("figures/trend_parameter_investigation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved investigation figure: {output_file}")
    return str(output_file)

def summarize_investigation(investigation_results):
    """Summarize the investigation findings"""
    print("\n" + "="*80)
    print("📊 INVESTIGATION SUMMARY")
    print("="*80)
    
    param_vs_actual_diffs = []
    consistent_stations = 0
    
    for result in investigation_results:
        diff = abs(result['ps02c_actual_slope'] - result['ps02c_param_trend'])
        param_vs_actual_diffs.append(diff)
        
        # Check if the parameter trend would give correct direction
        ps02c_ts = result['ps02c_ts']
        ts_direction = "up" if ps02c_ts[-1] > ps02c_ts[0] else "down"
        param_direction = "up" if result['ps02c_param_trend'] > 0 else "down"
        actual_direction = "up" if result['ps02c_actual_slope'] > 0 else "down"
        
        param_consistent = ts_direction == param_direction
        actual_consistent = ts_direction == actual_direction
        
        print(f"Station {result['station_idx']}:")
        print(f"  Time series trend: {ts_direction}")
        print(f"  Param trend direction: {param_direction} ({'✅' if param_consistent else '❌'})")
        print(f"  Actual slope direction: {actual_direction} ({'✅' if actual_consistent else '❌'})")
        print(f"  Difference (param vs actual): {diff:.3f} mm/year")
        
        if actual_consistent:
            consistent_stations += 1
    
    print(f"\n📈 OVERALL FINDINGS:")
    print(f"   Average difference (param vs actual): {np.mean(param_vs_actual_diffs):.3f} mm/year")
    print(f"   Max difference: {np.max(param_vs_actual_diffs):.3f} mm/year")
    print(f"   Stations where actual slope gives correct direction: {consistent_stations}/{len(investigation_results)}")
    
    # Recommendation
    if consistent_stations > len(investigation_results) * 0.8:  # 80% threshold
        print(f"\n✅ RECOMMENDATION: Use actual calculated slopes from time series")
        print(f"   The proposed fix would work for {consistent_stations}/{len(investigation_results)} stations")
    else:
        print(f"\n❌ RECOMMENDATION: Do NOT implement the fix")
        print(f"   The proposed fix would only work for {consistent_stations}/{len(investigation_results)} stations")
    
    return consistent_stations >= len(investigation_results) * 0.8

def main():
    """Main investigation workflow"""
    print("=" * 80)
    print("🔍 PS02C TREND PARAMETER INVESTIGATION")
    print("=" * 80)
    
    try:
        # Investigate specific stations
        investigation_results, time_years = investigate_specific_stations()
        
        if not investigation_results:
            print("❌ No data available for investigation")
            return False
        
        # Create visualization
        investigation_file = create_investigation_visualization(investigation_results, time_years)
        
        # Summarize findings
        should_implement_fix = summarize_investigation(investigation_results)
        
        print(f"\n📁 Generated investigation file: {Path(investigation_file).name}")
        
        if should_implement_fix:
            print(f"\n🎯 CONCLUSION: Investigation supports implementing the proposed fix")
            print(f"   - Calculate trend slopes directly from fitted time series")
            print(f"   - Replace fitted_params.trend with np.polyfit(time_years, fitted_ts, 1)[0]")
        else:
            print(f"\n⚠️  CONCLUSION: Investigation does NOT support implementing the fix")
            print(f"   - Current approach may be correct despite appearances")
            print(f"   - Further investigation needed")
        
        return should_implement_fix
        
    except Exception as e:
        print(f"❌ ERROR during investigation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
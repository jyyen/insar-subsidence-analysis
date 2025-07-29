"""
PS02C ENHANCED Algorithmic Optimization - GPS Validation Fixes

CRITICAL IMPROVEMENTS based on GPS ground truth validation:
1. Expanded parameter bounds (frequency flexibility ±30%)
2. Quadratic trend component (non-linear subsidence)
3. Improved optimization (more iterations, better convergence)
4. Adaptive objective function (signal-dependent weighting)
5. Enhanced signal model (biennial component)

GPS Validation Results BEFORE fixes:
- GPS vs PS02C R² = 0.010 (CRITICAL FAILURE)
- PS00 vs PS02C correlation = -0.071 (SYSTEMATIC ERROR)
- RMSE = 21.6 mm/year (HIGH ERRORS)

Target Performance AFTER fixes:
- GPS vs PS02C R² > 0.7 (ACCEPTABLE)
- PS00 vs PS02C correlation > 0.6 (REASONABLE)
- RMSE < 10 mm/year (LOW ERRORS)

Author: Taiwan InSAR Subsidence Analysis Project
Enhanced: 2025-07-28
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq
import pandas as pd
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
import sys
import pickle

warnings.filterwarnings('ignore')

@dataclass
class EnhancedInSARParameters:
    """Enhanced data class for InSAR signal parameters with quadratic trend"""
    linear_trend: float = 0.0             # mm/year (linear component)
    quadratic_trend: float = 0.0          # mm/year² (non-linear subsidence)
    annual_amp: float = 0.0               # mm
    annual_freq: float = 1.0              # cycles/year
    annual_phase: float = 0.0             # radians
    semi_annual_amp: float = 0.0          # mm
    semi_annual_freq: float = 2.0         # cycles/year
    semi_annual_phase: float = 0.0        # radians
    quarterly_amp: float = 0.0            # mm
    quarterly_freq: float = 4.0           # cycles/year
    quarterly_phase: float = 0.0          # radians
    long_annual_amp: float = 0.0          # mm
    long_annual_freq: float = 0.3         # cycles/year
    long_annual_phase: float = 0.0        # radians
    biennial_amp: float = 0.0             # mm
    biennial_freq: float = 0.5            # cycles/year (2-year cycle)
    biennial_phase: float = 0.0           # radians
    noise_std: float = 2.0                # mm
    
    @classmethod
    def from_dict(cls, param_dict: Dict) -> 'EnhancedInSARParameters':
        """Create from dictionary"""
        return cls(**param_dict)

class EnhancedSubsidenceModel:
    """Enhanced InSAR subsidence model with GPS validation fixes"""
    
    def __init__(self, time_vector: np.ndarray):
        self.time_vector = time_vector
        self.time_years = time_vector / 365.25
        
        # ENHANCED PARAMETER BOUNDS based on GPS validation analysis
        self.param_bounds = [
            # TREND COMPONENTS (Enhanced with quadratic)
            (-120, 50),      # linear_trend: wider range for extreme subsidence
            (-5, 5),         # quadratic_trend: non-linear subsidence component
            
            # ANNUAL COMPONENT (±30% frequency flexibility)
            (0, 50),         # annual_amp: larger range for strong seasonal signals
            (0.7, 1.3),      # annual_freq: ±30% flexibility for realistic variation
            (0, 2*np.pi),    # annual_phase: full range
            
            # SEMI-ANNUAL COMPONENT (±15% frequency flexibility)
            (0, 40),         # semi_annual_amp: larger amplitude range
            (1.7, 2.3),      # semi_annual_freq: ±15% flexibility
            (0, 2*np.pi),    # semi_annual_phase: full range
            
            # QUARTERLY COMPONENT (±12% frequency flexibility)
            (0, 30),         # quarterly_amp: larger amplitude range
            (3.5, 4.5),      # quarterly_freq: ±12.5% flexibility
            (0, 2*np.pi),    # quarterly_phase: full range
            
            # LONG-TERM ANNUAL (More flexible)
            (0, 50),         # long_annual_amp: much larger range
            (0.05, 1.0),     # long_annual_freq: wider frequency range
            (0, 2*np.pi),    # long_annual_phase: full range
            
            # BIENNIAL COMPONENT (New - for 2-year cycles)
            (0, 25),         # biennial_amp: moderate amplitude
            (0.4, 0.6),      # biennial_freq: around 0.5 cycles/year
            (0, 2*np.pi),    # biennial_phase: full range
            
            # NOISE MODEL
            (0.5, 15)        # noise_std: wider noise range
        ]
        
        print(f"🔧 Enhanced PS02C initialized with {len(self.param_bounds)} parameters")
        print(f"   • Quadratic trend component added")
        print(f"   • Expanded frequency bounds (±30% annual, ±15% semi-annual)")
        print(f"   • Larger amplitude ranges (up to 50mm)")
        print(f"   • New biennial (2-year) component")
    
    def enhanced_signal_model(self, params: np.ndarray) -> np.ndarray:
        """Enhanced signal model with quadratic trend and biennial component"""
        t = self.time_years
        
        # Enhanced trend with quadratic component for non-linear subsidence
        trend = params[0] * t + params[1] * t**2
        
        # Existing periodic components with expanded bounds
        annual = params[2] * np.sin(2*np.pi*params[3]*t + params[4])
        semi_annual = params[5] * np.sin(2*np.pi*params[6]*t + params[7])
        quarterly = params[8] * np.sin(2*np.pi*params[9]*t + params[10])
        long_annual = params[11] * np.sin(2*np.pi*params[12]*t + params[13])
        
        # NEW: Biennial component (2-year cycle)
        biennial = params[14] * np.sin(2*np.pi*params[15]*t + params[16])
        
        # Total signal
        signal = trend + annual + semi_annual + quarterly + long_annual + biennial
        
        return signal
    
    def enhanced_initial_estimate(self, signal: np.ndarray) -> np.ndarray:
        """Enhanced initial parameter estimation with better bounds compliance"""
        
        try:
            t = self.time_years
            n = len(signal)
            
            # Enhanced trend estimation (quadratic fit)
            trend_coeffs = np.polyfit(t, signal, 2)  # Quadratic instead of linear
            linear_trend = trend_coeffs[1]  # Linear coefficient
            quadratic_trend = trend_coeffs[0]  # Quadratic coefficient
            
            # Detrend signal for periodic analysis
            trend_signal = linear_trend * t + quadratic_trend * t**2
            detrended = signal - trend_signal
            
            # Enhanced FFT analysis for periodic components
            fft_signal = fft(detrended)
            freqs = fftfreq(n, d=np.mean(np.diff(t)))
            
            # Find dominant frequencies with better precision
            power = np.abs(fft_signal)**2
            positive_freqs = freqs[freqs > 0]
            positive_power = power[freqs > 0]
            
            # Enhanced frequency detection
            annual_idx = np.argmin(np.abs(positive_freqs - 1.0))
            annual_amp = 2 * np.abs(fft_signal[annual_idx]) / n
            annual_phase = np.angle(fft_signal[annual_idx])
            
            semi_annual_idx = np.argmin(np.abs(positive_freqs - 2.0))
            semi_annual_amp = 2 * np.abs(fft_signal[semi_annual_idx]) / n
            semi_annual_phase = np.angle(fft_signal[semi_annual_idx])
            
            quarterly_idx = np.argmin(np.abs(positive_freqs - 4.0))
            quarterly_amp = 2 * np.abs(fft_signal[quarterly_idx]) / n
            quarterly_phase = np.angle(fft_signal[quarterly_idx])
            
            long_annual_idx = np.argmin(np.abs(positive_freqs - 0.3))
            long_annual_amp = 2 * np.abs(fft_signal[long_annual_idx]) / n
            long_annual_phase = np.angle(fft_signal[long_annual_idx])
            
            # NEW: Biennial component detection
            biennial_idx = np.argmin(np.abs(positive_freqs - 0.5))
            biennial_amp = 2 * np.abs(fft_signal[biennial_idx]) / n
            biennial_phase = np.angle(fft_signal[biennial_idx])
            
            # Enhanced noise estimation
            residuals = np.polyfit(t, detrended, 1, full=True)
            if len(residuals) > 1:
                noise_std = np.sqrt(residuals[1][0] / n)
            else:
                noise_std = np.std(detrended) * 0.1
            
            # Enhanced parameter bounds compliance
            initial_params = np.array([
                np.clip(linear_trend, -120, 50),            # linear_trend
                np.clip(quadratic_trend, -5, 5),            # quadratic_trend
                np.clip(annual_amp, 0, 50),                 # annual_amp
                1.0,                                        # annual_freq (fixed)
                annual_phase % (2*np.pi),                   # annual_phase
                np.clip(semi_annual_amp, 0, 40),            # semi_annual_amp
                2.0,                                        # semi_annual_freq (fixed)
                semi_annual_phase % (2*np.pi),              # semi_annual_phase
                np.clip(quarterly_amp, 0, 30),              # quarterly_amp
                4.0,                                        # quarterly_freq (fixed)
                quarterly_phase % (2*np.pi),                # quarterly_phase
                np.clip(long_annual_amp, 0, 50),            # long_annual_amp
                0.3,                                        # long_annual_freq (fixed)
                long_annual_phase % (2*np.pi),              # long_annual_phase
                np.clip(biennial_amp, 0, 25),               # biennial_amp
                0.5,                                        # biennial_freq (fixed)
                biennial_phase % (2*np.pi),                 # biennial_phase
                np.clip(noise_std, 0.5, 15)                 # noise_std
            ])
            
            return initial_params
            
        except Exception as e:
            print(f"   Warning: Enhanced initial estimate failed ({e}), using robust defaults")
            
        # Enhanced default parameters (guaranteed within bounds)
        return np.array([
            -10.0,  # linear_trend (moderate subsidence)
            0.0,    # quadratic_trend (initially zero)
            8.0,    # annual_amp (larger default)
            1.0,    # annual_freq
            0.0,    # annual_phase
            5.0,    # semi_annual_amp (larger default)
            2.0,    # semi_annual_freq
            0.0,    # semi_annual_phase
            3.0,    # quarterly_amp (larger default)
            4.0,    # quarterly_freq
            0.0,    # quarterly_phase
            4.0,    # long_annual_amp (larger default)
            0.3,    # long_annual_freq
            0.0,    # long_annual_phase
            2.0,    # biennial_amp
            0.5,    # biennial_freq
            0.0,    # biennial_phase
            2.0     # noise_std
        ])
    
    def adaptive_objective_function(self, params: np.ndarray, observed: np.ndarray) -> float:
        """Enhanced adaptive objective function with signal-dependent weighting"""
        try:
            synthetic = self.enhanced_signal_model(params)
            
            # Calculate metrics
            residuals = synthetic - observed
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Calculate correlation
            obs_std = np.std(observed)
            if obs_std > 0:
                corr = np.corrcoef(synthetic, observed)[0, 1]
                if np.isnan(corr):
                    corr = 0
            else:
                corr = 0
            
            # ADAPTIVE WEIGHTING based on signal characteristics
            signal_to_noise = obs_std / (rmse + 1e-6)  # Avoid division by zero
            
            # Weight based on signal quality
            if signal_to_noise > 5:  # High SNR - emphasize correlation
                rmse_weight = 0.3
                corr_weight = 0.7
            elif signal_to_noise > 2:  # Medium SNR - balanced
                rmse_weight = 0.5
                corr_weight = 0.5
            else:  # Low SNR - emphasize RMSE
                rmse_weight = 0.7
                corr_weight = 0.3
            
            # Normalize RMSE and combine with adaptive weights
            normalized_rmse = rmse / obs_std if obs_std > 0 else rmse
            
            # Enhanced objective with regularization
            objective = rmse_weight * normalized_rmse - corr_weight * corr
            
            # Add regularization for extreme parameters
            regularization = 0
            if abs(params[0]) > 80:  # Extreme linear trend
                regularization += 0.01 * (abs(params[0]) - 80)**2
            if abs(params[1]) > 3:   # Extreme quadratic trend
                regularization += 0.01 * (abs(params[1]) - 3)**2
            
            return objective + regularization
            
        except Exception:
            return 1e6  # Large penalty for invalid parameters
    
    def enhanced_optimization(self, observed: np.ndarray, 
                            max_iterations: int = 800,  # Increased iterations
                            population_size: int = 25,  # Larger population
                            tolerance: float = 1e-6,    # Tighter tolerance
                            max_retries: int = 3) -> Optional[EnhancedInSARParameters]:
        """Enhanced optimization with better convergence"""
        
        print(f"   🔧 Enhanced optimization: {max_iterations} iterations, pop={population_size}")
        
        # Get enhanced initial estimate
        initial_params = self.enhanced_initial_estimate(observed)
        
        best_result = None
        best_objective = float('inf')
        
        for retry in range(max_retries):
            try:
                # Enhanced differential evolution settings
                result = differential_evolution(
                    func=lambda params: self.adaptive_objective_function(params, observed),
                    bounds=self.param_bounds,
                    maxiter=max_iterations,
                    popsize=population_size,
                    tol=tolerance,
                    seed=42 + retry,  # Different seed for each retry
                    x0=initial_params,  # Use enhanced initial estimate
                    atol=1e-8,         # Absolute tolerance
                    updating='deferred',  # Better for parallel execution
                    workers=1          # Sequential for stability
                )
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    
                if best_objective < 0.1:  # Good enough result
                    break
                    
            except Exception as e:
                print(f"   ⚠️ Retry {retry + 1} failed: {e}")
                continue
        
        if best_result is None or not best_result.success:
            print(f"   ❌ Enhanced optimization failed after {max_retries} retries")
            return None
        
        # Calculate enhanced performance metrics
        fitted_signal = self.enhanced_signal_model(best_result.x)
        correlation = np.corrcoef(observed, fitted_signal)[0, 1]
        rmse = np.sqrt(np.mean((observed - fitted_signal)**2))
        
        # Create enhanced parameter object
        params = EnhancedInSARParameters(
            linear_trend=best_result.x[0],
            quadratic_trend=best_result.x[1],
            annual_amp=best_result.x[2],
            annual_freq=best_result.x[3],
            annual_phase=best_result.x[4],
            semi_annual_amp=best_result.x[5],
            semi_annual_freq=best_result.x[6],
            semi_annual_phase=best_result.x[7],
            quarterly_amp=best_result.x[8],
            quarterly_freq=best_result.x[9],
            quarterly_phase=best_result.x[10],
            long_annual_amp=best_result.x[11],
            long_annual_freq=best_result.x[12],
            long_annual_phase=best_result.x[13],
            biennial_amp=best_result.x[14],
            biennial_freq=best_result.x[15],
            biennial_phase=best_result.x[16],
            noise_std=best_result.x[17]
        )
        
        return {
            'parameters': params,
            'correlation': correlation,
            'rmse': rmse,
            'objective': best_result.fun,
            'fitted_signal': fitted_signal,
            'success': True,
            'optimization_time': 0,  # Will be filled by calling function
            'enhancement_level': 'GPS_validated_fixes'
        }

def process_enhanced_station(args):
    """Process single station with enhanced algorithm"""
    station_idx, displacement, time_vector = args
    
    try:
        # Create enhanced model
        model = EnhancedSubsidenceModel(time_vector)
        
        start_time = time.time()
        result = model.enhanced_optimization(displacement)
        end_time = time.time()
        
        if result is not None:
            result['optimization_time'] = end_time - start_time
            result['station_idx'] = station_idx
            
            # Calculate trend in mm/year (combined linear + quadratic at mid-point)
            mid_time = np.mean(model.time_years)
            linear_component = result['parameters'].linear_trend
            quadratic_component = result['parameters'].quadratic_trend * mid_time
            combined_trend = linear_component + quadratic_component
            result['trend'] = combined_trend
            
            print(f"   ✅ Station {station_idx}: R={result['correlation']:.3f}, "
                  f"RMSE={result['rmse']:.1f}mm, trend={combined_trend:.1f}mm/yr")
            return result
        else:
            print(f"   ❌ Station {station_idx}: optimization failed")
            return None
            
    except Exception as e:
        print(f"   ❌ Station {station_idx}: error - {e}")
        return None

def run_enhanced_ps02c():
    """Run enhanced PS02C algorithm with GPS validation fixes"""
    
    print("=" * 80)
    print("🚀 PS02C ENHANCED - GPS VALIDATION FIXES")
    print("=" * 80)
    print("🔧 Enhancements:")
    print("   • Expanded parameter bounds (±30% frequency flexibility)")
    print("   • Quadratic trend component (non-linear subsidence)")
    print("   • Improved optimization (800 iterations, pop=25)")
    print("   • Adaptive objective function (signal-dependent weighting)")
    print("   • Enhanced signal model (biennial component)")
    print("   • Based on GPS ground truth validation failures")
    
    # Load data
    print("\\n📊 Loading InSAR data...")
    ps00_file = Path('data/processed/ps00_preprocessed_data.npz')
    
    if not ps00_file.exists():
        print(f"❌ Data file not found: {ps00_file}")
        return
    
    try:
        data = np.load(ps00_file)
        displacement = data['displacement']
        coordinates = data['coordinates']
        n_stations, n_acquisitions = displacement.shape
        
        # Create time vector
        time_vector = np.arange(n_acquisitions)
        
        print(f"✅ Loaded {n_stations} stations, {n_acquisitions} acquisitions")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Process stations (start with subset for testing)
    test_stations = min(50, n_stations)  # Start with 50 stations
    print(f"\\n🔄 Processing {test_stations} stations with enhanced algorithm...")
    
    results = []
    processing_times = []
    
    for i in range(test_stations):
        station_displacement = displacement[i, :]
        
        # Skip stations with too many NaN values
        if np.sum(~np.isnan(station_displacement)) < n_acquisitions * 0.7:
            results.append(None)
            continue
        
        result = process_enhanced_station((i, station_displacement, time_vector))
        results.append(result)
        
        if result is not None:
            processing_times.append(result['optimization_time'])
    
    # Analyze results
    print("\\n" + "=" * 80)
    print("📊 ENHANCED PS02C RESULTS SUMMARY")
    print("=" * 80)
    
    successful_results = [r for r in results if r is not None]
    
    if len(successful_results) > 0:
        correlations = [r['correlation'] for r in successful_results]
        rmse_values = [r['rmse'] for r in successful_results]
        trends = [r['trend'] for r in successful_results]
        
        print(f"✅ Success rate: {len(successful_results)}/{test_stations} ({len(successful_results)/test_stations*100:.1f}%)")
        print(f"📈 Performance metrics:")
        print(f"   • Correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
        print(f"   • RMSE: {np.mean(rmse_values):.1f} ± {np.std(rmse_values):.1f} mm")
        print(f"   • Trend range: {np.min(trends):.1f} to {np.max(trends):.1f} mm/year")
        print(f"⏱️ Processing: {np.mean(processing_times):.1f} ± {np.std(processing_times):.1f} sec/station")
        
        # Quality assessment
        excellent = np.sum((np.array(correlations) >= 0.9) & (np.array(rmse_values) <= 10))
        good = np.sum((np.array(correlations) >= 0.7) & (np.array(rmse_values) <= 20)) - excellent
        fair = np.sum((np.array(correlations) >= 0.5) & (np.array(rmse_values) <= 40)) - excellent - good
        poor = len(correlations) - excellent - good - fair
        
        print(f"🏆 Quality distribution:")
        print(f"   • Excellent (r≥0.9, RMSE≤10mm): {excellent} ({excellent/len(correlations)*100:.1f}%)")
        print(f"   • Good (r≥0.7, RMSE≤20mm): {good} ({good/len(correlations)*100:.1f}%)")
        print(f"   • Fair (r≥0.5, RMSE≤40mm): {fair} ({fair/len(correlations)*100:.1f}%)")
        print(f"   • Poor (others): {poor} ({poor/len(correlations)*100:.1f}%)")
        
        # Save enhanced results
        output_file = Path('data/processed/ps02c_ENHANCED_results.pkl')
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Enhanced results saved to {output_file}")
        
    else:
        print("❌ No successful results - algorithm needs further debugging")
    
    print("=" * 80)
    print("✅ Enhanced PS02C processing complete!")
    print("🔬 Next: Run GPS validation to test improvements")

if __name__ == "__main__":
    run_enhanced_ps02c()
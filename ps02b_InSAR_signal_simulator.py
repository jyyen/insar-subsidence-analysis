#!/usr/bin/env python3
"""
PS02B - InSAR Time Series Simulator and Fitter

Comprehensive tool for simulating and fitting InSAR deformation signals with multiple components:
- Linear trends (subsidence/uplift)
- Annual, semi-annual, quarterly, and long-period signals
- High-frequency noise and atmospheric effects
- Parameter optimization and component analysis

Author: Taiwan InSAR Subsidence Analysis Project
Created: 2025-07-26
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import pearsonr
from scipy import signal
import pandas as pd
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InSARParameters:
    """Data class for InSAR signal parameters"""
    trend: float = 0.0                    # mm/year
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
    noise_std: float = 2.0                # mm
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
    
    @classmethod
    def from_dict(cls, params: Dict):
        """Create from dictionary"""
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

class InSARTimeSeries:
    """InSAR time series simulation and analysis"""
    
    def __init__(self, time_vector: np.ndarray, time_unit: str = 'years'):
        """
        Initialize InSAR time series simulator
        
        Parameters:
        -----------
        time_vector : np.ndarray
            Time values (years for annual analysis, days for detailed)
        time_unit : str
            'years' or 'days'
        """
        self.time = np.array(time_vector)
        self.time_unit = time_unit
        
        # Convert to years if needed
        if time_unit == 'days':
            self.time_years = self.time / 365.25
        else:
            self.time_years = self.time
        
        # Define realistic parameter ranges for Taiwan InSAR
        self.param_ranges = {
            'trend': (-60, 20),               # mm/year (strong subsidence to slight uplift)
            'annual_amp': (0, 25),            # mm (seasonal groundwater)
            'annual_freq': (0.95, 1.05),      # cycles/year (around 365 days)
            'semi_annual_amp': (0, 15),       # mm (semi-annual patterns)
            'semi_annual_freq': (1.9, 2.1),   # cycles/year (around 183 days)
            'quarterly_amp': (0, 10),         # mm (quarterly pumping)
            'quarterly_freq': (3.8, 4.2),     # cycles/year (around 91 days)
            'long_annual_amp': (0, 30),       # mm (multi-year cycles)
            'long_annual_freq': (0.1, 0.8),   # cycles/year (1.25-10 year periods)
            'noise_std': (0.5, 8),            # mm (InSAR measurement noise)
            'annual_phase': (0, 2*np.pi),     # radians
            'semi_annual_phase': (0, 2*np.pi), # radians
            'quarterly_phase': (0, 2*np.pi),   # radians
            'long_annual_phase': (0, 2*np.pi)  # radians
        }
    
    def generate_signal(self, params: InSARParameters) -> Dict:
        """
        Generate synthetic InSAR time series with component breakdown
        
        Parameters:
        -----------
        params : InSARParameters
            Signal parameters
            
        Returns:
        --------
        Dict with 'signal', 'components', and metadata
        """
        t = self.time_years
        
        # Component calculations
        components = {}
        
        # 1. Linear trend (subsidence/uplift)
        components['trend'] = params.trend * t
        
        # 2. Annual signal (seasonal groundwater, thermal expansion)
        components['annual'] = params.annual_amp * np.sin(
            2 * np.pi * params.annual_freq * t + params.annual_phase
        )
        
        # 3. Semi-annual signal (bi-modal seasonal patterns)
        components['semi_annual'] = params.semi_annual_amp * np.sin(
            2 * np.pi * params.semi_annual_freq * t + params.semi_annual_phase
        )
        
        # 4. Quarterly signal (pumping schedules, irrigation)
        components['quarterly'] = params.quarterly_amp * np.sin(
            2 * np.pi * params.quarterly_freq * t + params.quarterly_phase
        )
        
        # 5. Long-period signal (multi-year climate/pumping cycles)
        components['long_annual'] = params.long_annual_amp * np.sin(
            2 * np.pi * params.long_annual_freq * t + params.long_annual_phase
        )
        
        # 6. High-frequency noise (atmospheric, measurement errors)
        np.random.seed(42)  # Reproducible noise
        components['noise'] = np.random.normal(0, params.noise_std, len(t))
        
        # Combine all components
        signal = sum(components.values())
        
        # Calculate component statistics
        component_stats = {}
        for name, comp in components.items():
            if name != 'noise':
                component_stats[name] = {
                    'amplitude': np.std(comp),
                    'contribution': np.var(comp) / np.var(signal - components['noise']),
                    'range': [np.min(comp), np.max(comp)]
                }
        
        return {
            'signal': signal,
            'components': components,
            'stats': component_stats,
            'snr': np.std(signal - components['noise']) / params.noise_std,
            'params': params
        }
    
    def plot_signal_breakdown(self, result: Dict, title: str = "InSAR Signal Breakdown", 
                            save_path: Optional[str] = None):
        """Plot signal with component breakdown"""
        
        signal = result['signal']
        components = result['components']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Complete signal
        ax = axes[0, 0]
        ax.plot(self.time, signal, 'k-', linewidth=2, alpha=0.8)
        ax.set_title('Complete Signal', fontweight='bold')
        ax.set_ylabel('Displacement (mm)')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'STD: {np.std(signal):.2f} mm\nRange: {np.ptp(signal):.1f} mm\nSNR: {result["snr"]:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Plot 2: Trend component
        ax = axes[0, 1]
        ax.plot(self.time, components['trend'], 'r-', linewidth=2, alpha=0.8)
        ax.set_title(f'Trend: {result["params"].trend:.1f} mm/year', fontweight='bold')
        ax.set_ylabel('Displacement (mm)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Annual component
        ax = axes[1, 0]
        ax.plot(self.time, components['annual'], 'b-', linewidth=2, alpha=0.8)
        period_days = 365.25 / result["params"].annual_freq
        ax.set_title(f'Annual: {result["params"].annual_amp:.1f} mm, {period_days:.0f} days', fontweight='bold')
        ax.set_ylabel('Displacement (mm)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Semi-annual component
        ax = axes[1, 1]
        ax.plot(self.time, components['semi_annual'], 'g-', linewidth=2, alpha=0.8)
        period_days = 365.25 / result["params"].semi_annual_freq
        ax.set_title(f'Semi-annual: {result["params"].semi_annual_amp:.1f} mm, {period_days:.0f} days', fontweight='bold')
        ax.set_ylabel('Displacement (mm)')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Quarterly + Long-period
        ax = axes[2, 0]
        ax.plot(self.time, components['quarterly'], 'orange', linewidth=2, alpha=0.8, label='Quarterly')
        ax.plot(self.time, components['long_annual'], 'purple', linewidth=2, alpha=0.8, label='Long-period')
        ax.set_title('Other Periodic Components', fontweight='bold')
        ax.set_ylabel('Displacement (mm)')
        ax.set_xlabel(f'Time ({self.time_unit})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Noise
        ax = axes[2, 1]
        ax.plot(self.time, components['noise'], 'gray', linewidth=1, alpha=0.7)
        ax.set_title(f'Noise: σ = {result["params"].noise_std:.1f} mm', fontweight='bold')
        ax.set_ylabel('Displacement (mm)')
        ax.set_xlabel(f'Time ({self.time_unit})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved signal breakdown: {save_path}")
        
        plt.show()
        
        return fig

class InSARFitter:
    """Fit InSAR models to observed data using optimization"""
    
    def __init__(self, time_vector: np.ndarray, observed_signal: np.ndarray, 
                 time_unit: str = 'years'):
        """
        Initialize fitter
        
        Parameters:
        -----------
        time_vector : np.ndarray
            Time values
        observed_signal : np.ndarray  
            Observed displacement values (mm)
        time_unit : str
            'years' or 'days'
        """
        self.simulator = InSARTimeSeries(time_vector, time_unit)
        self.observed = np.array(observed_signal)
        self.time = time_vector
        self.time_unit = time_unit
        
        # Results storage
        self.best_params = None
        self.best_result = None
        self.fit_history = []
        
        # Validation
        if len(self.observed) != len(time_vector):
            raise ValueError("Time vector and observed signal must have same length")
        
        # Remove NaN values
        valid_mask = ~np.isnan(self.observed)
        if np.sum(valid_mask) < len(self.observed) * 0.7:
            raise ValueError("Too many NaN values in observed signal")
        
        if np.any(~valid_mask):
            print(f"⚠️  Interpolating {np.sum(~valid_mask)} missing values")
            self.observed = np.interp(
                np.arange(len(self.observed)),
                np.where(valid_mask)[0],
                self.observed[valid_mask]
            )
    
    def objective_function(self, param_array: np.ndarray) -> float:
        """
        Objective function for optimization
        
        Parameters:
        -----------
        param_array : np.ndarray
            Parameter values in order
            
        Returns:
        --------
        float : objective value (lower is better)
        """
        try:
            # Convert array to parameters
            params = self.array_to_params(param_array)
            
            # Generate synthetic signal
            result = self.simulator.generate_signal(params)
            synthetic = result['signal']
            
            # Calculate metrics
            residuals = synthetic - self.observed
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Correlation coefficient
            corr, _ = pearsonr(synthetic, self.observed)
            if np.isnan(corr):
                corr = 0
            
            # Combined objective: minimize RMSE, maximize correlation
            # Normalize by observed signal std
            obs_std = np.std(self.observed)
            normalized_rmse = rmse / obs_std if obs_std > 0 else rmse
            
            objective = normalized_rmse - corr
            
            return objective
            
        except Exception as e:
            # Return large penalty for invalid parameters
            return 1e6
    
    def array_to_params(self, param_array: np.ndarray) -> InSARParameters:
        """Convert parameter array to InSARParameters object"""
        param_names = [
            'trend', 'annual_amp', 'annual_freq', 'annual_phase',
            'semi_annual_amp', 'semi_annual_freq', 'semi_annual_phase',
            'quarterly_amp', 'quarterly_freq', 'quarterly_phase',
            'long_annual_amp', 'long_annual_freq', 'long_annual_phase',
            'noise_std'
        ]
        
        param_dict = dict(zip(param_names, param_array))
        return InSARParameters.from_dict(param_dict)
    
    def fit(self, method: str = 'differential_evolution', maxiter: int = 1000) -> InSARParameters:
        """
        Fit parameters to observed signal
        
        Parameters:
        -----------
        method : str
            Optimization method ('differential_evolution' or 'minimize')
        maxiter : int
            Maximum iterations
            
        Returns:
        --------
        InSARParameters : best fitted parameters
        """
        print(f"🔍 Fitting InSAR signal model...")
        print(f"   Data points: {len(self.observed)}")
        print(f"   Time range: {self.time[0]:.2f} to {self.time[-1]:.2f} {self.time_unit}")
        print(f"   Signal range: {np.min(self.observed):.1f} to {np.max(self.observed):.1f} mm")
        
        # Set up bounds for optimization
        bounds = [
            self.simulator.param_ranges['trend'],
            self.simulator.param_ranges['annual_amp'],
            self.simulator.param_ranges['annual_freq'],
            self.simulator.param_ranges['annual_phase'],
            self.simulator.param_ranges['semi_annual_amp'],
            self.simulator.param_ranges['semi_annual_freq'],
            self.simulator.param_ranges['semi_annual_phase'],
            self.simulator.param_ranges['quarterly_amp'],
            self.simulator.param_ranges['quarterly_freq'],
            self.simulator.param_ranges['quarterly_phase'],
            self.simulator.param_ranges['long_annual_amp'],
            self.simulator.param_ranges['long_annual_freq'],
            self.simulator.param_ranges['long_annual_phase'],
            self.simulator.param_ranges['noise_std']
        ]
        
        start_time = time.time()
        
        # Run optimization with parallel workers
        if method == 'differential_evolution':
            # Use multiple workers for parallel optimization on M1 Ultra
            import multiprocessing as mp
            n_workers = min(8, mp.cpu_count())  # Conservative for stability
            
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=maxiter,
                popsize=15,
                tol=0.001,
                seed=42,
                disp=False,
                workers=n_workers,  # Enable parallel evaluation
                updating='deferred'  # Better for parallel workers
            )
        else:
            # Try multiple random starting points
            best_result = None
            best_obj = np.inf
            
            for i in range(10):
                # Random starting point within bounds
                x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
                
                try:
                    result = minimize(
                        self.objective_function,
                        x0,
                        bounds=bounds,
                        method='L-BFGS-B'
                    )
                    
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_result = result
                except:
                    continue
            
            result = best_result
        
        elapsed_time = time.time() - start_time
        
        if result is None or not hasattr(result, 'x'):
            raise RuntimeError("Optimization failed")
        
        # Store best parameters and generate result
        self.best_params = self.array_to_params(result.x)
        self.best_result = self.simulator.generate_signal(self.best_params)
        
        # Calculate quality metrics
        synthetic = self.best_result['signal']
        residuals = synthetic - self.observed
        rmse = np.sqrt(np.mean(residuals**2))
        corr, _ = pearsonr(synthetic, self.observed)
        
        print(f"✅ Optimization completed in {elapsed_time:.2f} seconds")
        print(f"   RMSE: {rmse:.2f} mm")
        print(f"   Correlation: {corr:.3f}")
        print(f"   Objective value: {result.fun:.4f}")
        
        return self.best_params
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot comprehensive fitting results"""
        
        if self.best_params is None:
            raise ValueError("Must run fit() first")
        
        synthetic = self.best_result['signal']
        components = self.best_result['components']
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        fig.suptitle('InSAR Signal Fitting Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Observed vs Fitted
        ax = axes[0, 0]
        ax.plot(self.time, self.observed, 'r.', label='Observed', markersize=3, alpha=0.7)
        ax.plot(self.time, synthetic, 'b-', label='Fitted', linewidth=2, alpha=0.8)
        ax.set_ylabel('Displacement (mm)')
        ax.legend()
        ax.set_title('Observed vs Fitted Signal')
        ax.grid(True, alpha=0.3)
        
        # Add fit statistics
        residuals = synthetic - self.observed
        rmse = np.sqrt(np.mean(residuals**2))
        corr, _ = pearsonr(synthetic, self.observed)
        stats_text = f'RMSE: {rmse:.2f} mm\nCorr: {corr:.3f}\nSNR: {self.best_result["snr"]:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Plot 2: Residuals
        ax = axes[0, 1]
        ax.plot(self.time, residuals, 'g.', markersize=3, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('Residuals (mm)')
        ax.set_title(f'Residuals (RMSE: {rmse:.2f} mm)')
        ax.grid(True, alpha=0.3)
        
        # Add residual statistics
        residual_std = np.std(residuals)
        residual_range = np.ptp(residuals)
        res_stats = f'STD: {residual_std:.2f} mm\nRange: {residual_range:.1f} mm'
        ax.text(0.02, 0.98, res_stats, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Plot 3: Trend component
        ax = axes[1, 0]
        ax.plot(self.time, components['trend'], 'r-', linewidth=2, alpha=0.8)
        ax.set_ylabel('Displacement (mm)')
        ax.set_title(f'Trend Component: {self.best_params.trend:.2f} mm/year')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Annual component
        ax = axes[1, 1]
        ax.plot(self.time, components['annual'], 'b-', linewidth=2, alpha=0.8)
        ax.set_ylabel('Displacement (mm)')
        period_days = 365.25 / self.best_params.annual_freq
        ax.set_title(f'Annual: {self.best_params.annual_amp:.1f} mm, {period_days:.0f} days')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Semi-annual component
        ax = axes[2, 0]
        ax.plot(self.time, components['semi_annual'], 'g-', linewidth=2, alpha=0.8)
        ax.set_ylabel('Displacement (mm)')
        period_days = 365.25 / self.best_params.semi_annual_freq
        ax.set_title(f'Semi-annual: {self.best_params.semi_annual_amp:.1f} mm, {period_days:.0f} days')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Quarterly component
        ax = axes[2, 1]
        ax.plot(self.time, components['quarterly'], 'orange', linewidth=2, alpha=0.8)
        ax.set_ylabel('Displacement (mm)')
        period_days = 365.25 / self.best_params.quarterly_freq
        ax.set_title(f'Quarterly: {self.best_params.quarterly_amp:.1f} mm, {period_days:.0f} days')
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Long-period component
        ax = axes[3, 0]
        ax.plot(self.time, components['long_annual'], 'purple', linewidth=2, alpha=0.8)
        ax.set_ylabel('Displacement (mm)')
        ax.set_xlabel(f'Time ({self.time_unit})')
        period_years = 1 / self.best_params.long_annual_freq
        ax.set_title(f'Long-period: {self.best_params.long_annual_amp:.1f} mm, {period_years:.1f} years')
        ax.grid(True, alpha=0.3)
        
        # Plot 8: All components together
        ax = axes[3, 1]
        if np.std(components['trend']) > 0.1:
            ax.plot(self.time, components['trend'], 'r-', label='Trend', alpha=0.7)
        if self.best_params.annual_amp > 0.1:
            ax.plot(self.time, components['annual'], 'b-', label='Annual', alpha=0.7)
        if self.best_params.semi_annual_amp > 0.1:
            ax.plot(self.time, components['semi_annual'], 'g-', label='Semi-annual', alpha=0.7)
        if self.best_params.quarterly_amp > 0.1:
            ax.plot(self.time, components['quarterly'], 'orange', label='Quarterly', alpha=0.7)
        if self.best_params.long_annual_amp > 0.1:
            ax.plot(self.time, components['long_annual'], 'purple', label='Long-period', alpha=0.7)
        
        ax.set_ylabel('Displacement (mm)')
        ax.set_xlabel(f'Time ({self.time_unit})')
        ax.set_title('All Components')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved fitting results: {save_path}")
        
        plt.show()
        
        return fig
    
    def print_parameters(self):
        """Print fitted parameters in readable format"""
        
        if self.best_params is None:
            raise ValueError("Must run fit() first")
        
        print("\n" + "="*60)
        print("🎯 FITTED InSAR SIGNAL PARAMETERS")
        print("="*60)
        
        # Trend
        print(f"\n📈 TREND COMPONENT:")
        print(f"   Rate: {self.best_params.trend:.2f} mm/year")
        if self.best_params.trend < -1:
            print(f"   → {abs(self.best_params.trend):.1f} mm/year SUBSIDENCE")
        elif self.best_params.trend > 1:
            print(f"   → {self.best_params.trend:.1f} mm/year UPLIFT")
        else:
            print(f"   → Stable (minimal trend)")
        
        # Annual
        print(f"\n🗓️  ANNUAL COMPONENT:")
        print(f"   Amplitude: {self.best_params.annual_amp:.2f} mm")
        print(f"   Frequency: {self.best_params.annual_freq:.3f} cycles/year")
        print(f"   Period: {365.25/self.best_params.annual_freq:.1f} days")
        print(f"   Phase: {self.best_params.annual_phase:.2f} rad ({self.best_params.annual_phase*180/np.pi:.0f}°)")
        
        # Semi-annual
        print(f"\n📅 SEMI-ANNUAL COMPONENT:")
        print(f"   Amplitude: {self.best_params.semi_annual_amp:.2f} mm")
        print(f"   Frequency: {self.best_params.semi_annual_freq:.3f} cycles/year")
        print(f"   Period: {365.25/self.best_params.semi_annual_freq:.1f} days")
        print(f"   Phase: {self.best_params.semi_annual_phase:.2f} rad ({self.best_params.semi_annual_phase*180/np.pi:.0f}°)")
        
        # Quarterly
        print(f"\n📋 QUARTERLY COMPONENT:")
        print(f"   Amplitude: {self.best_params.quarterly_amp:.2f} mm")
        print(f"   Frequency: {self.best_params.quarterly_freq:.3f} cycles/year")
        print(f"   Period: {365.25/self.best_params.quarterly_freq:.1f} days")
        print(f"   Phase: {self.best_params.quarterly_phase:.2f} rad ({self.best_params.quarterly_phase*180/np.pi:.0f}°)")
        
        # Long-period
        print(f"\n🔄 LONG-PERIOD COMPONENT:")
        print(f"   Amplitude: {self.best_params.long_annual_amp:.2f} mm")
        print(f"   Frequency: {self.best_params.long_annual_freq:.3f} cycles/year")
        print(f"   Period: {1/self.best_params.long_annual_freq:.2f} years")
        print(f"   Phase: {self.best_params.long_annual_phase:.2f} rad ({self.best_params.long_annual_phase*180/np.pi:.0f}°)")
        
        # Noise
        print(f"\n🔊 NOISE CHARACTERISTICS:")
        print(f"   Standard deviation: {self.best_params.noise_std:.2f} mm")
        print(f"   Signal-to-noise ratio: {self.best_result['snr']:.1f}")
        
        # Component importance ranking
        print(f"\n📊 COMPONENT IMPORTANCE RANKING:")
        component_amplitudes = [
            ('Trend', np.std(self.best_result['components']['trend'])),
            ('Annual', self.best_params.annual_amp),
            ('Semi-annual', self.best_params.semi_annual_amp),
            ('Quarterly', self.best_params.quarterly_amp),
            ('Long-period', self.best_params.long_annual_amp),
            ('Noise', self.best_params.noise_std)
        ]
        
        # Sort by amplitude
        component_amplitudes.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, amp) in enumerate(component_amplitudes, 1):
            print(f"   {i}. {name}: {amp:.2f} mm")
        
        print("\n" + "="*60)

def load_ps00_data():
    """Load ps00 preprocessed data"""
    try:
        # Try to load from existing preprocessed data
        data_file = Path("data/processed/ps00_preprocessed_data.npz")
        if data_file.exists():
            print("📂 Loading ps00 preprocessed data...")
            data = np.load(data_file)
            
            # Get basic info
            displacement = data['displacement']
            coordinates = data['coordinates']
            n_stations, n_times = displacement.shape
            
            print(f"   📍 {n_stations} stations")
            print(f"   📅 {n_times} time points")
            
            # Create time vector (6-day sampling, typical for InSAR)
            # Assuming ~4 years of data based on 215 acquisitions
            time_vector = np.arange(0, n_times * 6, 6)  # 6-day intervals
            
            print(f"   🗓️ Time range: {time_vector[0]:.1f} to {time_vector[-1]:.1f} days ({time_vector[-1]/365.25:.1f} years)")
            
            return {
                'time_vector': time_vector,
                'displacement': displacement,
                'coordinates': coordinates,
                'n_stations': n_stations
            }
        else:
            print("❌ ps00 data not found at data/processed/ps00_preprocessed_data.npz")
            return None
    except Exception as e:
        print(f"❌ Error loading ps00 data: {e}")
        return None

def select_random_stations(data, n_stations=100, min_valid_ratio=0.7):
    """Select random stations with sufficient valid data"""
    displacement = data['displacement']
    n_total = displacement.shape[0]
    
    # Find stations with sufficient valid data
    valid_stations = []
    for i in range(n_total):
        station_data = displacement[i, :]
        valid_ratio = np.sum(~np.isnan(station_data)) / len(station_data)
        if valid_ratio >= min_valid_ratio:
            valid_stations.append(i)
    
    print(f"📊 Found {len(valid_stations)}/{n_total} stations with ≥{min_valid_ratio*100:.0f}% valid data")
    
    if len(valid_stations) < n_stations:
        print(f"⚠️  Only {len(valid_stations)} stations available, using all")
        selected_indices = valid_stations
    else:
        # Randomly select stations
        np.random.seed(42)  # Reproducible selection
        selected_indices = np.random.choice(valid_stations, n_stations, replace=False)
    
    print(f"🎯 Selected {len(selected_indices)} stations for analysis")
    
    return sorted(selected_indices)

def create_comprehensive_analysis(time_vector, save_dir="figures"):
    """Create comprehensive analysis with multiple synthetic signals"""
    
    print("\n" + "="*70)
    print("🚀 PS02B - InSAR SIGNAL SIMULATOR ANALYSIS")
    print("="*70)
    
    # Create simulator
    simulator = InSARTimeSeries(time_vector, 'days')
    
    # Define several realistic parameter sets for Taiwan InSAR
    parameter_sets = [
        {
            'name': 'Coastal_Subsidence',
            'params': InSARParameters(
                trend=-25.0, annual_amp=12.0, annual_freq=1.0, annual_phase=np.pi/4,
                semi_annual_amp=6.0, semi_annual_freq=2.0, semi_annual_phase=np.pi/3,
                quarterly_amp=3.0, quarterly_freq=4.0, quarterly_phase=0,
                long_annual_amp=8.0, long_annual_freq=0.2, long_annual_phase=np.pi/2,
                noise_std=3.5
            )
        },
        {
            'name': 'Agricultural_Seasonal',
            'params': InSARParameters(
                trend=-8.0, annual_amp=18.0, annual_freq=0.98, annual_phase=0,
                semi_annual_amp=4.0, semi_annual_freq=2.1, semi_annual_phase=np.pi/6,
                quarterly_amp=7.0, quarterly_freq=3.9, quarterly_phase=np.pi/4,
                long_annual_amp=5.0, long_annual_freq=0.4, long_annual_phase=0,
                noise_std=2.8
            )
        },
        {
            'name': 'Urban_Stable',
            'params': InSARParameters(
                trend=-2.0, annual_amp=4.0, annual_freq=1.02, annual_phase=np.pi/2,
                semi_annual_amp=2.0, semi_annual_freq=1.95, semi_annual_phase=np.pi,
                quarterly_amp=1.0, quarterly_freq=4.1, quarterly_phase=np.pi/3,
                long_annual_amp=3.0, long_annual_freq=0.15, long_annual_phase=np.pi,
                noise_std=1.5
            )
        }
    ]
    
    # Generate and analyze each signal type
    results = []
    for i, param_set in enumerate(parameter_sets):
        print(f"\n🔬 Analyzing {param_set['name']} signal...")
        
        # Generate signal
        result = simulator.generate_signal(param_set['params'])
        
        # Plot breakdown
        save_path = Path(save_dir) / f"ps02b_fig0{i+1}_signal_{param_set['name'].lower()}.png"
        simulator.plot_signal_breakdown(
            result, 
            title=f"InSAR Signal: {param_set['name'].replace('_', ' ')}",
            save_path=str(save_path)
        )
        
        results.append({
            'name': param_set['name'],
            'result': result,
            'signal': result['signal']
        })
    
    # Create comparison figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('InSAR Signal Type Comparison', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green']
    
    for i, (result, color) in enumerate(zip(results, colors)):
        axes[i].plot(time_vector, result['signal'], color=color, linewidth=2, alpha=0.8)
        axes[i].set_title(f"{result['name'].replace('_', ' ')} Signal", fontweight='bold')
        axes[i].set_ylabel('Displacement (mm)')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        signal = result['signal']
        stats_text = (f'Range: {np.ptp(signal):.1f} mm\n'
                     f'STD: {np.std(signal):.1f} mm\n'
                     f'Trend: {result["result"]["params"].trend:.1f} mm/yr')
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    axes[2].set_xlabel('Time (days)')
    plt.tight_layout()
    
    save_path = Path(save_dir) / "ps02b_fig04_signal_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved signal comparison: {save_path}")
    
    return results

def analyze_100_stations(data, save_dir="figures"):
    """Analyze 100 randomly selected stations from ps00 data"""
    
    print("\n" + "="*70)
    print("🎯 PS02B - 100 STATION InSAR FITTING ANALYSIS")
    print("="*70)
    
    # Select random stations
    selected_indices = select_random_stations(data, n_stations=100)
    
    time_vector = data['time_vector']
    displacement = data['displacement']
    coordinates = data['coordinates']
    
    # Storage for results
    fitting_results = []
    successful_fits = 0
    failed_fits = 0
    
    print(f"\n🔄 Fitting signals for {len(selected_indices)} stations...")
    
    # Process each station
    for i, station_idx in enumerate(selected_indices):
        print(f"\n📍 Processing station {station_idx} ({i+1}/{len(selected_indices)})...")
        
        try:
            # Get station data
            station_data = displacement[station_idx, :]
            station_coords = coordinates[station_idx, :]
            
            # Check for sufficient valid data
            valid_mask = ~np.isnan(station_data)
            if np.sum(valid_mask) < len(station_data) * 0.7:
                print(f"   ⚠️  Insufficient valid data, skipping...")
                failed_fits += 1
                continue
            
            # Interpolate missing values if needed
            if np.any(~valid_mask):
                station_data = np.interp(
                    np.arange(len(station_data)),
                    np.where(valid_mask)[0],
                    station_data[valid_mask]
                )
            
            # Fit the signal
            fitter = InSARFitter(time_vector, station_data, 'days')
            fitted_params = fitter.fit(maxiter=500)  # Faster fitting
            
            # Calculate quality metrics
            synthetic = fitter.best_result['signal']
            residuals = synthetic - station_data
            rmse = np.sqrt(np.mean(residuals**2))
            corr, _ = pearsonr(synthetic, station_data)
            
            print(f"   ✅ Fit completed: RMSE={rmse:.2f}mm, Corr={corr:.3f}")
            
            # Store results
            result = {
                'station_idx': station_idx,
                'coordinates': station_coords,
                'observed_signal': station_data,
                'fitted_params': fitted_params,
                'fitted_result': fitter.best_result,
                'rmse': rmse,
                'correlation': corr,
                'components': fitter.best_result['components']
            }
            
            fitting_results.append(result)
            successful_fits += 1
            
        except Exception as e:
            print(f"   ❌ Fitting failed: {e}")
            failed_fits += 1
            continue
    
    print(f"\n📊 Fitting Summary:")
    print(f"   ✅ Successful fits: {successful_fits}")
    print(f"   ❌ Failed fits: {failed_fits}")
    print(f"   📈 Success rate: {successful_fits/(successful_fits+failed_fits)*100:.1f}%")
    
    if successful_fits == 0:
        print("❌ No successful fits, cannot create visualizations")
        return None
    
    # Create comprehensive visualizations
    create_100_station_visualizations(fitting_results, time_vector, save_dir)
    
    return fitting_results

def create_100_station_visualizations(results, time_vector, save_dir="figures"):
    """Create comprehensive visualizations for 100 stations"""
    
    print(f"\n🎨 Creating visualizations for {len(results)} stations...")
    
    # Create colormap for stations
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(results))))
    if len(results) > 20:
        # Use continuous colormap for more stations
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # 1. All components in separate subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(f'InSAR Signal Components - {len(results)} Stations', fontsize=16, fontweight='bold')
    
    component_names = ['trend', 'annual', 'semi_annual', 'quarterly', 'long_annual', 'noise']
    component_titles = ['Trend Component', 'Annual Component', 'Semi-annual Component', 
                       'Quarterly Component', 'Long-period Component', 'Noise Component']
    
    for idx, (comp_name, comp_title) in enumerate(zip(component_names, component_titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Plot all stations for this component
        amplitudes = []
        for i, result in enumerate(results):
            component = result['components'][comp_name]
            color = colors[i % len(colors)]
            ax.plot(time_vector, component, color=color, alpha=0.3, linewidth=0.8)
            amplitudes.append(np.std(component))
        
        # Calculate and plot ensemble statistics
        all_components = np.array([result['components'][comp_name] for result in results])
        mean_component = np.mean(all_components, axis=0)
        std_component = np.std(all_components, axis=0)
        
        ax.plot(time_vector, mean_component, 'red', linewidth=3, label=f'Ensemble Mean', alpha=0.9)
        ax.fill_between(time_vector, 
                       mean_component - std_component,
                       mean_component + std_component, 
                       color='red', alpha=0.2, label='±1 STD')
        
        ax.set_title(comp_title, fontweight='bold', fontsize=12)
        ax.set_ylabel('Displacement (mm)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics
        mean_amplitude = np.mean(amplitudes)
        amplitude_range = [np.min(amplitudes), np.max(amplitudes)]
        stats_text = f'Mean Amp: {mean_amplitude:.2f} mm\nRange: [{amplitude_range[0]:.1f}, {amplitude_range[1]:.1f}] mm'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        if row == 2:  # Bottom row
            ax.set_xlabel('Time (days)')
    
    plt.tight_layout()
    save_path = Path(save_dir) / "ps02b_fig06_100_stations_components.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved components plot: {save_path}")
    
    # 2. Parameter distribution analysis
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Parameter Distributions - {len(results)} Stations', fontsize=16, fontweight='bold')
    
    # Extract parameters
    trends = [r['fitted_params'].trend for r in results]
    annual_amps = [r['fitted_params'].annual_amp for r in results]
    annual_periods = [365.25/r['fitted_params'].annual_freq for r in results]
    semi_amps = [r['fitted_params'].semi_annual_amp for r in results]
    semi_periods = [365.25/r['fitted_params'].semi_annual_freq for r in results]
    quarterly_amps = [r['fitted_params'].quarterly_amp for r in results]
    long_amps = [r['fitted_params'].long_annual_amp for r in results]
    noise_stds = [r['fitted_params'].noise_std for r in results]
    correlations = [r['correlation'] for r in results]
    
    # Plot distributions
    param_data = [
        (trends, 'Trend Rate (mm/year)', axes[0, 0]),
        (annual_amps, 'Annual Amplitude (mm)', axes[0, 1]),
        (annual_periods, 'Annual Period (days)', axes[0, 2]),
        (semi_amps, 'Semi-annual Amplitude (mm)', axes[1, 0]),
        (semi_periods, 'Semi-annual Period (days)', axes[1, 1]),
        (quarterly_amps, 'Quarterly Amplitude (mm)', axes[1, 2]),
        (long_amps, 'Long-period Amplitude (mm)', axes[2, 0]),
        (noise_stds, 'Noise STD (mm)', axes[2, 1]),
        (correlations, 'Fit Correlation', axes[2, 2])
    ]
    
    for data, title, ax in param_data:
        ax.hist(data, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.legend()
        
        # Add text statistics
        stats_text = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nn = {len(data)}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    save_path = Path(save_dir) / "ps02b_fig07_parameter_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved parameter distributions: {save_path}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics for {len(results)} Stations:")
    print(f"   Trend rates: {np.mean(trends):.1f} ± {np.std(trends):.1f} mm/year")
    print(f"   Annual amplitudes: {np.mean(annual_amps):.1f} ± {np.std(annual_amps):.1f} mm")
    print(f"   Annual periods: {np.mean(annual_periods):.0f} ± {np.std(annual_periods):.0f} days")
    print(f"   Semi-annual amplitudes: {np.mean(semi_amps):.1f} ± {np.std(semi_amps):.1f} mm")
    print(f"   Fit correlations: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")

def main():
    """Main execution function"""
    
    # Load ps00 data
    data = load_ps00_data()
    if data is None:
        print("❌ Cannot proceed without ps00 data")
        return False
    
    # Create save directory
    save_dir = "figures"
    Path(save_dir).mkdir(exist_ok=True)
    
    # Analyze 100 stations
    results = analyze_100_stations(data, save_dir)
    
    if results is None:
        return False
    
    print(f"\n📁 All figures saved to: {save_dir}/")
    print(f"   - ps02b_fig01-03_signal_[type].png: Individual signal breakdowns")
    print(f"   - ps02b_fig04_signal_comparison.png: Signal type comparison")
    
    print(f"\n📁 All figures saved to: {save_dir}/")
    print(f"   - ps02b_fig06_100_stations_components.png: Component ensemble plots")
    print(f"   - ps02b_fig07_parameter_distributions.png: Parameter histograms")
    print(f"   - ps02b_fig08_geographic_parameters.png: Geographic parameter maps")
    print(f"   - ps02b_fig09_fitting_quality.png: Fit quality analysis")
    
    print(f"\n🎉 PS02B 100-station analysis completed successfully!")
    
    return True

if __name__ == "__main__":
    success = main()
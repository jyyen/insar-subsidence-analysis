#!/usr/bin/env python3
"""
ps02a_MCSSA_decomposition_test.py - Monte Carlo SSA Test Script

Purpose: Test Monte Carlo Singular Spectrum Analysis (MCSSA) on raw InSAR data
Input: Raw data from ps00 (or ps01) 
Output: MCSSA decomposition results and visualizations

Features:
- Load raw displacement data from ps00_preprocessed_data.npz
- Implement basic MCSSA decomposition
- Test on sample stations
- Generate publication-quality figures: ps02a_fig01_xxxxx.png

Date: 2025-01-26
Author: InSAR Analysis Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy import signal
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """Load raw InSAR displacement data from ps00"""
    print("📡 Loading raw InSAR data from ps00...")
    
    try:
        data_file = Path("data/processed/ps00_preprocessed_data.npz")
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return None
        
        data = np.load(data_file, allow_pickle=True)
        
        coordinates = data['coordinates']  # [N, 2] - [lon, lat]
        displacement = data['displacement']  # [N, T] - mm  
        subsidence_rates = data['subsidence_rates']  # [N] - mm/year
        n_stations = int(data['n_stations'])
        n_acquisitions = int(data['n_acquisitions'])
        
        print(f"✅ Loaded {n_stations} stations, {n_acquisitions} acquisitions")
        print(f"📊 Displacement range: {displacement.min():.1f} to {displacement.max():.1f} mm")
        
        # Create time vector (6-day sampling)
        time_vector = np.arange(n_acquisitions) * 6  # days
        
        return {
            'coordinates': coordinates,
            'displacement': displacement,
            'subsidence_rates': subsidence_rates,
            'time_vector': time_vector,
            'n_stations': n_stations,
            'n_acquisitions': n_acquisitions
        }
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def generate_synthetic_insar_signals(time_vector, n_surrogates=100):
    """Generate synthetic InSAR-like signals with realistic geophysical components"""
    N = len(time_vector)
    t_days = time_vector
    t_years = t_days / 365.25
    
    surrogates = []
    signal_metadata = []
    
    for i in range(n_surrogates):
        signal = np.zeros(N)
        components = {}
        
        # 1. Subsidence/uplift trend (80% probability)
        if np.random.random() < 0.8:
            # Linear trend rates (mm/year)
            trend_rate = np.random.uniform(-60, 15)  # Typical InSAR subsidence range
            trend_component = trend_rate * t_years
            signal += trend_component
            components['trend'] = {'rate': trend_rate, 'amplitude': np.std(trend_component)}
        
        # 2. Annual signal (90% probability)
        if np.random.random() < 0.9:
            annual_amplitude = np.random.uniform(2, 25)  # mm
            annual_phase = np.random.uniform(0, 2*np.pi)
            annual_period = np.random.uniform(350, 380)  # days (realistic variation)
            annual_component = annual_amplitude * np.sin(2*np.pi*t_days/annual_period + annual_phase)
            signal += annual_component
            components['annual'] = {'amplitude': annual_amplitude, 'period': annual_period, 'phase': annual_phase}
        
        # 3. Semi-annual signal (60% probability) 
        if np.random.random() < 0.6:
            semi_amplitude = np.random.uniform(1, 12)  # mm
            semi_phase = np.random.uniform(0, 2*np.pi)
            semi_period = np.random.uniform(175, 190)  # days
            semi_component = semi_amplitude * np.sin(2*np.pi*t_days/semi_period + semi_phase)
            signal += semi_component
            components['semi_annual'] = {'amplitude': semi_amplitude, 'period': semi_period, 'phase': semi_phase}
        
        # 4. Quarterly signal (40% probability)
        if np.random.random() < 0.4:
            quarterly_amplitude = np.random.uniform(0.5, 8)  # mm
            quarterly_phase = np.random.uniform(0, 2*np.pi)
            quarterly_period = np.random.uniform(85, 95)  # days
            quarterly_component = quarterly_amplitude * np.sin(2*np.pi*t_days/quarterly_period + quarterly_phase)
            signal += quarterly_component
            components['quarterly'] = {'amplitude': quarterly_amplitude, 'period': quarterly_period, 'phase': quarterly_phase}
        
        # 5. Bi-annual/2-year cycle (20% probability)
        if np.random.random() < 0.2:
            biannual_amplitude = np.random.uniform(3, 15)  # mm
            biannual_phase = np.random.uniform(0, 2*np.pi)
            biannual_period = np.random.uniform(700, 750)  # ~2 years
            biannual_component = biannual_amplitude * np.sin(2*np.pi*t_days/biannual_period + biannual_phase)
            signal += biannual_component
            components['biannual'] = {'amplitude': biannual_amplitude, 'period': biannual_period, 'phase': biannual_phase}
        
        # 6. High-frequency signals (30% probability)
        if np.random.random() < 0.3:
            hf_amplitude = np.random.uniform(0.5, 5)  # mm
            hf_period = np.random.uniform(12, 60)  # 12-60 days
            hf_phase = np.random.uniform(0, 2*np.pi)
            hf_component = hf_amplitude * np.sin(2*np.pi*t_days/hf_period + hf_phase)
            signal += hf_component
            components['high_frequency'] = {'amplitude': hf_amplitude, 'period': hf_period, 'phase': hf_phase}
        
        # 7. InSAR measurement noise
        noise_std = np.random.uniform(2, 12)  # mm
        noise_component = np.random.normal(0, noise_std, N)
        signal += noise_component
        components['noise'] = {'std': noise_std}
        
        # Store results
        surrogates.append(signal)
        signal_metadata.append({
            'index': i,
            'components': components,
            'total_std': np.std(signal),
            'snr': np.std(signal - noise_component) / noise_std if noise_std > 0 else np.inf
        })
    
    return np.array(surrogates), signal_metadata

def monte_carlo_significance_test(time_series, window_length, time_vector, n_surrogates=100, alpha=0.10):
    """Perform Monte Carlo significance testing using synthetic InSAR signals"""
    N = len(time_series)
    L = window_length
    K = N - L + 1
    
    print(f"   🎲 Generating {n_surrogates} synthetic InSAR signals...")
    
    # Generate synthetic InSAR signals (geophysical null model)
    surrogates, metadata = generate_synthetic_insar_signals(time_vector, n_surrogates)
    
    # Compute eigenvalues for original data
    X_orig = np.zeros((L, K))
    for i in range(L):
        X_orig[i, :] = time_series[i:i+K]
    C_orig = X_orig @ X_orig.T / K
    eigenvals_orig = np.linalg.eigvals(C_orig)
    eigenvals_orig = np.sort(eigenvals_orig)[::-1]  # Descending order
    
    # Compute eigenvalues for all surrogates
    surrogate_eigenvals = []
    for surrogate in surrogates:
        X_surr = np.zeros((L, K))
        for i in range(L):
            X_surr[i, :] = surrogate[i:i+K]
        C_surr = X_surr @ X_surr.T / K
        eigenvals_surr = np.linalg.eigvals(C_surr)
        eigenvals_surr = np.sort(eigenvals_surr)[::-1]
        surrogate_eigenvals.append(eigenvals_surr)
    
    surrogate_eigenvals = np.array(surrogate_eigenvals)
    
    # Determine significance thresholds
    significance_thresholds = np.percentile(surrogate_eigenvals, (1-alpha)*100, axis=0)
    
    # Find significant components
    significant_components = eigenvals_orig > significance_thresholds
    n_significant = np.sum(significant_components)
    
    print(f"   📊 Significance test: {n_significant}/{len(eigenvals_orig)} components above {alpha*100}% threshold")
    
    return {
        'eigenvals_orig': eigenvals_orig,
        'surrogate_eigenvals': surrogate_eigenvals,
        'significance_thresholds': significance_thresholds,
        'significant_components': significant_components,
        'n_significant': n_significant,
        'alpha': alpha,
        'synthetic_metadata': metadata
    }

def mcssa_decomposition(time_series, window_length=None, n_components=None, n_surrogates=100, alpha=0.10):
    """
    Monte Carlo Singular Spectrum Analysis implementation with significance testing
    
    Parameters:
    -----------
    time_series : np.ndarray
        Input time series data
    window_length : int, optional
        SSA window length (default: N//3)
    n_components : int, optional  
        Number of components to extract (default: from Monte Carlo significance)
    n_surrogates : int, optional
        Number of surrogate datasets for significance testing (default: 100)
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    dict : MCSSA decomposition results
    """
    N = len(time_series)
    
    # Set default window length
    if window_length is None:
        window_length = min(N//3, 36)  # Max 3 years for InSAR
    
    L = window_length
    K = N - L + 1
    
    if K < 2:
        raise ValueError(f"Time series too short: N={N}, L={L}")
    
    print(f"   SSA parameters: N={N}, L={L}, K={K}")
    
    # Step 1: Monte Carlo significance testing FIRST
    # Need to reconstruct time vector for this function
    time_vector_for_mc = np.arange(N) * 6  # 6-day sampling
    mc_results = monte_carlo_significance_test(time_series, L, time_vector_for_mc, n_surrogates, alpha)
    
    # Step 2: Embedding - Create trajectory matrix
    X = np.zeros((L, K))
    for i in range(L):
        X[i, :] = time_series[i:i+K]
    
    # Step 3: SVD decomposition
    # Use covariance matrix approach for better numerical stability
    C = X @ X.T / K
    eigenvals, eigenvecs = np.linalg.eigh(C)
    
    # Sort in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Step 4: Principal components
    pc_series = eigenvecs.T @ X
    
    # Step 5: Component selection based on Monte Carlo significance
    if n_components is None:
        # Use Monte Carlo significance testing results with minimum preservation
        n_components = mc_results['n_significant']
        
        # Ensure we preserve enough signal for InSAR analysis
        min_components = max(3, int(0.85 * np.sum(np.cumsum(eigenvals)/np.sum(eigenvals) <= 0.85)))
        n_components = max(n_components, min_components)
        
        n_components = min(n_components, L//2)  # Reasonable limit
        n_components = max(n_components, 1)     # At least 1 component
    
    print(f"   Using {n_components} significant components (captures {100*np.sum(eigenvals[:n_components])/np.sum(eigenvals):.1f}% energy)")
    
    # Step 5: Reconstruction
    components = []
    component_names = []
    
    for comp_idx in range(n_components):
        # Reconstruct trajectory matrix for this component
        eigenvec = eigenvecs[:, comp_idx]
        pc = pc_series[comp_idx, :]
        
        # Outer product reconstruction
        X_comp = np.outer(eigenvec, pc)
        
        # Anti-diagonal averaging (Hankelization)
        component_signal = np.zeros(N)
        normalization = np.zeros(N)
        
        for i in range(L):
            for j in range(K):
                time_idx = i + j
                if time_idx < N:
                    component_signal[time_idx] += X_comp[i, j]
                    normalization[time_idx] += 1
        
        # Normalize
        mask = normalization > 0
        component_signal[mask] /= normalization[mask]
        
        components.append(component_signal)
    
    # Group-based classification after all components are extracted
    significant_mask = mc_results['significant_components'][:n_components] if n_components <= len(mc_results['significant_components']) else None
    component_names = classify_ssa_components_grouped(components, significant_mask)
    
    return {
        'components': np.array(components),
        'component_names': component_names,
        'eigenvalues': eigenvals,
        'n_components': n_components,
        'reconstruction': np.sum(components, axis=0),
        'window_length': L,
        'explained_variance_ratio': np.sum(eigenvals[:n_components]) / np.sum(eigenvals),
        'monte_carlo_results': mc_results,
        'significant_components': mc_results['significant_components'][:n_components]
    }

def classify_ssa_components_grouped(components, significant_mask=None):
    """Classify SSA components using group-based analysis"""
    n_components = len(components)
    classifications = []
    
    # Analyze all components first to understand the spectrum
    periods = []
    energies = []
    
    for comp in components:
        try:
            # Use FFT only for periodic signals (after trend screening)
            # First check if this looks like a trend
            first_diff = np.diff(comp)
            sign_changes = np.sum(np.diff(np.sign(first_diff)) != 0)
            monotonic_score = 1 - (sign_changes / len(first_diff))
            
            if monotonic_score > 0.8:  # Likely trend - don't use FFT
                dominant_period = np.inf
                energy = np.var(comp)
            else:  # Likely periodic - use FFT
                freqs, psd = signal.periodogram(comp, fs=1/6)  # 6-day sampling
                if len(psd) > 1:
                    # Find dominant frequency (skip DC component)
                    dominant_freq_idx = np.argmax(psd[1:]) + 1
                    dominant_period = 1 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] > 0 else np.inf
                    energy = np.sum(psd)
                else:
                    dominant_period = np.inf
                    energy = np.var(comp)
        except:
            dominant_period = np.inf
            energy = np.var(comp)
        
        periods.append(dominant_period)
        energies.append(energy)
    
    # Sort components by energy (SSA natural ordering)
    energy_order = np.argsort(energies)[::-1]  # Highest energy first
    
    # Group-based classification  
    for i, comp in enumerate(components):
        period = periods[i]
        energy_rank = np.where(energy_order == i)[0][0]
        
        # Check if this component is Monte Carlo significant
        is_significant = significant_mask[i] if significant_mask is not None and i < len(significant_mask) else True
        
        if not is_significant:
            classifications.append('noise')
            continue
        
        # Trend detection with high-frequency screening (only for significant components)
        if energy_rank < 3:  # Top 3 components could be trends
            
            # FIRST: Use FFT to exclude high-frequency signals from trend classification
            freqs, psd = signal.periodogram(comp, fs=1/6)  # 6-day sampling
            if len(psd) > 1:
                # Find dominant frequency (skip DC component)
                dominant_freq_idx = np.argmax(psd[1:]) + 1
                dominant_period = 1 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] > 0 else np.inf
                
                # High-frequency cutoff: periods < 6 months are NOT trends
                if dominant_period < 180:  # Less than 6 months - too high frequency for geological trend
                    # Skip trend detection, continue to periodic classification
                    pass
                else:
                    # SECOND: For low-frequency signals, check trend characteristics
                    first_diff = np.diff(comp)
                    second_diff = np.diff(first_diff)
                    
                    # Monotonicity test: consistent sign in first derivative
                    sign_changes = np.sum(np.diff(np.sign(first_diff)) != 0)
                    monotonic_score = 1 - (sign_changes / len(first_diff))
                    
                    # Smoothness test: low second derivative variance (low curvature)
                    curvature_var = np.var(second_diff) if len(second_diff) > 0 else 0
                    total_var = np.var(comp)
                    smoothness_score = 1 - (curvature_var / total_var) if total_var > 0 else 0
                    
                    # Trend criteria: highly monotonic OR very smooth (AND not high frequency)
                    if monotonic_score > 0.8 or smoothness_score > 0.9:
                        classifications.append('trend')
                        continue
        
        # Classify by period ranges
        if 300 < period < 450:  # ~1 year (wider range for SSA pairs)
            classifications.append('annual')
        elif 150 < period < 250:  # ~6 months
            classifications.append('semi_annual')
        elif 75 < period < 130:   # ~3 months
            classifications.append('quarterly')
        elif period < 75:        # <2.5 months
            classifications.append('high_frequency')
        elif period > 450:       # Very long period
            classifications.append('trend')
        else:
            classifications.append(f'component_{i}')
    
    return classifications

def classify_ssa_component(component, component_index):
    """Legacy function - kept for compatibility"""
    # This will be called from the grouped function
    return f'component_{component_index}'

def create_mcssa_visualization(station_idx, time_series, time_vector, mcssa_result, save_dir="figures"):
    """Create comprehensive MCSSA visualization for a single station"""
    
    components = mcssa_result['components']
    component_names = mcssa_result['component_names']
    reconstruction = mcssa_result['reconstruction']
    eigenvalues = mcssa_result['eigenvalues']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 3 rows, 2 columns
    # Row 1: Original vs Reconstruction + Eigenvalue spectrum
    # Row 2-3: Individual components (2 per row)
    
    # Subplot 1: Original vs Reconstruction
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time_vector, time_series, 'k-', linewidth=1.5, label='Original', alpha=0.8)
    ax1.plot(time_vector, reconstruction, 'r-', linewidth=1.5, label='Reconstruction', alpha=0.8)
    ax1.set_title(f'Station {station_idx:04d} - Original vs MCSSA Reconstruction', fontweight='bold')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Displacement (mm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate reconstruction quality
    rmse = np.sqrt(np.mean((time_series - reconstruction)**2))
    correlation = np.corrcoef(time_series, reconstruction)[0, 1]
    ax1.text(0.02, 0.98, f'RMSE: {rmse:.2f} mm\nCorr: {correlation:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Subplot 2: Eigenvalue spectrum
    ax2 = plt.subplot(3, 2, 2)
    n_plot_eigenvals = min(20, len(eigenvalues))
    ax2.semilogy(range(1, n_plot_eigenvals+1), eigenvalues[:n_plot_eigenvals], 'bo-', markersize=4)
    ax2.axvline(mcssa_result['n_components'], color='red', linestyle='--', alpha=0.7, label='Cutoff')
    ax2.set_title('SSA Eigenvalue Spectrum', fontweight='bold')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add explained variance text
    explained_var = mcssa_result['explained_variance_ratio']
    ax2.text(0.02, 0.98, f'Explained Variance:\n{explained_var:.1%}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Subplots 3-6: Individual components (show first 4 components)
    n_components_to_plot = min(4, len(components))
    component_positions = [(3, 2, 3), (3, 2, 4), (3, 2, 5), (3, 2, 6)]
    
    colors = ['blue', 'green', 'orange', 'purple']
    
    for i in range(n_components_to_plot):
        ax = plt.subplot(*component_positions[i])
        component = components[i]
        component_name = component_names[i]
        color = colors[i % len(colors)]
        
        ax.plot(time_vector, component, color=color, linewidth=1.5, alpha=0.8)
        ax.set_title(f'Component {i+1}: {component_name.title()}', fontweight='bold')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Displacement (mm)')
        ax.grid(True, alpha=0.3)
        
        # Add component statistics
        amplitude = np.std(component)
        energy_percent = eigenvalues[i] / np.sum(eigenvalues) * 100
        ax.text(0.02, 0.98, f'Amplitude: {amplitude:.2f} mm\nEnergy: {energy_percent:.1f}%', 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig_file = save_path / f"ps02a_fig01_mcssa_station_{station_idx:04d}.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved MCSSA visualization: {fig_file}")
    return fig_file

def create_components_ensemble_plot(all_station_results, time_vector, save_dir="figures"):
    """Create ensemble plot showing all components from all stations together"""
    
    if not all_station_results:
        return None
    
    print(f"📊 Creating components ensemble plot with {len(all_station_results)} stations...")
    
    # Determine maximum number of components across all stations
    max_components = max(len(result['mcssa_result']['components']) for result in all_station_results)
    
    # Create figure with subplots for each component
    n_rows = (max_components + 1) // 2  # 2 components per row
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'MCSSA Components Ensemble - {len(all_station_results)} Stations (Every 10th)', 
                 fontsize=16, fontweight='bold')
    
    # Color map for stations
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(all_station_results))))
    
    component_stats = []
    
    for comp_idx in range(max_components):
        row = comp_idx // 2
        col = comp_idx % 2
        ax = axes[row, col]
        
        # Collect all components of this type
        all_components = []
        component_names = []
        
        for i, result in enumerate(all_station_results):
            mcssa_result = result['mcssa_result']
            if comp_idx < len(mcssa_result['components']):
                component = mcssa_result['components'][comp_idx]
                component_name = mcssa_result['component_names'][comp_idx]
                
                # Plot individual station component with transparency
                color = colors[i % len(colors)]
                ax.plot(time_vector, component, color=color, alpha=0.3, linewidth=0.8)
                
                all_components.append(component)
                component_names.append(component_name)
        
        if all_components:
            all_components = np.array(all_components)
            
            # Calculate ensemble statistics
            mean_component = np.mean(all_components, axis=0)
            std_component = np.std(all_components, axis=0)
            
            # Plot ensemble mean and std
            ax.plot(time_vector, mean_component, 'red', linewidth=2.5, label='Ensemble Mean')
            ax.fill_between(time_vector, 
                           mean_component - std_component,
                           mean_component + std_component, 
                           color='red', alpha=0.2, label='±1 STD')
            
            # Determine most common component name
            unique_names, counts = np.unique(component_names, return_counts=True)
            most_common_name = unique_names[np.argmax(counts)]
            
            # Title and labels
            ax.set_title(f'Component {comp_idx+1}: {most_common_name.title()} '
                        f'({len(all_components)} stations)', 
                        fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Displacement (mm)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistics
            mean_amplitude = np.mean(np.std(all_components, axis=1))
            amplitude_range = [np.min(np.std(all_components, axis=1)), 
                             np.max(np.std(all_components, axis=1))]
            
            stats_text = (f'Mean Amplitude: {mean_amplitude:.2f} mm\n' +
                          f'Range: [{amplitude_range[0]:.2f}, {amplitude_range[1]:.2f}] mm\n' +
                          f'Stations: {len(all_components)}')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            component_stats.append({
                'component_index': comp_idx,
                'component_name': most_common_name,
                'n_stations': len(all_components),
                'mean_amplitude': mean_amplitude,
                'amplitude_range': amplitude_range
            })
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Component {comp_idx+1}: No Data')
    
    # Hide empty subplots
    for comp_idx in range(max_components, n_rows * n_cols):
        row = comp_idx // 2
        col = comp_idx % 2
        if row < n_rows and col < n_cols:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save ensemble plot
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig_file = save_path / "ps02a_fig03_mcssa_components_ensemble.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved components ensemble plot: {fig_file}")
    
    # Print component statistics
    print(f"\n📊 Component Statistics:")
    for stat in component_stats:
        print(f"   Component {stat['component_index']+1} ({stat['component_name']}): "
              f"{stat['n_stations']} stations, "
              f"amplitude {stat['mean_amplitude']:.2f} mm")
    
    return fig_file

def create_synthetic_signals_visualization(time_vector, save_dir="figures"):
    """Create comprehensive visualization of synthetic InSAR signals"""
    
    print("🎨 Creating synthetic signals visualization...")
    
    # Generate a set of example signals for visualization
    synthetic_signals, metadata = generate_synthetic_insar_signals(time_vector, n_surrogates=20)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Example synthetic signals (top)
    ax1 = plt.subplot(4, 3, (1, 3))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(min(10, len(synthetic_signals))):
        ax1.plot(time_vector, synthetic_signals[i], color=colors[i], alpha=0.7, linewidth=1.5, 
                label=f'Signal {i+1}')
    
    ax1.set_title('Example Synthetic InSAR Signals', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Displacement (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Component statistics
    all_trends = []
    all_annual = []
    all_semi = []
    all_quarterly = []
    all_noise = []
    
    for meta in metadata:
        if 'trend' in meta['components']:
            all_trends.append(meta['components']['trend']['rate'])
        if 'annual' in meta['components']:
            all_annual.append(meta['components']['annual']['amplitude'])
        if 'semi_annual' in meta['components']:
            all_semi.append(meta['components']['semi_annual']['amplitude'])
        if 'quarterly' in meta['components']:
            all_quarterly.append(meta['components']['quarterly']['amplitude'])
        if 'noise' in meta['components']:
            all_noise.append(meta['components']['noise']['std'])
    
    # Plot distributions
    ax2 = plt.subplot(4, 3, 4)
    if all_trends:
        ax2.hist(all_trends, bins=15, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('Trend Rates Distribution')
    ax2.set_xlabel('Rate (mm/year)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(4, 3, 5)
    if all_annual:
        ax3.hist(all_annual, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_title('Annual Amplitude Distribution')
    ax3.set_xlabel('Amplitude (mm)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(4, 3, 6)
    if all_semi:
        ax4.hist(all_semi, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax4.set_title('Semi-annual Amplitude Distribution')
    ax4.set_xlabel('Amplitude (mm)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(4, 3, 7)
    if all_quarterly:
        ax5.hist(all_quarterly, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax5.set_title('Quarterly Amplitude Distribution')
    ax5.set_xlabel('Amplitude (mm)')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(4, 3, 8)
    if all_noise:
        ax6.hist(all_noise, bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax6.set_title('Noise Level Distribution')
    ax6.set_xlabel('Std Dev (mm)')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    # 3. Signal-to-noise ratios
    ax7 = plt.subplot(4, 3, 9)
    snr_values = [meta['snr'] for meta in metadata if np.isfinite(meta['snr'])]
    if snr_values:
        ax7.hist(snr_values, bins=15, alpha=0.7, color='brown', edgecolor='black')
    ax7.set_title('Signal-to-Noise Ratio Distribution')
    ax7.set_xlabel('SNR')
    ax7.set_ylabel('Frequency')
    ax7.grid(True, alpha=0.3)
    
    # 4. Component occurrence frequency
    ax8 = plt.subplot(4, 3, 10)
    component_counts = {
        'Trend': sum(1 for meta in metadata if 'trend' in meta['components']),
        'Annual': sum(1 for meta in metadata if 'annual' in meta['components']),
        'Semi-annual': sum(1 for meta in metadata if 'semi_annual' in meta['components']),
        'Quarterly': sum(1 for meta in metadata if 'quarterly' in meta['components']),
        'Bi-annual': sum(1 for meta in metadata if 'biannual' in meta['components']),
        'High-freq': sum(1 for meta in metadata if 'high_frequency' in meta['components'])
    }
    
    components = list(component_counts.keys())
    counts = list(component_counts.values())
    bars = ax8.bar(components, counts, alpha=0.7, color=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
    ax8.set_title('Component Occurrence Frequency')
    ax8.set_ylabel('Count')
    ax8.tick_params(axis='x', rotation=45)
    
    # Add percentages on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = count / len(metadata) * 100
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{percentage:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Example decomposed signal
    ax9 = plt.subplot(4, 3, (11, 12))
    
    # Take the first signal and decompose it manually to show components
    example_signal = synthetic_signals[0]
    example_meta = metadata[0]
    
    t_days = time_vector
    t_years = t_days / 365.25
    
    # Reconstruct individual components
    reconstructed_signal = np.zeros_like(t_days, dtype=float)
    legend_labels = []
    
    if 'trend' in example_meta['components']:
        trend_comp = example_meta['components']['trend']['rate'] * t_years
        ax9.plot(t_days, trend_comp, '--', linewidth=2, label='Trend', alpha=0.8)
        reconstructed_signal += trend_comp
        legend_labels.append('Trend')
    
    if 'annual' in example_meta['components']:
        annual_comp = (example_meta['components']['annual']['amplitude'] * 
                      np.sin(2*np.pi*t_days/example_meta['components']['annual']['period'] + 
                            example_meta['components']['annual']['phase']))
        ax9.plot(t_days, annual_comp, '--', linewidth=2, label='Annual', alpha=0.8)
        legend_labels.append('Annual')
    
    if 'semi_annual' in example_meta['components']:
        semi_comp = (example_meta['components']['semi_annual']['amplitude'] * 
                    np.sin(2*np.pi*t_days/example_meta['components']['semi_annual']['period'] + 
                          example_meta['components']['semi_annual']['phase']))
        ax9.plot(t_days, semi_comp, '--', linewidth=2, label='Semi-annual', alpha=0.8)
        legend_labels.append('Semi-annual')
    
    # Plot total signal
    ax9.plot(t_days, example_signal, 'k-', linewidth=2.5, label='Total Signal', alpha=0.9)
    
    ax9.set_title(f'Example Signal Decomposition (Signal 1)', fontweight='bold')
    ax9.set_xlabel('Time (days)')
    ax9.set_ylabel('Displacement (mm)')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig_file = save_path / "ps02a_fig04_synthetic_signals_analysis.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved synthetic signals analysis: {fig_file}")
    
    # Print summary statistics
    print(f"\n📊 Synthetic Signals Summary:")
    print(f"   Generated: {len(metadata)} signals")
    print(f"   Component occurrence rates:")
    for comp, count in component_counts.items():
        print(f"     {comp}: {count}/{len(metadata)} ({count/len(metadata)*100:.0f}%)")
    
    if all_trends:
        print(f"   Trend rates: {np.mean(all_trends):.1f} ± {np.std(all_trends):.1f} mm/year")
    if all_annual:
        print(f"   Annual amplitudes: {np.mean(all_annual):.1f} ± {np.std(all_annual):.1f} mm")
    if snr_values:
        print(f"   Signal-to-noise ratios: {np.mean(snr_values):.1f} ± {np.std(snr_values):.1f}")
    
    return fig_file

def create_summary_visualization(results_list, save_dir="figures"):
    """Create summary visualization across multiple stations"""
    
    if not results_list:
        return None
    
    # Extract data for summary
    station_indices = [r['station_idx'] for r in results_list]
    rmse_values = [r['rmse'] for r in results_list]
    correlation_values = [r['correlation'] for r in results_list]
    n_components_values = [r['n_components'] for r in results_list]
    explained_variance_values = [r['explained_variance_ratio'] for r in results_list]
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'MCSSA Decomposition Summary - {len(results_list)} Stations (Every 10th)', 
                 fontsize=14, fontweight='bold')
    
    # Subplot 1: RMSE distribution
    axes[0, 0].hist(rmse_values, bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Reconstruction RMSE Distribution')
    axes[0, 0].set_xlabel('RMSE (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(rmse_values), color='red', linestyle='--', label=f'Mean: {np.mean(rmse_values):.2f}')
    axes[0, 0].legend()
    
    # Subplot 2: Correlation distribution
    axes[0, 1].hist(correlation_values, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Reconstruction Correlation Distribution')
    axes[0, 1].set_xlabel('Correlation Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(correlation_values), color='red', linestyle='--', label=f'Mean: {np.mean(correlation_values):.3f}')
    axes[0, 1].legend()
    
    # Subplot 3: Number of components
    axes[1, 0].hist(n_components_values, bins=range(1, max(n_components_values)+2), alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Number of Components Distribution')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Explained variance
    axes[1, 1].hist(explained_variance_values, bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Explained Variance Distribution')
    axes[1, 1].set_xlabel('Explained Variance Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(np.mean(explained_variance_values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(explained_variance_values):.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save summary figure
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig_file = save_path / "ps02a_fig02_mcssa_summary.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved MCSSA summary: {fig_file}")
    return fig_file

def main():
    """Main execution function"""
    print("="*70)
    print("🚀 PS02A - MONTE CARLO SSA DECOMPOSITION TEST")
    print("="*70)
    
    # Load raw data
    data = load_raw_data()
    if data is None:
        print("❌ Failed to load data. Exiting.")
        return False
    
    coordinates = data['coordinates']
    displacement = data['displacement']
    time_vector = data['time_vector']
    n_stations = data['n_stations']
    
    # Test parameters - subsample every 10th station
    test_station_indices = np.arange(0, n_stations, 10)  # Every 10th station
    n_test_stations = len(test_station_indices)
    
    print(f"\n🧪 Testing MCSSA on {n_test_stations} stations...")
    print(f"   Station indices: {test_station_indices}")
    
    results_list = []
    all_station_results = []  # Store full results for ensemble plot
    
    for i, station_idx in enumerate(test_station_indices):
        print(f"\n🔄 Processing station {station_idx} ({i+1}/{n_test_stations})...")
        
        # Get station time series
        station_ts = displacement[station_idx, :]
        
        # Skip if too many NaN values
        valid_mask = ~np.isnan(station_ts)
        if np.sum(valid_mask) < len(station_ts) * 0.7:
            print(f"   ⚠️  Skipping station {station_idx}: too many NaN values")
            continue
        
        # Simple interpolation for missing values
        if np.any(np.isnan(station_ts)):
            station_ts = np.interp(
                np.arange(len(station_ts)),
                np.where(valid_mask)[0],
                station_ts[valid_mask]
            )
        
        try:
            # Perform MCSSA decomposition
            print(f"   🔍 Running MCSSA decomposition...")
            start_time = time.time()
            
            mcssa_result = mcssa_decomposition(station_ts, window_length=36, n_surrogates=50)  # Reduced for speed
            
            elapsed_time = time.time() - start_time
            print(f"   ✅ MCSSA completed in {elapsed_time:.2f} seconds")
            print(f"   📊 Found {mcssa_result['n_components']} components")
            print(f"   📈 Explained variance: {mcssa_result['explained_variance_ratio']:.1%}")
            
            # Calculate quality metrics
            reconstruction = mcssa_result['reconstruction']
            rmse = np.sqrt(np.mean((station_ts - reconstruction)**2))
            correlation = np.corrcoef(station_ts, reconstruction)[0, 1]
            
            print(f"   📊 RMSE: {rmse:.2f} mm, Correlation: {correlation:.3f}")
            
            # Create individual visualization (only for first 5 stations to avoid too many files)
            if i < 5:
                create_mcssa_visualization(station_idx, station_ts, time_vector, mcssa_result)
            
            # Store results for summary
            results_list.append({
                'station_idx': station_idx,
                'rmse': rmse,
                'correlation': correlation,
                'n_components': mcssa_result['n_components'],
                'explained_variance_ratio': mcssa_result['explained_variance_ratio'],
                'processing_time': elapsed_time
            })
            
            # Store full results for ensemble plot
            all_station_results.append({
                'station_idx': station_idx,
                'station_ts': station_ts,
                'mcssa_result': mcssa_result,
                'rmse': rmse,
                'correlation': correlation
            })
            
        except Exception as e:
            print(f"   ❌ MCSSA failed for station {station_idx}: {e}")
            continue
    
    # Create visualizations
    if results_list:
        print(f"\n📊 Creating visualizations...")
        
        # Create synthetic signals analysis first
        create_synthetic_signals_visualization(time_vector)
        
        # Create components ensemble plot (main new feature)
        create_components_ensemble_plot(all_station_results, time_vector)
        
        # Create summary statistics plot
        create_summary_visualization(results_list)
        
        # Print summary statistics
        print(f"\n✅ MCSSA Test Summary:")
        print(f"   Successfully processed: {len(results_list)}/{n_test_stations} stations")
        
        if len(results_list) > 1:
            rmse_values = [r['rmse'] for r in results_list]
            corr_values = [r['correlation'] for r in results_list]
            time_values = [r['processing_time'] for r in results_list]
            
            print(f"   RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} mm")
            print(f"   Correlation: {np.mean(corr_values):.3f} ± {np.std(corr_values):.3f}")
            print(f"   Processing time: {np.mean(time_values):.2f} ± {np.std(time_values):.2f} seconds/station")
        
        print(f"\n📁 Figures saved in: figures/")
        print(f"   - Individual stations (first 5): ps02a_fig01_mcssa_station_XXXX.png")
        print(f"   - Summary statistics: ps02a_fig02_mcssa_summary.png") 
        print(f"   - Components ensemble: ps02a_fig03_mcssa_components_ensemble.png")
        print(f"   - Synthetic signals analysis: ps02a_fig04_synthetic_signals_analysis.png")
        
    else:
        print("\n❌ No successful decompositions. Check input data and parameters.")
        return False
    
    print("\n🎉 PS02A MCSSA test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
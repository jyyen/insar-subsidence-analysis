#!/usr/bin/env python3
"""
ps02_26_evaluate_metrics.py: Metrics evaluation study
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

def evaluate_insar_metrics():
    """Evaluate different goodness-of-fit metrics for InSAR analysis"""
    
    # Load data
    try:
        ps00_data = np.load("data/processed/ps00_preprocessed_data.npz", allow_pickle=True)
        displacement = ps00_data['displacement'][:50]  # Use 50 stations
        subsidence_rates = ps00_data['subsidence_rates'][:50]
        
        # Create time vector
        n_timepoints = displacement.shape[1]
        time_years = np.arange(n_timepoints) * 6 / 365.25
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Simulate different types of fits
    metrics_comparison = {
        'station': [],
        'correlation': [],
        'r2_score': [],
        'rmse': [],
        'mae': [],
        'rate_accuracy': [],
        'seasonal_capture': [],
        'noise_level': []
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx in range(min(9, len(displacement))):
        ax = axes[idx]
        signal = displacement[idx]
        
        # Decompose signal for analysis
        # 1. Linear trend
        coeffs = np.polyfit(time_years, signal, 1)
        linear_trend = np.polyval(coeffs, time_years)
        fitted_rate = coeffs[0]
        
        # 2. Detrended signal
        detrended = signal - linear_trend
        
        # 3. Estimate seasonal component (simple FFT approach)
        fft_signal = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended), d=6/365.25)  # frequencies in 1/years
        
        # Keep only seasonal frequencies (0.5-4 cycles/year)
        seasonal_mask = (np.abs(freqs) >= 0.5) & (np.abs(freqs) <= 4)
        fft_seasonal = fft_signal.copy()
        fft_seasonal[~seasonal_mask] = 0
        seasonal_component = np.real(np.fft.ifft(fft_seasonal))
        
        # 4. Residual (noise)
        residual = detrended - seasonal_component
        
        # Create different quality fits
        # Good fit: trend + seasonal
        good_fit = linear_trend + seasonal_component
        
        # Poor fit: just trend
        poor_fit = linear_trend
        
        # Calculate metrics for good fit
        correlation = np.corrcoef(signal, good_fit)[0, 1]
        r2 = r2_score(signal, good_fit)
        rmse = np.sqrt(mean_squared_error(signal, good_fit))
        mae = mean_absolute_error(signal, good_fit)
        
        # Rate accuracy (most important for subsidence!)
        true_rate = subsidence_rates[idx]
        rate_error = abs(fitted_rate - true_rate)
        rate_accuracy = 1 - (rate_error / (abs(true_rate) + 1e-6))
        
        # Seasonal capture ratio
        seasonal_variance = np.var(seasonal_component)
        total_variance = np.var(detrended)
        seasonal_capture = seasonal_variance / (total_variance + 1e-6)
        
        # Noise level
        noise_level = np.std(residual)
        
        # Store metrics
        metrics_comparison['station'].append(idx)
        metrics_comparison['correlation'].append(correlation)
        metrics_comparison['r2_score'].append(r2)
        metrics_comparison['rmse'].append(rmse)
        metrics_comparison['mae'].append(mae)
        metrics_comparison['rate_accuracy'].append(rate_accuracy)
        metrics_comparison['seasonal_capture'].append(seasonal_capture)
        metrics_comparison['noise_level'].append(noise_level)
        
        # Plot
        ax.plot(time_years, signal, 'b-', alpha=0.7, linewidth=1, label='Original')
        ax.plot(time_years, good_fit, 'r-', linewidth=2, label='Fitted')
        ax.plot(time_years, linear_trend, 'k--', linewidth=1.5, alpha=0.7, 
                label=f'Trend: {fitted_rate:.1f} mm/yr')
        
        # Add metrics text
        metrics_text = (f'R={correlation:.3f}, R²={r2:.3f}\n'
                       f'RMSE={rmse:.1f}mm\n'
                       f'Rate Acc={rate_accuracy:.2f}')
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Displacement (mm)')
        ax.set_title(f'Station {idx+1}')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('📊 InSAR Metrics Evaluation: What Matters Most?', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("figures")
    plt.savefig(output_dir / "ps02c_metrics_evaluation.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: figures/ps02c_metrics_evaluation.png")
    
    # Create comprehensive metrics comparison
    create_metrics_importance_analysis(metrics_comparison)

def create_metrics_importance_analysis(metrics_data):
    """Analyze which metrics matter most for InSAR subsidence analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Convert to arrays
    correlation = np.array(metrics_data['correlation'])
    r2 = np.array(metrics_data['r2_score'])
    rmse = np.array(metrics_data['rmse'])
    rate_accuracy = np.array(metrics_data['rate_accuracy'])
    seasonal_capture = np.array(metrics_data['seasonal_capture'])
    noise_level = np.array(metrics_data['noise_level'])
    
    # 1. Correlation vs R² comparison
    ax1 = axes[0]
    ax1.scatter(correlation, r2, alpha=0.7, s=100)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('Correlation (R)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Correlation vs R²: Similar but R² penalizes bias')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.1, 0.9, 'R² < R when there\'s offset bias', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 2. RMSE distribution
    ax2 = axes[1]
    ax2.hist(rmse, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(rmse), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rmse):.1f} mm')
    ax2.set_xlabel('RMSE (mm)')
    ax2.set_ylabel('Count')
    ax2.set_title('RMSE: Absolute fit quality in physical units')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rate Accuracy - MOST IMPORTANT!
    ax3 = axes[2]
    ax3.hist(rate_accuracy, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(rate_accuracy), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rate_accuracy):.3f}')
    ax3.set_xlabel('Rate Accuracy (1 = perfect)')
    ax3.set_ylabel('Count')
    ax3.set_title('⭐ RATE ACCURACY: Most Critical for Subsidence!')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Seasonal Capture Ratio
    ax4 = axes[3]
    ax4.hist(seasonal_capture, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(seasonal_capture), color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(seasonal_capture):.3f}')
    ax4.set_xlabel('Seasonal Capture Ratio')
    ax4.set_ylabel('Count')
    ax4.set_title('Seasonal Pattern Recognition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Metric Correlations
    ax5 = axes[4]
    metric_names = ['R', 'R²', 'RMSE', 'Rate\nAcc', 'Seasonal\nCapture']
    metric_values = [correlation, r2, rmse, rate_accuracy, seasonal_capture]
    
    # Calculate correlation matrix
    corr_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = np.corrcoef(metric_values[i], metric_values[j])[0, 1]
    
    im = ax5.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax5.set_xticks(range(5))
    ax5.set_yticks(range(5))
    ax5.set_xticklabels(metric_names)
    ax5.set_yticklabels(metric_names)
    ax5.set_title('Metric Correlations')
    
    # Add correlation values
    for i in range(5):
        for j in range(5):
            text = ax5.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax5)
    
    # 6. Recommended Composite Score
    ax6 = axes[5]
    
    # Create composite score emphasizing what matters for InSAR
    composite_score = (
        0.40 * rate_accuracy +           # 40% weight on rate accuracy
        0.20 * (1 - rmse/100) +          # 20% weight on RMSE (normalized)
        0.20 * seasonal_capture +         # 20% weight on seasonal capture
        0.10 * r2 +                      # 10% weight on R²
        0.10 * (1 - noise_level/50)      # 10% weight on noise suppression
    )
    
    ax6.hist(composite_score, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean(composite_score), color='purple', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(composite_score):.3f}')
    ax6.set_xlabel('Composite InSAR Score')
    ax6.set_ylabel('Count')
    ax6.set_title('🎯 Recommended Composite Score\n(40% Rate, 20% RMSE, 20% Seasonal, 20% Other)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('🔍 InSAR Metrics Analysis: What Really Matters for Subsidence Monitoring', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("figures")
    plt.savefig(output_dir / "ps02c_metrics_importance.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: figures/ps02c_metrics_importance.png")
    
    # Print recommendations
    print("\n" + "="*80)
    print("📊 METRIC RECOMMENDATIONS FOR InSAR SUBSIDENCE ANALYSIS:")
    print("="*80)
    print("\n🎯 PRIMARY METRICS (Most Important):")
    print("   1. SUBSIDENCE RATE ACCURACY - How well we capture the long-term trend")
    print("   2. RMSE in mm - Physical displacement error")
    print("   3. SEASONAL PATTERN CAPTURE - Groundwater pumping cycles")
    print("\n📈 SECONDARY METRICS:")
    print("   4. R² Score - Better than correlation (penalizes offset bias)")
    print("   5. Signal-to-Noise Ratio - Denoising effectiveness")
    print("\n⚠️ WHY CORRELATION ALONE IS INSUFFICIENT:")
    print("   - Doesn't capture rate accuracy (most critical for hazard assessment)")
    print("   - Can be high even with poor seasonal pattern capture")
    print("   - Doesn't reflect physical displacement errors in mm")
    print("\n✅ RECOMMENDED COMPOSITE SCORE:")
    print("   40% Rate Accuracy + 20% RMSE + 20% Seasonal + 10% R² + 10% SNR")
    print("="*80)

if __name__ == "__main__":
    print("🔍 Evaluating InSAR goodness-of-fit metrics...")
    evaluate_insar_metrics()
    plt.close('all')
    print("\n✅ Metric evaluation complete!")
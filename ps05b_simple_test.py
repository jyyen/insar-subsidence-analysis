#!/usr/bin/env python3
"""
ps05b_simple_test.py - Simplified test of advanced change detection methods

This is a simplified version to test the working methods without multiprocessing issues.
"""

import sys
import subprocess
import os

def check_and_switch_environment():
    """Check if we're in the correct environment for TensorFlow"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
        return True
    except ImportError:
        pass
    
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    if current_env != 'tensorflow-env' and '--auto-switched' not in sys.argv:
        print(f"📋 Current environment: {current_env}")
        print("🔄 Switching to tensorflow-env...")
        
        script_path = os.path.abspath(__file__)
        cmd_args = ['conda', 'run', '-n', 'tensorflow-env', 'python', script_path, '--auto-switched'] + sys.argv[1:]
        
        result = subprocess.run(cmd_args, check=False)
        sys.exit(result.returncode)
    
    return True

if __name__ == "__main__":
    check_and_switch_environment()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pywt
from scipy import signal

class SimpleAdvancedChangeDetection:
    """Simplified version focusing on working methods"""
    
    def __init__(self):
        self.coordinates = None
        self.displacement = None
        self.time_vector = None
        self.results = {}
        
        # Setup directories
        self.results_dir = Path("data/processed/ps05b_advanced")
        self.figures_dir = Path("figures")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self):
        """Load raw displacement data from ps00 preprocessing"""
        print("📡 Loading RAW displacement data from ps00...")
        
        try:
            preprocessed_data = np.load("data/processed/ps00_preprocessed_data.npz")
            
            self.coordinates = preprocessed_data['coordinates']
            self.displacement = preprocessed_data['displacement']
            
            n_times = self.displacement.shape[1]
            self.time_vector = np.arange(n_times) * 6  # 6-day sampling
            
            print(f"✅ Loaded coordinates for {len(self.coordinates)} stations")
            print(f"✅ Loaded RAW displacement: {self.displacement.shape}")
            print(f"✅ Time vector: {len(self.time_vector)} points ({self.time_vector[-1]:.0f} days)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading raw data: {e}")
            return False
    
    def method1_wavelet_analysis(self):
        """Simplified wavelet decomposition analysis"""
        print("\n" + "="*60)
        print("🌊 METHOD 1: WAVELET DECOMPOSITION")
        print("="*60)
        
        n_stations = len(self.coordinates)
        change_points = []
        
        print(f"🔄 Analyzing {n_stations} stations...")
        
        for station_idx in range(n_stations):
            if station_idx % 500 == 0:
                print(f"   Processing station {station_idx}/{n_stations}")
            
            try:
                signal_data = self.displacement[station_idx, :]
                
                # Multi-level wavelet decomposition with appropriate level
                max_level = pywt.dwt_max_level(len(signal_data), 'db4')
                decomp_level = min(4, max_level)  # Use 4 levels or max possible, whichever is smaller
                coeffs = pywt.wavedec(signal_data, 'db4', level=decomp_level)
                
                # Analyze each scale
                for level, coeff in enumerate(coeffs):
                    if len(coeff) < 10:
                        continue
                    
                    # Find significant peaks
                    coeff_abs = np.abs(coeff)
                    threshold = np.mean(coeff_abs) + 2.0 * np.std(coeff_abs)
                    peaks, _ = signal.find_peaks(coeff_abs, height=threshold)
                    
                    for peak in peaks:
                        time_idx = int(peak * (2**level))
                        if time_idx < len(self.time_vector):
                            change_points.append({
                                'station_idx': station_idx,
                                'time_idx': time_idx,
                                'time_days': self.time_vector[time_idx],
                                'scale_level': level,
                                'significance_score': coeff_abs[peak] / threshold
                            })
            
            except Exception as e:
                if station_idx < 10:  # Only print first few errors
                    print(f"   ⚠️  Error for station {station_idx}: {e}")
        
        self.results['wavelet'] = {
            'change_points': change_points,
            'n_change_points': len(change_points)
        }
        
        print(f"✅ Wavelet analysis completed")
        print(f"   Found {len(change_points)} change points")
        
        return True
    
    def method2_simple_bayesian(self):
        """Simplified Bayesian-style change detection"""
        print("\n" + "="*60)
        print("🎲 METHOD 2: SIMPLIFIED BAYESIAN")
        print("="*60)
        
        n_stations = len(self.coordinates)
        change_points = []
        
        print(f"🔄 Analyzing {n_stations} stations...")
        
        for station_idx in range(n_stations):
            if station_idx % 500 == 0:
                print(f"   Processing station {station_idx}/{n_stations}")
            
            try:
                signal_data = self.displacement[station_idx, :]
                
                # Simple change detection based on derivatives
                differences = np.diff(signal_data)
                change_indicators = np.abs(differences) > (2 * np.std(differences))
                
                for t, is_change in enumerate(change_indicators):
                    if is_change:
                        magnitude = np.abs(differences[t])
                        probability = min(0.9, magnitude / (3 * np.std(differences)))
                        
                        change_points.append({
                            'station_idx': station_idx,
                            'time_idx': t,
                            'time_days': self.time_vector[t] if t < len(self.time_vector) else t * 6,
                            'probability': probability,
                            'detection_method': 'simplified_bayesian'
                        })
            
            except Exception as e:
                if station_idx < 10:
                    print(f"   ⚠️  Error for station {station_idx}: {e}")
        
        self.results['bayesian'] = {
            'change_points': change_points,
            'n_change_points': len(change_points)
        }
        
        print(f"✅ Bayesian analysis completed")
        print(f"   Found {len(change_points)} change points")
        
        return True
    
    def method3_ensemble_consensus(self):
        """Simplified ensemble method combining PELT, BCA, and CUSUM"""
        print("\n" + "="*60)
        print("🗳️ METHOD 3: ENSEMBLE CONSENSUS")
        print("="*60)
        
        n_stations = len(self.coordinates)
        consensus_points = []
        
        print(f"🔄 Running ensemble analysis on {n_stations} stations...")
        
        # Check if we have ruptures for PELT
        try:
            import ruptures as rpt
            has_ruptures = True
        except ImportError:
            has_ruptures = False
            print("   ⚠️  Ruptures not available, using simplified methods")
        
        for station_idx in range(n_stations):
            if station_idx % 500 == 0:
                print(f"   Processing station {station_idx}/{n_stations}")
            
            try:
                signal_data = self.displacement[station_idx, :]
                method_results = {}
                
                # Method 1: PELT (if available)
                if has_ruptures:
                    try:
                        algo = rpt.Pelt(model="rbf").fit(signal_data.reshape(-1, 1))
                        pelt_changes = algo.predict(pen=10)
                        pelt_changes = [cp for cp in pelt_changes if cp < len(signal_data)]
                        method_results['pelt'] = {'changes': pelt_changes, 'weight': 0.4, 'confidence': 0.9}
                    except:
                        method_results['pelt'] = {'changes': [], 'weight': 0.0, 'confidence': 0.0}
                
                # Method 2: Simple Bayesian (reuse from method 2)
                differences = np.diff(signal_data)
                bca_changes = []
                change_indicators = np.abs(differences) > (2 * np.std(differences))
                for t, is_change in enumerate(change_indicators):
                    if is_change:
                        bca_changes.append(t)
                method_results['bca'] = {'changes': bca_changes, 'weight': 0.3, 'confidence': 0.7}
                
                # Method 3: Robust CUSUM
                def robust_cusum(data, threshold=2.0):
                    median_val = np.median(data)
                    mad = np.median(np.abs(data - median_val))
                    robust_std = 1.4826 * mad
                    
                    changes = []
                    cusum_pos = cusum_neg = 0
                    
                    for i in range(1, len(data)):
                        diff = data[i] - median_val
                        cusum_pos = max(0, cusum_pos + diff - robust_std/2)
                        cusum_neg = max(0, cusum_neg - diff - robust_std/2)
                        
                        if cusum_pos > threshold * robust_std or cusum_neg > threshold * robust_std:
                            changes.append(i)
                            cusum_pos = cusum_neg = 0
                    
                    return changes
                
                cusum_changes = robust_cusum(signal_data)
                method_results['cusum'] = {'changes': cusum_changes, 'weight': 0.3, 'confidence': 0.6}
                
                # Consensus voting: find change points detected by ≥2 methods
                all_changes = []
                for method, results in method_results.items():
                    for cp in results['changes']:
                        all_changes.append({
                            'time_idx': cp,
                            'method': method,
                            'weight': results['weight'],
                            'confidence': results['confidence']
                        })
                
                # Group nearby changes (within 3 time steps)
                processed_times = set()
                for change in sorted(all_changes, key=lambda x: x['time_idx']):
                    if change['time_idx'] in processed_times:
                        continue
                    
                    nearby_changes = [c for c in all_changes 
                                    if abs(c['time_idx'] - change['time_idx']) <= 3]
                    
                    if len(nearby_changes) >= 2:  # At least 2 methods agree
                        avg_time = int(np.mean([c['time_idx'] for c in nearby_changes]))
                        consensus_score = sum(c['weight'] * c['confidence'] for c in nearby_changes)
                        
                        consensus_points.append({
                            'station_idx': station_idx,
                            'time_idx': avg_time,
                            'time_days': self.time_vector[avg_time] if avg_time < len(self.time_vector) else avg_time * 6,
                            'consensus_score': consensus_score,
                            'n_methods_agree': len(nearby_changes),
                            'agreeing_methods': [c['method'] for c in nearby_changes]
                        })
                        
                        for c in nearby_changes:
                            processed_times.add(c['time_idx'])
            
            except Exception as e:
                if station_idx < 10:
                    print(f"   ⚠️  Error for station {station_idx}: {e}")
        
        self.results['ensemble'] = {
            'change_points': consensus_points,
            'n_change_points': len(consensus_points)
        }
        
        print(f"✅ Ensemble analysis completed")
        print(f"   Found {len(consensus_points)} consensus change points")
        
        return True
    
    def method4_lstm_simplified(self):
        """Simplified LSTM-style anomaly detection"""
        print("\n" + "="*60)
        print("🧠 METHOD 4: SIMPLIFIED LSTM-STYLE DETECTION")
        print("="*60)
        
        # Check if TensorFlow is available
        try:
            import tensorflow as tf
            print(f"   Using TensorFlow {tf.__version__}")
        except ImportError:
            print("   ⚠️  TensorFlow not available, using statistical approximation")
            return self._statistical_anomaly_detection()
        
        n_stations = len(self.coordinates)
        anomaly_points = []
        
        print(f"🔄 Analyzing {n_stations} stations with LSTM-style detection...")
        
        # Use a simple sliding window approach instead of full LSTM training
        window_size = 30
        
        for station_idx in range(n_stations):
            if station_idx % 500 == 0:
                print(f"   Processing station {station_idx}/{n_stations}")
            
            try:
                station_data = self.displacement[station_idx, :]
                
                # Normalize data
                normalized_data = (station_data - np.mean(station_data)) / (np.std(station_data) + 1e-8)
                
                # Simple reconstruction error approximation
                for i in range(window_size, len(normalized_data) - window_size):
                    window = normalized_data[i-window_size:i+window_size]
                    
                    # Simple prediction: linear trend from first half to second half
                    first_half = window[:window_size]
                    second_half = window[window_size:]
                    
                    # Linear prediction
                    trend = np.linspace(first_half[0], first_half[-1], window_size)
                    reconstruction_error = np.mean(np.square(second_half - trend))
                    
                    # Threshold based on local statistics
                    if reconstruction_error > 2.0:  # Simple threshold
                        anomaly_points.append({
                            'station_idx': station_idx,
                            'time_idx': i,
                            'time_days': self.time_vector[i] if i < len(self.time_vector) else i * 6,
                            'reconstruction_error': reconstruction_error,
                            'anomaly_score': reconstruction_error / 2.0
                        })
            
            except Exception as e:
                if station_idx < 10:
                    print(f"   ⚠️  Error for station {station_idx}: {e}")
        
        self.results['lstm'] = {
            'change_points': anomaly_points,
            'n_change_points': len(anomaly_points)
        }
        
        print(f"✅ LSTM-style analysis completed")
        print(f"   Found {len(anomaly_points)} anomalous patterns")
        
        return True
    
    def method5_lombscargle_analysis(self):
        """Lomb-Scargle periodogram analysis for periodic pattern changes"""
        print("\n" + "="*60)
        print("📊 METHOD 5: LOMB-SCARGLE PERIODOGRAM")
        print("="*60)
        
        # Check if Astropy is available
        try:
            from astropy.timeseries import LombScargle
            print("   Using Astropy LombScargle")
        except ImportError:
            print("   ⚠️  Astropy not available, using simple periodicity detection")
            return self._simple_periodicity_detection()
        
        n_stations = len(self.coordinates)
        periodic_changes = []
        
        print(f"🔄 Analyzing periodic patterns in {n_stations} stations...")
        
        for station_idx in range(n_stations):
            if station_idx % 500 == 0:
                print(f"   Processing station {station_idx}/{n_stations}")
            
            try:
                station_data = self.displacement[station_idx, :]
                time_days = self.time_vector
                
                # Full series periodogram
                frequency, power = LombScargle(time_days, station_data).autopower(
                    minimum_frequency=1/365.25,  # 1 year period
                    maximum_frequency=1/30       # 1 month period
                )
                
                # Simple sliding window analysis to detect changes
                window_size = 90  # days
                window_step = 30  # days
                n_windows = max(1, (len(time_days) - window_size) // window_step)
                
                if n_windows > 2:
                    window_periods = []
                    
                    for w in range(n_windows):
                        start_idx = w * window_step
                        end_idx = min(start_idx + window_size, len(time_days))
                        
                        if end_idx - start_idx < 30:
                            continue
                        
                        window_time = time_days[start_idx:end_idx]
                        window_data = station_data[start_idx:end_idx]
                        
                        try:
                            freq_win, power_win = LombScargle(window_time, window_data).autopower(
                                minimum_frequency=1/365.25,
                                maximum_frequency=1/30
                            )
                            
                            # Find dominant period
                            max_power_idx = np.argmax(power_win)
                            dominant_period = 1 / freq_win[max_power_idx]
                            
                            window_periods.append({
                                'window_idx': w,
                                'start_time': window_time[0],
                                'dominant_period': dominant_period,
                                'max_power': power_win[max_power_idx]
                            })
                        except:
                            continue
                    
                    # Detect significant changes in dominant periods
                    for i in range(1, len(window_periods)):
                        prev_period = window_periods[i-1]['dominant_period']
                        curr_period = window_periods[i]['dominant_period']
                        
                        period_change = abs(curr_period - prev_period) / prev_period
                        
                        if period_change > 0.2:  # 20% change threshold
                            change_time = window_periods[i]['start_time']
                            change_idx = np.argmin(np.abs(time_days - change_time))
                            
                            periodic_changes.append({
                                'station_idx': station_idx,
                                'time_idx': change_idx,
                                'time_days': change_time,
                                'old_period_days': prev_period,
                                'new_period_days': curr_period,
                                'relative_change': period_change
                            })
            
            except Exception as e:
                if station_idx < 10:
                    print(f"   ⚠️  Error for station {station_idx}: {e}")
        
        self.results['lombscargle'] = {
            'change_points': periodic_changes,
            'n_change_points': len(periodic_changes)
        }
        
        print(f"✅ Lomb-Scargle analysis completed")
        print(f"   Found {len(periodic_changes)} periodic pattern changes")
        
        return True
    
    def _statistical_anomaly_detection(self):
        """Fallback statistical anomaly detection when TensorFlow unavailable"""
        print("   Using statistical anomaly detection fallback...")
        
        n_stations = len(self.coordinates)
        anomaly_points = []
        
        for station_idx in range(n_stations):
            if station_idx % 1000 == 0:
                print(f"   Processing station {station_idx}/{n_stations}")
            
            try:
                station_data = self.displacement[station_idx, :]
                
                # Z-score based anomaly detection
                z_scores = np.abs((station_data - np.mean(station_data)) / np.std(station_data))
                
                anomaly_indices = np.where(z_scores > 3.0)[0]  # 3-sigma threshold
                
                for idx in anomaly_indices:
                    anomaly_points.append({
                        'station_idx': station_idx,
                        'time_idx': idx,
                        'time_days': self.time_vector[idx],
                        'z_score': z_scores[idx],
                        'anomaly_score': z_scores[idx] / 3.0
                    })
            
            except Exception as e:
                continue
        
        self.results['lstm'] = {
            'change_points': anomaly_points,
            'n_change_points': len(anomaly_points)
        }
        
        print(f"✅ Statistical anomaly detection completed")
        print(f"   Found {len(anomaly_points)} statistical anomalies")
        
        return True
    
    def _simple_periodicity_detection(self):
        """Fallback periodicity detection when Astropy unavailable"""
        print("   Using simple periodicity detection fallback...")
        
        n_stations = len(self.coordinates)
        periodic_changes = []
        
        for station_idx in range(n_stations):
            if station_idx % 1000 == 0:
                print(f"   Processing station {station_idx}/{n_stations}")
            
            try:
                station_data = self.displacement[station_idx, :]
                
                # Simple FFT-based periodicity detection
                fft_result = np.fft.fft(station_data)
                frequencies = np.fft.fftfreq(len(station_data), d=6)  # 6-day sampling
                
                # Find dominant frequency
                power_spectrum = np.abs(fft_result)
                max_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                dominant_period = 1 / frequencies[max_freq_idx] if frequencies[max_freq_idx] > 0 else 0
                
                # Simple change detection based on variance changes
                window_size = 50
                for i in range(window_size, len(station_data) - window_size, window_size//2):
                    window1 = station_data[i-window_size:i]
                    window2 = station_data[i:i+window_size]
                    
                    var_change = abs(np.var(window2) - np.var(window1)) / (np.var(window1) + 1e-8)
                    
                    if var_change > 0.5:  # 50% variance change
                        periodic_changes.append({
                            'station_idx': station_idx,
                            'time_idx': i,
                            'time_days': self.time_vector[i],
                            'variance_change': var_change,
                            'estimated_period': dominant_period
                        })
            
            except Exception as e:
                continue
        
        self.results['lombscargle'] = {
            'change_points': periodic_changes,
            'n_change_points': len(periodic_changes)
        }
        
        print(f"✅ Simple periodicity detection completed")
        print(f"   Found {len(periodic_changes)} variance-based changes")
        
        return True
    
    def create_summary_figure(self):
        """Create summary comparison figure for all 5 methods"""
        print("\n📊 Creating comprehensive summary figure...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Advanced Change Detection Results - All 5 Methods', fontsize=16, fontweight='bold')
        
        # Method 1: Wavelet
        ax = axes[0, 0]
        if 'wavelet' in self.results:
            times = [cp['time_days'] for cp in self.results['wavelet']['change_points']]
            scores = [cp['significance_score'] for cp in self.results['wavelet']['change_points']]
            ax.scatter(times, scores, alpha=0.6, c='blue', s=15)
            ax.set_title(f"Wavelet Analysis\n({len(times)} change points)")
        else:
            ax.text(0.5, 0.5, 'No Results', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Wavelet Analysis\n(No Results)")
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Significance Score')
        ax.grid(True, alpha=0.3)
        
        # Method 2: Bayesian
        ax = axes[0, 1]
        if 'bayesian' in self.results:
            times = [cp['time_days'] for cp in self.results['bayesian']['change_points']]
            probs = [cp['probability'] for cp in self.results['bayesian']['change_points']]
            ax.scatter(times, probs, alpha=0.6, c='red', s=15)
            ax.set_title(f"Bayesian Analysis\n({len(times)} change points)")
        else:
            ax.text(0.5, 0.5, 'No Results', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Bayesian Analysis\n(No Results)")
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)
        
        # Method 3: Ensemble
        ax = axes[0, 2]
        if 'ensemble' in self.results:
            times = [cp['time_days'] for cp in self.results['ensemble']['change_points']]
            scores = [cp['consensus_score'] for cp in self.results['ensemble']['change_points']]
            ax.scatter(times, scores, alpha=0.6, c='green', s=15)
            ax.set_title(f"Ensemble Consensus\n({len(times)} change points)")
        else:
            ax.text(0.5, 0.5, 'No Results', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Ensemble Consensus\n(No Results)")
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Consensus Score')
        ax.grid(True, alpha=0.3)
        
        # Method 4: LSTM
        ax = axes[1, 0]
        if 'lstm' in self.results and self.results['lstm']['change_points']:
            try:
                times = []
                scores = []
                for cp in self.results['lstm']['change_points']:
                    if cp['time_days'] <= 1300:  # Limit to reasonable time range
                        times.append(cp['time_days'])
                        scores.append(cp.get('anomaly_score', cp.get('reconstruction_error', 1.0)))
                
                if times and scores:
                    ax.scatter(times, scores, alpha=0.6, c='purple', s=15)
                    ax.set_xlim(0, 1300)
                    ax.set_title(f"LSTM Anomaly Detection\n({len(times)} change points)")
                else:
                    ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title("LSTM Anomaly Detection\n(No Valid Data)")
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title("LSTM Anomaly Detection\n(Error)")
        else:
            ax.text(0.5, 0.5, 'No Results', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("LSTM Anomaly Detection\n(No Results)")
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Anomaly Score')
        ax.grid(True, alpha=0.3)
        
        # Method 5: Lomb-Scargle
        ax = axes[1, 1]
        if 'lombscargle' in self.results and self.results['lombscargle']['change_points']:
            try:
                times = []
                changes = []
                for cp in self.results['lombscargle']['change_points']:
                    if cp['time_days'] <= 1300:  # Limit to reasonable time range
                        times.append(cp['time_days'])
                        changes.append(cp.get('relative_change', cp.get('variance_change', 0.5)))
                
                if times and changes:
                    ax.scatter(times, changes, alpha=0.6, c='orange', s=15)
                    ax.set_xlim(0, 1300)
                    ax.set_title(f"Lomb-Scargle Periodogram\n({len(times)} change points)")
                else:
                    ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title("Lomb-Scargle Periodogram\n(No Valid Data)")
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Lomb-Scargle Periodogram\n(Error)")
        else:
            ax.text(0.5, 0.5, 'No Results', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Lomb-Scargle Periodogram\n(No Results)")
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Relative Period Change')
        ax.grid(True, alpha=0.3)
        
        # Method Summary
        ax = axes[1, 2]
        method_names = []
        method_counts = []
        method_colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, (method, color) in enumerate(zip(['wavelet', 'bayesian', 'ensemble', 'lstm', 'lombscargle'], method_colors)):
            if method in self.results:
                method_names.append(method.capitalize())
                method_counts.append(self.results[method]['n_change_points'])
            else:
                method_names.append(method.capitalize())
                method_counts.append(0)
        
        bars = ax.bar(method_names, method_counts, color=method_colors, alpha=0.7)
        ax.set_title('Change Points by Method')
        ax.set_ylabel('Number of Change Points')
        ax.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, method_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(method_counts)*0.01,
                       str(count), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        figure_file = self.figures_dir / "ps05b_all_methods_results.png"
        plt.savefig(figure_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive summary figure saved: {figure_file}")
    
    def save_results(self):
        """Save results to file"""
        summary = {
            'timestamp': str(np.datetime64('now')),
            'n_stations': len(self.coordinates),
            'methods': list(self.results.keys()),
            'total_detections': sum(result['n_change_points'] for result in self.results.values())
        }
        
        for method, result in self.results.items():
            np.savez_compressed(
                self.results_dir / f"{method}_simple_results.npz",
                change_points=result['change_points']
            )
        
        with open(self.results_dir / "simple_test_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to: {self.results_dir}")
        print(f"📊 Total detections: {summary['total_detections']}")

def main():
    """Main execution"""
    print("="*60)
    print("🚀 PS05B SIMPLE TEST - ADVANCED CHANGE DETECTION")
    print("="*60)
    
    detector = SimpleAdvancedChangeDetection()
    
    # Load data
    if not detector.load_raw_data():
        return False
    
    # Run all 5 methods
    detector.method1_wavelet_analysis()
    detector.method2_simple_bayesian()
    detector.method3_ensemble_consensus()
    detector.method4_lstm_simplified()
    detector.method5_lombscargle_analysis()
    
    # Create visualizations and save results
    detector.create_summary_figure()
    detector.save_results()
    
    print("\n" + "="*60)
    print("✅ SIMPLE TEST COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
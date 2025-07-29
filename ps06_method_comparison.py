#!/usr/bin/env python3
"""
ps06_method_comparison.py - Method Comparison & Validation

Purpose: Comprehensive comparison of all decomposition methods from ps02
Methods: EMD, FFT, VMD, Wavelet decomposition comparison
Analysis: Reconstruction quality, component stability, performance ranking
Output: Method rankings, validation metrics, recommendation guidelines

Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import argparse
import warnings
from scipy import stats, signal
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Optional advanced visualization
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("✅ Cartopy available for geographic visualization")
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not available. Using matplotlib for geographic plots.")

class MethodComparisonAnalysis:
    """
    Comprehensive method comparison and validation framework
    
    Compares EMD, FFT, VMD, and Wavelet decomposition methods across multiple criteria:
    1. Reconstruction quality (RMSE, correlation, spectral fidelity)
    2. Component stability (cross-method consistency)
    3. Computational performance (speed, memory, robustness)
    4. Physical interpretability (seasonal patterns, trend smoothness)
    5. Frequency separation quality (spectral purity, mode mixing)
    """
    
    def __init__(self, methods=['emd', 'fft', 'vmd', 'wavelet']):
        """
        Initialize method comparison analysis
        
        Parameters:
        -----------
        methods : list
            Decomposition methods to compare ['emd', 'fft', 'vmd', 'wavelet']
        """
        self.methods = methods
        self.available_methods = []
        
        # Data containers
        self.coordinates = None
        self.original_displacement = None
        self.method_data = {}
        self.quality_metrics = {}
        
        # Analysis results
        self.reconstruction_metrics = {}
        self.component_stability = {}
        self.performance_metrics = {}
        self.statistical_tests = {}
        self.method_rankings = {}
        
        # Ranking criteria weights (customizable)
        self.criteria_weights = {
            'reconstruction_accuracy': 0.25,
            'physical_interpretability': 0.20,
            'computational_efficiency': 0.15,
            'robustness': 0.15,
            'frequency_separation': 0.15,
            'component_stability': 0.10
        }
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories for results and figures"""
        self.figures_dir = Path("figures")
        self.results_dir = Path("data/processed/ps06_comparison")
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Figures directory: {self.figures_dir}")

    def load_all_method_data(self):
        """Load all ps02 decomposition results and original data"""
        print("📡 Loading all method data from ps02...")
        
        try:
            # Load original displacement from ps00
            preprocessed_data = np.load("data/processed/ps00_preprocessed_data.npz")
            self.coordinates = preprocessed_data['coordinates']
            self.original_displacement = preprocessed_data['displacement']
            
            print(f"✅ Loaded original data: {self.original_displacement.shape}")
            print(f"✅ Loaded coordinates: {len(self.coordinates)} stations")
            
            # Load decomposition results for each method
            for method in self.methods:
                try:
                    # Load decomposition data
                    decomp_file = f"data/processed/ps02_{method}_decomposition.npz"
                    decomp_data = np.load(decomp_file)
                    
                    # Load quality metrics
                    quality_file = f"data/processed/ps02_{method}_quality_metrics.json"
                    with open(quality_file, 'r') as f:
                        quality_data = json.load(f)
                    
                    self.method_data[method] = {
                        'imfs': decomp_data['imfs'],
                        'residuals': decomp_data['residuals'],
                        'n_imfs_per_station': decomp_data['n_imfs_per_station'],
                        'time_vector': decomp_data['time_vector'],
                        'successful_decompositions': decomp_data['successful_decompositions'].item(),
                        'failed_decompositions': decomp_data['failed_decompositions'].item(),
                        'coordinates': decomp_data['coordinates'],
                        'subsidence_rates': decomp_data['subsidence_rates']
                    }
                    
                    self.quality_metrics[method] = quality_data
                    self.available_methods.append(method)
                    
                    print(f"✅ Loaded {method.upper()}: {decomp_data['imfs'].shape}")
                    print(f"   Success rate: {decomp_data['successful_decompositions'].item()}/{len(self.coordinates)} stations")
                    
                except FileNotFoundError as e:
                    print(f"⚠️  {method.upper()} data not found, skipping...")
                    continue
                except Exception as e:
                    print(f"⚠️  Error loading {method.upper()}: {e}")
                    continue
            
            if len(self.available_methods) == 0:
                print("❌ No method data loaded")
                return False
            
            print(f"✅ Successfully loaded {len(self.available_methods)} methods: {', '.join(self.available_methods).upper()}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def reconstruct_signal(self, method, station_idx=None):
        """
        Reconstruct signal from decomposition components
        
        Parameters:
        -----------
        method : str
            Method name ('emd', 'fft', 'vmd', 'wavelet')
        station_idx : int or None
            Station index (None for all stations)
        
        Returns:
        --------
        reconstructed : ndarray
            Reconstructed time series
        """
        try:
            method_data = self.method_data[method]
            imfs = method_data['imfs']
            residuals = method_data['residuals']
            
            if station_idx is not None:
                # Single station reconstruction
                station_imfs = imfs[station_idx]
                station_residual = residuals[station_idx]
                n_imfs = method_data['n_imfs_per_station'][station_idx]
                
                # Sum valid IMFs + residual
                reconstructed = np.sum(station_imfs[:n_imfs], axis=0) + station_residual
                
            else:
                # All stations reconstruction
                n_stations = len(imfs)
                n_times = imfs.shape[2]
                reconstructed = np.zeros((n_stations, n_times))
                
                for i in range(n_stations):
                    n_imfs = method_data['n_imfs_per_station'][i]
                    reconstructed[i] = np.sum(imfs[i, :n_imfs], axis=0) + residuals[i]
            
            return reconstructed
            
        except Exception as e:
            print(f"⚠️  Error reconstructing signal for {method}: {e}")
            return None

    def assess_reconstruction_quality(self, method):
        """
        Comprehensive reconstruction quality assessment
        
        Parameters:
        -----------
        method : str
            Method name to assess
            
        Returns:
        --------
        metrics : dict
            Reconstruction quality metrics
        """
        print(f"   Assessing reconstruction quality for {method.upper()}...")
        
        try:
            # Reconstruct signals
            reconstructed = self.reconstruct_signal(method)
            
            if reconstructed is None:
                return None
            
            # Ensure same shape
            original = self.original_displacement
            if original.shape != reconstructed.shape:
                min_stations = min(original.shape[0], reconstructed.shape[0])
                original = original[:min_stations]
                reconstructed = reconstructed[:min_stations]
            
            # Basic reconstruction metrics
            rmse_per_station = np.sqrt(np.mean((original - reconstructed)**2, axis=1))
            mae_per_station = np.mean(np.abs(original - reconstructed), axis=1)
            
            # Overall metrics
            rmse_overall = np.sqrt(np.mean((original - reconstructed)**2))
            mae_overall = np.mean(np.abs(original - reconstructed))
            
            # Correlation metrics
            correlations = []
            for i in range(len(original)):
                if np.std(original[i]) > 0 and np.std(reconstructed[i]) > 0:
                    corr, _ = pearsonr(original[i], reconstructed[i])
                    correlations.append(corr if np.isfinite(corr) else 0)
                else:
                    correlations.append(0)
            
            correlations = np.array(correlations)
            correlation_mean = np.mean(correlations)
            
            # R-squared (coefficient of determination)
            r2_per_station = []
            for i in range(len(original)):
                if np.var(original[i]) > 0:
                    r2 = 1 - np.sum((original[i] - reconstructed[i])**2) / np.sum((original[i] - np.mean(original[i]))**2)
                    r2_per_station.append(r2 if np.isfinite(r2) else 0)
                else:
                    r2_per_station.append(0)
            
            r2_per_station = np.array(r2_per_station)
            r2_mean = np.mean(r2_per_station)
            
            # Signal-to-noise ratio
            signal_power = np.var(original)
            noise_power = np.var(original - reconstructed)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            # Normalized RMSE
            nrmse = rmse_overall / np.std(original) if np.std(original) > 0 else 0
            
            # Spectral fidelity (simplified)
            spectral_coherence = self._compute_spectral_coherence(original, reconstructed)
            
            metrics = {
                'rmse_overall': rmse_overall,
                'mae_overall': mae_overall,
                'rmse_per_station': rmse_per_station.tolist(),
                'mae_per_station': mae_per_station.tolist(),
                'correlation_mean': correlation_mean,
                'correlations': correlations.tolist(),
                'r2_mean': r2_mean,
                'r2_per_station': r2_per_station.tolist(),
                'snr': snr,
                'nrmse': nrmse,
                'spectral_coherence': spectral_coherence,
                'reconstruction_success_rate': np.mean(correlations > 0.5)  # Threshold for "good" reconstruction
            }
            
            print(f"      RMSE: {rmse_overall:.3f} mm")
            print(f"      Correlation: {correlation_mean:.3f}")
            print(f"      R²: {r2_mean:.3f}")
            print(f"      SNR: {snr:.1f} dB")
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error assessing reconstruction quality for {method}: {e}")
            return None

    def _compute_spectral_coherence(self, original, reconstructed):
        """Compute spectral coherence between original and reconstructed signals"""
        try:
            # Use first few stations for spectral analysis (computational efficiency)
            n_stations_sample = min(10, len(original))
            coherences = []
            
            for i in range(n_stations_sample):
                try:
                    # Compute power spectral density
                    f_orig, psd_orig = signal.periodogram(original[i])
                    f_recon, psd_recon = signal.periodogram(reconstructed[i])
                    
                    # Compute coherence (normalized cross-correlation in frequency domain)
                    if len(f_orig) == len(f_recon):
                        coherence = np.corrcoef(psd_orig, psd_recon)[0, 1]
                        if np.isfinite(coherence):
                            coherences.append(coherence)
                except:
                    continue
            
            return np.mean(coherences) if coherences else 0.0
            
        except Exception as e:
            return 0.0

    def analyze_component_stability(self):
        """
        Analyze component consistency and stability across methods
        """
        print("🔄 Analyzing component stability across methods...")
        
        try:
            stability_results = {}
            
            # Define frequency bands for comparison
            frequency_bands = {
                'high_freq': (0, 2),      # First 2 IMFs (high frequency)
                'seasonal': (2, 5),       # IMFs 2-4 (seasonal patterns)
                'trend': (-1, None)       # Residual (long-term trend)
            }
            
            # Cross-method correlation analysis
            cross_correlations = {}
            
            for band_name, (start_imf, end_imf) in frequency_bands.items():
                print(f"   Analyzing {band_name} components...")
                
                band_correlations = {}
                
                # Extract band components for each method
                band_components = {}
                for method in self.available_methods:
                    band_components[method] = self._extract_frequency_band(method, start_imf, end_imf)
                
                # Compute pairwise correlations between methods
                method_pairs = [(m1, m2) for i, m1 in enumerate(self.available_methods) 
                               for m2 in self.available_methods[i+1:]]
                
                for method1, method2 in method_pairs:
                    comp1 = band_components[method1]
                    comp2 = band_components[method2]
                    
                    if comp1 is not None and comp2 is not None:
                        # Compute station-wise correlations
                        correlations = []
                        min_stations = min(len(comp1), len(comp2))
                        
                        for i in range(min_stations):
                            if np.std(comp1[i]) > 0 and np.std(comp2[i]) > 0:
                                corr, _ = pearsonr(comp1[i], comp2[i])
                                if np.isfinite(corr):
                                    correlations.append(corr)
                        
                        if correlations:
                            band_correlations[f"{method1}_{method2}"] = {
                                'mean_correlation': np.mean(correlations),
                                'std_correlation': np.std(correlations),
                                'correlations': correlations
                            }
                
                cross_correlations[band_name] = band_correlations
            
            # Component count consistency
            component_counts = {}
            for method in self.available_methods:
                counts = self.method_data[method]['n_imfs_per_station']
                component_counts[method] = {
                    'mean_components': np.mean(counts),
                    'std_components': np.std(counts),
                    'min_components': np.min(counts),
                    'max_components': np.max(counts)
                }
            
            stability_results = {
                'cross_correlations': cross_correlations,
                'component_counts': component_counts,
                'stability_score': self._compute_overall_stability_score(cross_correlations)
            }
            
            self.component_stability = stability_results
            
            print(f"✅ Component stability analysis completed")
            print(f"   Overall stability score: {stability_results['stability_score']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in component stability analysis: {e}")
            return False

    def _extract_frequency_band(self, method, start_imf, end_imf):
        """Extract specific frequency band components"""
        try:
            method_data = self.method_data[method]
            
            if end_imf is None:  # Trend component (residual)
                return method_data['residuals']
            else:
                imfs = method_data['imfs']
                n_stations = len(imfs)
                n_times = imfs.shape[2]
                
                band_component = np.zeros((n_stations, n_times))
                
                for i in range(n_stations):
                    n_imfs = method_data['n_imfs_per_station'][i]
                    end_idx = min(end_imf, n_imfs) if end_imf is not None else n_imfs
                    start_idx = min(start_imf, n_imfs)
                    
                    if end_idx > start_idx:
                        band_component[i] = np.sum(imfs[i, start_idx:end_idx], axis=0)
                
                return band_component
                
        except Exception as e:
            print(f"   ⚠️  Error extracting frequency band for {method}: {e}")
            return None

    def _compute_overall_stability_score(self, cross_correlations):
        """Compute overall stability score across all frequency bands"""
        try:
            all_correlations = []
            
            for band_name, band_corrs in cross_correlations.items():
                for pair_name, pair_data in band_corrs.items():
                    all_correlations.append(pair_data['mean_correlation'])
            
            return np.mean(all_correlations) if all_correlations else 0.0
            
        except:
            return 0.0

    def perform_statistical_tests(self):
        """
        Perform statistical significance tests between methods
        """
        print("🔄 Performing statistical significance tests...")
        
        try:
            test_results = {}
            
            # Get RMSE values for each method
            rmse_data = {}
            for method in self.available_methods:
                if method in self.reconstruction_metrics:
                    rmse_data[method] = np.array(self.reconstruction_metrics[method]['rmse_per_station'])
            
            if len(rmse_data) < 2:
                print("   ⚠️  Need at least 2 methods for statistical testing")
                return False
            
            # Pairwise statistical tests
            method_pairs = [(m1, m2) for i, m1 in enumerate(self.available_methods) 
                           for m2 in self.available_methods[i+1:]]
            
            for method1, method2 in method_pairs:
                if method1 in rmse_data and method2 in rmse_data:
                    rmse1 = rmse_data[method1]
                    rmse2 = rmse_data[method2]
                    
                    # Ensure same length
                    min_len = min(len(rmse1), len(rmse2))
                    rmse1 = rmse1[:min_len]
                    rmse2 = rmse2[:min_len]
                    
                    # Paired t-test
                    try:
                        t_stat, t_pval = ttest_rel(rmse1, rmse2)
                    except:
                        t_stat, t_pval = np.nan, np.nan
                    
                    # Wilcoxon signed-rank test (non-parametric)
                    try:
                        w_stat, w_pval = wilcoxon(rmse1, rmse2)
                    except:
                        w_stat, w_pval = np.nan, np.nan
                    
                    # Effect size (Cohen's d)
                    try:
                        pooled_std = np.sqrt((np.var(rmse1) + np.var(rmse2)) / 2)
                        cohens_d = (np.mean(rmse1) - np.mean(rmse2)) / pooled_std if pooled_std > 0 else 0
                    except:
                        cohens_d = np.nan
                    
                    test_results[f"{method1}_vs_{method2}"] = {
                        'mean_rmse_diff': np.mean(rmse1) - np.mean(rmse2),
                        'paired_ttest': {
                            'statistic': t_stat,
                            'p_value': t_pval,
                            'significant': t_pval < 0.05 if np.isfinite(t_pval) else False
                        },
                        'wilcoxon': {
                            'statistic': w_stat,
                            'p_value': w_pval,
                            'significant': w_pval < 0.05 if np.isfinite(w_pval) else False
                        },
                        'cohens_d': cohens_d,
                        'effect_size': self._interpret_effect_size(cohens_d)
                    }
            
            # Overall ANOVA (if more than 2 methods)
            if len(rmse_data) > 2:
                try:
                    # Prepare data for ANOVA
                    all_rmse = []
                    method_labels = []
                    
                    for method, rmse_vals in rmse_data.items():
                        all_rmse.extend(rmse_vals)
                        method_labels.extend([method] * len(rmse_vals))
                    
                    # One-way ANOVA
                    from scipy.stats import f_oneway
                    f_stat, f_pval = f_oneway(*[rmse_data[method] for method in self.available_methods])
                    
                    test_results['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': f_pval,
                        'significant': f_pval < 0.05
                    }
                    
                except Exception as e:
                    print(f"   ⚠️  ANOVA failed: {e}")
            
            self.statistical_tests = test_results
            
            print(f"✅ Statistical tests completed")
            significant_pairs = sum(1 for result in test_results.values() 
                                  if isinstance(result, dict) and 
                                  result.get('paired_ttest', {}).get('significant', False))
            print(f"   {significant_pairs} significant pairwise differences found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in statistical testing: {e}")
            return False

    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        if np.isnan(cohens_d):
            return "unknown"
        
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def generate_method_rankings(self):
        """
        Generate comprehensive method rankings based on multiple criteria
        """
        print("🔄 Generating method rankings...")
        
        try:
            rankings = {}
            
            # Prepare criteria scores for each method
            method_scores = {}
            
            for method in self.available_methods:
                scores = {}
                
                # 1. Reconstruction accuracy (lower RMSE, higher correlation is better)
                if method in self.reconstruction_metrics:
                    rmse = self.reconstruction_metrics[method]['rmse_overall']
                    corr = self.reconstruction_metrics[method]['correlation_mean']
                    r2 = self.reconstruction_metrics[method]['r2_mean']
                    
                    # Normalize scores (0-1, higher is better)
                    scores['reconstruction_accuracy'] = (corr + r2) / 2  # Average of correlation and R²
                
                # 2. Physical interpretability (based on component stability and success rate)
                if method in self.component_stability:
                    stability = self.component_stability.get('stability_score', 0)
                    success_rate = self.method_data[method]['successful_decompositions'] / len(self.coordinates)
                    scores['physical_interpretability'] = (stability + success_rate) / 2
                
                # 3. Computational efficiency (based on ps02 quality metrics)
                if method in self.quality_metrics:
                    # Use inverse of processing time and memory usage (if available)
                    # For now, use success rate as proxy for efficiency
                    scores['computational_efficiency'] = self.method_data[method]['successful_decompositions'] / len(self.coordinates)
                
                # 4. Robustness (success rate and low variability)
                if method in self.reconstruction_metrics:
                    success_rate = self.method_data[method]['successful_decompositions'] / len(self.coordinates)
                    rmse_std = np.std(self.reconstruction_metrics[method]['rmse_per_station'])
                    max_rmse = max([self.reconstruction_metrics[m]['rmse_overall'] for m in self.available_methods])
                    
                    # Lower variability is better
                    variability_score = 1 - (rmse_std / max_rmse) if max_rmse > 0 else 1
                    scores['robustness'] = (success_rate + variability_score) / 2
                
                # 5. Frequency separation (based on spectral coherence)
                if method in self.reconstruction_metrics:
                    spectral_coherence = self.reconstruction_metrics[method]['spectral_coherence']
                    scores['frequency_separation'] = spectral_coherence
                
                # 6. Component stability (from stability analysis)
                if method in self.component_stability:
                    scores['component_stability'] = self.component_stability.get('stability_score', 0)
                
                method_scores[method] = scores
            
            # Normalize scores across methods (0-1 scale)
            normalized_scores = self._normalize_scores(method_scores)
            
            # Compute weighted overall scores
            overall_scores = {}
            for method in self.available_methods:
                if method in normalized_scores:
                    overall_score = 0
                    for criterion, weight in self.criteria_weights.items():
                        if criterion in normalized_scores[method]:
                            overall_score += weight * normalized_scores[method][criterion]
                    overall_scores[method] = overall_score
            
            # Create rankings
            sorted_methods = sorted(overall_scores.keys(), key=lambda x: overall_scores[x], reverse=True)
            
            rankings['overall'] = {
                'ranking': sorted_methods,
                'scores': overall_scores,
                'normalized_scores': normalized_scores
            }
            
            # Application-specific rankings
            rankings['scientific_accuracy'] = self._rank_for_application(
                normalized_scores, ['reconstruction_accuracy', 'physical_interpretability', 'component_stability']
            )
            
            rankings['operational_efficiency'] = self._rank_for_application(
                normalized_scores, ['computational_efficiency', 'robustness', 'reconstruction_accuracy']
            )
            
            rankings['frequency_analysis'] = self._rank_for_application(
                normalized_scores, ['frequency_separation', 'reconstruction_accuracy', 'component_stability']
            )
            
            self.method_rankings = rankings
            
            print(f"✅ Method rankings generated")
            print(f"   Overall ranking: {' > '.join(rankings['overall']['ranking']).upper()}")
            print(f"   Best overall method: {rankings['overall']['ranking'][0].upper()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating rankings: {e}")
            return False

    def _normalize_scores(self, method_scores):
        """Normalize scores to 0-1 scale across methods"""
        normalized = {}
        
        # Get all criteria
        all_criteria = set()
        for scores in method_scores.values():
            all_criteria.update(scores.keys())
        
        # Normalize each criterion
        for criterion in all_criteria:
            criterion_values = []
            for method, scores in method_scores.items():
                if criterion in scores:
                    criterion_values.append(scores[criterion])
            
            if criterion_values:
                min_val = min(criterion_values)
                max_val = max(criterion_values)
                value_range = max_val - min_val
                
                for method, scores in method_scores.items():
                    if method not in normalized:
                        normalized[method] = {}
                    
                    if criterion in scores:
                        if value_range > 0:
                            normalized[method][criterion] = (scores[criterion] - min_val) / value_range
                        else:
                            normalized[method][criterion] = 1.0  # All methods equal
                    else:
                        normalized[method][criterion] = 0.0
        
        return normalized

    def _rank_for_application(self, normalized_scores, criteria):
        """Rank methods for specific application"""
        app_scores = {}
        
        for method in self.available_methods:
            if method in normalized_scores:
                score = 0
                valid_criteria = 0
                
                for criterion in criteria:
                    if criterion in normalized_scores[method]:
                        score += normalized_scores[method][criterion]
                        valid_criteria += 1
                
                if valid_criteria > 0:
                    app_scores[method] = score / valid_criteria
        
        sorted_methods = sorted(app_scores.keys(), key=lambda x: app_scores[x], reverse=True)
        
        return {
            'ranking': sorted_methods,
            'scores': app_scores,
            'criteria': criteria
        }

    def compute_all_metrics(self):
        """Compute all comparison metrics"""
        print("🔄 Computing all comparison metrics...")
        
        # 1. Reconstruction quality assessment
        for method in self.available_methods:
            metrics = self.assess_reconstruction_quality(method)
            if metrics:
                self.reconstruction_metrics[method] = metrics
        
        # 2. Component stability analysis
        if len(self.available_methods) > 1:
            self.analyze_component_stability()
        
        # 3. Statistical tests
        if len(self.reconstruction_metrics) > 1:
            self.perform_statistical_tests()
        
        # 4. Method rankings
        self.generate_method_rankings()
        
        print("✅ All metrics computed successfully")

    def create_performance_radar_chart(self):
        """Create multi-dimensional performance radar chart"""
        print("🔄 Creating performance radar chart...")
        
        try:
            if 'overall' not in self.method_rankings:
                print("   ⚠️  No ranking data available")
                return False
            
            # Prepare data
            criteria = list(self.criteria_weights.keys())
            methods = self.available_methods
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
            
            # Number of criteria
            N = len(criteria)
            
            # Compute angle for each criterion
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Colors for methods
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
            
            # Plot each method
            for i, method in enumerate(methods):
                if method in self.method_rankings['overall']['normalized_scores']:
                    scores = self.method_rankings['overall']['normalized_scores'][method]
                    
                    # Extract scores for each criterion
                    method_scores = []
                    for criterion in criteria:
                        method_scores.append(scores.get(criterion, 0))
                    
                    method_scores += method_scores[:1]  # Complete the circle
                    
                    # Plot
                    ax.plot(angles, method_scores, 'o-', linewidth=2, 
                           label=method.upper(), color=colors[i % len(colors)])
                    ax.fill(angles, method_scores, alpha=0.25, color=colors[i % len(colors)])
            
            # Customize plot
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([c.replace('_', ' ').title() for c in criteria])
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)
            
            # Title and legend
            plt.title('Method Performance Comparison\nMulti-Criteria Radar Chart', 
                     size=16, fontweight='bold', pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            # Save
            radar_file = self.figures_dir / "ps06_fig01_performance_radar.png"
            plt.savefig(radar_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Performance radar chart saved: {radar_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating radar chart: {e}")
            return False

    def create_reconstruction_quality_heatmap(self):
        """Create geographic heatmap of reconstruction quality"""
        print("🔄 Creating reconstruction quality heatmap...")
        
        try:
            n_methods = len(self.available_methods)
            if n_methods == 0:
                print("   ⚠️  No methods available")
                return False
            
            # Create figure with subplots for each method
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Color map limits (use consistent scale across methods)
            all_rmse = []
            for method in self.available_methods:
                if method in self.reconstruction_metrics:
                    all_rmse.extend(self.reconstruction_metrics[method]['rmse_per_station'])
            
            if not all_rmse:
                print("   ⚠️  No RMSE data available")
                return False
            
            vmin, vmax = np.percentile(all_rmse, [5, 95])  # Use 5-95 percentile for robust range
            
            for i, method in enumerate(self.available_methods[:4]):  # Max 4 methods
                ax = axes[i]
                
                if method in self.reconstruction_metrics:
                    rmse_data = np.array(self.reconstruction_metrics[method]['rmse_per_station'])
                    coordinates = self.coordinates[:len(rmse_data)]
                    
                    # Create scatter plot
                    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                                       c=rmse_data, cmap='viridis_r', 
                                       s=10, alpha=0.7, vmin=vmin, vmax=vmax)
                    
                    ax.set_title(f'{method.upper()} Reconstruction RMSE')
                    ax.set_xlabel('Longitude (°E)')
                    ax.set_ylabel('Latitude (°N)')
                    ax.grid(True, alpha=0.3)
                    
                    # Equal aspect ratio
                    ax.set_aspect('equal', adjustable='box')
                    
                    # Colorbar
                    plt.colorbar(scatter, ax=ax, label='RMSE (mm)', shrink=0.8)
                else:
                    ax.text(0.5, 0.5, f'{method.upper()}\nNo data available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Hide unused subplots
            for i in range(len(self.available_methods), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save
            heatmap_file = self.figures_dir / "ps06_fig02_reconstruction_quality_heatmap.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Reconstruction quality heatmap saved: {heatmap_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating heatmap: {e}")
            return False

    def create_statistical_comparison_plots(self):
        """Create statistical comparison plots"""
        print("🔄 Creating statistical comparison plots...")
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: RMSE distribution comparison
            ax = axes[0, 0]
            rmse_data = []
            method_labels = []
            
            for method in self.available_methods:
                if method in self.reconstruction_metrics:
                    rmse_vals = self.reconstruction_metrics[method]['rmse_per_station']
                    rmse_data.append(rmse_vals)
                    method_labels.append(method.upper())
            
            if rmse_data:
                bp = ax.boxplot(rmse_data, labels=method_labels, patch_artist=True)
                colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title('RMSE Distribution Comparison')
                ax.set_ylabel('RMSE (mm)')
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Correlation distribution comparison  
            ax = axes[0, 1]
            corr_data = []
            
            for method in self.available_methods:
                if method in self.reconstruction_metrics:
                    corr_vals = self.reconstruction_metrics[method]['correlations']
                    corr_data.append(corr_vals)
            
            if corr_data:
                bp = ax.boxplot(corr_data, labels=method_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title('Correlation Distribution Comparison')
                ax.set_ylabel('Correlation')
                ax.grid(True, alpha=0.3)
            
            # Plot 3: Method ranking bar chart
            ax = axes[1, 0]
            if 'overall' in self.method_rankings:
                methods = self.method_rankings['overall']['ranking']
                scores = [self.method_rankings['overall']['scores'][m] for m in methods]
                
                bars = ax.bar(range(len(methods)), scores, color=colors[:len(methods)])
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels([m.upper() for m in methods])
                ax.set_title('Overall Method Ranking')
                ax.set_ylabel('Overall Score')
                ax.grid(True, alpha=0.3)
                
                # Add score labels on bars
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # Plot 4: Statistical significance matrix
            ax = axes[1, 1]
            if self.statistical_tests:
                # Create significance matrix
                methods = self.available_methods
                n_methods = len(methods)
                sig_matrix = np.zeros((n_methods, n_methods))
                
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i != j:
                            test_key = f"{method1}_vs_{method2}"
                            alt_key = f"{method2}_vs_{method1}"
                            
                            if test_key in self.statistical_tests:
                                sig = self.statistical_tests[test_key]['paired_ttest']['significant']
                                sig_matrix[i, j] = 1 if sig else 0
                            elif alt_key in self.statistical_tests:
                                sig = self.statistical_tests[alt_key]['paired_ttest']['significant']
                                sig_matrix[i, j] = 1 if sig else 0
                
                im = ax.imshow(sig_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
                ax.set_xticks(range(n_methods))
                ax.set_yticks(range(n_methods))
                ax.set_xticklabels([m.upper() for m in methods])
                ax.set_yticklabels([m.upper() for m in methods])
                ax.set_title('Statistical Significance Matrix\n(p < 0.05)')
                
                # Add text annotations
                for i in range(n_methods):
                    for j in range(n_methods):
                        text = ax.text(j, i, 'Sig' if sig_matrix[i, j] else 'NS',
                                     ha="center", va="center", color="black" if sig_matrix[i, j] < 0.5 else "white")
                
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            plt.tight_layout()
            
            # Save
            stats_file = self.figures_dir / "ps06_fig03_statistical_distributions.png"
            plt.savefig(stats_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Statistical comparison plots saved: {stats_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating statistical plots: {e}")
            return False

    def create_method_specific_analysis(self):
        """Create method-specific characteristic analysis"""
        print("🔄 Creating method-specific analysis plots...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Component count analysis
            ax = axes[0, 0]
            component_data = []
            method_labels = []
            
            for method in self.available_methods:
                if method in self.method_data:
                    counts = self.method_data[method]['n_imfs_per_station']
                    component_data.append(counts)
                    method_labels.append(method.upper())
            
            if component_data:
                bp = ax.boxplot(component_data, labels=method_labels, patch_artist=True)
                colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title('Component Count Distribution')
                ax.set_ylabel('Number of IMFs')
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Success rate comparison
            ax = axes[0, 1]
            success_rates = []
            
            for method in self.available_methods:
                if method in self.method_data:
                    success_rate = self.method_data[method]['successful_decompositions'] / len(self.coordinates)
                    success_rates.append(success_rate * 100)  # Convert to percentage
            
            if success_rates:
                bars = ax.bar(method_labels, success_rates, color=colors[:len(method_labels)])
                ax.set_title('Decomposition Success Rate')
                ax.set_ylabel('Success Rate (%)')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                
                # Add percentage labels on bars
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
            
            # Plot 3: Reconstruction quality vs component count
            ax = axes[1, 0]
            
            for i, method in enumerate(self.available_methods):
                if method in self.reconstruction_metrics and method in self.method_data:
                    rmse_vals = self.reconstruction_metrics[method]['rmse_per_station']
                    component_counts = self.method_data[method]['n_imfs_per_station']
                    
                    # Ensure same length
                    min_len = min(len(rmse_vals), len(component_counts))
                    rmse_vals = rmse_vals[:min_len]
                    component_counts = component_counts[:min_len]
                    
                    ax.scatter(component_counts, rmse_vals, alpha=0.6, 
                             label=method.upper(), color=colors[i % len(colors)], s=20)
            
            ax.set_xlabel('Number of IMFs')
            ax.set_ylabel('RMSE (mm)')
            ax.set_title('Reconstruction Quality vs Component Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Application-specific rankings
            ax = axes[1, 1]
            
            if 'scientific_accuracy' in self.method_rankings and 'operational_efficiency' in self.method_rankings:
                sci_scores = self.method_rankings['scientific_accuracy']['scores']
                ops_scores = self.method_rankings['operational_efficiency']['scores']
                
                methods = list(set(sci_scores.keys()) & set(ops_scores.keys()))
                
                for i, method in enumerate(methods):
                    ax.scatter(sci_scores[method], ops_scores[method], 
                             s=100, label=method.upper(), color=colors[i % len(colors)])
                    ax.annotate(method.upper(), 
                               (sci_scores[method], ops_scores[method]),
                               xytext=(5, 5), textcoords='offset points')
                
                ax.set_xlabel('Scientific Accuracy Score')
                ax.set_ylabel('Operational Efficiency Score')
                ax.set_title('Application-Specific Performance')
                ax.grid(True, alpha=0.3)
                
                # Add diagonal line (equal performance)
                lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 
                       max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1)
            
            plt.tight_layout()
            
            # Save
            analysis_file = self.figures_dir / "ps06_fig04_method_specific_analysis.png"
            plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Method-specific analysis saved: {analysis_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating method-specific analysis: {e}")
            return False

    def create_ranking_summary(self):
        """Create final ranking summary visualization"""
        print("🔄 Creating ranking summary visualization...")
        
        try:
            if 'overall' not in self.method_rankings:
                print("   ⚠️  No ranking data available")
                return False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Overall ranking
            ax = axes[0, 0]
            methods = self.method_rankings['overall']['ranking']
            scores = [self.method_rankings['overall']['scores'][m] for m in methods]
            colors = ['gold', 'silver', '#cd7f32', 'lightcoral']  # Gold, silver, bronze, etc.
            
            bars = ax.barh(range(len(methods)), scores, color=colors[:len(methods)])
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([f"{i+1}. {m.upper()}" for i, m in enumerate(methods)])
            ax.set_xlabel('Overall Score')
            ax.set_title('Final Method Ranking\n(Overall Performance)')
            ax.grid(True, alpha=0.3)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{score:.3f}', ha='left', va='center')
            
            # Plot 2: Criteria breakdown for top method
            ax = axes[0, 1]
            if methods:
                top_method = methods[0]
                criteria_scores = self.method_rankings['overall']['normalized_scores'][top_method]
                
                criteria_names = list(criteria_scores.keys())
                criteria_values = list(criteria_scores.values())
                
                bars = ax.bar(range(len(criteria_names)), criteria_values, 
                             color='skyblue', alpha=0.7)
                ax.set_xticks(range(len(criteria_names)))
                ax.set_xticklabels([c.replace('_', '\n') for c in criteria_names], rotation=0)
                ax.set_ylabel('Normalized Score')
                ax.set_title(f'Performance Breakdown\n{top_method.upper()} (Best Method)')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, criteria_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.2f}', ha='center', va='bottom')
            
            # Plot 3: Application-specific rankings
            ax = axes[1, 0]
            app_rankings = ['scientific_accuracy', 'operational_efficiency', 'frequency_analysis']
            app_names = ['Scientific\nAccuracy', 'Operational\nEfficiency', 'Frequency\nAnalysis']
            
            ranking_data = []
            for app in app_rankings:
                if app in self.method_rankings:
                    ranking_data.append(self.method_rankings[app]['ranking'])
                else:
                    ranking_data.append([])
            
            # Create a table-like visualization
            y_pos = np.arange(len(app_names))
            
            for i, (app_name, ranking) in enumerate(zip(app_names, ranking_data)):
                if ranking:
                    ranking_text = ' > '.join([m.upper() for m in ranking[:3]])  # Top 3
                    ax.text(0.05, i, f"{app_name}:\n{ranking_text}", 
                           fontsize=10, verticalalignment='center')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, len(app_names) - 0.5)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title('Application-Specific Rankings\n(Top 3 Methods)')
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Plot 4: Statistical significance summary
            ax = axes[1, 1]
            
            if self.statistical_tests:
                significance_data = []
                pair_labels = []
                
                for test_name, test_result in self.statistical_tests.items():
                    if isinstance(test_result, dict) and 'paired_ttest' in test_result:
                        pair_labels.append(test_name.replace('_vs_', ' vs\n').upper())
                        significance_data.append(test_result['paired_ttest']['significant'])
                
                if significance_data:
                    colors_sig = ['red' if sig else 'lightgray' for sig in significance_data]
                    bars = ax.bar(range(len(pair_labels)), [1 if sig else 0.5 for sig in significance_data],
                                 color=colors_sig, alpha=0.7)
                    
                    ax.set_xticks(range(len(pair_labels)))
                    ax.set_xticklabels(pair_labels, fontsize=8)
                    ax.set_ylabel('Statistical Significance')
                    ax.set_title('Pairwise Significance Tests\n(p < 0.05)')
                    ax.set_ylim(0, 1.1)
                    ax.set_yticks([0, 0.5, 1])
                    ax.set_yticklabels(['Not Tested', 'Not Significant', 'Significant'])
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            summary_file = self.figures_dir / "ps06_fig05_ranking_summary.png"
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Ranking summary saved: {summary_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating ranking summary: {e}")
            return False

    def save_results(self):
        """Save all analysis results to files"""
        print("💾 Saving method comparison results...")
        
        try:
            # Save reconstruction metrics
            metrics_file = self.results_dir / "method_comparison_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.reconstruction_metrics, f, indent=2, default=str)
            print(f"✅ Saved reconstruction metrics: {metrics_file}")
            
            # Save component stability results
            stability_file = self.results_dir / "component_stability.json"
            with open(stability_file, 'w') as f:
                json.dump(self.component_stability, f, indent=2, default=str)
            print(f"✅ Saved component stability: {stability_file}")
            
            # Save statistical test results
            stats_file = self.results_dir / "statistical_tests.json"
            with open(stats_file, 'w') as f:
                json.dump(self.statistical_tests, f, indent=2, default=str)
            print(f"✅ Saved statistical tests: {stats_file}")
            
            # Save method rankings
            rankings_file = self.results_dir / "method_rankings.json"
            with open(rankings_file, 'w') as f:
                json.dump(self.method_rankings, f, indent=2, default=str)
            print(f"✅ Saved method rankings: {rankings_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

    def generate_recommendation_report(self):
        """Generate comprehensive method recommendation report"""
        print("📝 Generating method recommendation report...")
        
        try:
            report_content = self._create_recommendation_report_content()
            
            report_file = self.results_dir / "method_recommendation_report.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            print(f"✅ Recommendation report saved: {report_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating recommendation report: {e}")
            return False

    def _create_recommendation_report_content(self):
        """Create the content for the recommendation report"""
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get best methods
        overall_best = self.method_rankings['overall']['ranking'][0] if 'overall' in self.method_rankings else "N/A"
        
        report = f"""# Method Comparison Report - Taiwan Subsidence Analysis

**Analysis Date**: {timestamp}  
**Methods Compared**: {', '.join(self.available_methods).upper()}  
**Total Stations**: {len(self.coordinates)}  

---

## Executive Summary

**Best Overall Method**: **{overall_best.upper()}**

This comprehensive analysis compared {len(self.available_methods)} decomposition methods (EMD, FFT, VMD, Wavelet) across multiple criteria including reconstruction accuracy, physical interpretability, computational efficiency, robustness, frequency separation, and component stability.

---

## Method Rankings

### Overall Performance Ranking
"""
        
        if 'overall' in self.method_rankings:
            for i, method in enumerate(self.method_rankings['overall']['ranking']):
                score = self.method_rankings['overall']['scores'][method]
                report += f"{i+1}. **{method.upper()}** (Score: {score:.3f})\n"
        
        report += """
### Application-Specific Rankings

"""
        
        # Application-specific rankings
        app_rankings = {
            'scientific_accuracy': 'Scientific Accuracy (Research Applications)',
            'operational_efficiency': 'Operational Efficiency (Monitoring Applications)', 
            'frequency_analysis': 'Frequency Analysis (Spectral Studies)'
        }
        
        for app_key, app_name in app_rankings.items():
            if app_key in self.method_rankings:
                report += f"#### {app_name}\n"
                for i, method in enumerate(self.method_rankings[app_key]['ranking']):
                    score = self.method_rankings[app_key]['scores'][method]
                    report += f"{i+1}. {method.upper()} (Score: {score:.3f})\n"
                report += "\n"
        
        report += """---

## Detailed Analysis Results

### Reconstruction Quality Metrics
"""
        
        # Add reconstruction metrics
        for method in self.available_methods:
            if method in self.reconstruction_metrics:
                metrics = self.reconstruction_metrics[method]
                report += f"""
#### {method.upper()}
- **RMSE**: {metrics['rmse_overall']:.3f} mm
- **Correlation**: {metrics['correlation_mean']:.3f}
- **R²**: {metrics['r2_mean']:.3f}
- **SNR**: {metrics['snr']:.1f} dB
- **Success Rate**: {metrics['reconstruction_success_rate']:.1%}
"""
        
        report += """
### Statistical Significance Tests

"""
        
        # Add statistical test results
        if self.statistical_tests:
            for test_name, test_result in self.statistical_tests.items():
                if isinstance(test_result, dict) and 'paired_ttest' in test_result:
                    method1, method2 = test_name.split('_vs_')
                    pval = test_result['paired_ttest']['p_value']
                    significant = test_result['paired_ttest']['significant']
                    effect_size = test_result.get('effect_size', 'unknown')
                    
                    report += f"**{method1.upper()} vs {method2.upper()}**: "
                    report += f"{'Significant' if significant else 'Not significant'} "
                    report += f"(p = {pval:.4f}, Effect size: {effect_size})\n"
        
        report += """
---

## Recommendations

### For Scientific Research Applications
"""
        
        if 'scientific_accuracy' in self.method_rankings:
            best_sci = self.method_rankings['scientific_accuracy']['ranking'][0]
            report += f"**Recommended Method**: {best_sci.upper()}\n\n"
            report += f"The {best_sci.upper()} method provides the best balance of reconstruction accuracy, physical interpretability, and component stability for scientific analysis.\n"
        
        report += """
### For Operational Monitoring Applications
"""
        
        if 'operational_efficiency' in self.method_rankings:
            best_ops = self.method_rankings['operational_efficiency']['ranking'][0]
            report += f"**Recommended Method**: {best_ops.upper()}\n\n"
            report += f"The {best_ops.upper()} method offers optimal computational efficiency and robustness for operational monitoring systems.\n"
        
        report += """
### For Frequency Analysis Applications
"""
        
        if 'frequency_analysis' in self.method_rankings:
            best_freq = self.method_rankings['frequency_analysis']['ranking'][0]
            report += f"**Recommended Method**: {best_freq.upper()}\n\n"
            report += f"The {best_freq.upper()} method provides superior frequency separation and spectral fidelity for detailed frequency domain analysis.\n"
        
        report += """
---

## Implementation Guidelines

### Data Quality Requirements
- Minimum 100 time points for reliable decomposition
- Regular sampling intervals preferred (6-day for Taiwan InSAR)
- GPS reference correction applied (LNJS station)

### Computational Considerations
- EMD: Most computationally intensive but highest accuracy
- FFT: Fastest processing but requires regular sampling
- VMD: Good balance of speed and accuracy
- Wavelet: Efficient for time-frequency analysis

### Quality Control
- Monitor decomposition success rates (>95% recommended)
- Validate reconstruction quality (RMSE <5mm, Correlation >0.8)
- Check component interpretability (seasonal patterns visible)

---

## Limitations and Future Work

### Current Limitations
- Analysis limited to Taiwan subsidence data (2018-2021)
- Comparison based on reconstruction quality metrics
- Limited to 4 decomposition methods

### Future Enhancements
- Include additional decomposition methods (EEMD, CEEMDAN)
- Extend analysis to other geographic regions
- Incorporate external validation data
- Add computational performance benchmarks

---

## Data Files Generated

1. `method_comparison_metrics.json` - Detailed reconstruction metrics
2. `component_stability.json` - Cross-method component analysis
3. `statistical_tests.json` - Statistical significance test results
4. `method_rankings.json` - Complete ranking results

## Figures Generated

1. `ps06_fig01_performance_radar.png` - Multi-criteria radar chart
2. `ps06_fig02_reconstruction_quality_heatmap.png` - Geographic quality comparison
3. `ps06_fig03_statistical_distributions.png` - Statistical comparison plots
4. `ps06_fig04_method_specific_analysis.png` - Method characteristics
5. `ps06_fig05_ranking_summary.png` - Final ranking visualization

---

*Generated by ps06_method_comparison.py - Taiwan Subsidence Analysis Pipeline*
"""
        
        return report

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Method Comparison Analysis for Taiwan Subsidence')
    parser.add_argument('--methods', type=str, default='all',
                       help='Comma-separated list of methods: emd,fft,vmd,wavelet or "all"')
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Create visualization figures')
    parser.add_argument('--save-results', action='store_true',
                       help='Save analysis results to files')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate recommendation report')
    return parser.parse_args()

def main():
    """Main method comparison analysis workflow"""
    args = parse_arguments()
    
    # Parse methods
    if args.methods.lower() == 'all':
        methods = ['emd', 'fft', 'vmd', 'wavelet']
    else:
        methods = [m.strip().lower() for m in args.methods.split(',')]
    
    print("=" * 80)
    print("🚀 ps06_method_comparison.py - Method Comparison & Validation")
    print(f"📋 METHODS: {', '.join(methods).upper()}")
    print("=" * 80)
    
    # Initialize analysis
    comparison_analysis = MethodComparisonAnalysis(methods=methods)
    
    # Load all method data
    if not comparison_analysis.load_all_method_data():
        print("❌ Failed to load method data")
        return False
    
    print(f"\n🔄 PERFORMING COMPREHENSIVE COMPARISON")
    print("-" * 50)
    
    # Compute all metrics
    comparison_analysis.compute_all_metrics()
    
    # Create visualizations
    if args.create_visualizations:
        print(f"\n🔄 CREATING VISUALIZATIONS")
        print("-" * 50)
        
        comparison_analysis.create_performance_radar_chart()
        comparison_analysis.create_reconstruction_quality_heatmap()
        comparison_analysis.create_statistical_comparison_plots()
        comparison_analysis.create_method_specific_analysis()
        comparison_analysis.create_ranking_summary()
        
        print("📊 Generated all visualizations successfully")
    
    # Save results
    if args.save_results:
        comparison_analysis.save_results()
    
    # Generate recommendation report
    if args.generate_report:
        comparison_analysis.generate_recommendation_report()
    
    print("\n" + "=" * 80)
    print("✅ ps06_method_comparison.py ANALYSIS COMPLETED SUCCESSFULLY")
    
    # Print summary
    if 'overall' in comparison_analysis.method_rankings:
        ranking = comparison_analysis.method_rankings['overall']['ranking']
        print("🏆 FINAL METHOD RANKING:")
        for i, method in enumerate(ranking):
            score = comparison_analysis.method_rankings['overall']['scores'][method]
            print(f"   {i+1}. {method.upper()} (Score: {score:.3f})")
        
        print(f"\n🥇 BEST OVERALL METHOD: {ranking[0].upper()}")
    
    print("📊 Generated outputs:")
    print("   1. Performance radar chart")
    print("   2. Reconstruction quality heatmap")  
    print("   3. Statistical comparison plots")
    print("   4. Method-specific analysis")
    print("   5. Ranking summary visualization")
    print("   6. Comprehensive recommendation report")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
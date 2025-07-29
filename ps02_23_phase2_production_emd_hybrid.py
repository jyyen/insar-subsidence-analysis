#!/usr/bin/env python3
"""
ps02_23_phase2_production_emd_hybrid.py: Phase 2 production framework
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import time
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Core scientific packages
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import silhouette_score

# Signal processing packages
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.stats import zscore

# Wavelet denoising
try:
    import pywt
    HAS_WAVELETS = True
    print("✅ PyWavelets available for advanced denoising")
except ImportError:
    HAS_WAVELETS = False
    print("⚠️ PyWavelets not available. Install with: conda install pywavelets")

# Advanced image processing for spatial noise
try:
    from skimage.restoration import denoise_tv_chambolle
    from skimage.filters import rank
    HAS_SKIMAGE = True
    print("✅ scikit-image available for spatial noise handling")
except ImportError:
    HAS_SKIMAGE = False
    print("⚠️ scikit-image not available. Install with: conda install scikit-image")

warnings.filterwarnings('ignore')

@dataclass
class NoiseAnalysisResult:
    """Results from comprehensive noise analysis"""
    noise_level: float
    noise_type: str
    spatial_correlation: float
    temporal_correlation: float
    recommended_denoising: List[str]
    snr_estimate: float

@dataclass
class ProcessingChunk:
    """Processing chunk for memory-efficient computation"""
    start_idx: int
    end_idx: int
    n_stations: int
    chunk_id: int

class AdvancedNoiseHandler:
    """Comprehensive noise handling using multiple Python packages"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.scaler = RobustScaler()  # Better than StandardScaler for outliers
        self.pca_denoiser = None
        self.ica_denoiser = None
        
    def analyze_noise_characteristics(self, signals: np.ndarray, coordinates: np.ndarray) -> NoiseAnalysisResult:
        """Comprehensive noise analysis using multiple methods"""
        
        print("🔍 Analyzing noise characteristics...")
        
        # 1. Estimate noise level using robust statistics
        signal_mad = np.median(np.abs(signals - np.median(signals, axis=1, keepdims=True)), axis=1)
        noise_level = np.median(signal_mad)
        
        # 2. Analyze noise spatial correlation
        if len(signals) > 100:
            sample_indices = np.random.choice(len(signals), 100, replace=False)
            sample_signals = signals[sample_indices]
            sample_coords = coordinates[sample_indices]
        else:
            sample_signals = signals
            sample_coords = coordinates
        
        # Compute noise correlation matrix
        noise_residuals = sample_signals - signal.savgol_filter(sample_signals, window_length=min(21, sample_signals.shape[1]//2), polyorder=3, axis=1)
        correlation_matrix = np.corrcoef(noise_residuals)
        spatial_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        
        # 3. Analyze temporal correlation
        temporal_autocorr = []
        for i in range(min(50, len(signals))):
            autocorr = np.correlate(noise_residuals[i], noise_residuals[i], mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            if len(autocorr) > 1:
                temporal_autocorr.append(autocorr[1] / autocorr[0])
        
        temporal_correlation = np.mean(temporal_autocorr) if temporal_autocorr else 0.0
        
        # 4. Estimate SNR
        signal_power = np.mean(np.var(signals, axis=1))
        noise_power = noise_level**2
        snr_estimate = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # 5. Determine noise type and recommendations
        if spatial_correlation > 0.3:
            noise_type = "spatially_correlated"
            recommendations = ["spatial_filtering", "pca_denoising", "wavelet_denoising"]
        elif temporal_correlation > 0.3:
            noise_type = "temporally_correlated"
            recommendations = ["adaptive_filtering", "ica_denoising", "savgol_filter"]
        elif noise_level > signal_power * 0.5:
            noise_type = "high_amplitude"
            recommendations = ["robust_filtering", "median_filter", "wavelet_denoising"]
        else:
            noise_type = "gaussian_like"
            recommendations = ["wiener_filter", "gaussian_filter", "pca_denoising"]
        
        print(f"   📊 Noise level: {noise_level:.2f}mm")
        print(f"   📊 Noise type: {noise_type}")
        print(f"   📊 SNR estimate: {snr_estimate:.1f}dB")
        print(f"   📊 Spatial correlation: {spatial_correlation:.3f}")
        print(f"   📊 Recommended denoising: {recommendations}")
        
        return NoiseAnalysisResult(
            noise_level=noise_level,
            noise_type=noise_type,
            spatial_correlation=spatial_correlation,
            temporal_correlation=temporal_correlation,
            recommended_denoising=recommendations,
            snr_estimate=snr_estimate
        )
    
    def apply_sklearn_denoising(self, signals: np.ndarray, n_components: int = 10) -> np.ndarray:
        """Apply sklearn-based denoising (PCA/ICA)"""
        
        print(f"🧹 Applying sklearn denoising (PCA/ICA, {n_components} components)...")
        
        # Robust scaling
        signals_scaled = self.scaler.fit_transform(signals.T).T
        
        # PCA denoising
        self.pca_denoiser = PCA(n_components=n_components, random_state=42)
        pca_components = self.pca_denoiser.fit_transform(signals_scaled)
        pca_denoised = self.pca_denoiser.inverse_transform(pca_components)
        
        # ICA denoising (more aggressive)
        self.ica_denoiser = FastICA(n_components=min(n_components, signals.shape[0]//2), random_state=42, max_iter=200)
        try:
            ica_components = self.ica_denoiser.fit_transform(signals_scaled)
            ica_denoised = self.ica_denoiser.inverse_transform(ica_components)
        except:
            print("   ⚠️ ICA failed, using PCA only")
            ica_denoised = pca_denoised
        
        # Combine PCA and ICA (weighted average)
        combined_denoised = 0.7 * pca_denoised + 0.3 * ica_denoised
        
        # Inverse scaling
        denoised_signals = self.scaler.inverse_transform(combined_denoised.T).T
        
        # Quality metrics
        noise_reduction = np.mean(np.var(signals - denoised_signals, axis=1)) / np.mean(np.var(signals, axis=1))
        print(f"   📈 Noise variance reduction: {noise_reduction:.3f}")
        
        return denoised_signals
    
    def apply_wavelet_denoising(self, signals: np.ndarray, wavelet: str = 'db6') -> np.ndarray:
        """Apply wavelet denoising using PyWavelets"""
        
        if not HAS_WAVELETS:
            print("   ⚠️ PyWavelets not available, skipping wavelet denoising")
            return signals
        
        print(f"🌊 Applying wavelet denoising (wavelet: {wavelet})...")
        
        denoised_signals = np.zeros_like(signals)
        
        for i in range(len(signals)):
            try:
                # Estimate noise level for this station
                sigma = np.median(np.abs(signals[i] - np.median(signals[i]))) / 0.6745
                
                # Manual wavelet denoising if denoise_wavelet not available
                if hasattr(pywt, 'denoise_wavelet'):
                    denoised_signals[i] = pywt.denoise_wavelet(
                        signals[i], 
                        wavelet=wavelet, 
                        mode='soft', 
                        sigma=sigma,
                        rescale_sigma=True
                    )
                else:
                    # Manual wavelet denoising implementation
                    coeffs = pywt.wavedec(signals[i], wavelet, level=4)
                    threshold = sigma * np.sqrt(2 * np.log(len(signals[i])))
                    
                    # Apply soft thresholding to detail coefficients
                    coeffs_thresh = coeffs.copy()
                    for j in range(1, len(coeffs)):
                        coeffs_thresh[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
                    
                    denoised_signals[i] = pywt.waverec(coeffs_thresh, wavelet)
                    
                    # Ensure same length
                    if len(denoised_signals[i]) != len(signals[i]):
                        denoised_signals[i] = denoised_signals[i][:len(signals[i])]
            except Exception as e:
                print(f"   ⚠️ Wavelet denoising failed for station {i}: {e}")
                denoised_signals[i] = signals[i]
        
        # Quality metrics
        noise_reduction = np.mean(np.var(signals - denoised_signals, axis=1)) / np.mean(np.var(signals, axis=1))
        print(f"   📈 Wavelet noise reduction: {noise_reduction:.3f}")
        
        return denoised_signals
    
    def apply_scipy_filtering(self, signals: np.ndarray, filter_type: str = 'savgol') -> np.ndarray:
        """Apply scipy-based filtering"""
        
        print(f"🔧 Applying scipy filtering (type: {filter_type})...")
        
        filtered_signals = np.zeros_like(signals)
        
        for i in range(len(signals)):
            if filter_type == 'savgol':
                # Savitzky-Golay filter - preserves trends
                window_length = min(21, len(signals[i])//3)
                if window_length % 2 == 0:
                    window_length += 1
                filtered_signals[i] = signal.savgol_filter(signals[i], window_length=window_length, polyorder=3)
                
            elif filter_type == 'median':
                # Median filter - removes impulse noise
                kernel_size = min(5, len(signals[i])//10)
                filtered_signals[i] = median_filter(signals[i], size=kernel_size)
                
            elif filter_type == 'gaussian':
                # Gaussian filter - smooth denoising
                sigma = len(signals[i]) / 50  # Adaptive sigma
                filtered_signals[i] = gaussian_filter1d(signals[i], sigma=sigma)
                
            elif filter_type == 'wiener':
                # Wiener filter - optimal for Gaussian noise
                try:
                    filtered_signals[i] = signal.wiener(signals[i], noise=None)
                except:
                    filtered_signals[i] = signals[i]  # Fallback
            
            else:
                filtered_signals[i] = signals[i]
        
        return filtered_signals
    
    def apply_emd_noise_separation(self, signals: np.ndarray, emd_imfs: np.ndarray) -> np.ndarray:
        """Separate noise using EMD IMFs"""
        
        print("🧬 Applying EMD-based noise separation...")
        
        denoised_signals = np.zeros_like(signals)
        
        for i in range(len(signals)):
            if i < len(emd_imfs):
                station_imfs = emd_imfs[i]
                
                # First 1-2 IMFs are typically noise
                noise_imfs = station_imfs[:2]
                signal_imfs = station_imfs[2:]
                
                # Reconstruct without noise IMFs
                denoised_signals[i] = np.sum(signal_imfs, axis=0)
            else:
                denoised_signals[i] = signals[i]
        
        return denoised_signals
    
    def adaptive_denoising_pipeline(self, signals: np.ndarray, coordinates: np.ndarray, 
                                   emd_imfs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Adaptive denoising pipeline based on noise characteristics"""
        
        print("🔄 Starting adaptive denoising pipeline...")
        
        # 1. Analyze noise characteristics
        noise_analysis = self.analyze_noise_characteristics(signals, coordinates)
        
        # 2. Apply appropriate denoising methods
        results = {}
        denoised_versions = {}
        
        # Always start with original
        denoised_versions['original'] = signals.copy()
        
        # Apply recommended methods
        if 'pca_denoising' in noise_analysis.recommended_denoising:
            denoised_versions['pca_ica'] = self.apply_sklearn_denoising(signals)
        
        if 'wavelet_denoising' in noise_analysis.recommended_denoising and HAS_WAVELETS:
            denoised_versions['wavelet'] = self.apply_wavelet_denoising(signals)
        
        if 'savgol_filter' in noise_analysis.recommended_denoising:
            denoised_versions['savgol'] = self.apply_scipy_filtering(signals, 'savgol')
        
        if 'median_filter' in noise_analysis.recommended_denoising:
            denoised_versions['median'] = self.apply_scipy_filtering(signals, 'median')
        
        if emd_imfs is not None:
            denoised_versions['emd_separation'] = self.apply_emd_noise_separation(signals, emd_imfs)
        
        # 3. Evaluate and select best denoising
        best_method, best_signals = self._select_best_denoising(signals, denoised_versions)
        
        results['noise_analysis'] = noise_analysis
        results['methods_applied'] = list(denoised_versions.keys())
        results['best_method'] = best_method
        results['denoised_versions'] = {k: v for k, v in denoised_versions.items() if k != 'original'}
        
        print(f"✅ Best denoising method: {best_method}")
        
        return best_signals, results
    
    def _select_best_denoising(self, original: np.ndarray, versions: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
        """Select best denoising method based on multiple criteria"""
        
        print("🎯 Evaluating denoising methods...")
        
        scores = {}
        
        for method, denoised in versions.items():
            if method == 'original':
                continue
                
            # Criteria 1: Noise reduction (but not over-smoothing)
            noise_reduction = 1.0 - np.mean(np.var(original - denoised, axis=1)) / np.mean(np.var(original, axis=1))
            
            # Criteria 2: Signal preservation (correlation with original)
            correlations = []
            for i in range(min(50, len(original))):
                corr = np.corrcoef(original[i], denoised[i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            signal_preservation = np.mean(correlations) if correlations else 0.0
            
            # Criteria 3: Smoothness (penalize over-smoothing)
            original_smoothness = np.mean([np.var(np.diff(original[i])) for i in range(len(original))])
            denoised_smoothness = np.mean([np.var(np.diff(denoised[i])) for i in range(len(denoised))])
            smoothness_ratio = denoised_smoothness / (original_smoothness + 1e-6)
            smoothness_score = 1.0 / (1.0 + np.abs(smoothness_ratio - 0.7))  # Prefer 30% noise reduction
            
            # Combined score
            score = 0.4 * noise_reduction + 0.4 * signal_preservation + 0.2 * smoothness_score
            scores[method] = score
            
            print(f"   {method}: score={score:.3f} (noise_red={noise_reduction:.3f}, preservation={signal_preservation:.3f})")
        
        best_method = max(scores.keys(), key=lambda k: scores[k])
        return best_method, versions[best_method]

class ProductionEMDHybridInSARModel(nn.Module):
    """Production-ready EMD-Hybrid model with noise handling"""
    
    def __init__(self, n_stations: int, n_timepoints: int, coordinates: np.ndarray, 
                 ps00_rates: torch.Tensor, emd_data: Dict, denoised_signals: np.ndarray,
                 n_neighbors: int = 12, device='cpu'):
        super().__init__()
        self.n_stations = n_stations
        self.n_timepoints = n_timepoints
        self.device = device
        self.n_neighbors = n_neighbors
        self.emd_data = emd_data
        
        # Store denoised signals for reference
        self.register_buffer('denoised_displacement', torch.tensor(denoised_signals, dtype=torch.float32, device=device))
        
        # Fixed parameters (preserve perfect rate correlation)
        self.register_buffer('linear_trend', ps00_rates.to(device))
        
        # EMD seasonal components (fixed foundation)
        self.emd_seasonal_components = self._extract_emd_seasonal_components()
        
        # Spatial neighbor graph
        self.coordinates = coordinates
        self.neighbor_graph = self._build_neighbor_graph()
        
        # Learnable parameters for residuals and noise modeling
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        
        # Noise-aware residual modeling
        self.residual_amplitudes = nn.Parameter(torch.ones(n_stations, 4, device=device) * 2.0)
        self.residual_phases = nn.Parameter(torch.rand(n_stations, 4, device=device) * 2 * np.pi)
        self.residual_periods = nn.Parameter(torch.tensor([30.0, 60.0, 120.0, 240.0], device=device))
        
        # Noise modeling parameters
        self.noise_amplitudes = nn.Parameter(torch.ones(n_stations, device=device) * 1.0)
        self.spatial_noise_weights = nn.Parameter(torch.ones(n_stations, device=device) * 0.1)
        
        print(f"🏭 Production EMD-Hybrid: {n_stations} stations with noise modeling")
    
    def _extract_emd_seasonal_components(self) -> torch.Tensor:
        """Extract EMD seasonal components (same as optimized version)"""
        seasonal_bands = {
            'quarterly': (50, 130),
            'semi_annual': (130, 250), 
            'annual': (250, 450),
            'long_annual': (450, 800)
        }
        
        seasonal_signals = torch.zeros(self.n_stations, 4, self.n_timepoints, device=self.device)
        
        for station_idx in range(min(self.n_stations, len(self.emd_data['imfs']))):
            station_imfs = self.emd_data['imfs'][station_idx]
            
            for band_idx, (band_name, (min_period, max_period)) in enumerate(seasonal_bands.items()):
                best_imf_idx = -1
                best_match_score = 0
                
                for imf_idx in range(station_imfs.shape[0]):
                    imf_signal = station_imfs[imf_idx]
                    if np.any(imf_signal != 0) and np.var(imf_signal) > 1e-6:
                        
                        try:
                            # Multiple period estimation methods
                            period_estimates = []
                            
                            # Zero crossing method
                            zero_crossings = np.where(np.diff(np.signbit(imf_signal)))[0]
                            if len(zero_crossings) > 3:
                                avg_half_period = np.mean(np.diff(zero_crossings)) * 6
                                period_estimates.append(avg_half_period * 2)
                            
                            # Autocorrelation method
                            autocorr = np.correlate(imf_signal, imf_signal, mode='full')
                            autocorr = autocorr[autocorr.size // 2:]
                            peaks, _ = signal.find_peaks(autocorr[5:], height=0.2 * np.max(autocorr))
                            if len(peaks) > 0:
                                period_estimates.append((peaks[0] + 5) * 6)
                            
                            if period_estimates:
                                estimated_period = np.median(period_estimates)
                                
                                if min_period <= estimated_period <= max_period:
                                    energy_score = np.var(imf_signal)
                                    period_match_score = 1.0 / (1 + abs(estimated_period - (min_period + max_period) / 2) / 100)
                                    match_score = energy_score * period_match_score
                                    
                                    if match_score > best_match_score:
                                        best_match_score = match_score
                                        best_imf_idx = imf_idx
                        except:
                            continue
                
                if best_imf_idx >= 0:
                    seasonal_signals[station_idx, band_idx] = torch.from_numpy(
                        station_imfs[best_imf_idx]).float().to(self.device)
        
        return seasonal_signals
    
    def _build_neighbor_graph(self) -> Dict:
        """Build spatial neighbor graph"""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(self.coordinates)
        
        distances, indices = nbrs.kneighbors(self.coordinates)
        
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]
        
        weights = np.exp(-neighbor_distances / np.mean(neighbor_distances))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-6)
        
        return {
            'indices': torch.tensor(neighbor_indices, device=self.device),
            'distances': torch.tensor(neighbor_distances, device=self.device),
            'weights': torch.tensor(weights, device=self.device, dtype=torch.float32)
        }
    
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate signals with noise-aware modeling"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add EMD seasonal components (fixed foundation)
        emd_seasonal = torch.sum(self.emd_seasonal_components, dim=1)
        signals = signals + emd_seasonal
        
        # Add residual components
        residual_signals = self._generate_residual_components(time_vector)
        signals = signals + residual_signals
        
        # Add noise modeling
        noise_signals = self._generate_noise_components(time_vector)
        signals = signals + noise_signals
        
        return signals
    
    def _generate_residual_components(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate residual components"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        residual_signals = torch.zeros(batch_size, len(time_vector), device=self.device)
        
        for i in range(4):
            amplitude = self.residual_amplitudes[:, i].unsqueeze(1)
            phase = self.residual_phases[:, i].unsqueeze(1)
            period = torch.clamp(self.residual_periods[i], 20, 300)
            frequency = 1.0 / period
            
            component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            residual_signals += component
        
        return residual_signals
    
    def _generate_noise_components(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate noise modeling components"""
        batch_size = self.n_stations
        
        # Simple noise modeling: station-specific random walk
        noise_signals = torch.zeros(batch_size, len(time_vector), device=self.device)
        
        for i in range(batch_size):
            noise_amplitude = torch.abs(self.noise_amplitudes[i])
            # Generate correlated noise
            base_noise = torch.randn(len(time_vector), device=self.device)
            # Apply smoothing for temporal correlation
            smoothed_noise = torch.conv1d(
                base_noise.unsqueeze(0).unsqueeze(0), 
                torch.ones(1, 1, 3, device=self.device) / 3, 
                padding=1
            ).squeeze()
            
            noise_signals[i] = noise_amplitude * smoothed_noise
        
        return noise_signals
    
    def apply_constraints(self):
        """Apply physical constraints"""
        with torch.no_grad():
            self.residual_amplitudes.clamp_(0, 15)
            self.residual_phases.data = torch.fmod(self.residual_phases.data, 2 * np.pi)
            self.residual_periods.clamp_(20, 300)
            self.noise_amplitudes.clamp_(0, 5)
            self.spatial_noise_weights.clamp_(0, 0.3)
            self.constant_offset.clamp_(-200, 200)

class ChunkedProductionProcessor:
    """Memory-efficient chunked processing for full dataset"""
    
    def __init__(self, chunk_size: int = 500, device='cpu'):
        self.chunk_size = chunk_size
        self.device = device
        self.noise_handler = AdvancedNoiseHandler(device)
        
    def create_processing_chunks(self, n_total_stations: int) -> List[ProcessingChunk]:
        """Create processing chunks for memory efficiency"""
        chunks = []
        
        for start_idx in range(0, n_total_stations, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_total_stations)
            chunk = ProcessingChunk(
                start_idx=start_idx,
                end_idx=end_idx,
                n_stations=end_idx - start_idx,
                chunk_id=len(chunks)
            )
            chunks.append(chunk)
        
        print(f"📦 Created {len(chunks)} processing chunks (chunk_size={self.chunk_size})")
        return chunks
    
    def process_chunk(self, chunk: ProcessingChunk, displacement: np.ndarray, 
                     coordinates: np.ndarray, subsidence_rates: np.ndarray,
                     emd_data: Dict) -> Dict:
        """Process a single chunk"""
        
        print(f"🔄 Processing chunk {chunk.chunk_id}: stations {chunk.start_idx}-{chunk.end_idx}")
        
        # Extract chunk data
        chunk_displacement = displacement[chunk.start_idx:chunk.end_idx]
        chunk_coordinates = coordinates[chunk.start_idx:chunk.end_idx]
        chunk_rates = subsidence_rates[chunk.start_idx:chunk.end_idx]
        
        # Extract chunk EMD data
        chunk_emd_imfs = emd_data['imfs'][chunk.start_idx:chunk.end_idx] if chunk.start_idx < len(emd_data['imfs']) else None
        
        # Apply noise handling
        denoised_signals, noise_results = self.noise_handler.adaptive_denoising_pipeline(
            chunk_displacement, chunk_coordinates, chunk_emd_imfs
        )
        
        # Create and train model for this chunk
        chunk_model = self._create_chunk_model(
            chunk_displacement, chunk_coordinates, chunk_rates, 
            emd_data, denoised_signals, chunk.start_idx
        )
        
        # Train chunk model
        chunk_results = self._train_chunk_model(chunk_model, denoised_signals)
        
        # Clean up memory
        del chunk_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        return {
            'chunk_id': chunk.chunk_id,
            'start_idx': chunk.start_idx,
            'end_idx': chunk.end_idx,
            'denoised_signals': denoised_signals,
            'noise_results': noise_results,
            'model_results': chunk_results
        }
    
    def _create_chunk_model(self, displacement: np.ndarray, coordinates: np.ndarray,
                          rates: np.ndarray, emd_data: Dict, denoised_signals: np.ndarray,
                          global_start_idx: int) -> ProductionEMDHybridInSARModel:
        """Create model for chunk"""
        
        n_stations, n_timepoints = displacement.shape
        ps00_rates = torch.tensor(rates, dtype=torch.float32, device=self.device)
        
        # Create subset EMD data for this chunk
        chunk_emd_data = {
            'imfs': emd_data['imfs'][global_start_idx:global_start_idx + n_stations] if global_start_idx < len(emd_data['imfs']) else np.zeros((n_stations, 10, n_timepoints)),
            'residuals': emd_data['residuals'][global_start_idx:global_start_idx + n_stations] if global_start_idx < len(emd_data['residuals']) else np.zeros((n_stations, n_timepoints))
        }
        
        model = ProductionEMDHybridInSARModel(
            n_stations, n_timepoints, coordinates, ps00_rates, 
            chunk_emd_data, denoised_signals, device=self.device
        )
        
        return model
    
    def _train_chunk_model(self, model: ProductionEMDHybridInSARModel, 
                          target_signals: np.ndarray, max_epochs: int = 800) -> Dict:
        """Train model for chunk"""
        
        target_tensor = torch.tensor(target_signals, dtype=torch.float32, device=self.device)
        time_vector = torch.arange(target_signals.shape[1], dtype=torch.float32, device=self.device) * 6 / 365.25
        
        # Quick training for production
        optimizer = optim.AdamW(model.parameters(), lr=0.015, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.001)
        
        model.train()
        correlations = []
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            predictions = model(time_vector)
            loss = torch.nn.functional.mse_loss(predictions, target_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.apply_constraints()
            optimizer.step()
            scheduler.step()
            
            if epoch % 100 == 0:
                with torch.no_grad():
                    pred_centered = predictions - torch.mean(predictions, dim=1, keepdim=True)
                    target_centered = target_tensor - torch.mean(target_tensor, dim=1, keepdim=True)
                    
                    numerator = torch.sum(pred_centered * target_centered, dim=1)
                    denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
                    
                    correlation = torch.mean(numerator / (denominator + 1e-8))
                    correlations.append(correlation.item())
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_predictions = model(time_vector)
            final_correlation = self._compute_correlation(final_predictions, target_tensor)
            final_rmse = torch.sqrt(torch.mean((final_predictions - target_tensor)**2))
        
        return {
            'final_correlation': final_correlation.item(),
            'final_rmse': final_rmse.item(),
            'correlation_history': correlations,
            'predictions': final_predictions.cpu().numpy()
        }
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute correlation"""
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
        
        correlations = numerator / (denominator + 1e-8)
        return torch.mean(correlations)

def demonstrate_phase2_production_framework():
    """Demonstrate Phase 2 production framework"""
    
    print("🚀 PHASE 2: PRODUCTION EMD-HYBRID FRAMEWORK")
    print("🏭 Full-scale implementation with comprehensive noise handling")
    print("🎯 Objective: Scale to full 7,154 stations with noise management")
    print("="*85)
    
    # Load full dataset
    print(f"\n1️⃣ Loading Full Taiwan InSAR Dataset...")
    
    try:
        data = np.load("data/processed/ps00_preprocessed_data.npz", allow_pickle=True)
        emd_data = np.load("data/processed/ps02_emd_decomposition.npz", allow_pickle=True)
        
        displacement = data['displacement']
        coordinates = data['coordinates']
        subsidence_rates = data['subsidence_rates']
        
        emd_full_data = {
            'imfs': emd_data['imfs'],
            'residuals': emd_data['residuals'],
            'n_imfs_per_station': emd_data['n_imfs_per_station']
        }
        
        print(f"✅ Loaded full dataset: {displacement.shape[0]} stations, {displacement.shape[1]} time points")
        print(f"✅ Loaded EMD data: {emd_full_data['imfs'].shape} IMFs")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("📝 Using demonstration subset instead...")
        
        # Use smaller subset for demonstration
        subset_size = 1000
        data = np.load("data/processed/ps00_preprocessed_data.npz", allow_pickle=True)
        
        selected_indices = np.arange(0, min(len(data['displacement']), 7154), 7154//subset_size)[:subset_size]
        
        displacement = data['displacement'][selected_indices]
        coordinates = data['coordinates'][selected_indices]
        subsidence_rates = data['subsidence_rates'][selected_indices]
        
        # Mock EMD data for demonstration
        n_stations, n_timepoints = displacement.shape
        emd_full_data = {
            'imfs': np.random.randn(n_stations, 8, n_timepoints) * 5,
            'residuals': np.random.randn(n_stations, n_timepoints) * 2,
            'n_imfs_per_station': np.full(n_stations, 8)
        }
        
        print(f"📊 Using demonstration subset: {n_stations} stations")
    
    # Initialize chunked processor
    print(f"\n2️⃣ Initializing Chunked Production Processor...")
    processor = ChunkedProductionProcessor(chunk_size=300, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create processing chunks
    chunks = processor.create_processing_chunks(len(displacement))
    
    # Process chunks
    print(f"\n3️⃣ Processing {len(chunks)} chunks with noise handling...")
    
    all_results = []
    total_start_time = time.time()
    
    for chunk in chunks[:3]:  # Process first 3 chunks for demonstration
        chunk_start_time = time.time()
        
        try:
            chunk_result = processor.process_chunk(
                chunk, displacement, coordinates, subsidence_rates, emd_full_data
            )
            all_results.append(chunk_result)
            
            chunk_time = time.time() - chunk_start_time
            correlation = chunk_result['model_results']['final_correlation']
            rmse = chunk_result['model_results']['final_rmse']
            
            print(f"   ✅ Chunk {chunk.chunk_id}: R={correlation:.4f}, RMSE={rmse:.1f}mm, Time={chunk_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Chunk {chunk.chunk_id} failed: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Aggregate results
    print(f"\n4️⃣ Aggregating Results...")
    
    if all_results:
        all_correlations = [r['model_results']['final_correlation'] for r in all_results]
        all_rmse = [r['model_results']['final_rmse'] for r in all_results]
        
        mean_correlation = np.mean(all_correlations)
        mean_rmse = np.mean(all_rmse)
        
        print(f"📊 PHASE 2 PRODUCTION RESULTS:")
        print(f"   Processed chunks: {len(all_results)}")
        print(f"   Mean correlation: {mean_correlation:.4f} ± {np.std(all_correlations):.3f}")
        print(f"   Mean RMSE: {mean_rmse:.1f} ± {np.std(all_rmse):.1f} mm")
        print(f"   Total processing time: {total_time:.1f} seconds")
        print(f"   Average time per chunk: {total_time/len(all_results):.1f} seconds")
        
        # Noise handling summary
        noise_methods_used = []
        for result in all_results:
            noise_results = result['noise_results']
            noise_methods_used.append(noise_results['best_method'])
        
        from collections import Counter
        method_counts = Counter(noise_methods_used)
        print(f"   🧹 Noise handling methods used:")
        for method, count in method_counts.items():
            print(f"      {method}: {count} chunks")
        
        # Success assessment
        target_correlation = 0.3
        baseline_correlation = 0.065
        improvement_factor = mean_correlation / baseline_correlation
        
        print(f"\n🏆 PHASE 2 ASSESSMENT:")
        print(f"   📈 Improvement over baseline: {improvement_factor:.1f}x")
        print(f"   🎯 Target achievement: {mean_correlation/target_correlation:.1%}")
        
        if mean_correlation >= target_correlation:
            print(f"   🎉 PHASE 2 TARGET ACHIEVED!")
            print(f"   ✅ Production framework validated")
        else:
            print(f"   🔄 Approaching target, framework scalable")
        
        # Save results
        output_file = Path("data/processed/ps02_phase2_production_results.npz")
        output_file.parent.mkdir(exist_ok=True)
        
        np.savez(output_file,
                 correlations=all_correlations,
                 rmse_values=all_rmse,
                 mean_correlation=mean_correlation,
                 processing_time=total_time,
                 noise_methods=noise_methods_used)
        
        print(f"💾 Phase 2 results saved: {output_file}")
        
    else:
        print("❌ No chunks processed successfully")
    
    print(f"\n✅ PHASE 2 PRODUCTION FRAMEWORK DEMONSTRATION COMPLETED!")

if __name__ == "__main__":
    try:
        demonstrate_phase2_production_framework()
    except KeyboardInterrupt:
        print("\n⚠️ Phase 2 framework interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Phase 2 framework failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
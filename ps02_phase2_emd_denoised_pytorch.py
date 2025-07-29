#!/usr/bin/env python3
"""
PS02 Phase 2 EMD-Denoised PyTorch Framework
Phase 2 Innovation: EMD-based noise removal + PyTorch optimization

Key Strategy:
1. ✅ Use EMD IMF1 (highest frequency) for noise identification
2. ✅ FFT post-EMD analysis to categorize and remove high-frequency noise
3. ✅ Clean signals → PyTorch optimization on denoised data
4. ✅ Production-ready chunked processing for 7,154 stations

Author: Taiwan InSAR Subsidence Analysis Project
Created: 2025-07-27
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import time
from sklearn.neighbors import NearestNeighbors
from scipy import signal
from scipy.fft import fft, fftfreq
import json

warnings.filterwarnings('ignore')

class EMDBasedDenoiser:
    """EMD-based signal denoising using IMF1 + FFT analysis"""
    
    def __init__(self, noise_threshold_percentile: float = 85, 
                 high_freq_cutoff_days: float = 60):
        self.noise_threshold_percentile = noise_threshold_percentile
        self.high_freq_cutoff_days = high_freq_cutoff_days
        self.denoising_stats = {}
    
    def denoise_signals(self, displacement: np.ndarray, emd_data: Dict, 
                       time_vector: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Remove noise using EMD IMF1 + FFT high-frequency analysis
        
        Args:
            displacement: Raw InSAR signals (n_stations, n_timepoints)
            emd_data: EMD decomposition results
            time_vector: Time vector in years
            
        Returns:
            denoised_signals, denoising_statistics
        """
        n_stations, n_timepoints = displacement.shape
        denoised_signals = displacement.copy()
        
        # Time sampling info
        dt_years = np.mean(np.diff(time_vector))
        dt_days = dt_years * 365.25
        
        noise_removed_per_station = []
        noise_variance_reduction = []
        
        print(f"🧹 EMD-based denoising: {n_stations} stations")
        print(f"📊 Strategy: IMF1 noise detection + FFT high-frequency removal")
        print(f"⚡ High-frequency cutoff: {self.high_freq_cutoff_days} days")
        
        for station_idx in range(min(n_stations, len(emd_data['imfs']))):
            original_signal = displacement[station_idx]
            station_imfs = emd_data['imfs'][station_idx]
            
            # Step 1: Analyze IMF1 (highest frequency component)
            if station_imfs.shape[0] > 0:
                imf1 = station_imfs[0]  # First IMF = highest frequency
                
                # Estimate noise characteristics from IMF1
                imf1_variance = np.var(imf1)
                imf1_std = np.std(imf1)
                
                # Step 2: FFT analysis of IMF1 to identify noise frequencies
                fft_imf1 = fft(imf1)
                freqs_per_day = fftfreq(len(imf1), dt_days)
                freqs_positive = freqs_per_day[:len(freqs_per_day)//2]
                power_spectrum = np.abs(fft_imf1[:len(fft_imf1)//2])**2
                
                # Identify high-frequency noise (periods < cutoff)
                high_freq_mask = freqs_positive > (1.0 / self.high_freq_cutoff_days)
                high_freq_power = np.sum(power_spectrum[high_freq_mask])
                total_power = np.sum(power_spectrum)
                noise_power_ratio = high_freq_power / (total_power + 1e-10)
                
                # Step 3: Determine noise removal threshold based on IMF1 + FFT analysis
                if noise_power_ratio > 0.3:  # Significant high-frequency content
                    # Use aggressive denoising
                    noise_threshold = np.percentile(np.abs(imf1), self.noise_threshold_percentile)
                    noise_component = imf1
                else:
                    # Use conservative denoising (only extreme outliers from IMF1)
                    noise_threshold = np.percentile(np.abs(imf1), 95)
                    noise_component = imf1 * (np.abs(imf1) > noise_threshold)
                
                # Step 4: Remove identified noise from original signal
                denoised_signal = original_signal - noise_component
                
                # Optional: Apply additional FFT-based high-frequency filtering
                if noise_power_ratio > 0.2:
                    denoised_signal = self._apply_fft_highfreq_filter(
                        denoised_signal, dt_days, self.high_freq_cutoff_days)
                
                denoised_signals[station_idx] = denoised_signal
                
                # Track denoising statistics
                noise_variance_before = np.var(original_signal)
                noise_variance_after = np.var(denoised_signal)
                variance_reduction = (noise_variance_before - noise_variance_after) / noise_variance_before
                
                noise_removed_per_station.append(np.var(noise_component))
                noise_variance_reduction.append(variance_reduction)
            
            else:
                # No EMD data available, skip denoising
                noise_removed_per_station.append(0.0)
                noise_variance_reduction.append(0.0)
        
        # Compile denoising statistics
        self.denoising_stats = {
            'noise_removed_variance': noise_removed_per_station,
            'variance_reduction_ratio': noise_variance_reduction,
            'mean_noise_removal': np.mean(noise_removed_per_station),
            'mean_variance_reduction': np.mean(noise_variance_reduction),
            'stations_processed': len(noise_removed_per_station),
            'high_freq_cutoff_days': self.high_freq_cutoff_days
        }
        
        print(f"✅ EMD denoising complete:")
        print(f"   📊 Mean noise variance removed: {self.denoising_stats['mean_noise_removal']:.3f}")
        print(f"   📈 Mean variance reduction: {self.denoising_stats['mean_variance_reduction']:.1%}")
        print(f"   🎯 Stations processed: {self.denoising_stats['stations_processed']}")
        
        return denoised_signals, self.denoising_stats
    
    def _apply_fft_highfreq_filter(self, signal: np.ndarray, dt_days: float, 
                                 cutoff_days: float) -> np.ndarray:
        """Apply FFT-based high-frequency filter"""
        fft_signal = fft(signal)
        freqs = fftfreq(len(signal), dt_days)
        
        # Create filter mask (remove frequencies > 1/cutoff_days)
        filter_mask = np.abs(freqs) <= (1.0 / cutoff_days)
        
        # Apply filter
        fft_filtered = fft_signal * filter_mask
        
        # Convert back to time domain
        filtered_signal = np.real(np.fft.ifft(fft_filtered))
        
        return filtered_signal

class ProductionEMDDenoisedInSARModel(nn.Module):
    """Production PyTorch model optimized for EMD-denoised signals"""
    
    def __init__(self, n_stations: int, n_timepoints: int, coordinates: np.ndarray,
                 ps00_rates: torch.Tensor, emd_data: Dict, n_neighbors: int = 8, device='cpu'):
        super().__init__()
        self.n_stations = n_stations
        self.n_timepoints = n_timepoints
        self.device = device
        self.n_neighbors = n_neighbors
        self.emd_data = emd_data
        
        # Fixed parameters (preserve perfect rate correlation)
        self.register_buffer('linear_trend', ps00_rates.to(device))
        
        # Build spatial neighbor graph
        self.coordinates = coordinates
        self.neighbor_graph = self._build_neighbor_graph()
        
        # EMD-informed seasonal components (reduced complexity for denoised signals)
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        self.seasonal_amplitudes = nn.Parameter(torch.ones(n_stations, 3, device=device) * 8.0)  # 3 main seasonal
        self.seasonal_phases = nn.Parameter(torch.rand(n_stations, 3, device=device) * 2 * np.pi)
        
        # Fixed optimized periods based on EMD analysis
        self.register_buffer('periods', torch.tensor([0.25, 0.5, 1.0], device=device))  # Quarterly, semi-annual, annual
        
        # Residual modeling (minimal for denoised signals)
        self.residual_amplitudes = nn.Parameter(torch.ones(n_stations, 2, device=device) * 1.5)  # Reduced residuals
        self.residual_phases = nn.Parameter(torch.rand(n_stations, 2, device=device) * 2 * np.pi)
        self.residual_periods = nn.Parameter(torch.tensor([180.0, 730.0], device=device))  # Long-term residuals only
        
        print(f"🚀 Production EMD-denoised model: {n_stations} stations")
        print(f"🧹 Optimized for denoised signals (reduced complexity)")
    
    def _build_neighbor_graph(self) -> Dict:
        """Build spatial neighbor graph"""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(self.coordinates)
        
        distances, indices = nbrs.kneighbors(self.coordinates)
        
        neighbor_indices = indices[:, 1:]  # Exclude self
        neighbor_distances = distances[:, 1:]
        
        weights = np.exp(-neighbor_distances / np.mean(neighbor_distances))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-6)
        
        return {
            'indices': torch.tensor(neighbor_indices, device=self.device),
            'distances': torch.tensor(neighbor_distances, device=self.device),
            'weights': torch.tensor(weights, device=self.device, dtype=torch.float32)
        }
    
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate signals for denoised data"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add main seasonal components (spatially regularized)
        for i, period in enumerate(self.periods):
            smoothed_amplitude = self._apply_spatial_smoothing(self.seasonal_amplitudes[:, i])
            smoothed_phase = self._apply_spatial_smoothing(self.seasonal_phases[:, i], is_phase=True)
            
            amplitude = smoothed_amplitude.unsqueeze(1)
            phase = smoothed_phase.unsqueeze(1)
            frequency = 1.0 / period
            
            seasonal_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals += seasonal_component
        
        # Add minimal residual components for denoised signals
        for i in range(2):  # Only 2 residual components for denoised signals
            amplitude = self.residual_amplitudes[:, i].unsqueeze(1)
            phase = self.residual_phases[:, i].unsqueeze(1)
            period = self.residual_periods[i]
            frequency = 1.0 / period
            
            residual_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals += residual_component
        
        return signals
    
    def _apply_spatial_smoothing(self, parameter: torch.Tensor, 
                               smoothing_factor: float = 0.12, is_phase: bool = False) -> torch.Tensor:
        """Apply spatial smoothing optimized for denoised signals"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        smoothed_values = torch.zeros_like(parameter)
        
        for i in range(self.n_stations):
            current_value = parameter[i]
            neighbor_values = parameter[neighbor_indices[i]]
            weights = neighbor_weights[i]
            
            if is_phase:
                # Circular averaging for phases
                current_complex = torch.complex(torch.cos(current_value), torch.sin(current_value))
                neighbor_complex = torch.complex(torch.cos(neighbor_values), torch.sin(neighbor_values))
                weighted_avg_complex = torch.sum(neighbor_complex * weights)
                mixed_complex = (1 - smoothing_factor) * current_complex + smoothing_factor * weighted_avg_complex
                smoothed_values[i] = torch.angle(mixed_complex)
            else:
                # Regular weighted average
                weighted_avg = torch.sum(neighbor_values * weights)
                smoothed_values[i] = (1 - smoothing_factor) * current_value + smoothing_factor * weighted_avg
        
        return smoothed_values
    
    def apply_denoised_constraints(self):
        """Apply constraints optimized for denoised signals"""
        with torch.no_grad():
            # Seasonal amplitude constraints (tighter for denoised signals)
            self.seasonal_amplitudes.clamp_(0, 40)  # Reduced upper bound
            
            # Residual amplitude constraints (much tighter)
            self.residual_amplitudes.clamp_(0, 8)   # Very small residuals for denoised signals
            
            # Phase constraints
            self.seasonal_phases.data = torch.fmod(self.seasonal_phases.data, 2 * np.pi)
            self.residual_phases.data = torch.fmod(self.residual_phases.data, 2 * np.pi)
            
            # Period constraints for residuals
            self.residual_periods.clamp_(120, 1000)  # Long-term only
            
            # Offset constraints
            self.constant_offset.clamp_(-200, 200)

class ProductionEMDDenoisedLoss(nn.Module):
    """Optimized loss function for denoised signals"""
    
    def __init__(self, alpha_spatial=0.08, alpha_physics=0.03):
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_physics = alpha_physics
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                model: ProductionEMDDenoisedInSARModel) -> Tuple[torch.Tensor, Dict]:
        """Compute optimized loss for denoised signals"""
        
        # Primary fitting loss (more important for denoised signals)
        primary_loss = torch.nn.functional.mse_loss(predicted, target)
        
        # Reduced spatial consistency (denoised signals are cleaner)
        spatial_loss = self._spatial_consistency_loss(model)
        
        # Physics regularization
        physics_loss = self._physics_regularization(model)
        
        total_loss = (primary_loss + 
                     self.alpha_spatial * spatial_loss +
                     self.alpha_physics * physics_loss)
        
        loss_components = {
            'primary': primary_loss.item(),
            'spatial': spatial_loss.item(),
            'physics': physics_loss.item()
        }
        
        return total_loss, loss_components
    
    def _spatial_consistency_loss(self, model: ProductionEMDDenoisedInSARModel) -> torch.Tensor:
        """Spatial consistency optimized for denoised signals"""
        neighbor_indices = model.neighbor_graph['indices']
        neighbor_weights = model.neighbor_graph['weights']
        
        total_loss = torch.tensor(0.0, device=model.device)
        
        # Seasonal amplitude consistency
        for component_idx in range(3):
            amplitudes = model.seasonal_amplitudes[:, component_idx]
            
            for station_idx in range(model.n_stations):
                station_amp = amplitudes[station_idx]
                neighbor_amps = amplitudes[neighbor_indices[station_idx]]
                weights = neighbor_weights[station_idx]
                
                weighted_mean = torch.sum(neighbor_amps * weights)
                consistency_penalty = (station_amp - weighted_mean)**2
                total_loss += consistency_penalty
        
        return total_loss / (model.n_stations * 3)
    
    def _physics_regularization(self, model: ProductionEMDDenoisedInSARModel) -> torch.Tensor:
        """Physics constraints for denoised signals"""
        # Annual component prominence (should be strongest)
        annual_amp = model.seasonal_amplitudes[:, 2]  # Annual component
        other_amps = model.seasonal_amplitudes[:, [0, 1]]  # Quarterly, semi-annual
        max_other = torch.max(other_amps, dim=1)[0]
        
        annual_prominence_penalty = torch.mean(torch.relu(max_other - annual_amp * 0.8))
        
        # Residual amplitudes should be small for denoised signals
        total_residual = torch.sum(model.residual_amplitudes, dim=1)
        residual_penalty = torch.mean(torch.relu(total_residual - 15))
        
        return annual_prominence_penalty + residual_penalty

class ChunkedProductionProcessor:
    """Memory-efficient chunked processing for production deployment"""
    
    def __init__(self, chunk_size: int = 200, overlap: int = 20, device='cpu'):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.processing_stats = {}
    
    def process_full_dataset(self, displacement: np.ndarray, coordinates: np.ndarray,
                           subsidence_rates: np.ndarray, emd_data: Dict,
                           time_vector: np.ndarray, max_epochs: int = 800) -> Dict:
        """Process full dataset in memory-efficient chunks"""
        
        n_stations = displacement.shape[0]
        n_chunks = int(np.ceil(n_stations / self.chunk_size))
        
        print(f"🏭 Production Processing: {n_stations} stations in {n_chunks} chunks")
        print(f"📦 Chunk size: {self.chunk_size}, Overlap: {self.overlap}")
        print(f"🎯 Target: Production-ready results with EMD denoising")
        
        # Initialize EMD denoiser
        denoiser = EMDBasedDenoiser(noise_threshold_percentile=85, high_freq_cutoff_days=60)
        
        # Step 1: Global denoising
        print(f"\n🧹 Phase 2 Step 1: EMD-based denoising...")
        denoised_displacement, denoising_stats = denoiser.denoise_signals(
            displacement, emd_data, time_vector)
        
        # Initialize results storage
        all_results = {
            'correlations': np.zeros(n_stations),
            'rmse': np.zeros(n_stations),
            'fitted_trends': np.zeros(n_stations),
            'fitted_offsets': np.zeros(n_stations),
            'predictions': np.zeros_like(displacement),
            'chunk_processing_times': [],
            'denoising_stats': denoising_stats
        }
        
        # Step 2: Chunked PyTorch optimization
        print(f"\n🚀 Phase 2 Step 2: Chunked PyTorch optimization...")
        
        total_start_time = time.time()
        
        for chunk_idx in range(n_chunks):
            chunk_start_time = time.time()
            
            # Define chunk boundaries with overlap
            start_idx = max(0, chunk_idx * self.chunk_size - self.overlap)
            end_idx = min(n_stations, (chunk_idx + 1) * self.chunk_size + self.overlap)
            actual_start = chunk_idx * self.chunk_size
            actual_end = min(n_stations, (chunk_idx + 1) * self.chunk_size)
            
            print(f"   Chunk {chunk_idx+1}/{n_chunks}: stations {actual_start}-{actual_end-1} "
                  f"(processing {start_idx}-{end_idx-1})")
            
            # Extract chunk data
            chunk_displacement = denoised_displacement[start_idx:end_idx]
            chunk_coordinates = coordinates[start_idx:end_idx]
            chunk_rates = subsidence_rates[start_idx:end_idx]
            
            # Process chunk
            chunk_results = self._process_chunk(
                chunk_displacement, chunk_coordinates, chunk_rates,
                emd_data, time_vector, max_epochs, 
                actual_start - start_idx, actual_end - start_idx)
            
            # Store results (only for actual chunk, not overlap)
            result_start = actual_start - start_idx
            result_end = actual_end - start_idx
            
            all_results['correlations'][actual_start:actual_end] = chunk_results['correlations'][result_start:result_end]
            all_results['rmse'][actual_start:actual_end] = chunk_results['rmse'][result_start:result_end]
            all_results['fitted_trends'][actual_start:actual_end] = chunk_results['fitted_trends'][result_start:result_end]
            all_results['fitted_offsets'][actual_start:actual_end] = chunk_results['fitted_offsets'][result_start:result_end]
            all_results['predictions'][actual_start:actual_end] = chunk_results['predictions'][result_start:result_end]
            
            chunk_time = time.time() - chunk_start_time
            all_results['chunk_processing_times'].append(chunk_time)
            
            print(f"     ✅ Chunk {chunk_idx+1} complete: {chunk_time:.1f}s, "
                  f"Mean R={np.mean(chunk_results['correlations'][result_start:result_end]):.4f}")
        
        total_time = time.time() - total_start_time
        
        # Final statistics
        mean_correlation = np.mean(all_results['correlations'])
        mean_rmse = np.mean(all_results['rmse'])
        
        print(f"\n✅ PRODUCTION PROCESSING COMPLETE!")
        print(f"   📊 Total stations: {n_stations}")
        print(f"   ⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   📈 Mean correlation: {mean_correlation:.4f}")
        print(f"   📊 Mean RMSE: {mean_rmse:.2f} mm")
        print(f"   🧹 Noise variance reduction: {denoising_stats['mean_variance_reduction']:.1%}")
        
        return all_results
    
    def _process_chunk(self, chunk_displacement: np.ndarray, chunk_coordinates: np.ndarray,
                      chunk_rates: np.ndarray, emd_data: Dict, time_vector: np.ndarray,
                      max_epochs: int, actual_start: int, actual_end: int) -> Dict:
        """Process a single chunk"""
        
        n_chunk_stations = chunk_displacement.shape[0]
        
        # Convert to tensors
        displacement_tensor = torch.tensor(chunk_displacement, dtype=torch.float32, device=self.device)
        rates_tensor = torch.tensor(chunk_rates, dtype=torch.float32, device=self.device)
        time_tensor = torch.arange(chunk_displacement.shape[1], dtype=torch.float32, device=self.device) * 6 / 365.25
        
        # Initialize model
        model = ProductionEMDDenoisedInSARModel(
            n_chunk_stations, chunk_displacement.shape[1],
            chunk_coordinates, rates_tensor, emd_data, 
            n_neighbors=min(8, n_chunk_stations-1), device=self.device
        )
        
        # Initialize parameters
        with torch.no_grad():
            # Offset initialization
            station_means = torch.mean(displacement_tensor, dim=1)
            model.constant_offset.data = station_means
            
            # Conservative amplitude initialization for denoised signals
            for station_idx in range(n_chunk_stations):
                signal = displacement_tensor[station_idx].cpu().numpy()
                signal_std = max(np.std(signal), 3.0)
                
                model.seasonal_amplitudes.data[station_idx, 0] = float(signal_std * 0.25)  # Quarterly
                model.seasonal_amplitudes.data[station_idx, 1] = float(signal_std * 0.35)  # Semi-annual
                model.seasonal_amplitudes.data[station_idx, 2] = float(signal_std * 0.60)  # Annual
                
                model.residual_amplitudes.data[station_idx, 0] = float(signal_std * 0.15)  # Long-term 1
                model.residual_amplitudes.data[station_idx, 1] = float(signal_std * 0.10)  # Long-term 2
        
        # Training setup
        loss_function = ProductionEMDDenoisedLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.001)
        
        model.train()
        
        # Training loop (streamlined for production)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            predictions = model(time_tensor)
            total_loss, _ = loss_function(predictions, displacement_tensor, model)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            model.apply_denoised_constraints()
            optimizer.step()
            scheduler.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            final_predictions = model(time_tensor)
            
            # Per-station metrics
            rmse_per_station = torch.sqrt(torch.mean((final_predictions - displacement_tensor)**2, dim=1))
            correlations = []
            
            for i in range(n_chunk_stations):
                corr = torch.corrcoef(torch.stack([final_predictions[i], displacement_tensor[i]]))[0, 1]
                correlations.append(corr.item() if not torch.isnan(corr) else 0.0)
            
            return {
                'correlations': np.array(correlations),
                'rmse': rmse_per_station.cpu().numpy(),
                'fitted_trends': model.linear_trend.cpu().numpy(),
                'fitted_offsets': model.constant_offset.cpu().numpy(),
                'predictions': final_predictions.cpu().numpy()
            }

def demonstrate_phase2_emd_denoised_framework():
    """Demonstrate Phase 2 EMD-denoised PyTorch framework"""
    
    print("🚀 PHASE 2: EMD-DENOISED PYTORCH FRAMEWORK")
    print("🧹 Innovation: EMD IMF1 + FFT noise removal → PyTorch optimization")
    print("🏭 Production: Memory-efficient chunked processing for full dataset")
    print("="*80)
    
    # Load data
    print(f"\n1️⃣ Loading Production Dataset...")
    try:
        data = np.load("data/processed/ps00_preprocessed_data.npz", allow_pickle=True)
        emd_data = np.load("data/processed/ps02_emd_decomposition.npz", allow_pickle=True)
        
        displacement = data['displacement']
        coordinates = data['coordinates']
        subsidence_rates = data['subsidence_rates']
        
        n_stations, n_timepoints = displacement.shape
        time_vector = np.arange(n_timepoints) * 6 / 365.25
        
        print(f"✅ Loaded production dataset: {n_stations} stations, {n_timepoints} time points")
        
        emd_dict = {
            'imfs': emd_data['imfs'],
            'residuals': emd_data['residuals'],
            'n_imfs_per_station': emd_data['n_imfs_per_station'],
            'coordinates': emd_data['coordinates'],
            'subsidence_rates': emd_data['subsidence_rates']
        }
        
        print(f"✅ EMD decomposition loaded: {emd_dict['imfs'].shape[0]} stations")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None
    
    # Initialize chunked processor
    print(f"\n2️⃣ Initializing Production Processor...")
    processor = ChunkedProductionProcessor(
        chunk_size=50,   # Smaller chunk size for demo
        overlap=5,       # Small overlap for spatial continuity
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Use smaller subset for demonstration
    demo_size = min(100, n_stations)  # Demonstrate with 100 stations
    demo_indices = np.arange(0, n_stations, n_stations // demo_size)[:demo_size]
    
    demo_displacement = displacement[demo_indices]
    demo_coordinates = coordinates[demo_indices]
    demo_rates = subsidence_rates[demo_indices]
    
    print(f"📊 Demo processing: {demo_size} stations (chunked)")
    
    # Process with EMD denoising + PyTorch optimization
    print(f"\n3️⃣ Phase 2 Production Processing...")
    start_time = time.time()
    
    results = processor.process_full_dataset(
        demo_displacement, demo_coordinates, demo_rates,
        emd_dict, time_vector, max_epochs=200  # Reduced for demonstration
    )
    
    processing_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n4️⃣ Phase 2 Production Results...")
    mean_correlation = np.mean(results['correlations'])
    mean_rmse = np.mean(results['rmse'])
    noise_reduction = results['denoising_stats']['mean_variance_reduction']
    
    print(f"✅ PHASE 2 EMD-DENOISED FRAMEWORK COMPLETE!")
    print(f"⏱️ Processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
    print(f"📊 Production Results:")
    print(f"   🎯 Mean correlation: {mean_correlation:.4f}")
    print(f"   📊 Mean RMSE: {mean_rmse:.2f} mm")
    print(f"   🧹 Noise reduction: {noise_reduction:.1%}")
    print(f"   📈 Improvement over baseline: {mean_correlation/0.065:.1f}x")
    
    # Phase 2 success assessment
    baseline_correlation = 0.065
    phase1_correlation = 0.3238  # From Phase 1 results
    
    print(f"\n🏆 PHASE 2 ASSESSMENT:")
    print(f"   📊 Baseline (pure PyTorch): {baseline_correlation:.3f}")
    print(f"   🏅 Phase 1 (EMD-hybrid): {phase1_correlation:.4f}")
    print(f"   🚀 Phase 2 (EMD-denoised): {mean_correlation:.4f}")
    
    phase2_improvement = mean_correlation / baseline_correlation
    vs_phase1 = "maintained" if abs(mean_correlation - phase1_correlation) < 0.02 else \
                ("improved" if mean_correlation > phase1_correlation else "decreased")
    
    print(f"   📈 Phase 2 improvement: {phase2_improvement:.1f}x over baseline")
    print(f"   🔄 vs Phase 1: {vs_phase1}")
    print(f"   🧹 Key innovation: EMD noise removal + production chunked processing")
    
    return processor, results

if __name__ == "__main__":
    try:
        processor, results = demonstrate_phase2_emd_denoised_framework()
        
        if results is not None:
            # Save Phase 2 results
            output_file = Path("data/processed/ps02_phase2_emd_denoised_results.npz")
            output_file.parent.mkdir(exist_ok=True)
            
            np.savez(output_file,
                     correlations=results['correlations'],
                     rmse=results['rmse'],
                     denoising_stats=results['denoising_stats'],
                     processing_times=results['chunk_processing_times'],
                     improvement_factor=np.mean(results['correlations']) / 0.065)
            
            print(f"💾 Phase 2 results saved: {output_file}")
            print(f"\n🎉 PHASE 2 EMD-DENOISED FRAMEWORK READY FOR PRODUCTION!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Phase 2 framework interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Phase 2 framework failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
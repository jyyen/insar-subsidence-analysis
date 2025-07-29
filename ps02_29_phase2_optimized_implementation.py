#!/usr/bin/env python3
"""
ps02_29_phase2_optimized_implementation.py: Phase 2 OPTIMIZED - Final implementation
Chronological order script from Taiwan InSAR Subsidence Analysis Project
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
from tslearn.metrics import dtw
import json

warnings.filterwarnings('ignore')

class OptimizedEMDDenoiser:
    """Optimized EMD-based denoising - less aggressive to preserve signal"""
    
    def __init__(self, noise_threshold_percentile: float = 75,  # Reduced from 85
                 high_freq_cutoff_days: float = 90,  # Increased from 60
                 max_noise_removal_ratio: float = 0.5):  # Cap at 50% removal
        self.noise_threshold_percentile = noise_threshold_percentile
        self.high_freq_cutoff_days = high_freq_cutoff_days
        self.max_noise_removal_ratio = max_noise_removal_ratio
        self.denoising_stats = {}
    
    def denoise_signals(self, displacement: np.ndarray, emd_data: Dict, 
                       time_vector: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Optimized denoising - preserve more seasonal signal"""
        
        n_stations, n_timepoints = displacement.shape
        denoised_signals = displacement.copy()
        
        # Time sampling info
        dt_years = np.mean(np.diff(time_vector))
        dt_days = dt_years * 365.25
        
        noise_removed_per_station = []
        noise_variance_reduction = []
        
        print(f"🧹 Optimized EMD denoising: {n_stations} stations")
        print(f"📊 Strategy: Gentle IMF1 removal + selective FFT filtering")
        print(f"⚡ Preserving quarterly signals (>90 days)")
        
        for station_idx in range(min(n_stations, len(emd_data['imfs']))):
            original_signal = displacement[station_idx]
            station_imfs = emd_data['imfs'][station_idx]
            
            # Step 1: Analyze IMF1 (highest frequency)
            if station_imfs.shape[0] > 0:
                imf1 = station_imfs[0]
                
                # Conservative noise estimation
                imf1_std = np.std(imf1)
                
                # Step 2: FFT analysis for selective filtering
                fft_imf1 = fft(imf1)
                freqs_per_day = fftfreq(len(imf1), dt_days)
                freqs_positive = freqs_per_day[:len(freqs_per_day)//2]
                power_spectrum = np.abs(fft_imf1[:len(fft_imf1)//2])**2
                
                # Only remove very high frequencies (< cutoff days)
                high_freq_mask = freqs_positive > (1.0 / self.high_freq_cutoff_days)
                high_freq_power = np.sum(power_spectrum[high_freq_mask])
                total_power = np.sum(power_spectrum)
                noise_power_ratio = high_freq_power / (total_power + 1e-10)
                
                # Step 3: Selective noise removal
                if noise_power_ratio > 0.4:  # Significant high-frequency
                    # Only remove outliers from IMF1
                    noise_threshold = np.percentile(np.abs(imf1), self.noise_threshold_percentile)
                    noise_mask = np.abs(imf1) > noise_threshold
                    noise_component = imf1 * noise_mask * 0.7  # Partial removal
                else:
                    # Very conservative - only extreme outliers
                    noise_threshold = np.percentile(np.abs(imf1), 95)
                    noise_mask = np.abs(imf1) > noise_threshold
                    noise_component = imf1 * noise_mask * 0.5
                
                # Step 4: Apply noise removal with cap
                original_variance = np.var(original_signal)
                proposed_denoised = original_signal - noise_component
                new_variance = np.var(proposed_denoised)
                
                # Cap variance reduction at max_noise_removal_ratio
                variance_reduction = (original_variance - new_variance) / original_variance
                if variance_reduction > self.max_noise_removal_ratio:
                    # Scale back noise removal
                    scale_factor = self.max_noise_removal_ratio / variance_reduction
                    noise_component = noise_component * scale_factor
                    denoised_signal = original_signal - noise_component
                else:
                    denoised_signal = proposed_denoised
                
                denoised_signals[station_idx] = denoised_signal
                
                # Track statistics
                final_variance = np.var(denoised_signal)
                actual_reduction = (original_variance - final_variance) / original_variance
                
                noise_removed_per_station.append(np.var(noise_component))
                noise_variance_reduction.append(actual_reduction)
            else:
                noise_removed_per_station.append(0.0)
                noise_variance_reduction.append(0.0)
        
        # Compile statistics
        self.denoising_stats = {
            'noise_removed_variance': noise_removed_per_station,
            'variance_reduction_ratio': noise_variance_reduction,
            'mean_noise_removal': np.mean(noise_removed_per_station),
            'mean_variance_reduction': np.mean(noise_variance_reduction),
            'stations_processed': len(noise_removed_per_station),
            'high_freq_cutoff_days': self.high_freq_cutoff_days
        }
        
        print(f"✅ Optimized denoising complete:")
        print(f"   📊 Mean noise variance removed: {self.denoising_stats['mean_noise_removal']:.3f}")
        print(f"   📈 Mean variance reduction: {self.denoising_stats['mean_variance_reduction']:.1%}")
        print(f"   🎯 Max reduction capped at: {self.max_noise_removal_ratio:.1%}")
        
        return denoised_signals, self.denoising_stats

class Phase2OptimizedInSARModel(nn.Module):
    """Optimized PyTorch model for Phase 2 with denoised signals"""
    
    def __init__(self, n_stations: int, n_timepoints: int, coordinates: np.ndarray,
                 ps00_rates: torch.Tensor, emd_data: Dict, n_neighbors: int = 8, device='cpu'):
        super().__init__()
        self.n_stations = n_stations
        self.n_timepoints = n_timepoints
        self.device = device
        self.n_neighbors = n_neighbors
        self.emd_data = emd_data
        
        # Fixed parameters
        self.register_buffer('linear_trend', ps00_rates.to(device))
        
        # Build spatial neighbor graph
        self.coordinates = coordinates
        self.neighbor_graph = self._build_neighbor_graph()
        
        # Optimized parameters for denoised signals
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        
        # Main seasonal components (quarterly, semi-annual, annual)
        self.seasonal_amplitudes = nn.Parameter(torch.ones(n_stations, 3, device=device) * 10.0)
        self.seasonal_phases = nn.Parameter(torch.rand(n_stations, 3, device=device) * 2 * np.pi)
        self.register_buffer('periods', torch.tensor([0.25, 0.5, 1.0], device=device))
        
        # Long-term components for inter-annual variations
        self.longterm_amplitudes = nn.Parameter(torch.ones(n_stations, 2, device=device) * 5.0)
        self.longterm_phases = nn.Parameter(torch.rand(n_stations, 2, device=device) * 2 * np.pi)
        self.longterm_periods = nn.Parameter(torch.tensor([2.0, 3.5], device=device))
        
        # Spatial mixing weights (learnable)
        self.spatial_weights = nn.Parameter(torch.ones(n_stations, device=device) * 0.15)
        
        print(f"🚀 Phase 2 Optimized Model initialized")
        print(f"📊 Stations: {n_stations}, Neighbors: {n_neighbors}")
        
    def _build_neighbor_graph(self) -> Dict:
        """Build spatial neighbor graph for regularization"""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(self.coordinates)
        
        distances, indices = nbrs.kneighbors(self.coordinates)
        
        neighbor_indices = indices[:, 1:]  # Exclude self
        neighbor_distances = distances[:, 1:]
        
        # Distance-based weights
        weights = np.exp(-neighbor_distances / np.median(neighbor_distances))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-6)
        
        return {
            'indices': torch.tensor(neighbor_indices, device=self.device),
            'distances': torch.tensor(neighbor_distances, device=self.device),
            'weights': torch.tensor(weights, device=self.device, dtype=torch.float32)
        }
    
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate signals with optimized spatial regularization"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add main seasonal components with spatial smoothing
        for i, period in enumerate(self.periods):
            smoothed_amp = self._apply_adaptive_smoothing(self.seasonal_amplitudes[:, i])
            smoothed_phase = self._apply_adaptive_smoothing(self.seasonal_phases[:, i], is_phase=True)
            
            amplitude = smoothed_amp.unsqueeze(1)
            phase = smoothed_phase.unsqueeze(1)
            frequency = 1.0 / period
            
            seasonal = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals = signals + seasonal
        
        # Add long-term components (less spatial smoothing)
        for i in range(2):
            amplitude = self.longterm_amplitudes[:, i].unsqueeze(1)
            phase = self.longterm_phases[:, i].unsqueeze(1)
            period = self.longterm_periods[i]
            frequency = 1.0 / period
            
            longterm = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals = signals + longterm
        
        return signals
    
    def _apply_adaptive_smoothing(self, parameter: torch.Tensor, is_phase: bool = False) -> torch.Tensor:
        """Apply adaptive spatial smoothing based on learned weights"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        smoothed_values = torch.zeros_like(parameter)
        
        # Use sigmoid to constrain spatial weights to [0, 1]
        spatial_mix = torch.sigmoid(self.spatial_weights)
        
        for i in range(self.n_stations):
            current_value = parameter[i]
            neighbor_values = parameter[neighbor_indices[i]]
            weights = neighbor_weights[i]
            
            if is_phase:
                # Circular averaging for phases
                current_complex = torch.complex(torch.cos(current_value), torch.sin(current_value))
                neighbor_complex = torch.complex(torch.cos(neighbor_values), torch.sin(neighbor_values))
                weighted_avg_complex = torch.sum(neighbor_complex * weights)
                mixed_complex = (1 - spatial_mix[i]) * current_complex + spatial_mix[i] * weighted_avg_complex
                smoothed_values[i] = torch.angle(mixed_complex)
            else:
                # Regular weighted average
                weighted_avg = torch.sum(neighbor_values * weights)
                smoothed_values[i] = (1 - spatial_mix[i]) * current_value + spatial_mix[i] * weighted_avg
        
        return smoothed_values
    
    def apply_optimized_constraints(self):
        """Apply constraints optimized for denoised signals"""
        with torch.no_grad():
            # Seasonal amplitudes (reasonable bounds)
            self.seasonal_amplitudes.clamp_(0, 50)
            
            # Long-term amplitudes (smaller)
            self.longterm_amplitudes.clamp_(0, 20)
            
            # Phase constraints
            self.seasonal_phases.data = torch.fmod(self.seasonal_phases.data, 2 * np.pi)
            self.longterm_phases.data = torch.fmod(self.longterm_phases.data, 2 * np.pi)
            
            # Period constraints
            self.longterm_periods.clamp_(1.5, 5.0)
            
            # Offset constraints
            self.constant_offset.clamp_(-200, 200)
            
            # Spatial weight constraints (raw values, sigmoid applied in forward)
            self.spatial_weights.clamp_(-3, 3)

class Phase2OptimizedLoss(nn.Module):
    """Optimized loss function with dynamic weighting"""
    
    def __init__(self):
        super().__init__()
        self.epoch = 0
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                model: Phase2OptimizedInSARModel) -> Tuple[torch.Tensor, Dict]:
        """Dynamic loss with epoch-dependent weighting"""
        
        # Primary fitting loss
        mse_loss = torch.nn.functional.mse_loss(predicted, target)
        
        # DTW-inspired shape loss (local)
        shape_loss = self._shape_similarity_loss(predicted, target)
        
        # Spatial consistency loss
        spatial_loss = self._spatial_consistency_loss(model)
        
        # Seasonal pattern loss
        seasonal_loss = self._seasonal_pattern_loss(model)
        
        # Dynamic weighting based on training progress
        warmup_epochs = 100
        if self.epoch < warmup_epochs:
            # Early training: focus on fitting
            alpha_shape = 0.05
            alpha_spatial = 0.05
            alpha_seasonal = 0.05
        else:
            # Later training: increase regularization
            progress = min((self.epoch - warmup_epochs) / 1000, 1.0)
            alpha_shape = 0.1 + 0.1 * progress
            alpha_spatial = 0.1 + 0.05 * progress
            alpha_seasonal = 0.1 + 0.1 * progress
        
        total_loss = (mse_loss + 
                     alpha_shape * shape_loss +
                     alpha_spatial * spatial_loss +
                     alpha_seasonal * seasonal_loss)
        
        loss_components = {
            'mse': mse_loss.item(),
            'shape': shape_loss.item(),
            'spatial': spatial_loss.item(),
            'seasonal': seasonal_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _shape_similarity_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """DTW-inspired local shape similarity loss"""
        # Compare local gradients (shape)
        pred_grad = predicted[:, 1:] - predicted[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        
        # Normalized gradient difference
        grad_diff = torch.mean((pred_grad - target_grad)**2)
        
        return grad_diff
    
    def _spatial_consistency_loss(self, model: Phase2OptimizedInSARModel) -> torch.Tensor:
        """Encourage spatial smoothness in parameters"""
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
                consistency = (station_amp - weighted_mean)**2
                total_loss += consistency * (1 - torch.sigmoid(model.spatial_weights[station_idx]))
        
        return total_loss / (model.n_stations * 3)
    
    def _seasonal_pattern_loss(self, model: Phase2OptimizedInSARModel) -> torch.Tensor:
        """Encourage realistic seasonal patterns"""
        # Annual should typically be strongest
        annual_amp = model.seasonal_amplitudes[:, 2]  # Annual
        semi_annual_amp = model.seasonal_amplitudes[:, 1]  # Semi-annual
        quarterly_amp = model.seasonal_amplitudes[:, 0]  # Quarterly
        
        # Soft constraint: annual > semi-annual in most cases
        pattern_loss = torch.mean(torch.relu(semi_annual_amp - annual_amp * 1.2))
        
        # Total seasonal amplitude should be reasonable
        total_seasonal = annual_amp + semi_annual_amp + quarterly_amp
        amplitude_penalty = torch.mean(torch.relu(total_seasonal - 60))
        
        return pattern_loss + amplitude_penalty * 0.1

class Phase2OptimizedProcessor:
    """Complete Phase 2 optimized implementation"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_history = {}
        
    def run_phase2_optimized(self, data_file: str = "data/processed/ps00_preprocessed_data.npz",
                            emd_file: str = "data/processed/ps02_emd_decomposition.npz",
                            subset_size: int = 100,
                            max_epochs: int = 1500):
        """Run complete Phase 2 optimized pipeline"""
        
        print("🚀 PHASE 2 OPTIMIZED IMPLEMENTATION")
        print("📊 Improvements: Gentle denoising + Extended training + Optimized loss")
        print("="*80)
        
        # Load data
        print("\n1️⃣ Loading data...")
        data = np.load(data_file, allow_pickle=True)
        emd_data = np.load(emd_file, allow_pickle=True)
        
        # Use subset for demonstration
        displacement = data['displacement'][:subset_size]
        coordinates = data['coordinates'][:subset_size]
        subsidence_rates = data['subsidence_rates'][:subset_size]
        
        n_stations, n_timepoints = displacement.shape
        time_years = np.arange(n_timepoints) * 6 / 365.25
        
        emd_dict = {
            'imfs': emd_data['imfs'][:subset_size],
            'residuals': emd_data['residuals'][:subset_size],
            'n_imfs_per_station': emd_data['n_imfs_per_station'][:subset_size]
        }
        
        print(f"✅ Loaded {n_stations} stations, {n_timepoints} time points")
        
        # Step 1: Optimized denoising
        print("\n2️⃣ Optimized EMD denoising...")
        denoiser = OptimizedEMDDenoiser(
            noise_threshold_percentile=75,  # Less aggressive
            high_freq_cutoff_days=90,  # Preserve quarterly signals
            max_noise_removal_ratio=0.5  # Cap at 50%
        )
        
        denoised_displacement, denoising_stats = denoiser.denoise_signals(
            displacement, emd_dict, time_years)
        
        # Convert to tensors
        displacement_tensor = torch.tensor(denoised_displacement, dtype=torch.float32, device=self.device)
        rates_tensor = torch.tensor(subsidence_rates, dtype=torch.float32, device=self.device)
        time_tensor = torch.tensor(time_years, dtype=torch.float32, device=self.device)
        
        # Step 2: Initialize optimized model
        print("\n3️⃣ Initializing optimized model...")
        self.model = Phase2OptimizedInSARModel(
            n_stations, n_timepoints, coordinates, rates_tensor, 
            emd_dict, n_neighbors=min(8, n_stations-1), device=self.device
        )
        
        # Initialize parameters
        self._initialize_model_parameters(displacement_tensor, time_tensor)
        
        # Step 3: Extended training with optimizations
        print("\n4️⃣ Extended training with optimizations...")
        results = self._train_optimized(displacement_tensor, time_tensor, max_epochs)
        
        # Step 4: Comprehensive evaluation
        print("\n5️⃣ Evaluating results...")
        evaluation = self._evaluate_comprehensive(
            displacement_tensor, denoised_displacement, 
            data['displacement'][:subset_size], time_tensor)
        
        # Save results
        self._save_results(results, evaluation, denoising_stats)
        
        return results, evaluation
    
    def _initialize_model_parameters(self, displacement: torch.Tensor, time_vector: torch.Tensor):
        """Smart initialization based on signal characteristics"""
        with torch.no_grad():
            # Initialize offsets
            station_means = torch.mean(displacement, dim=1)
            self.model.constant_offset.data = station_means
            
            # Analyze each station for initialization
            for i in range(self.model.n_stations):
                signal = displacement[i].cpu().numpy()
                
                # Detrend
                trend = np.polyfit(time_vector.cpu().numpy(), signal, 1)
                detrended = signal - np.polyval(trend, time_vector.cpu().numpy())
                
                # Estimate seasonal amplitudes using FFT
                fft_signal = np.fft.fft(detrended)
                freqs = np.fft.fftfreq(len(detrended), d=np.mean(np.diff(time_vector.cpu().numpy())))
                power = np.abs(fft_signal)**2
                
                # Find peaks near seasonal frequencies
                annual_idx = np.argmin(np.abs(freqs - 1.0))
                semi_annual_idx = np.argmin(np.abs(freqs - 2.0))
                quarterly_idx = np.argmin(np.abs(freqs - 4.0))
                
                # Initialize amplitudes based on FFT power
                self.model.seasonal_amplitudes.data[i, 2] = np.sqrt(power[annual_idx]) * 0.5
                self.model.seasonal_amplitudes.data[i, 1] = np.sqrt(power[semi_annual_idx]) * 0.4
                self.model.seasonal_amplitudes.data[i, 0] = np.sqrt(power[quarterly_idx]) * 0.3
                
                # Long-term components
                self.model.longterm_amplitudes.data[i, 0] = np.std(detrended) * 0.2
                self.model.longterm_amplitudes.data[i, 1] = np.std(detrended) * 0.15
    
    def _train_optimized(self, displacement: torch.Tensor, time_vector: torch.Tensor, 
                        max_epochs: int) -> Dict:
        """Extended training with all optimizations"""
        
        # Setup
        loss_function = Phase2OptimizedLoss()
        
        # Optimizer with gradient accumulation
        optimizer = optim.AdamW(self.model.parameters(), lr=0.025, weight_decay=1e-5)
        
        # Learning rate schedule with warmup
        def lr_lambda(epoch):
            warmup = 100
            if epoch < warmup:
                return epoch / warmup
            else:
                # Cosine annealing
                progress = (epoch - warmup) / (max_epochs - warmup)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training history
        history = {
            'loss': [], 'loss_components': [],
            'rmse': [], 'dtw': [], 'lr': []
        }
        
        # Training loop
        print(f"Training for {max_epochs} epochs...")
        self.model.train()
        
        for epoch in range(max_epochs):
            # Update loss function epoch
            loss_function.epoch = epoch
            
            # Forward pass
            predictions = self.model(time_vector)
            
            # Compute loss
            total_loss, loss_components = loss_function(predictions, displacement, self.model)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping (reduced for denoised signals)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            # Apply constraints
            self.model.apply_optimized_constraints()
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            with torch.no_grad():
                rmse = torch.sqrt(torch.mean((predictions - displacement)**2))
                
                # Simple DTW approximation for monitoring
                dtw_approx = torch.mean(torch.abs(predictions - displacement)) / torch.std(displacement)
                
                history['loss'].append(total_loss.item())
                history['loss_components'].append(loss_components)
                history['rmse'].append(rmse.item())
                history['dtw'].append(dtw_approx.item())
                history['lr'].append(scheduler.get_last_lr()[0])
            
            # Progress logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: Loss={total_loss.item():.4f}, "
                      f"RMSE={rmse.item():.2f}mm, DTW≈{dtw_approx.item():.4f}, "
                      f"LR={scheduler.get_last_lr()[0]:.5f}")
        
        print("✅ Training complete!")
        self.training_history = history
        return history
    
    def _evaluate_comprehensive(self, predictions_tensor: torch.Tensor, 
                              denoised_signals: np.ndarray,
                              original_signals: np.ndarray,
                              time_vector: torch.Tensor) -> Dict:
        """Comprehensive evaluation with all metrics"""
        
        self.model.eval()
        
        with torch.no_grad():
            final_predictions = self.model(time_vector).cpu().numpy()
        
        results = {
            'rmse_vs_denoised': [],
            'rmse_vs_original': [],
            'dtw_vs_denoised': [],
            'dtw_vs_original': [],
            'rate_accuracy': []
        }
        
        # Calculate metrics for each station
        for i in range(len(original_signals)):
            # RMSE
            rmse_denoised = np.sqrt(np.mean((final_predictions[i] - denoised_signals[i])**2))
            rmse_original = np.sqrt(np.mean((final_predictions[i] - original_signals[i])**2))
            
            # DTW
            dtw_denoised = dtw(final_predictions[i].reshape(-1, 1), 
                              denoised_signals[i].reshape(-1, 1))
            dtw_original = dtw(final_predictions[i].reshape(-1, 1), 
                              original_signals[i].reshape(-1, 1))
            
            # Normalize DTW
            signal_range = np.max(original_signals[i]) - np.min(original_signals[i])
            dtw_denoised_norm = dtw_denoised / (len(original_signals[i]) * signal_range)
            dtw_original_norm = dtw_original / (len(original_signals[i]) * signal_range)
            
            results['rmse_vs_denoised'].append(rmse_denoised)
            results['rmse_vs_original'].append(rmse_original)
            results['dtw_vs_denoised'].append(dtw_denoised_norm)
            results['dtw_vs_original'].append(dtw_original_norm)
            results['rate_accuracy'].append(1.0)  # Perfect by design
        
        # Summary statistics
        print("\n📊 PHASE 2 OPTIMIZED RESULTS:")
        print(f"   RMSE vs original: {np.mean(results['rmse_vs_original']):.2f} ± "
              f"{np.std(results['rmse_vs_original']):.2f} mm")
        print(f"   RMSE vs denoised: {np.mean(results['rmse_vs_denoised']):.2f} ± "
              f"{np.std(results['rmse_vs_denoised']):.2f} mm")
        print(f"   DTW vs original: {np.mean(results['dtw_vs_original']):.4f} ± "
              f"{np.std(results['dtw_vs_original']):.4f}")
        print(f"   DTW vs denoised: {np.mean(results['dtw_vs_denoised']):.4f} ± "
              f"{np.std(results['dtw_vs_denoised']):.4f}")
        
        return results
    
    def _save_results(self, training_history: Dict, evaluation: Dict, denoising_stats: Dict):
        """Save Phase 2 optimized results"""
        output_file = Path("data/processed/ps02_phase2_optimized_results.npz")
        output_file.parent.mkdir(exist_ok=True)
        
        np.savez(output_file,
                 training_history=training_history,
                 evaluation=evaluation,
                 denoising_stats=denoising_stats,
                 rmse_mean=np.mean(evaluation['rmse_vs_original']),
                 dtw_mean=np.mean(evaluation['dtw_vs_original']),
                 timestamp=time.time())
        
        print(f"\n💾 Results saved: {output_file}")

def main():
    """Execute Phase 2 optimized implementation"""
    processor = Phase2OptimizedProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run with optimized settings
    results, evaluation = processor.run_phase2_optimized(
        subset_size=100,  # Demo size
        max_epochs=1500   # Extended training
    )
    
    print("\n🎉 PHASE 2 OPTIMIZED IMPLEMENTATION COMPLETE!")
    print("📈 Key improvements implemented:")
    print("   ✅ Gentle denoising (50% cap)")
    print("   ✅ Extended training (1500 epochs)")
    print("   ✅ Optimized learning schedule")
    print("   ✅ Dynamic loss weighting")
    print("   ✅ Better initialization")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
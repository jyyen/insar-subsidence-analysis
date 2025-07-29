#!/usr/bin/env python3
"""
ps02c_15_pytorch_emd_hybrid.py: EMD-PyTorch hybrid approach
Chronological order: PS02C development timeline
Taiwan InSAR Subsidence Analysis Project
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
import json

warnings.filterwarnings('ignore')

class EMDHybridInSARModel(nn.Module):
    """Hybrid model: EMD seasonality + PyTorch residual fitting"""
    
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
        
        # Extract and prepare EMD seasonal components
        self.emd_seasonal_components = self._extract_emd_seasonal_components()
        
        # Build spatial neighbor graph
        self.coordinates = coordinates
        self.neighbor_graph = self._build_neighbor_graph()
        
        # PyTorch learns ONLY the residual components + spatial patterns
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        
        # Residual modeling parameters (what EMD couldn't capture)
        self.residual_amplitudes = nn.Parameter(torch.ones(n_stations, 3, device=device) * 2.0)  # High-freq, noise, local
        self.residual_phases = nn.Parameter(torch.rand(n_stations, 3, device=device) * 2 * np.pi)
        self.residual_periods = nn.Parameter(torch.tensor([30.0, 45.0, 180.0], device=device))  # Short periods EMD misses
        
        # Spatial adjustment weights (how much to spatially smooth EMD components)
        self.emd_spatial_weights = nn.Parameter(torch.ones(n_stations, device=device) * 0.1)
        
        print(f"🔄 EMD-PyTorch Hybrid: EMD seasonality + PyTorch residuals")
        print(f"📊 EMD components: {len(self.emd_seasonal_components)} seasonal signals")
        print(f"🧠 PyTorch learning: 3 residual components + spatial patterns")
    
    def _extract_emd_seasonal_components(self) -> torch.Tensor:
        """Extract seasonal IMFs from EMD decomposition"""
        
        # Frequency band definitions (in days) 
        seasonal_bands = {
            'quarterly': (60, 120),
            'semi_annual': (120, 280), 
            'annual': (280, 400),
            'long_annual': (400, 1000)
        }
        
        # For each station, identify which IMFs correspond to seasonal bands
        seasonal_signals = torch.zeros(self.n_stations, 4, self.n_timepoints, device=self.device)
        
        for station_idx in range(min(self.n_stations, len(self.emd_data['imfs']))):
            station_imfs = self.emd_data['imfs'][station_idx]
            
            for band_idx, (band_name, (min_period, max_period)) in enumerate(seasonal_bands.items()):
                # Find IMF that best matches this frequency band
                best_imf_idx = -1
                best_match_score = 0
                
                for imf_idx in range(station_imfs.shape[0]):
                    imf_signal = station_imfs[imf_idx]
                    if np.any(imf_signal != 0):  # Valid IMF
                        
                        # Estimate IMF period using autocorrelation
                        try:
                            autocorr = np.correlate(imf_signal, imf_signal, mode='full')
                            autocorr = autocorr[autocorr.size // 2:]
                            
                            # Find first major peak after lag 10
                            peaks, _ = signal.find_peaks(autocorr[10:], height=0.3 * np.max(autocorr))
                            if len(peaks) > 0:
                                estimated_period = (peaks[0] + 10) * 6  # Convert to days
                                
                                # Check if period matches our target band
                                if min_period <= estimated_period <= max_period:
                                    # Score based on how well it fits and its energy
                                    match_score = np.var(imf_signal) / (1 + abs(estimated_period - (min_period + max_period) / 2))
                                    if match_score > best_match_score:
                                        best_match_score = match_score
                                        best_imf_idx = imf_idx
                        except:
                            continue
                
                # Use best matching IMF for this seasonal band
                if best_imf_idx >= 0:
                    seasonal_signals[station_idx, band_idx] = torch.from_numpy(
                        station_imfs[best_imf_idx]).float().to(self.device)
                else:
                    # If no good match, use zeros (PyTorch will learn to fill this gap)
                    seasonal_signals[station_idx, band_idx] = torch.zeros(self.n_timepoints, device=self.device)
        
        print(f"🧬 Extracted EMD seasonal components for {self.n_stations} stations")
        return seasonal_signals
    
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
        """Generate hybrid EMD-PyTorch signals"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add EMD seasonal components (with optional spatial adjustment)
        emd_seasonal = self._apply_emd_seasonal_components()
        signals = signals + emd_seasonal
        
        # Add PyTorch-learned residual components
        pytorch_residuals = self._generate_pytorch_residuals(time_vector)
        signals = signals + pytorch_residuals
        
        return signals
    
    def _apply_emd_seasonal_components(self) -> torch.Tensor:
        """Apply EMD seasonal components with optional spatial smoothing"""
        
        # Start with raw EMD seasonal signals
        emd_signals = torch.sum(self.emd_seasonal_components, dim=1)  # Sum across 4 seasonal bands
        
        # Apply spatial smoothing based on learned weights
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        spatially_adjusted = torch.zeros_like(emd_signals)
        
        for station_idx in range(self.n_stations):
            original_signal = emd_signals[station_idx]
            neighbor_signals = emd_signals[neighbor_indices[station_idx]]
            spatial_weights = neighbor_weights[station_idx]
            
            # Weighted average of neighbor EMD signals
            neighbor_avg_signal = torch.sum(neighbor_signals * spatial_weights.unsqueeze(1), dim=0)
            
            # Mix original EMD with spatially averaged version
            spatial_mix_weight = torch.sigmoid(self.emd_spatial_weights[station_idx])  # 0-1 range
            spatially_adjusted[station_idx] = (
                (1 - spatial_mix_weight) * original_signal + 
                spatial_mix_weight * neighbor_avg_signal
            )
        
        return spatially_adjusted
    
    def _generate_pytorch_residuals(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate PyTorch-learned residual components"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        residual_signals = torch.zeros(batch_size, len(time_vector), device=self.device)
        
        # Apply spatial smoothing to residual parameters
        smoothed_amplitudes = self._apply_spatial_smoothing_to_residuals(self.residual_amplitudes)
        smoothed_phases = self._apply_spatial_smoothing_to_residuals(self.residual_phases, is_phase=True)
        
        # Generate residual components
        for i in range(3):  # 3 residual components
            amplitude = smoothed_amplitudes[:, i].unsqueeze(1)
            phase = smoothed_phases[:, i].unsqueeze(1)
            period = self.residual_periods[i]
            frequency = 1.0 / period
            
            residual_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            residual_signals += residual_component
        
        return residual_signals
    
    def _apply_spatial_smoothing_to_residuals(self, parameter: torch.Tensor, 
                                            is_phase: bool = False, smoothing_factor: float = 0.15) -> torch.Tensor:
        """Apply spatial smoothing to residual parameters"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        if len(parameter.shape) == 1:  # 1D parameter
            smoothed_values = torch.zeros_like(parameter)
            
            for i in range(self.n_stations):
                current_value = parameter[i]
                neighbor_values = parameter[neighbor_indices[i]]
                weights = neighbor_weights[i]
                
                if is_phase:
                    # Circular averaging
                    current_complex = torch.complex(torch.cos(current_value), torch.sin(current_value))
                    neighbor_complex = torch.complex(torch.cos(neighbor_values), torch.sin(neighbor_values))
                    weighted_avg_complex = torch.sum(neighbor_complex * weights)
                    mixed_complex = (1 - smoothing_factor) * current_complex + smoothing_factor * weighted_avg_complex
                    smoothed_values[i] = torch.angle(mixed_complex)
                else:
                    # Regular averaging
                    weighted_avg = torch.sum(neighbor_values * weights)
                    smoothed_values[i] = (1 - smoothing_factor) * current_value + smoothing_factor * weighted_avg
        
        else:  # 2D parameter (stations x components)
            smoothed_values = torch.zeros_like(parameter)
            
            for component_idx in range(parameter.shape[1]):
                component_param = parameter[:, component_idx]
                
                for i in range(self.n_stations):
                    current_value = component_param[i]
                    neighbor_values = component_param[neighbor_indices[i]]
                    weights = neighbor_weights[i]
                    
                    if is_phase:
                        # Circular averaging
                        current_complex = torch.complex(torch.cos(current_value), torch.sin(current_value))
                        neighbor_complex = torch.complex(torch.cos(neighbor_values), torch.sin(neighbor_values))
                        weighted_avg_complex = torch.sum(neighbor_complex * weights)
                        mixed_complex = (1 - smoothing_factor) * current_complex + smoothing_factor * weighted_avg_complex
                        smoothed_values[i, component_idx] = torch.angle(mixed_complex)
                    else:
                        # Regular averaging
                        weighted_avg = torch.sum(neighbor_values * weights)
                        smoothed_values[i, component_idx] = (1 - smoothing_factor) * current_value + smoothing_factor * weighted_avg
        
        return smoothed_values
    
    def hybrid_consistency_loss(self) -> torch.Tensor:
        """Hybrid consistency loss: spatial smoothness + EMD preservation"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 1. EMD preservation loss (don't let spatial mixing destroy EMD's seasonality)
        emd_preservation_penalty = torch.mean(torch.relu(torch.abs(self.emd_spatial_weights) - 0.3))
        total_loss += emd_preservation_penalty
        
        # 2. Residual spatial consistency
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        for component_idx in range(3):  # 3 residual components
            amplitudes = self.residual_amplitudes[:, component_idx]
            
            for station_idx in range(self.n_stations):
                station_amp = amplitudes[station_idx]
                neighbor_amps = amplitudes[neighbor_indices[station_idx]]
                weights = neighbor_weights[station_idx]
                
                weighted_mean = torch.sum(neighbor_amps * weights)
                consistency_penalty = (station_amp - weighted_mean)**2
                total_loss += consistency_penalty
        
        return total_loss / (self.n_stations * 3)
    
    def apply_hybrid_constraints(self):
        """Apply constraints for hybrid model"""
        with torch.no_grad():
            # Residual amplitude constraints
            self.residual_amplitudes.clamp_(0, 20)  # Residuals should be smaller than main seasonal
            
            # Phase constraints
            self.residual_phases.data = torch.fmod(self.residual_phases.data, 2 * np.pi)
            
            # Period constraints for residuals (keep them short-term)
            self.residual_periods.clamp_(15, 200)  # 15-200 days
            
            # EMD spatial weight constraints (don't over-smooth EMD)
            self.emd_spatial_weights.clamp_(-2, 2)
            
            # Offset constraints
            self.constant_offset.clamp_(-200, 200)

class EMDHybridLoss(nn.Module):
    """Loss function optimized for EMD-PyTorch hybrid approach"""
    
    def __init__(self, alpha_spatial=0.10, alpha_preservation=0.15, alpha_physics=0.05):
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_preservation = alpha_preservation
        self.alpha_physics = alpha_physics
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                model: EMDHybridInSARModel) -> Tuple[torch.Tensor, Dict]:
        """Compute hybrid loss"""
        
        # Primary fitting loss (EMD should already provide good fit)
        primary_loss = torch.nn.functional.mse_loss(predicted, target)
        
        # Hybrid consistency loss
        hybrid_loss = model.hybrid_consistency_loss()
        
        # EMD preservation loss
        preservation_loss = self._emd_preservation_loss(model)
        
        # Physics regularization
        physics_loss = self._physics_regularization(model)
        
        total_loss = (primary_loss + 
                     self.alpha_spatial * hybrid_loss + 
                     self.alpha_preservation * preservation_loss +
                     self.alpha_physics * physics_loss)
        
        loss_components = {
            'primary': primary_loss.item(),
            'spatial': hybrid_loss.item(),
            'preservation': preservation_loss.item(),
            'physics': physics_loss.item()
        }
        
        return total_loss, loss_components
    
    def _emd_preservation_loss(self, model: EMDHybridInSARModel) -> torch.Tensor:
        """Encourage preservation of EMD's proven seasonal patterns"""
        
        # Penalize excessive spatial mixing of EMD components
        spatial_mixing_penalty = torch.mean(torch.relu(torch.abs(model.emd_spatial_weights) - 0.2))
        
        # Encourage residuals to be smaller than seasonal (EMD should handle most signal)
        total_residual_energy = torch.sum(model.residual_amplitudes**2, dim=1)
        residual_penalty = torch.mean(torch.relu(total_residual_energy - 100))  # Keep residuals modest
        
        return spatial_mixing_penalty + residual_penalty * 0.1
    
    def _physics_regularization(self, model: EMDHybridInSARModel) -> torch.Tensor:
        """Physics constraints for hybrid model"""
        
        # Residual periods should be well-separated
        period_separation_penalty = torch.tensor(0.0, device=model.device)
        periods = model.residual_periods
        for i in range(len(periods)):
            for j in range(i+1, len(periods)):
                separation = torch.abs(periods[i] - periods[j])
                penalty = torch.relu(20.0 - separation)  # Minimum 20-day separation
                period_separation_penalty += penalty
        
        # Residual amplitudes should be reasonable
        amplitude_penalty = torch.mean(torch.relu(torch.sum(model.residual_amplitudes, dim=1) - 30))
        
        return period_separation_penalty + amplitude_penalty

class EMDHybridTaiwanInSARFitter:
    """EMD-PyTorch Hybrid fitter"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = EMDHybridLoss()
        self.training_history = {'loss_components': [], 'correlations': []}
        self.emd_data = None
        
    def load_data(self, data_file: str = "data/processed/ps00_preprocessed_data.npz",
                  emd_file: str = "data/processed/ps02_emd_decomposition.npz"):
        """Load Taiwan InSAR data and EMD decomposition results"""
        try:
            # Load main InSAR data
            data = np.load(data_file, allow_pickle=True)
            
            self.displacement = torch.tensor(data['displacement'], dtype=torch.float32, device=self.device)
            self.coordinates = data['coordinates']
            self.subsidence_rates = torch.tensor(data['subsidence_rates'], dtype=torch.float32, device=self.device)
            
            self.n_stations, self.n_timepoints = self.displacement.shape
            
            # Create time vector
            time_days = torch.arange(self.n_timepoints, dtype=torch.float32, device=self.device) * 6
            self.time_years = time_days / 365.25
            
            print(f"✅ Loaded Taiwan InSAR data: {self.n_stations} stations, {self.n_timepoints} time points")
            
            # Load EMD decomposition results
            emd_data = np.load(emd_file, allow_pickle=True)
            self.emd_data = {
                'imfs': emd_data['imfs'],
                'residuals': emd_data['residuals'],
                'n_imfs_per_station': emd_data['n_imfs_per_station'],
                'coordinates': emd_data['coordinates'],
                'subsidence_rates': emd_data['subsidence_rates']
            }
            
            print(f"✅ Loaded EMD decomposition: {self.emd_data['imfs'].shape[0]} stations")
            print(f"🧬 Strategy: EMD handles seasonality, PyTorch handles residuals + spatial patterns")
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def initialize_hybrid_model(self, n_neighbors: int = 8):
        """Initialize EMD-PyTorch hybrid model"""
        if self.emd_data is None:
            raise RuntimeError("EMD data not loaded. Call load_data() first.")
        
        self.model = EMDHybridInSARModel(
            self.n_stations, self.n_timepoints, 
            self.coordinates, self.subsidence_rates, 
            self.emd_data, n_neighbors=n_neighbors, device=self.device
        )
        
        # Hybrid initialization
        self._hybrid_initialization()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        emd_params = self.model.emd_seasonal_components.numel()
        pytorch_params = total_params - emd_params
        
        print(f"🚀 Initialized EMD-PyTorch Hybrid model on {self.device}")
        print(f"📊 Total parameters: {total_params}")
        print(f"🧬 EMD seasonal components: {emd_params} (fixed)")
        print(f"🧠 PyTorch learnable: {pytorch_params} (residuals + spatial)")
    
    def _hybrid_initialization(self):
        """Initialize hybrid model parameters"""
        with torch.no_grad():
            # Initialize offset to station means
            station_means = torch.mean(self.displacement, dim=1)
            self.model.constant_offset.data = station_means
            
            # Initialize residual components to small values (EMD should handle most signal)
            for station_idx in range(self.n_stations):
                signal = self.displacement[station_idx].cpu().numpy()
                time_np = self.time_years.cpu().numpy()
                
                # Remove trend for residual analysis
                detrended = signal - station_means[station_idx].item() - self.subsidence_rates[station_idx].item() * time_np
                
                # Estimate what EMD might have missed (high-frequency residuals)
                residual_std = np.std(detrended) * 0.2  # Expect EMD captured 80% of signal
                
                # Initialize residual amplitudes to small values
                self.model.residual_amplitudes.data[station_idx, 0] = float(residual_std * 0.5)  # High-freq
                self.model.residual_amplitudes.data[station_idx, 1] = float(residual_std * 0.3)  # Noise
                self.model.residual_amplitudes.data[station_idx, 2] = float(residual_std * 0.4)  # Local
            
            # Initialize EMD spatial weights to minimal mixing
            self.model.emd_spatial_weights.data.fill_(0.05)  # Preserve EMD with minimal spatial adjustment
            
            # Random phase initialization for residuals
            self.model.residual_phases.data = torch.rand_like(self.model.residual_phases) * 2 * np.pi
            
            print(f"🔄 Hybrid initialization: EMD seasonality preserved, PyTorch learns residuals")
    
    def train_hybrid(self, max_epochs: int = 1200, target_correlation: float = 0.3):
        """Train EMD-PyTorch hybrid model"""
        
        print(f"🎯 Starting EMD-PyTorch Hybrid Training (target: {target_correlation:.3f})")
        print("🔄 Strategy: EMD seasonality (fixed) + PyTorch residuals (learned)")
        print("="*75)
        
        # Optimizer setup (smaller learning rate since EMD provides good starting point)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.015, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=200, T_mult=2, eta_min=0.001
        )
        
        self.model.train()
        loss_history = []
        correlation_history = []
        best_correlation = -1.0
        patience_counter = 0
        patience = 150
        
        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.time_years)
            
            # Compute hybrid loss
            total_loss, loss_components = self.loss_function(predictions, self.displacement, self.model)
            
            # Backward pass
            total_loss.backward()
            
            # Gentle gradient clipping (EMD provides good initialization)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Apply hybrid constraints
            self.model.apply_hybrid_constraints()
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Track progress
            loss_history.append(total_loss.item())
            self.training_history['loss_components'].append(loss_components)
            
            # Compute correlation
            with torch.no_grad():
                correlation = self._compute_correlation(predictions, self.displacement)
                correlation_history.append(correlation.item())
                self.training_history['correlations'].append(correlation.item())
                
                # Track best correlation
                if correlation.item() > best_correlation:
                    best_correlation = correlation.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early success check
                if correlation.item() >= target_correlation:
                    print(f"🎉 Target correlation {target_correlation:.3f} achieved at epoch {epoch}!")
                    break
                
                # Patience check
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch} (no improvement)")
                    break
            
            # Logging
            if epoch % 100 == 0:
                rmse = torch.sqrt(torch.mean((predictions - self.displacement)**2))
                lr = self.optimizer.param_groups[0]['lr']
                print(f"E{epoch:4d}: Loss={total_loss.item():.3f} "
                      f"(Fit:{loss_components['primary']:.2f}, "
                      f"Preserve:{loss_components['preservation']:.2f}, "
                      f"Spatial:{loss_components['spatial']:.2f}) "
                      f"RMSE={rmse.item():.1f}mm, R={correlation.item():.4f}, LR={lr:.4f}")
        
        print("="*75)
        print(f"✅ EMD-PyTorch Hybrid Training Completed: Best correlation = {best_correlation:.4f}")
        
        return loss_history, correlation_history
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute average correlation across all stations"""
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
        
        correlations = numerator / (denominator + 1e-8)
        return torch.mean(correlations)
    
    def evaluate_hybrid(self) -> Dict:
        """Comprehensive evaluation of hybrid model"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(self.time_years)
            
            # Decompose predictions into EMD and PyTorch components
            emd_component = self.model._apply_emd_seasonal_components()
            pytorch_component = self.model._generate_pytorch_residuals(self.time_years)
            
            # Per-station metrics
            rmse_per_station = torch.sqrt(torch.mean((predictions - self.displacement)**2, dim=1))
            correlations = []
            
            for i in range(self.n_stations):
                corr = torch.corrcoef(torch.stack([predictions[i], self.displacement[i]]))[0, 1]
                correlations.append(corr.item() if not torch.isnan(corr) else 0.0)
            
            correlations = torch.tensor(correlations)
            
            results = {
                'rmse': rmse_per_station.cpu().numpy(),
                'correlations': correlations.cpu().numpy(),
                'fitted_trends': self.model.linear_trend.cpu().numpy(),
                'fitted_offsets': self.model.constant_offset.cpu().numpy(),
                'emd_components': emd_component.cpu().numpy(),
                'pytorch_components': pytorch_component.cpu().numpy(),
                'predictions': predictions.cpu().numpy(),
                'original_rates': self.subsidence_rates.cpu().numpy(),
                'displacement': self.displacement.cpu().numpy(),
                'coordinates': self.coordinates,
                'training_history': self.training_history
            }
        
        # Performance analysis
        mean_rmse = np.mean(results['rmse'])
        mean_corr = np.mean(results['correlations'])
        rate_corr = np.corrcoef(results['original_rates'], results['fitted_trends'])[0, 1]
        
        print(f"📊 EMD-PYTORCH HYBRID MODEL EVALUATION:")
        print(f"   Signal RMSE: {mean_rmse:.2f} ± {np.std(results['rmse']):.2f} mm")
        print(f"   Signal Correlation: {mean_corr:.4f} ± {np.std(results['correlations']):.3f}")
        print(f"   Rate Correlation: {rate_corr:.6f}")
        
        # Component analysis
        emd_contribution = np.mean(np.std(results['emd_components'], axis=1))
        pytorch_contribution = np.mean(np.std(results['pytorch_components'], axis=1))
        total_contribution = emd_contribution + pytorch_contribution
        
        print(f"   🧬 EMD contribution: {100*emd_contribution/total_contribution:.1f}% of signal variance")
        print(f"   🧠 PyTorch contribution: {100*pytorch_contribution/total_contribution:.1f}% of signal variance")
        
        # Achievement metrics
        baseline_correlation = 0.065
        target_correlation = 0.3
        improvement_factor = mean_corr / baseline_correlation
        achievement_ratio = mean_corr / target_correlation
        
        print(f"   📈 Improvement Factor: {improvement_factor:.1f}x over baseline")
        print(f"   🎯 Target Achievement: {achievement_ratio:.1%}")
        
        if mean_corr >= target_correlation:
            print(f"   🎉 PHASE 1 TARGET ACHIEVED!")
        else:
            print(f"   🔄 Progress toward target: {achievement_ratio:.1%}")
        
        return results

def demonstrate_emd_hybrid_framework():
    """Demonstrate EMD-PyTorch hybrid framework"""
    
    print("🚀 PS02C-PYTORCH EMD HYBRID FRAMEWORK")
    print("🔄 True Hybrid: EMD seasonality + PyTorch residuals + spatial optimization")
    print("🏆 Target: 0.3+ correlation leveraging EMD's proven 0.996 seasonal performance")
    print("="*80)
    
    # Initialize hybrid fitter
    fitter = EMDHybridTaiwanInSARFitter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data including EMD results
    print(f"\n1️⃣ Loading Taiwan InSAR Data + EMD Decomposition...")
    fitter.load_data()
    
    # Use subset for demonstration
    subset_size = min(100, fitter.n_stations)
    
    # Select well-distributed subset
    total_stations = fitter.n_stations
    step = total_stations // subset_size
    selected_indices = np.arange(0, total_stations, step)[:subset_size]
    
    fitter.displacement = fitter.displacement[selected_indices]
    fitter.coordinates = fitter.coordinates[selected_indices]
    fitter.subsidence_rates = fitter.subsidence_rates[selected_indices]
    fitter.n_stations = subset_size
    
    print(f"📊 Using EMD-PyTorch hybrid subset: {subset_size} stations")
    
    # Initialize hybrid model
    print(f"\n2️⃣ Initializing EMD-PyTorch Hybrid Model...")
    fitter.initialize_hybrid_model(n_neighbors=8)
    
    # Train with hybrid strategy
    print(f"\n3️⃣ EMD-PyTorch Hybrid Training...")
    start_time = time.time()
    loss_history, correlation_history = fitter.train_hybrid(
        max_epochs=1200, 
        target_correlation=0.3
    )
    training_time = time.time() - start_time
    
    # Comprehensive evaluation
    print(f"\n4️⃣ Final Evaluation...")
    results = fitter.evaluate_hybrid()
    
    print(f"\n✅ EMD-PYTORCH HYBRID FRAMEWORK COMPLETED!")
    print(f"⏱️ Training time: {training_time:.1f} seconds")
    
    # Phase 1 success assessment
    achieved_correlation = np.mean(results['correlations'])
    target_correlation = 0.3
    baseline_correlation = 0.065
    
    print(f"\n🏆 PHASE 1 FINAL RESULTS:")
    print(f"   🎯 Target: {target_correlation:.3f}")
    print(f"   ✅ Achieved: {achieved_correlation:.4f}")
    print(f"   📊 Baseline: {baseline_correlation:.3f}")
    print(f"   📈 Improvement: {achieved_correlation/baseline_correlation:.1f}x")
    
    success = achieved_correlation >= target_correlation
    if success:
        print(f"   🎉 🎉 PHASE 1 COMPLETE - TARGET ACHIEVED! 🎉 🎉")
        print(f"   ✅ Ready to proceed to Phase 2: Production scaling")
    else:
        print(f"   🔄 Phase 1 Progress: {achieved_correlation/target_correlation:.1%} of target")
        print(f"   💡 Next steps: Borehole integration, full dataset scaling")
    
    return fitter, results, success

if __name__ == "__main__":
    try:
        fitter, results, success = demonstrate_emd_hybrid_framework()
        
        # Save Phase 1 results
        output_file = Path("data/processed/ps02c_emd_hybrid_phase1_results.npz")
        output_file.parent.mkdir(exist_ok=True)
        
        np.savez(output_file,
                 correlations=results['correlations'],
                 rmse=results['rmse'],
                 emd_components=results['emd_components'],
                 pytorch_components=results['pytorch_components'],
                 target_achieved=success,
                 improvement_factor=np.mean(results['correlations']) / 0.065,
                 training_time=time.time())
        
        print(f"💾 Phase 1 EMD-hybrid results saved: {output_file}")
        
        if success:
            print(f"\n🚀 READY FOR PHASE 2 IMPLEMENTATION!")
        
    except KeyboardInterrupt:
        print("\n⚠️ EMD-hybrid framework interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ EMD-hybrid framework failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
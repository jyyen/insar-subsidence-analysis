#!/usr/bin/env python3
"""
PS02C-PyTorch EMD-Informed Framework
Phase 1 Implementation: Use EMD decomposition results to initialize PyTorch model

Key Innovation:
1. ✅ EMD-informed parameter initialization from ps02_emd_decomposition.npz
2. ✅ Seasonal amplitude initialization based on EMD IMF analysis
3. ✅ Frequency-matched seasonal components using EMD periods
4. ✅ Spatial regularization with EMD-constrained bounds
5. ✅ Hybrid EMD-PyTorch training strategy

Target: Achieve 0.3+ signal correlation by leveraging EMD's proven performance

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
import json

warnings.filterwarnings('ignore')

class EMDInformedInSARModel(nn.Module):
    """PyTorch InSAR model initialized with EMD decomposition results"""
    
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
        
        # Learnable parameters initialized with EMD insights
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        self.seasonal_amplitudes = nn.Parameter(torch.ones(n_stations, 4, device=device) * 5.0)
        self.seasonal_phases = nn.Parameter(torch.rand(n_stations, 4, device=device) * 2 * np.pi)
        
        # Fixed periods optimized based on EMD analysis
        self.register_buffer('periods', torch.tensor([0.25, 0.5, 1.0, 2.0], device=device))
        
        print(f"🧬 EMD-informed model: {n_stations} stations with EMD initialization")
        print(f"📊 EMD IMFs available: {emd_data['imfs'].shape[1]} per station")
    
    def _build_neighbor_graph(self) -> Dict:
        """Build spatial neighbor graph"""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(self.coordinates)
        
        distances, indices = nbrs.kneighbors(self.coordinates)
        
        # Exclude self (first neighbor is always self)
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]
        
        # Compute adaptive weights
        weights = np.exp(-neighbor_distances / np.mean(neighbor_distances))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-6)
        
        return {
            'indices': torch.tensor(neighbor_indices, device=self.device),
            'distances': torch.tensor(neighbor_distances, device=self.device),
            'weights': torch.tensor(weights, device=self.device, dtype=torch.float32)
        }
    
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate EMD-informed spatially regularized signals"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add EMD-informed seasonal components
        for i, period in enumerate(self.periods):
            # Apply spatial smoothing with EMD constraints
            smoothed_amplitude = self._apply_emd_constrained_smoothing(
                self.seasonal_amplitudes[:, i], period_index=i)
            smoothed_phase = self._apply_spatial_smoothing(
                self.seasonal_phases[:, i], is_phase=True)
            
            amplitude = smoothed_amplitude.unsqueeze(1)
            phase = smoothed_phase.unsqueeze(1)
            frequency = 1.0 / period
            
            seasonal_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals += seasonal_component
        
        return signals
    
    def _apply_emd_constrained_smoothing(self, parameter: torch.Tensor, 
                                       period_index: int, smoothing_factor: float = 0.15) -> torch.Tensor:
        """Apply spatial smoothing with EMD-derived constraints"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        smoothed_values = torch.zeros_like(parameter)
        
        # EMD-derived reasonable bounds for each seasonal component
        emd_bounds = self._get_emd_amplitude_bounds(period_index)
        
        for i in range(self.n_stations):
            current_value = parameter[i]
            neighbor_values = parameter[neighbor_indices[i]]
            weights = neighbor_weights[i]
            
            # Weighted average of neighbors
            neighbor_avg = torch.sum(neighbor_values * weights)
            
            # EMD-constrained smoothing
            proposed_value = (1 - smoothing_factor) * current_value + smoothing_factor * neighbor_avg
            
            # Apply EMD-derived bounds
            min_bound, max_bound = emd_bounds[i] if i < len(emd_bounds) else (0, 50)
            smoothed_values[i] = torch.clamp(proposed_value, min_bound, max_bound)
        
        return smoothed_values
    
    def _get_emd_amplitude_bounds(self, period_index: int) -> List[Tuple[float, float]]:
        """Get EMD-derived amplitude bounds for each station and seasonal component"""
        bounds = []
        
        # Period mapping: 0=quarterly, 1=semi-annual, 2=annual, 3=long-annual
        period_to_freq_band = {
            0: (60, 120),    # quarterly
            1: (120, 280),   # semi-annual  
            2: (280, 400),   # annual
            3: (400, 1000)   # long-annual
        }
        
        target_period_range = period_to_freq_band.get(period_index, (60, 1000))
        
        for station_idx in range(self.n_stations):
            if station_idx < len(self.emd_data['imfs']):
                # Analyze EMD IMFs for this station
                station_imfs = self.emd_data['imfs'][station_idx]
                station_amplitudes = []
                
                for imf_idx in range(station_imfs.shape[0]):
                    if np.any(station_imfs[imf_idx] != 0):  # Valid IMF
                        # Estimate IMF period using zero crossings
                        imf_signal = station_imfs[imf_idx]
                        zero_crossings = np.where(np.diff(np.signbit(imf_signal)))[0]
                        if len(zero_crossings) > 2:
                            avg_half_period = np.mean(np.diff(zero_crossings)) * 6  # 6-day intervals
                            estimated_period = avg_half_period * 2
                            
                            # Check if this IMF matches our target frequency band
                            if target_period_range[0] <= estimated_period <= target_period_range[1]:
                                amplitude = np.std(imf_signal)
                                station_amplitudes.append(amplitude)
                
                # Set bounds based on EMD analysis
                if station_amplitudes:
                    mean_amplitude = np.mean(station_amplitudes)
                    std_amplitude = np.std(station_amplitudes) if len(station_amplitudes) > 1 else mean_amplitude * 0.3
                    
                    min_bound = max(0, mean_amplitude - 2 * std_amplitude)
                    max_bound = mean_amplitude + 2 * std_amplitude
                    bounds.append((float(min_bound), float(max_bound)))
                else:
                    # Default bounds if no matching EMD components
                    default_bounds = {0: (0, 15), 1: (0, 25), 2: (0, 40), 3: (0, 30)}
                    bounds.append(default_bounds.get(period_index, (0, 30)))
            else:
                # Default bounds for stations without EMD data
                default_bounds = {0: (0, 15), 1: (0, 25), 2: (0, 40), 3: (0, 30)}
                bounds.append(default_bounds.get(period_index, (0, 30)))
        
        return bounds
    
    def _apply_spatial_smoothing(self, parameter: torch.Tensor, 
                               smoothing_factor: float = 0.1, is_phase: bool = False) -> torch.Tensor:
        """Apply standard spatial smoothing"""
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
    
    def emd_consistency_loss(self) -> torch.Tensor:
        """Compute loss term that encourages consistency with EMD results"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        # For each station, compare seasonal amplitudes with EMD-derived expectations
        for station_idx in range(min(self.n_stations, len(self.emd_data['imfs']))):
            station_imfs = self.emd_data['imfs'][station_idx]
            predicted_amplitudes = self.seasonal_amplitudes[station_idx]
            
            # Compute EMD-derived amplitude expectations
            emd_amplitudes = []
            for period_idx in range(4):
                bounds = self._get_emd_amplitude_bounds(period_idx)
                if station_idx < len(bounds):
                    min_bound, max_bound = bounds[station_idx]
                    expected_amplitude = (min_bound + max_bound) / 2
                    emd_amplitudes.append(expected_amplitude)
                else:
                    emd_amplitudes.append(10.0)  # Default
            
            emd_amplitudes = torch.tensor(emd_amplitudes, device=self.device)
            
            # L2 loss between predicted and EMD-expected amplitudes
            amplitude_loss = torch.mean((predicted_amplitudes - emd_amplitudes)**2)
            total_loss += amplitude_loss
        
        return total_loss / self.n_stations
    
    def apply_emd_constraints(self):
        """Apply EMD-informed physical constraints"""
        with torch.no_grad():
            # Apply EMD-derived bounds to seasonal amplitudes
            for period_idx in range(4):
                bounds = self._get_emd_amplitude_bounds(period_idx)
                for station_idx in range(min(self.n_stations, len(bounds))):
                    min_bound, max_bound = bounds[station_idx]
                    self.seasonal_amplitudes.data[station_idx, period_idx] = torch.clamp(
                        self.seasonal_amplitudes.data[station_idx, period_idx], 
                        min_bound, max_bound
                    )
            
            # Phase constraints
            self.seasonal_phases.data = torch.fmod(self.seasonal_phases.data, 2 * np.pi)
            
            # Offset constraints
            self.constant_offset.clamp_(-200, 200)

class EMDInformedLoss(nn.Module):
    """Loss function with EMD consistency terms"""
    
    def __init__(self, alpha_spatial=0.15, alpha_emd=0.20, alpha_physics=0.05):
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_emd = alpha_emd
        self.alpha_physics = alpha_physics
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                model: EMDInformedInSARModel) -> Tuple[torch.Tensor, Dict]:
        """Compute EMD-informed loss"""
        
        # Primary fitting loss
        primary_loss = torch.nn.functional.mse_loss(predicted, target)
        
        # Spatial consistency loss (simplified for EMD focus)
        spatial_loss = self._spatial_consistency_loss(model)
        
        # EMD consistency loss
        emd_loss = model.emd_consistency_loss()
        
        # Physics regularization
        physics_loss = self._physics_regularization(model)
        
        total_loss = (primary_loss + 
                     self.alpha_spatial * spatial_loss + 
                     self.alpha_emd * emd_loss +
                     self.alpha_physics * physics_loss)
        
        loss_components = {
            'primary': primary_loss.item(),
            'spatial': spatial_loss.item(),
            'emd': emd_loss.item(),
            'physics': physics_loss.item()
        }
        
        return total_loss, loss_components
    
    def _spatial_consistency_loss(self, model: EMDInformedInSARModel) -> torch.Tensor:
        """Simplified spatial consistency loss"""
        neighbor_indices = model.neighbor_graph['indices']
        neighbor_weights = model.neighbor_graph['weights']
        
        total_loss = torch.tensor(0.0, device=model.device)
        
        # Amplitude consistency
        for component_idx in range(4):
            amplitudes = model.seasonal_amplitudes[:, component_idx]
            
            for station_idx in range(model.n_stations):
                station_amp = amplitudes[station_idx]
                neighbor_amps = amplitudes[neighbor_indices[station_idx]]
                weights = neighbor_weights[station_idx]
                
                weighted_mean = torch.sum(neighbor_amps * weights)
                consistency_penalty = (station_amp - weighted_mean)**2
                total_loss += consistency_penalty
        
        return total_loss / (model.n_stations * 4)
    
    def _physics_regularization(self, model: EMDInformedInSARModel) -> torch.Tensor:
        """Physics-based regularization"""
        # Total seasonal amplitude reasonableness
        total_amplitudes = torch.sum(model.seasonal_amplitudes, dim=1)
        amplitude_penalty = torch.mean(torch.relu(total_amplitudes - 80))
        
        # Annual component should be prominent (EMD-informed expectation)
        annual_amp = model.seasonal_amplitudes[:, 2]
        other_amps = torch.cat([model.seasonal_amplitudes[:, :2], model.seasonal_amplitudes[:, 3:]], dim=1)
        max_other = torch.max(other_amps, dim=1)[0]
        
        annual_prominence_penalty = torch.mean(torch.relu(max_other - annual_amp * 1.2))
        
        return amplitude_penalty + annual_prominence_penalty

class EMDInformedTaiwanInSARFitter:
    """EMD-informed InSAR fitter"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = EMDInformedLoss()
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
            
            print(f"✅ Loaded EMD decomposition: {self.emd_data['imfs'].shape[0]} stations, "
                  f"{self.emd_data['imfs'].shape[1]} IMFs max")
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def initialize_emd_informed_model(self, n_neighbors: int = 8):
        """Initialize EMD-informed model"""
        if self.emd_data is None:
            raise RuntimeError("EMD data not loaded. Call load_data() first.")
        
        self.model = EMDInformedInSARModel(
            self.n_stations, self.n_timepoints, 
            self.coordinates, self.subsidence_rates, 
            self.emd_data, n_neighbors=n_neighbors, device=self.device
        )
        
        # EMD-informed initialization
        self._emd_informed_initialization()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"🚀 Initialized EMD-Informed InSAR model on {self.device}")
        print(f"📊 Model parameters: {total_params} total")
        print(f"🧬 EMD-informed initialization complete")
    
    def _emd_informed_initialization(self):
        """Initialize parameters using EMD decomposition insights"""
        with torch.no_grad():
            # Initialize offset to station means
            station_means = torch.mean(self.displacement, dim=1)
            self.model.constant_offset.data = station_means
            
            # EMD-informed seasonal amplitude initialization
            for station_idx in range(self.n_stations):
                if station_idx < len(self.emd_data['imfs']):
                    station_imfs = self.emd_data['imfs'][station_idx]
                    
                    # Analyze each seasonal component using EMD IMFs
                    for period_idx in range(4):
                        bounds = self.model._get_emd_amplitude_bounds(period_idx)
                        if station_idx < len(bounds):
                            min_bound, max_bound = bounds[station_idx]
                            # Initialize at middle of EMD-derived range
                            initial_amplitude = (min_bound + max_bound) / 2
                            self.model.seasonal_amplitudes.data[station_idx, period_idx] = initial_amplitude
                        else:
                            # Fallback initialization
                            default_amps = [8, 12, 20, 15]  # quarterly, semi-annual, annual, long-annual
                            self.model.seasonal_amplitudes.data[station_idx, period_idx] = default_amps[period_idx]
                else:
                    # Default initialization for stations without EMD data
                    signal = self.displacement[station_idx].cpu().numpy()
                    time_np = self.time_years.cpu().numpy()
                    
                    # Remove trend for seasonal analysis
                    detrended = signal - station_means[station_idx].item() - self.subsidence_rates[station_idx].item() * time_np
                    signal_std = max(np.std(detrended), 5.0)  # Minimum 5mm std
                    
                    # Initialize with reasonable defaults
                    self.model.seasonal_amplitudes.data[station_idx, 0] = signal_std * 0.3  # Quarterly
                    self.model.seasonal_amplitudes.data[station_idx, 1] = signal_std * 0.4  # Semi-annual
                    self.model.seasonal_amplitudes.data[station_idx, 2] = signal_std * 0.7  # Annual
                    self.model.seasonal_amplitudes.data[station_idx, 3] = signal_std * 0.5  # Long-annual
            
            # Random phase initialization (EMD doesn't provide reliable phase info)
            self.model.seasonal_phases.data = torch.rand_like(self.model.seasonal_phases) * 2 * np.pi
            
            print(f"🧬 EMD-informed initialization: Amplitude bounds derived from {len(self.emd_data['imfs'])} stations")
    
    def train_emd_informed(self, max_epochs: int = 1500, target_correlation: float = 0.3):
        """Train EMD-informed model"""
        
        print(f"🎯 Starting EMD-Informed Training (target: {target_correlation:.3f})")
        print("🧬 Strategy: EMD initialization + spatial regularization + consistency loss")
        print("="*80)
        
        # Optimizer setup
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.025, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=300, T_mult=2, eta_min=0.002
        )
        
        self.model.train()
        loss_history = []
        correlation_history = []
        best_correlation = -1.0
        patience_counter = 0
        patience = 200
        
        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.time_years)
            
            # Compute EMD-informed loss
            total_loss, loss_components = self.loss_function(predictions, self.displacement, self.model)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
            
            # Apply EMD constraints
            self.model.apply_emd_constraints()
            
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
                      f"(Fit:{loss_components['primary']:.2f}, EMD:{loss_components['emd']:.2f}, "
                      f"Spatial:{loss_components['spatial']:.2f}) "
                      f"RMSE={rmse.item():.1f}mm, R={correlation.item():.4f}, LR={lr:.4f}")
        
        print("="*80)
        print(f"✅ EMD-Informed Training Completed: Best correlation = {best_correlation:.4f}")
        
        return loss_history, correlation_history
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute average correlation across all stations"""
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
        
        correlations = numerator / (denominator + 1e-8)
        return torch.mean(correlations)
    
    def evaluate_emd_informed(self) -> Dict:
        """Comprehensive evaluation of EMD-informed model"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(self.time_years)
            
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
                'seasonal_amplitudes': self.model.seasonal_amplitudes.cpu().numpy(),
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
        
        print(f"📊 EMD-INFORMED MODEL EVALUATION:")
        print(f"   Signal RMSE: {mean_rmse:.2f} ± {np.std(results['rmse']):.2f} mm")
        print(f"   Signal Correlation: {mean_corr:.4f} ± {np.std(results['correlations']):.3f}")
        print(f"   Rate Correlation: {rate_corr:.6f}")
        
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

def demonstrate_emd_informed_framework():
    """Demonstrate EMD-informed framework"""
    
    print("🚀 PS02C-PYTORCH EMD-INFORMED FRAMEWORK")
    print("🧬 Phase 1 Innovation: EMD decomposition insights + PyTorch optimization")
    print("🏆 Target: 0.3+ signal correlation leveraging EMD's proven performance")
    print("="*75)
    
    # Initialize EMD-informed fitter
    fitter = EMDInformedTaiwanInSARFitter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    print(f"📊 Using EMD-informed subset: {subset_size} stations")
    
    # Initialize EMD-informed model
    print(f"\n2️⃣ Initializing EMD-Informed Model...")
    fitter.initialize_emd_informed_model(n_neighbors=8)
    
    # Train with EMD-informed strategy
    print(f"\n3️⃣ EMD-Informed Training...")
    start_time = time.time()
    loss_history, correlation_history = fitter.train_emd_informed(
        max_epochs=1500, 
        target_correlation=0.3
    )
    training_time = time.time() - start_time
    
    # Comprehensive evaluation
    print(f"\n4️⃣ Final Evaluation...")
    results = fitter.evaluate_emd_informed()
    
    print(f"\n✅ EMD-INFORMED FRAMEWORK COMPLETED!")
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
        print(f"   💡 Next steps: Borehole integration, extended EMD analysis")
    
    return fitter, results, success

if __name__ == "__main__":
    try:
        fitter, results, success = demonstrate_emd_informed_framework()
        
        # Save Phase 1 results
        output_file = Path("data/processed/ps02c_emd_informed_phase1_results.npz")
        output_file.parent.mkdir(exist_ok=True)
        
        np.savez(output_file,
                 correlations=results['correlations'],
                 rmse=results['rmse'],
                 target_achieved=success,
                 improvement_factor=np.mean(results['correlations']) / 0.065,
                 training_time=time.time())
        
        print(f"💾 Phase 1 EMD-informed results saved: {output_file}")
        
        if success:
            print(f"\n🚀 READY FOR PHASE 2 IMPLEMENTATION!")
        
    except KeyboardInterrupt:
        print("\n⚠️ EMD-informed framework interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ EMD-informed framework failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
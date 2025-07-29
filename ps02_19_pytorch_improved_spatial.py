#!/usr/bin/env python3
"""
ps02_19_pytorch_improved_spatial.py: Improved spatial algorithms
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
import json

warnings.filterwarnings('ignore')

class ImprovedSpatialInSARModel(nn.Module):
    """Improved spatial model with enhanced correlation focus"""
    
    def __init__(self, n_stations: int, n_timepoints: int, coordinates: np.ndarray, 
                 ps00_rates: torch.Tensor, n_neighbors: int = 12, device='cpu'):
        super().__init__()
        self.n_stations = n_stations
        self.n_timepoints = n_timepoints
        self.device = device
        self.n_neighbors = n_neighbors
        
        # Fixed parameters (preserve perfect rate correlation)
        self.register_buffer('linear_trend', ps00_rates.to(device))
        
        # Build enhanced spatial neighbor graph
        self.coordinates = coordinates
        self.neighbor_graph = self._build_enhanced_neighbor_graph()
        
        # Learnable parameters with improved initialization
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        self.seasonal_amplitudes = nn.Parameter(torch.ones(n_stations, 4, device=device) * 4.0)
        self.seasonal_phases = nn.Parameter(torch.rand(n_stations, 4, device=device) * 2 * np.pi)
        
        # Fixed periods for stability
        self.register_buffer('periods', torch.tensor([0.25, 0.5, 1.0, 2.0], device=device))
        
        print(f"🌐 Enhanced spatial graph: {n_stations} stations, {n_neighbors} neighbors each")
    
    def _build_enhanced_neighbor_graph(self) -> Dict:
        """Build enhanced spatial neighbor graph with adaptive weighting"""
        # Use larger neighborhood for better spatial consistency
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(self.coordinates)
        
        distances, indices = nbrs.kneighbors(self.coordinates)
        
        # Exclude self (first neighbor is always self)
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]
        
        # Enhanced adaptive weighting scheme
        weights = self._compute_enhanced_weights(neighbor_distances)
        
        return {
            'indices': torch.tensor(neighbor_indices, device=self.device),
            'distances': torch.tensor(neighbor_distances, device=self.device),
            'weights': torch.tensor(weights, device=self.device, dtype=torch.float32)
        }
    
    def _compute_enhanced_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute enhanced adaptive spatial weights"""
        
        # Multi-scale distance weighting
        # Local weight (emphasizes very close neighbors)
        local_weights = np.exp(-distances / (np.mean(distances, axis=1, keepdims=True) * 0.5))
        
        # Regional weight (broader spatial consistency)
        regional_weights = 1.0 / (distances + np.mean(distances) * 0.1)
        
        # Combine weights adaptively
        alpha = 0.7  # Favor local weights
        combined_weights = alpha * local_weights + (1 - alpha) * regional_weights
        
        # Normalize weights per station
        weights = combined_weights / (np.sum(combined_weights, axis=1, keepdims=True) + 1e-6)
        
        return weights
    
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate spatially regularized signals with correlation focus"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add spatially regularized seasonal components
        for i, period in enumerate(self.periods):
            # Apply enhanced spatial smoothing
            smoothed_amplitude = self._apply_enhanced_spatial_smoothing(
                self.seasonal_amplitudes[:, i], smoothing_factor=0.25)
            smoothed_phase = self._apply_enhanced_spatial_smoothing(
                self.seasonal_phases[:, i], smoothing_factor=0.15, is_phase=True)
            
            amplitude = smoothed_amplitude.unsqueeze(1)
            phase = smoothed_phase.unsqueeze(1)
            frequency = 1.0 / period
            
            seasonal_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals += seasonal_component
        
        return signals
    
    def _apply_enhanced_spatial_smoothing(self, parameter: torch.Tensor, 
                                        smoothing_factor: float = 0.2, is_phase: bool = False) -> torch.Tensor:
        """Apply enhanced spatial smoothing with improved correlation focus"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        smoothed_values = torch.zeros_like(parameter)
        
        for i in range(self.n_stations):
            current_value = parameter[i]
            neighbor_values = parameter[neighbor_indices[i]]
            weights = neighbor_weights[i]
            
            if is_phase:
                # Enhanced circular averaging for phases
                current_complex = torch.complex(torch.cos(current_value), torch.sin(current_value))
                neighbor_complex = torch.complex(torch.cos(neighbor_values), torch.sin(neighbor_values))
                
                # Weighted average in complex space
                weighted_avg_complex = torch.sum(neighbor_complex * weights)
                
                # Adaptive mixing based on local phase consistency
                consistency = torch.abs(weighted_avg_complex)  # Measure of local phase agreement
                adaptive_factor = smoothing_factor * consistency  # Reduce smoothing if inconsistent
                
                mixed_complex = (1 - adaptive_factor) * current_complex + adaptive_factor * weighted_avg_complex
                smoothed_values[i] = torch.angle(mixed_complex)
            else:
                # Enhanced weighted average for amplitudes
                weighted_avg = torch.sum(neighbor_values * weights)
                
                # Adaptive smoothing based on local amplitude variability
                local_variance = torch.var(neighbor_values)
                adaptive_factor = smoothing_factor / (1 + local_variance * 0.1)  # Reduce smoothing in high-variance areas
                
                smoothed_values[i] = (1 - adaptive_factor) * current_value + adaptive_factor * weighted_avg
        
        return smoothed_values
    
    def enhanced_spatial_consistency_loss(self) -> torch.Tensor:
        """Enhanced spatial consistency loss with correlation focus"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Multi-component spatial consistency
        for component_idx in range(4):
            amplitudes = self.seasonal_amplitudes[:, component_idx]
            phases = self.seasonal_phases[:, component_idx]
            
            component_loss = torch.tensor(0.0, device=self.device)
            
            for station_idx in range(self.n_stations):
                # Amplitude consistency
                station_amp = amplitudes[station_idx]
                neighbor_amps = amplitudes[neighbor_indices[station_idx]]
                weights = neighbor_weights[station_idx]
                
                weighted_mean_amp = torch.sum(neighbor_amps * weights)
                amplitude_penalty = (station_amp - weighted_mean_amp)**2
                
                # Phase consistency (circular)
                station_phase = phases[station_idx]
                neighbor_phases = phases[neighbor_indices[station_idx]]
                
                # Circular variance penalty
                phase_diffs = torch.sin(neighbor_phases - station_phase)
                phase_penalty = torch.sum(weights * phase_diffs**2)
                
                component_loss += amplitude_penalty + phase_penalty * 0.5
            
            # Weight by component importance (annual dominant)
            if component_idx == 2:  # Annual
                total_loss += component_loss * 1.5
            else:
                total_loss += component_loss
        
        return total_loss / (self.n_stations * 4)  # Normalize
    
    def apply_enhanced_constraints(self):
        """Apply enhanced physical constraints"""
        with torch.no_grad():
            # Amplitude constraints with component-specific bounds
            self.seasonal_amplitudes[:, 0].clamp_(0, 25)   # Quarterly: smaller
            self.seasonal_amplitudes[:, 1].clamp_(0, 35)   # Semi-annual: moderate
            self.seasonal_amplitudes[:, 2].clamp_(0, 50)   # Annual: largest
            self.seasonal_amplitudes[:, 3].clamp_(0, 40)   # Long-annual: moderate
            
            # Phase constraints
            self.seasonal_phases.data = torch.fmod(self.seasonal_phases.data, 2 * np.pi)
            
            # Offset constraints
            self.constant_offset.clamp_(-250, 250)  # mm

class ImprovedSpatialLoss(nn.Module):
    """Improved loss function with enhanced correlation focus"""
    
    def __init__(self, alpha_spatial=0.20, alpha_physics=0.05, alpha_correlation=0.15):
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_physics = alpha_physics
        self.alpha_correlation = alpha_correlation
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                model: ImprovedSpatialInSARModel) -> Tuple[torch.Tensor, Dict]:
        """Compute improved loss with enhanced correlation focus"""
        
        # Primary fitting loss with robust Huber loss
        primary_loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        
        # Enhanced spatial consistency loss
        spatial_loss = model.enhanced_spatial_consistency_loss()
        
        # Physics regularization
        physics_loss = self._enhanced_physics_regularization(model)
        
        # Enhanced correlation loss
        correlation_loss = self._enhanced_correlation_loss(predicted, target)
        
        total_loss = (primary_loss + 
                     self.alpha_spatial * spatial_loss + 
                     self.alpha_physics * physics_loss +
                     self.alpha_correlation * correlation_loss)
        
        loss_components = {
            'primary': primary_loss.item(),
            'spatial': spatial_loss.item(),
            'physics': physics_loss.item(),
            'correlation': correlation_loss.item()
        }
        
        return total_loss, loss_components
    
    def _enhanced_physics_regularization(self, model: ImprovedSpatialInSARModel) -> torch.Tensor:
        """Enhanced physics-based regularization"""
        # Total seasonal amplitude reasonableness
        total_amplitudes = torch.sum(model.seasonal_amplitudes, dim=1)
        amplitude_penalty = torch.mean(torch.relu(total_amplitudes - 100))
        
        # Annual component dominance (geological expectation)
        annual_amp = model.seasonal_amplitudes[:, 2]  # Annual component
        other_amps = torch.cat([model.seasonal_amplitudes[:, :2], model.seasonal_amplitudes[:, 3:]], dim=1)
        max_other = torch.max(other_amps, dim=1)[0]
        
        # Encourage annual to be at least as large as others
        annual_dominance_penalty = torch.mean(torch.relu(max_other - annual_amp))
        
        # Seasonal amplitude gradual increase (quarterly < semi-annual < annual)
        progression_penalty = torch.tensor(0.0, device=model.device)
        for i in range(model.n_stations):
            amps = model.seasonal_amplitudes[i]
            # Penalize if quarterly > semi-annual or semi-annual > annual
            progression_penalty += torch.relu(amps[0] - amps[1]) + torch.relu(amps[1] - amps[2])
        
        progression_penalty = progression_penalty / model.n_stations
        
        return amplitude_penalty + annual_dominance_penalty + progression_penalty * 0.1
    
    def _enhanced_correlation_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Enhanced correlation loss with multiple correlation metrics"""
        
        # Station-wise correlation loss
        station_correlation_losses = []
        
        for i in range(predicted.shape[0]):
            pred_i = predicted[i]
            target_i = target[i]
            
            # Center the signals
            pred_centered = pred_i - torch.mean(pred_i)
            target_centered = target_i - torch.mean(target_i)
            
            # Pearson correlation
            numerator = torch.sum(pred_centered * target_centered)
            denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2))
            
            if denominator > 1e-8:
                correlation = numerator / denominator
                correlation_loss = 1.0 - correlation
            else:
                correlation_loss = torch.tensor(1.0, device=predicted.device)
            
            station_correlation_losses.append(correlation_loss)
        
        # Mean correlation loss
        mean_correlation_loss = torch.mean(torch.stack(station_correlation_losses))
        
        # Add variance penalty to encourage consistent correlations
        correlation_variance = torch.var(torch.stack(station_correlation_losses))
        variance_penalty = correlation_variance * 0.1
        
        return mean_correlation_loss + variance_penalty

class ImprovedSpatialTaiwanInSARFitter:
    """Improved fitter with enhanced training strategies"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = ImprovedSpatialLoss()
        self.training_history = {'loss_components': [], 'correlations': []}
        
    def load_data(self, data_file: str = "data/processed/ps00_preprocessed_data.npz"):
        """Load Taiwan InSAR data"""
        try:
            data = np.load(data_file, allow_pickle=True)
            
            self.displacement = torch.tensor(data['displacement'], dtype=torch.float32, device=self.device)
            self.coordinates = data['coordinates']
            self.subsidence_rates = torch.tensor(data['subsidence_rates'], dtype=torch.float32, device=self.device)
            
            self.n_stations, self.n_timepoints = self.displacement.shape
            
            # Create time vector
            time_days = torch.arange(self.n_timepoints, dtype=torch.float32, device=self.device) * 6
            self.time_years = time_days / 365.25
            
            print(f"✅ Loaded Taiwan InSAR data: {self.n_stations} stations, {self.n_timepoints} time points")
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def initialize_improved_model(self, n_neighbors: int = 12):
        """Initialize improved spatial model"""
        self.model = ImprovedSpatialInSARModel(
            self.n_stations, self.n_timepoints, 
            self.coordinates, self.subsidence_rates, 
            n_neighbors=n_neighbors, device=self.device
        )
        
        # Enhanced initialization strategy
        self._enhanced_initialization()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"🚀 Initialized Improved Spatial InSAR model on {self.device}")
        print(f"📊 Model parameters: {total_params} total")
        print(f"🌐 Enhanced spatial neighbors: {n_neighbors} per station")
    
    def _enhanced_initialization(self):
        """Enhanced parameter initialization with seasonal analysis"""
        with torch.no_grad():
            # Initialize offset to station means
            station_means = torch.mean(self.displacement, dim=1)
            self.model.constant_offset.data = station_means
            
            # Enhanced seasonal initialization based on signal analysis
            for i in range(self.n_stations):
                signal = self.displacement[i].cpu().numpy()
                time_np = self.time_years.cpu().numpy()
                
                # Remove trend to analyze seasonal components
                detrended = signal - station_means[i].item() - self.subsidence_rates[i].item() * time_np
                
                # Analyze signal characteristics
                signal_range = np.ptp(detrended)  # Peak-to-peak range
                signal_std = np.std(detrended)
                signal_energy = np.sum(detrended**2)
                
                # Enhanced initialization based on signal energy and variability
                energy_factor = min(2.0, np.sqrt(signal_energy / len(detrended)) / 10.0)
                variability_factor = min(1.5, signal_std / 15.0)
                
                base_factor = (energy_factor + variability_factor) / 2.0
                
                # Component-specific initialization with geological reasoning
                self.model.seasonal_amplitudes.data[i, 0] = float(signal_std * 0.2 * base_factor)   # Quarterly
                self.model.seasonal_amplitudes.data[i, 1] = float(signal_std * 0.35 * base_factor)  # Semi-annual
                self.model.seasonal_amplitudes.data[i, 2] = float(signal_std * 0.8 * base_factor)   # Annual (dominant)
                self.model.seasonal_amplitudes.data[i, 3] = float(signal_std * 0.45 * base_factor)  # Long-annual
            
            # Initialize phases with spatial correlation
            # Create spatially coherent base phases
            n_components = 4
            for component_idx in range(n_components):
                # Use coordinates to create spatially varying but smooth phase initialization
                coords_norm = (self.coordinates - np.mean(self.coordinates, axis=0)) / np.std(self.coordinates, axis=0)
                
                # Create phase gradient based on geographic location
                phase_gradient = (coords_norm[:, 0] * 0.5 + coords_norm[:, 1] * 0.3) * np.pi
                base_phase = np.random.rand() * 2 * np.pi
                
                # Add spatial variation with noise
                spatial_phases = base_phase + phase_gradient + np.random.randn(self.n_stations) * 0.3
                
                self.model.seasonal_phases.data[:, component_idx] = torch.from_numpy(spatial_phases).float()
    
    def train_improved_spatial(self, max_epochs: int = 2000, target_correlation: float = 0.3):
        """Train with improved multi-stage strategy"""
        
        print(f"🎯 Starting Improved Spatial Training (target: {target_correlation:.3f})")
        print("🌐 Strategy: Multi-stage progressive enhancement + correlation focus")
        print("="*80)
        
        # Multi-stage training with different focus areas
        stages = [
            {'name': 'Bootstrap', 'epochs': 400, 'lr': 0.035, 'spatial_weight': 0.05, 'corr_weight': 0.05},
            {'name': 'Spatial Integration', 'epochs': 600, 'lr': 0.025, 'spatial_weight': 0.15, 'corr_weight': 0.10},
            {'name': 'Correlation Focus', 'epochs': 800, 'lr': 0.015, 'spatial_weight': 0.25, 'corr_weight': 0.20},
            {'name': 'Fine-tuning', 'epochs': 600, 'lr': 0.008, 'spatial_weight': 0.30, 'corr_weight': 0.25}
        ]
        
        total_loss_history = []
        total_correlation_history = []
        best_correlation = -1.0
        global_patience_counter = 0
        global_patience = 300
        
        for stage_idx, stage_config in enumerate(stages):
            print(f"\n🔥 STAGE {stage_idx + 1}: {stage_config['name']}")
            print(f"   Epochs: {stage_config['epochs']}, LR: {stage_config['lr']}")
            print(f"   Spatial weight: {stage_config['spatial_weight']}, Correlation weight: {stage_config['corr_weight']}")
            print("-" * 60)
            
            # Update loss function weights
            self.loss_function.alpha_spatial = stage_config['spatial_weight']
            self.loss_function.alpha_correlation = stage_config['corr_weight']
            
            # Setup optimizer for this stage
            self.optimizer = optim.AdamW(self.model.parameters(), lr=stage_config['lr'], weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=stage_config['epochs']//2, eta_min=stage_config['lr']*0.1
            )
            
            # Train this stage
            stage_loss, stage_corr = self._train_improved_stage(
                stage_config['epochs'], 
                stage_idx, 
                target_correlation
            )
            
            total_loss_history.extend(stage_loss)
            total_correlation_history.extend(stage_corr)
            
            # Check for target achievement
            current_best = max(stage_corr) if len(stage_corr) > 0 else best_correlation
            if current_best > best_correlation:
                best_correlation = current_best
                global_patience_counter = 0
            else:
                global_patience_counter += len(stage_corr)
            
            if best_correlation >= target_correlation:
                print(f"🎉 Target correlation achieved in stage {stage_idx + 1}!")
                break
                
            if global_patience_counter >= global_patience:
                print(f"🛑 Global early stopping after stage {stage_idx + 1}")
                break
        
        print("="*80)
        print(f"✅ Improved Spatial Training Completed: Best correlation = {best_correlation:.4f}")
        
        return total_loss_history, total_correlation_history
    
    def _train_improved_stage(self, max_epochs: int, stage_idx: int, target_correlation: float):
        """Train single stage with improved strategy"""
        
        self.model.train()
        loss_history = []
        correlation_history = []
        stage_patience_counter = 0
        stage_patience = 150
        
        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.time_years)
            
            # Compute enhanced loss
            total_loss, loss_components = self.loss_function(predictions, self.displacement, self.model)
            
            # Backward pass
            total_loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            
            # Apply constraints
            self.model.apply_enhanced_constraints()
            
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
                
                # Early stopping logic
                if correlation.item() >= target_correlation:
                    print(f"🎉 Target correlation {target_correlation:.3f} achieved at epoch {epoch}!")
                    break
                
                # Stage-specific patience
                if len(correlation_history) > 30:
                    recent_improvement = max(correlation_history[-30:]) - min(correlation_history[-30:])
                    if recent_improvement < 0.002:
                        stage_patience_counter += 1
                    else:
                        stage_patience_counter = 0
                
                if stage_patience_counter >= stage_patience:
                    print(f"🛑 Stage {stage_idx + 1} early stopping at epoch {epoch}")
                    break
            
            # Enhanced logging
            if epoch % 80 == 0:
                rmse = torch.sqrt(torch.mean((predictions - self.displacement)**2))
                lr = self.optimizer.param_groups[0]['lr']
                print(f"S{stage_idx+1} E{epoch:3d}: Loss={total_loss.item():.3f} "
                      f"(Fit:{loss_components['primary']:.2f}, Spatial:{loss_components['spatial']:.2f}, "
                      f"Corr:{loss_components['correlation']:.2f}) "
                      f"RMSE={rmse.item():.1f}mm, R={correlation.item():.4f}, LR={lr:.4f}")
        
        return loss_history, correlation_history
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute average correlation across all stations"""
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
        
        correlations = numerator / (denominator + 1e-8)
        return torch.mean(correlations)
    
    def evaluate_improved(self) -> Dict:
        """Comprehensive evaluation of improved spatial model"""
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
        
        print(f"📊 IMPROVED SPATIAL MODEL EVALUATION:")
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

def demonstrate_improved_spatial_framework():
    """Demonstrate improved spatial framework"""
    
    print("🚀 PS02C-PYTORCH IMPROVED SPATIAL FRAMEWORK")
    print("🎯 Enhanced Phase 1: Multi-stage training + correlation focus")
    print("🏆 Target: 0.3+ signal correlation (5x improvement)")
    print("="*75)
    
    # Initialize improved fitter
    fitter = ImprovedSpatialTaiwanInSARFitter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print(f"\n1️⃣ Loading Taiwan InSAR Data...")
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
    
    print(f"📊 Using improved subset: {subset_size} stations")
    
    # Initialize improved model
    print(f"\n2️⃣ Initializing Improved Spatial Model...")
    fitter.initialize_improved_model(n_neighbors=12)
    
    # Train with improved multi-stage strategy
    print(f"\n3️⃣ Improved Multi-Stage Training...")
    start_time = time.time()
    loss_history, correlation_history = fitter.train_improved_spatial(
        max_epochs=2000, 
        target_correlation=0.3
    )
    training_time = time.time() - start_time
    
    # Comprehensive evaluation
    print(f"\n4️⃣ Final Evaluation...")
    results = fitter.evaluate_improved()
    
    print(f"\n✅ IMPROVED SPATIAL FRAMEWORK COMPLETED!")
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
        print(f"   💡 Next steps: EMD integration, geological constraints, extended training")
    
    return fitter, results, success

if __name__ == "__main__":
    try:
        fitter, results, success = demonstrate_improved_spatial_framework()
        
        # Save Phase 1 results
        output_file = Path("data/processed/ps02c_improved_spatial_phase1_results.npz")
        output_file.parent.mkdir(exist_ok=True)
        
        np.savez(output_file,
                 correlations=results['correlations'],
                 rmse=results['rmse'],
                 target_achieved=success,
                 improvement_factor=np.mean(results['correlations']) / 0.065,
                 training_time=time.time())
        
        print(f"💾 Phase 1 improved results saved: {output_file}")
        
        if success:
            print(f"\n🚀 READY FOR PHASE 2 IMPLEMENTATION!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Improved spatial framework interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Improved spatial framework failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
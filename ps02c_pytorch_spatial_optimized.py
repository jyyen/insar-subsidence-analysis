#!/usr/bin/env python3
"""
PS02C-PyTorch Spatial Optimized Framework
Simplified but effective spatial regularization for Phase 1 target achievement

Focus: Core spatial improvements without complexity that causes memory issues
Target: Signal correlation 0.065 → 0.3+ (5x improvement)

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
import json

warnings.filterwarnings('ignore')

class OptimizedSpatialInSARModel(nn.Module):
    """Optimized spatial model focused on core improvements"""
    
    def __init__(self, n_stations: int, n_timepoints: int, coordinates: np.ndarray, 
                 ps00_rates: torch.Tensor, n_neighbors: int = 8, device='cpu'):
        super().__init__()
        self.n_stations = n_stations
        self.n_timepoints = n_timepoints
        self.device = device
        self.n_neighbors = n_neighbors
        
        # Fixed parameters (preserve perfect rate correlation)
        self.register_buffer('linear_trend', ps00_rates.to(device))
        
        # Build efficient spatial neighbor graph
        self.coordinates = coordinates
        self.neighbor_graph = self._build_neighbor_graph()
        
        # Learnable parameters
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        self.seasonal_amplitudes = nn.Parameter(torch.ones(n_stations, 4, device=device) * 3.0)
        self.seasonal_phases = nn.Parameter(torch.rand(n_stations, 4, device=device) * 2 * np.pi)
        
        # Fixed periods for stability
        self.register_buffer('periods', torch.tensor([0.25, 0.5, 1.0, 2.0], device=device))
        
        print(f"🌐 Optimized spatial graph: {n_stations} stations, {n_neighbors} neighbors each")
    
    def _build_neighbor_graph(self) -> Dict:
        """Build efficient spatial neighbor graph"""
        # Use sklearn for efficient neighbor finding
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(self.coordinates)
        
        distances, indices = nbrs.kneighbors(self.coordinates)
        
        # Exclude self (first neighbor is always self)
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]
        
        # Compute inverse distance weights
        weights = 1.0 / (neighbor_distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)  # Normalize
        
        return {
            'indices': torch.tensor(neighbor_indices, device=self.device),
            'distances': torch.tensor(neighbor_distances, device=self.device),
            'weights': torch.tensor(weights, device=self.device, dtype=torch.float32)
        }
    
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate signals with spatial smoothing"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add spatially smoothed seasonal components
        for i, period in enumerate(self.periods):
            # Apply spatial smoothing to seasonal parameters
            smoothed_amplitude = self._apply_spatial_smoothing(self.seasonal_amplitudes[:, i])
            smoothed_phase = self._apply_spatial_smoothing(self.seasonal_phases[:, i], is_phase=True)
            
            amplitude = smoothed_amplitude.unsqueeze(1)
            phase = smoothed_phase.unsqueeze(1)
            frequency = 1.0 / period
            
            seasonal_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals += seasonal_component
        
        return signals
    
    def _apply_spatial_smoothing(self, parameter: torch.Tensor, is_phase: bool = False, 
                                smoothing_factor: float = 0.2) -> torch.Tensor:
        """Apply spatial smoothing using neighbor graph"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        smoothed_values = torch.zeros_like(parameter)
        
        for i in range(self.n_stations):
            current_value = parameter[i]
            neighbor_values = parameter[neighbor_indices[i]]
            weights = neighbor_weights[i]
            
            if is_phase:
                # Handle circular nature of phases using complex representation
                current_complex = torch.complex(torch.cos(current_value), torch.sin(current_value))
                neighbor_complex = torch.complex(torch.cos(neighbor_values), torch.sin(neighbor_values))
                
                # Weighted average in complex space
                weighted_avg_complex = torch.sum(neighbor_complex * weights)
                smoothed_phase = torch.angle(weighted_avg_complex)
                
                # Mix with original
                original_complex = current_complex
                mixed_complex = (1 - smoothing_factor) * original_complex + smoothing_factor * weighted_avg_complex
                smoothed_values[i] = torch.angle(mixed_complex)
            else:
                # Regular weighted average for amplitudes
                weighted_avg = torch.sum(neighbor_values * weights)
                smoothed_values[i] = (1 - smoothing_factor) * current_value + smoothing_factor * weighted_avg
        
        return smoothed_values
    
    def spatial_consistency_loss(self) -> torch.Tensor:
        """Compute spatial consistency loss"""
        neighbor_indices = self.neighbor_graph['indices']
        neighbor_weights = self.neighbor_graph['weights']
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Consistency loss for seasonal amplitudes
        for component_idx in range(4):
            amplitudes = self.seasonal_amplitudes[:, component_idx]
            
            for station_idx in range(self.n_stations):
                station_amp = amplitudes[station_idx]
                neighbor_amps = amplitudes[neighbor_indices[station_idx]]
                weights = neighbor_weights[station_idx]
                
                # Weighted variance penalty
                weighted_mean = torch.sum(neighbor_amps * weights)
                consistency_penalty = (station_amp - weighted_mean)**2
                total_loss += consistency_penalty
        
        # Consistency loss for seasonal phases (accounting for circularity)
        for component_idx in range(4):
            phases = self.seasonal_phases[:, component_idx]
            
            for station_idx in range(self.n_stations):
                station_phase = phases[station_idx]
                neighbor_phases = phases[neighbor_indices[station_idx]]
                weights = neighbor_weights[station_idx]
                
                # Circular difference penalty
                phase_diffs = torch.sin(neighbor_phases - station_phase)
                weighted_phase_penalty = torch.sum(weights * phase_diffs**2)
                total_loss += weighted_phase_penalty
        
        return total_loss / (self.n_stations * 8)  # Normalize
    
    def apply_constraints(self):
        """Apply physical constraints"""
        with torch.no_grad():
            # Amplitude constraints
            self.seasonal_amplitudes.clamp_(0, 60)  # mm
            
            # Phase constraints
            self.seasonal_phases.data = torch.fmod(self.seasonal_phases.data, 2 * np.pi)
            
            # Offset constraints
            self.constant_offset.clamp_(-200, 200)  # mm

class OptimizedSpatialLoss(nn.Module):
    """Optimized loss function with effective spatial regularization"""
    
    def __init__(self, alpha_spatial=0.15, alpha_physics=0.05, alpha_correlation=0.1):
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_physics = alpha_physics
        self.alpha_correlation = alpha_correlation
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                model: OptimizedSpatialInSARModel) -> Tuple[torch.Tensor, Dict]:
        """Compute optimized loss with correlation focus"""
        
        # Primary fitting loss - use MSE for gradient properties
        primary_loss = torch.nn.functional.mse_loss(predicted, target)
        
        # Spatial consistency loss
        spatial_loss = model.spatial_consistency_loss()
        
        # Physics regularization
        physics_loss = self._physics_regularization(model)
        
        # Correlation enhancement loss
        correlation_loss = self._correlation_enhancement_loss(predicted, target)
        
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
    
    def _physics_regularization(self, model: OptimizedSpatialInSARModel) -> torch.Tensor:
        """Physics-based regularization"""
        # Total seasonal amplitude reasonableness
        total_amplitudes = torch.sum(model.seasonal_amplitudes, dim=1)
        amplitude_penalty = torch.mean(torch.relu(total_amplitudes - 80))
        
        # Annual component should be reasonably large (geological expectation)
        annual_amp = model.seasonal_amplitudes[:, 2]  # Annual component
        annual_penalty = torch.mean(torch.relu(5.0 - annual_amp))  # Encourage at least 5mm annual
        
        return amplitude_penalty + annual_penalty * 0.1
    
    def _correlation_enhancement_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss term specifically designed to enhance correlation"""
        
        # Compute per-station correlations
        correlation_losses = []
        
        for i in range(predicted.shape[0]):
            pred_i = predicted[i]
            target_i = target[i]
            
            # Center the signals
            pred_centered = pred_i - torch.mean(pred_i)
            target_centered = target_i - torch.mean(target_i)
            
            # Compute correlation
            numerator = torch.sum(pred_centered * target_centered)
            denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2))
            
            if denominator > 1e-8:
                correlation = numerator / denominator
                # Convert correlation to loss (higher correlation = lower loss)
                correlation_loss = 1.0 - correlation
            else:
                correlation_loss = torch.tensor(1.0, device=predicted.device)
            
            correlation_losses.append(correlation_loss)
        
        # Average correlation loss across stations
        return torch.mean(torch.stack(correlation_losses))

class OptimizedSpatialTaiwanInSARFitter:
    """Optimized fitter focused on correlation improvement"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = OptimizedSpatialLoss()
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
    
    def initialize_optimized_model(self, n_neighbors: int = 8):
        """Initialize optimized spatial model"""
        self.model = OptimizedSpatialInSARModel(
            self.n_stations, self.n_timepoints, 
            self.coordinates, self.subsidence_rates, 
            n_neighbors=n_neighbors, device=self.device
        )
        
        # Smart initialization
        self._smart_initialization()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"🚀 Initialized Optimized Spatial InSAR model on {self.device}")
        print(f"📊 Model parameters: {total_params} total")
        print(f"🌐 Spatial neighbors: {n_neighbors} per station")
    
    def _smart_initialization(self):
        """Smart parameter initialization for better convergence"""
        with torch.no_grad():
            # Initialize offset to station means
            station_means = torch.mean(self.displacement, dim=1)
            self.model.constant_offset.data = station_means
            
            # Data-driven seasonal initialization
            for i in range(self.n_stations):
                signal = self.displacement[i].cpu().numpy()
                time_np = self.time_years.cpu().numpy()
                
                # Remove trend to analyze seasonal components
                detrended = signal - station_means[i].item() - self.subsidence_rates[i].item() * time_np
                
                # Estimate seasonal amplitudes from data variability
                signal_range = np.ptp(detrended)  # Peak-to-peak range
                signal_std = np.std(detrended)
                
                # Initialize with data-informed values
                self.model.seasonal_amplitudes.data[i, 0] = float(signal_std * 0.3)   # Quarterly
                self.model.seasonal_amplitudes.data[i, 1] = float(signal_std * 0.4)   # Semi-annual
                self.model.seasonal_amplitudes.data[i, 2] = float(signal_std * 0.7)   # Annual (dominant)
                self.model.seasonal_amplitudes.data[i, 3] = float(signal_std * 0.4)   # Long-annual
            
            # Initialize phases with some spatial correlation
            base_phases = torch.rand(4) * 2 * np.pi
            for i in range(self.n_stations):
                # Add station-specific variation to base phases
                phase_noise = torch.randn(4) * 0.5
                self.model.seasonal_phases.data[i] = base_phases + phase_noise
    
    def train_optimized_spatial(self, max_epochs: int = 1200, target_correlation: float = 0.3):
        """Train with optimized spatial strategy"""
        
        print(f"🎯 Starting Optimized Spatial Training (target: {target_correlation:.3f})")
        print("🌐 Strategy: Progressive spatial regularization + correlation focus")
        print("="*75)
        
        # Adaptive training strategy
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.025, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=200, T_mult=2, eta_min=0.001
        )
        
        self.model.train()
        loss_history = []
        correlation_history = []
        best_correlation = -1.0
        patience_counter = 0
        patience = 200
        
        # Progressive spatial weight schedule
        spatial_schedule = self._create_spatial_schedule(max_epochs)
        
        for epoch in range(max_epochs):
            # Update spatial regularization weight
            current_spatial_weight = spatial_schedule[min(epoch, len(spatial_schedule) - 1)]
            self.loss_function.alpha_spatial = current_spatial_weight
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.time_years)
            
            # Compute loss
            total_loss, loss_components = self.loss_function(predictions, self.displacement, self.model)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Apply constraints
            self.model.apply_constraints()
            
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
                
                # Track best
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
                      f"(Fit:{loss_components['primary']:.2f}, Spatial:{loss_components['spatial']:.2f}, "
                      f"Corr:{loss_components['correlation']:.2f}) "
                      f"RMSE={rmse.item():.1f}mm, R={correlation.item():.4f}, LR={lr:.4f}")
        
        print("="*75)
        print(f"✅ Optimized Spatial Training Completed: Best correlation = {best_correlation:.4f}")
        
        return loss_history, correlation_history
    
    def _create_spatial_schedule(self, max_epochs: int) -> List[float]:
        """Create progressive spatial regularization schedule"""
        schedule = []
        
        # Warm-up phase (first 20%)
        warmup_epochs = max_epochs // 5
        for epoch in range(warmup_epochs):
            weight = 0.05 + (0.1 - 0.05) * (epoch / warmup_epochs)
            schedule.append(weight)
        
        # Main training phase (next 60%)
        main_epochs = int(max_epochs * 0.6)
        for epoch in range(main_epochs):
            weight = 0.1 + (0.2 - 0.1) * (epoch / main_epochs)
            schedule.append(weight)
        
        # Fine-tuning phase (last 20%)
        finetune_epochs = max_epochs - warmup_epochs - main_epochs
        for epoch in range(finetune_epochs):
            weight = 0.2 + (0.25 - 0.2) * (epoch / finetune_epochs)
            schedule.append(weight)
        
        return schedule
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute average correlation across all stations"""
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
        
        correlations = numerator / (denominator + 1e-8)
        return torch.mean(correlations)
    
    def evaluate_optimized(self) -> Dict:
        """Comprehensive evaluation of optimized model"""
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
        
        print(f"📊 OPTIMIZED SPATIAL MODEL EVALUATION:")
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

def demonstrate_optimized_spatial_framework():
    """Demonstrate optimized spatial framework"""
    
    print("🚀 PS02C-PYTORCH OPTIMIZED SPATIAL FRAMEWORK")
    print("🎯 Streamlined Phase 1: Effective spatial regularization")
    print("🏆 Target: 0.3+ signal correlation (5x improvement)")
    print("="*70)
    
    # Initialize optimized fitter
    fitter = OptimizedSpatialTaiwanInSARFitter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    print(f"📊 Using optimized subset: {subset_size} stations")
    
    # Initialize optimized model
    print(f"\n2️⃣ Initializing Optimized Spatial Model...")
    fitter.initialize_optimized_model(n_neighbors=8)
    
    # Train with optimized spatial strategy
    print(f"\n3️⃣ Optimized Spatial Training...")
    start_time = time.time()
    loss_history, correlation_history = fitter.train_optimized_spatial(
        max_epochs=1200, 
        target_correlation=0.3
    )
    training_time = time.time() - start_time
    
    # Comprehensive evaluation
    print(f"\n4️⃣ Final Evaluation...")
    results = fitter.evaluate_optimized()
    
    print(f"\n✅ OPTIMIZED SPATIAL FRAMEWORK COMPLETED!")
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
        print(f"   💡 Next steps: EMD integration, extended training, larger datasets")
    
    return fitter, results, success

if __name__ == "__main__":
    try:
        fitter, results, success = demonstrate_optimized_spatial_framework()
        
        # Save Phase 1 results
        output_file = Path("data/processed/ps02c_optimized_spatial_phase1_results.npz")
        output_file.parent.mkdir(exist_ok=True)
        
        np.savez(output_file,
                 correlations=results['correlations'],
                 rmse=results['rmse'],
                 target_achieved=success,
                 improvement_factor=np.mean(results['correlations']) / 0.065,
                 training_time=time.time())
        
        print(f"💾 Phase 1 optimized results saved: {output_file}")
        
        if success:
            print(f"\n🚀 READY FOR PHASE 2 IMPLEMENTATION!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimized spatial framework interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Optimized spatial framework failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
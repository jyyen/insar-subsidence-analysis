#!/usr/bin/env python3
"""
PS02C-PyTorch: InSAR Signal Simulation and Fitting Framework
Fresh start implementation for Taiwan subsidence analysis

Author: Taiwan InSAR Subsidence Analysis Project  
Created: 2025-07-27
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

class InSARSignalModel(nn.Module):
    """PyTorch model for InSAR time series simulation and fitting"""
    
    def __init__(self, n_stations: int, n_timepoints: int, device='cpu'):
        super().__init__()
        self.n_stations = n_stations
        self.n_timepoints = n_timepoints
        self.device = device
        
        # Initialize parameters with physics-based priors
        self.linear_trend = nn.Parameter(torch.zeros(n_stations, device=device))
        
        # Seasonal component amplitudes [quarterly, semi-annual, annual, long-annual]
        self.seasonal_amplitudes = nn.Parameter(torch.ones(n_stations, 4, device=device) * 2.0)
        
        # Seasonal component phases [quarterly, semi-annual, annual, long-annual]  
        self.seasonal_phases = nn.Parameter(torch.rand(n_stations, 4, device=device) * 2 * np.pi)
        
        # Long-annual period parameter (1.5-3 years, learnable)
        self.long_period = nn.Parameter(torch.ones(n_stations, device=device) * 2.0)  # Default 2 years
        
        # Noise standard deviation (learnable per station)
        self.noise_std = nn.Parameter(torch.ones(n_stations, device=device) * 1.0)
        
        # Fixed seasonal periods (in years)
        self.register_buffer('fixed_periods', torch.tensor([0.25, 0.5, 1.0], device=device))
        
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """
        Generate simulated InSAR signals
        
        Args:
            time_vector: Time points in years [n_timepoints]
            
        Returns:
            signals: Simulated signals [n_stations, n_timepoints]
        """
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)  # [n_stations, n_timepoints]
        
        # Linear trend component
        signals = self.linear_trend.unsqueeze(1) * time_expanded
        
        # Fixed seasonal components (quarterly, semi-annual, annual)
        for i, period in enumerate(self.fixed_periods):
            amplitude = self.seasonal_amplitudes[:, i].unsqueeze(1)
            phase = self.seasonal_phases[:, i].unsqueeze(1)
            frequency = 1.0 / period  # Convert period to frequency
            
            seasonal_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals += seasonal_component
        
        # Long-annual component (learnable period)
        long_amplitude = self.seasonal_amplitudes[:, 3].unsqueeze(1)
        long_phase = self.seasonal_phases[:, 3].unsqueeze(1)
        long_frequency = 1.0 / self.long_period.unsqueeze(1)
        
        long_component = long_amplitude * torch.sin(2 * np.pi * long_frequency * time_expanded + long_phase)
        signals += long_component
        
        return signals
    
    def add_noise(self, signals: torch.Tensor, time_vector: torch.Tensor) -> torch.Tensor:
        """Add realistic high-frequency noise to signals"""
        if self.training:
            # Generate correlated noise (atmospheric effects have temporal correlation)
            noise_base = torch.randn_like(signals)
            
            # Apply simple temporal smoothing for realistic atmospheric correlation
            kernel_size = 3
            padding = kernel_size // 2
            noise_smoothed = torch.nn.functional.conv1d(
                noise_base.unsqueeze(1), 
                torch.ones(1, 1, kernel_size, device=self.device) / kernel_size,
                padding=padding
            ).squeeze(1)
            
            # Scale by station-specific noise standard deviation
            noise_scaled = noise_smoothed * self.noise_std.unsqueeze(1)
            return signals + noise_scaled
        else:
            return signals
    
    def apply_constraints(self):
        """Apply physical constraints to parameters"""
        with torch.no_grad():
            # Linear trend constraints (Taiwan typical range)
            self.linear_trend.clamp_(-100, 50)  # mm/year
            
            # Amplitude constraints (reasonable seasonal amplitudes)
            self.seasonal_amplitudes.clamp_(0, 50)  # mm
            
            # Phase constraints
            self.seasonal_phases.data = torch.fmod(self.seasonal_phases.data, 2 * np.pi)
            
            # Long period constraints (1.5-4 years)
            self.long_period.clamp_(1.5, 4.0)
            
            # Noise constraints
            self.noise_std.clamp_(0.1, 10.0)


class InSARLoss(nn.Module):
    """Custom loss function for InSAR signal fitting"""
    
    def __init__(self, alpha_physics=0.1, alpha_smoothness=0.05):
        super().__init__()
        self.alpha_physics = alpha_physics
        self.alpha_smoothness = alpha_smoothness
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, model: InSARSignalModel) -> torch.Tensor:
        """
        Compute comprehensive loss for InSAR fitting
        
        Args:
            predicted: Model predictions [n_stations, n_timepoints]
            target: Observed InSAR data [n_stations, n_timepoints]
            model: InSAR model for accessing parameters
            
        Returns:
            total_loss: Combined loss value
        """
        # Primary fitting loss (Huber loss for robustness to outliers)
        primary_loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        
        # Physics regularization
        physics_loss = self._physics_regularization(model)
        
        # Smoothness regularization
        smoothness_loss = self._smoothness_regularization(model)
        
        total_loss = primary_loss + self.alpha_physics * physics_loss + self.alpha_smoothness * smoothness_loss
        
        return total_loss
    
    def _physics_regularization(self, model: InSARSignalModel) -> torch.Tensor:
        """Penalize unphysical parameter combinations"""
        # Penalize extremely large seasonal amplitudes relative to trend
        trend_magnitude = torch.abs(model.linear_trend)
        seasonal_magnitude = torch.sum(model.seasonal_amplitudes, dim=1)
        
        # Seasonal shouldn't dominate linear trend unreasonably
        physics_penalty = torch.mean(torch.relu(seasonal_magnitude - 3 * trend_magnitude - 10))
        
        return physics_penalty
    
    def _smoothness_regularization(self, model: InSARSignalModel) -> torch.Tensor:
        """Encourage spatial smoothness in parameters (neighboring stations similar)"""
        # Simple smoothness penalty (could be improved with actual spatial neighbors)
        trend_smoothness = torch.mean((model.linear_trend[1:] - model.linear_trend[:-1])**2)
        amplitude_smoothness = torch.mean((model.seasonal_amplitudes[1:] - model.seasonal_amplitudes[:-1])**2)
        
        return trend_smoothness + amplitude_smoothness


class TaiwanInSARFitter:
    """Main interface for Taiwan InSAR signal fitting"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = InSARLoss()
        
    def load_data(self, data_file: str = "data/processed/ps00_preprocessed_data.npz"):
        """Load Taiwan InSAR data"""
        try:
            data = np.load(data_file, allow_pickle=True)
            
            self.displacement = torch.tensor(data['displacement'], dtype=torch.float32, device=self.device)
            self.coordinates = torch.tensor(data['coordinates'], dtype=torch.float32, device=self.device)
            self.subsidence_rates = torch.tensor(data['subsidence_rates'], dtype=torch.float32, device=self.device)
            
            self.n_stations, self.n_timepoints = self.displacement.shape
            
            # Create time vector (assuming 6-day intervals)
            time_days = torch.arange(self.n_timepoints, dtype=torch.float32, device=self.device) * 6
            self.time_years = time_days / 365.25
            
            print(f"✅ Loaded Taiwan InSAR data: {self.n_stations} stations, {self.n_timepoints} time points")
            print(f"📊 Time span: {self.time_years[-1]:.2f} years")
            print(f"📊 Subsidence rates range: {self.subsidence_rates.min():.1f} to {self.subsidence_rates.max():.1f} mm/year")
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def initialize_model(self, physics_based_init: bool = True):
        """Initialize PyTorch model with optional physics-based initialization"""
        self.model = InSARSignalModel(self.n_stations, self.n_timepoints, self.device)
        
        if physics_based_init:
            self._physics_based_initialization()
        
        print(f"🚀 Initialized InSAR model on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters())} total")
    
    def _physics_based_initialization(self):
        """Initialize parameters based on physical understanding"""
        with torch.no_grad():
            # Initialize linear trends from PS00 rates
            self.model.linear_trend.data = self.subsidence_rates.clone()
            
            # Initialize seasonal amplitudes based on data variability
            data_std = torch.std(self.displacement, dim=1)
            self.model.seasonal_amplitudes.data = data_std.unsqueeze(1).expand(-1, 4) * 0.3
            
            # Random phase initialization
            self.model.seasonal_phases.data = torch.rand_like(self.model.seasonal_phases) * 2 * np.pi
            
            # Initialize long period around 2 years
            self.model.long_period.data.fill_(2.0)
            
            # Initialize noise based on data
            self.model.noise_std.data = data_std * 0.1
    
    def setup_optimization(self, learning_rate: float = 0.01, weight_decay: float = 1e-4):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=100, factor=0.7
        )
        
        print(f"🔧 Setup optimization: Adam lr={learning_rate}, weight_decay={weight_decay}")
    
    def train(self, max_epochs: int = 2000, patience: int = 200, min_delta: float = 1e-6):
        """Train the model to fit InSAR signals"""
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialized first")
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        loss_history = []
        
        print(f"🎯 Starting training: max_epochs={max_epochs}, patience={patience}")
        print("=" * 60)
        
        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.time_years)
            
            # Add noise during training
            predictions_with_noise = self.model.add_noise(predictions, self.time_years)
            
            # Compute loss
            loss = self.loss_function(predictions_with_noise, self.displacement, self.model)
            
            # Backward pass
            loss.backward()
            
            # Apply constraints
            self.model.apply_constraints()
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step(loss)
            
            # Track progress
            loss_history.append(loss.item())
            
            # Check for improvement
            if loss.item() < best_loss - min_delta:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 20 == 0:
                with torch.no_grad():
                    predictions_clean = self.model(self.time_years)
                    rmse = torch.sqrt(torch.mean((predictions_clean - self.displacement)**2))
                    correlation = self._compute_correlation(predictions_clean, self.displacement)
                
                print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, RMSE={rmse.item():.3f}mm, Corr={correlation.item():.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch} (patience exceeded)")
                break
        
        print("=" * 60)
        print(f"✅ Training completed: Final loss={loss_history[-1]:.6f}")
        
        return loss_history
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute average correlation across all stations"""
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
        
        correlations = numerator / (denominator + 1e-8)
        return torch.mean(correlations)
    
    def evaluate(self) -> Dict:
        """Evaluate model performance"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(self.time_years)
            
            # Compute metrics
            rmse = torch.sqrt(torch.mean((predictions - self.displacement)**2, dim=1))
            correlations = []
            
            for i in range(self.n_stations):
                corr = torch.corrcoef(torch.stack([predictions[i], self.displacement[i]]))[0, 1]
                correlations.append(corr.item() if not torch.isnan(corr) else 0.0)
            
            correlations = torch.tensor(correlations)
            
            # Extract fitted parameters
            fitted_trends = self.model.linear_trend.cpu().numpy()
            seasonal_amplitudes = self.model.seasonal_amplitudes.cpu().numpy()
            
            results = {
                'rmse': rmse.cpu().numpy(),
                'correlations': correlations.cpu().numpy(), 
                'fitted_trends': fitted_trends,
                'seasonal_amplitudes': seasonal_amplitudes,
                'predictions': predictions.cpu().numpy(),
                'original_rates': self.subsidence_rates.cpu().numpy()
            }
        
        print(f"📊 Evaluation Results:")
        print(f"   Mean RMSE: {np.mean(results['rmse']):.2f} ± {np.std(results['rmse']):.2f} mm")
        print(f"   Mean Correlation: {np.mean(results['correlations']):.3f} ± {np.std(results['correlations']):.3f}")
        print(f"   Trend fitting RMSE: {np.sqrt(np.mean((fitted_trends - results['original_rates'])**2)):.2f} mm/year")
        
        return results


if __name__ == "__main__":
    # Example usage
    print("🚀 PS02C-PyTorch Framework Initialization")
    
    fitter = TaiwanInSARFitter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    fitter.load_data()
    
    # Initialize model
    fitter.initialize_model(physics_based_init=True)
    
    # Setup optimization
    fitter.setup_optimization(learning_rate=0.01)
    
    # Train model
    loss_history = fitter.train(max_epochs=1000)
    
    # Evaluate results
    results = fitter.evaluate()
    
    print("✅ Framework demonstration complete!")
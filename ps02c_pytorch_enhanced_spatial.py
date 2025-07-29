#!/usr/bin/env python3
"""
PS02C-PyTorch Enhanced Spatial Framework
Advanced spatial regularization with multiple strategies for Phase 1 target achievement

Enhanced Features:
1. ✅ Multi-scale spatial regularization (local + regional)
2. ✅ Adaptive spatial weights based on geological similarity
3. ✅ Graph Neural Network inspired message passing
4. ✅ Multi-stage training with progressive spatial constraints
5. ✅ Enhanced loss weighting strategies

Target: Achieve 0.3+ signal correlation (5x improvement from 0.065 baseline)

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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import json

warnings.filterwarnings('ignore')

class EnhancedSpatialInSARModel(nn.Module):
    """Advanced spatial model with multi-scale regularization"""
    
    def __init__(self, n_stations: int, n_timepoints: int, coordinates: np.ndarray, 
                 ps00_rates: torch.Tensor, device='cpu'):
        super().__init__()
        self.n_stations = n_stations
        self.n_timepoints = n_timepoints
        self.device = device
        
        # Fixed parameters (preserve perfect rate correlation) - FIRST
        self.register_buffer('linear_trend', ps00_rates.to(device))
        
        # Build multi-scale spatial graphs
        self.coordinates = coordinates
        self.spatial_graphs = self._build_multiscale_spatial_graphs()
        
        # Detect spatial clusters for adaptive regularization
        self.spatial_clusters = self._detect_spatial_clusters()
        
        # Learnable parameters with adaptive initialization
        self.constant_offset = nn.Parameter(torch.zeros(n_stations, device=device))
        self.seasonal_amplitudes = nn.Parameter(torch.ones(n_stations, 4, device=device) * 3.0)
        self.seasonal_phases = nn.Parameter(torch.rand(n_stations, 4, device=device) * 2 * np.pi)
        
        # Learnable spatial adaptation weights
        self.spatial_adaptation_weights = nn.Parameter(torch.ones(n_stations, device=device))
        
        # Fixed periods
        self.register_buffer('periods', torch.tensor([0.25, 0.5, 1.0, 2.0], device=device))
        
        print(f"🌐 Built enhanced spatial framework: multi-scale graphs + adaptive weights")
    
    def _build_multiscale_spatial_graphs(self) -> Dict:
        """Build multi-scale spatial graphs for different regularization scales"""
        graphs = {}
        
        # Local scale (5 nearest neighbors)
        local_nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
        local_nbrs.fit(self.coordinates)
        local_distances, local_indices = local_nbrs.kneighbors(self.coordinates)
        
        graphs['local'] = {
            'indices': torch.tensor(local_indices[:, 1:], device=self.device),  # Exclude self
            'distances': torch.tensor(local_distances[:, 1:], device=self.device),
            'weights': self._compute_adaptive_weights(local_distances[:, 1:], scale='local')
        }
        
        # Regional scale (15 nearest neighbors)
        regional_nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree')
        regional_nbrs.fit(self.coordinates)
        regional_distances, regional_indices = regional_nbrs.kneighbors(self.coordinates)
        
        graphs['regional'] = {
            'indices': torch.tensor(regional_indices[:, 1:], device=self.device),
            'distances': torch.tensor(regional_distances[:, 1:], device=self.device),
            'weights': self._compute_adaptive_weights(regional_distances[:, 1:], scale='regional')
        }
        
        # Distance-based threshold graph
        distance_threshold = np.percentile(local_distances[:, -1], 75)  # 75th percentile of max local distances
        distance_matrix = cdist(self.coordinates, self.coordinates)
        
        threshold_connections = []
        threshold_weights = []
        
        for i in range(self.n_stations):
            connected_stations = np.where(distance_matrix[i] <= distance_threshold)[0]
            connected_stations = connected_stations[connected_stations != i]  # Exclude self
            
            if len(connected_stations) > 0:
                distances = distance_matrix[i, connected_stations]
                weights = self._compute_adaptive_weights(distances.reshape(1, -1), scale='threshold')[0]
                
                threshold_connections.append(connected_stations)
                threshold_weights.append(weights)
            else:
                threshold_connections.append(np.array([]))
                threshold_weights.append(np.array([]))
        
        graphs['threshold'] = {
            'connections': threshold_connections,
            'weights': threshold_weights
        }
        
        return graphs
    
    def _compute_adaptive_weights(self, distances: np.ndarray, scale: str) -> torch.Tensor:
        """Compute adaptive spatial weights based on scale and geological similarity"""
        
        # Base inverse distance weighting
        weights = 1.0 / (distances + 1e-6)
        
        # Scale-specific adjustments
        if scale == 'local':
            # Strong local correlation
            weights = np.exp(-distances / np.mean(distances) * 2.0)
        elif scale == 'regional':
            # Moderate regional correlation
            weights = np.exp(-distances / np.mean(distances) * 1.0)
        elif scale == 'threshold':
            # Distance-based threshold
            weights = np.exp(-distances / np.mean(distances) * 1.5)
        
        # Normalize weights per station
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-6)
        
        return torch.tensor(weights, device=self.device, dtype=torch.float32)
    
    def _detect_spatial_clusters(self) -> Dict:
        """Detect spatial clusters for adaptive regularization"""
        # Use K-means clustering on coordinates + subsidence rates
        features = np.column_stack([
            self.coordinates,
            self.linear_trend.cpu().numpy()
        ])
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Optimal number of clusters (3-5 for Taiwan)
        n_clusters = min(5, max(3, self.n_stations // 20))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        clusters = {
            'labels': torch.tensor(cluster_labels, device=self.device),
            'n_clusters': n_clusters,
            'centroids': torch.tensor(kmeans.cluster_centers_, device=self.device)
        }
        
        print(f"🔍 Detected {n_clusters} spatial clusters for adaptive regularization")
        return clusters
    
    def forward(self, time_vector: torch.Tensor) -> torch.Tensor:
        """Generate signals with enhanced spatial regularization"""
        batch_size = self.n_stations
        time_expanded = time_vector.unsqueeze(0).expand(batch_size, -1)
        
        # Start with constant offset
        signals = self.constant_offset.unsqueeze(1).expand(-1, len(time_vector))
        
        # Add fixed linear trend
        signals = signals + self.linear_trend.unsqueeze(1) * time_expanded
        
        # Add spatially regularized seasonal components
        for i, period in enumerate(self.periods):
            # Apply multi-scale spatial message passing
            amplitude = self._spatial_message_passing(self.seasonal_amplitudes[:, i], component_type='amplitude')
            phase = self._spatial_message_passing(self.seasonal_phases[:, i], component_type='phase')
            
            amplitude = amplitude.unsqueeze(1)
            phase = phase.unsqueeze(1)
            frequency = 1.0 / period
            
            seasonal_component = amplitude * torch.sin(2 * np.pi * frequency * time_expanded + phase)
            signals += seasonal_component
        
        return signals
    
    def _spatial_message_passing(self, parameter: torch.Tensor, component_type: str = 'amplitude') -> torch.Tensor:
        """Graph neural network inspired spatial message passing"""
        
        # Local message passing
        local_updated = self._apply_local_message_passing(parameter, component_type)
        
        # Regional message passing
        regional_updated = self._apply_regional_message_passing(parameter, component_type)
        
        # Cluster-aware message passing
        cluster_updated = self._apply_cluster_message_passing(parameter, component_type)
        
        # Adaptive combination based on learned weights
        adaptation_weights = torch.softmax(self.spatial_adaptation_weights, dim=0)
        
        # Weighted combination of different scales
        local_weight = 0.5
        regional_weight = 0.3
        cluster_weight = 0.2
        
        # Combine with adaptive weighting
        combined = (local_weight * local_updated + 
                   regional_weight * regional_updated + 
                   cluster_weight * cluster_updated)
        
        # Final adaptive mixing with original parameter
        mixing_factor = 0.3  # How much spatial regularization to apply
        final_parameter = (1 - mixing_factor) * parameter + mixing_factor * combined
        
        return final_parameter
    
    def _apply_local_message_passing(self, parameter: torch.Tensor, component_type: str) -> torch.Tensor:
        """Apply local spatial message passing"""
        local_graph = self.spatial_graphs['local']
        neighbor_indices = local_graph['indices']
        neighbor_weights = local_graph['weights']
        
        updated_values = torch.zeros_like(parameter)
        
        for i in range(self.n_stations):
            neighbor_vals = parameter[neighbor_indices[i]]
            weights = neighbor_weights[i]
            
            if component_type == 'phase':
                # Handle circular nature of phases
                neighbor_complex = torch.complex(torch.cos(neighbor_vals), torch.sin(neighbor_vals))
                weighted_complex = torch.sum(neighbor_complex * weights.unsqueeze(-1))
                updated_values[i] = torch.angle(weighted_complex)
            else:
                # Regular weighted average for amplitudes
                updated_values[i] = torch.sum(neighbor_vals * weights)
        
        return updated_values
    
    def _apply_regional_message_passing(self, parameter: torch.Tensor, component_type: str) -> torch.Tensor:
        """Apply regional spatial message passing"""
        regional_graph = self.spatial_graphs['regional']
        neighbor_indices = regional_graph['indices']
        neighbor_weights = regional_graph['weights']
        
        updated_values = torch.zeros_like(parameter)
        
        for i in range(self.n_stations):
            neighbor_vals = parameter[neighbor_indices[i]]
            weights = neighbor_weights[i]
            
            if component_type == 'phase':
                # Handle circular nature of phases
                neighbor_complex = torch.complex(torch.cos(neighbor_vals), torch.sin(neighbor_vals))
                weighted_complex = torch.sum(neighbor_complex * weights.unsqueeze(-1))
                updated_values[i] = torch.angle(weighted_complex)
            else:
                # Regional smoothing with lower weights
                updated_values[i] = torch.sum(neighbor_vals * weights * 0.7)  # Reduced regional influence
        
        return updated_values
    
    def _apply_cluster_message_passing(self, parameter: torch.Tensor, component_type: str) -> torch.Tensor:
        """Apply cluster-aware message passing"""
        cluster_labels = self.spatial_clusters['labels']
        n_clusters = self.spatial_clusters['n_clusters']
        
        updated_values = torch.zeros_like(parameter)
        
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_stations = torch.where(cluster_mask)[0]
            
            if len(cluster_stations) > 1:
                cluster_params = parameter[cluster_stations]
                
                if component_type == 'phase':
                    # Circular mean for phases
                    cluster_complex = torch.complex(torch.cos(cluster_params), torch.sin(cluster_params))
                    cluster_mean_complex = torch.mean(cluster_complex)
                    cluster_mean = torch.angle(cluster_mean_complex)
                else:
                    # Regular mean for amplitudes
                    cluster_mean = torch.mean(cluster_params)
                
                # Update all stations in this cluster
                updated_values[cluster_stations] = cluster_mean
            else:
                # Single station cluster - no change
                updated_values[cluster_stations] = parameter[cluster_stations]
        
        return updated_values
    
    def enhanced_spatial_loss(self) -> torch.Tensor:
        """Compute enhanced spatial consistency loss"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Multi-scale spatial consistency
        for scale_name, graph in self.spatial_graphs.items():
            if scale_name == 'threshold':
                continue  # Skip threshold graph for loss computation
                
            neighbor_indices = graph['indices']
            neighbor_weights = graph['weights']
            
            # Amplitude consistency across scales
            for component_idx in range(4):
                amplitudes = self.seasonal_amplitudes[:, component_idx]
                
                scale_loss = torch.tensor(0.0, device=self.device)
                for station_idx in range(self.n_stations):
                    station_amp = amplitudes[station_idx]
                    neighbor_amps = amplitudes[neighbor_indices[station_idx]]
                    weights = neighbor_weights[station_idx]
                    
                    # Weighted variance penalty
                    weighted_mean = torch.sum(neighbor_amps * weights)
                    weighted_diff = (station_amp - weighted_mean)**2
                    scale_loss += weighted_diff
                
                # Scale-specific weighting
                if scale_name == 'local':
                    total_loss += scale_loss * 1.0  # Strong local consistency
                elif scale_name == 'regional':
                    total_loss += scale_loss * 0.5  # Moderate regional consistency
        
        # Cluster-based consistency
        cluster_loss = self._compute_cluster_consistency_loss()
        total_loss += cluster_loss * 0.3
        
        return total_loss / (self.n_stations * 4)  # Normalize
    
    def _compute_cluster_consistency_loss(self) -> torch.Tensor:
        """Compute within-cluster consistency loss"""
        cluster_labels = self.spatial_clusters['labels']
        n_clusters = self.spatial_clusters['n_clusters']
        
        cluster_loss = torch.tensor(0.0, device=self.device)
        
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_stations = torch.where(cluster_mask)[0]
            
            if len(cluster_stations) > 1:
                # Variance within cluster for each seasonal component
                for component_idx in range(4):
                    cluster_amps = self.seasonal_amplitudes[cluster_stations, component_idx]
                    cluster_variance = torch.var(cluster_amps)
                    cluster_loss += cluster_variance
        
        return cluster_loss
    
    def apply_constraints(self):
        """Apply enhanced physical constraints"""
        with torch.no_grad():
            # Amplitude constraints with cluster-aware bounds
            self.seasonal_amplitudes.clamp_(0, 50)  # mm
            
            # Phase constraints
            self.seasonal_phases.data = torch.fmod(self.seasonal_phases.data, 2 * np.pi)
            
            # Offset constraints
            self.constant_offset.clamp_(-200, 200)  # mm
            
            # Spatial adaptation weights constraints
            self.spatial_adaptation_weights.clamp_(-2, 2)

class EnhancedSpatialLoss(nn.Module):
    """Enhanced loss function with multi-component spatial regularization"""
    
    def __init__(self, alpha_spatial=0.2, alpha_physics=0.05, alpha_temporal=0.1):
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_physics = alpha_physics
        self.alpha_temporal = alpha_temporal
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                model: EnhancedSpatialInSARModel) -> Tuple[torch.Tensor, Dict]:
        """Compute enhanced loss with multiple regularization terms"""
        
        # Primary fitting loss (use L1 loss for robustness)
        primary_loss = torch.nn.functional.l1_loss(predicted, target)
        
        # Enhanced spatial consistency loss
        spatial_loss = model.enhanced_spatial_loss()
        
        # Physics regularization
        physics_loss = self._physics_regularization(model)
        
        # Temporal consistency loss
        temporal_loss = self._temporal_consistency_loss(predicted)
        
        total_loss = (primary_loss + 
                     self.alpha_spatial * spatial_loss + 
                     self.alpha_physics * physics_loss +
                     self.alpha_temporal * temporal_loss)
        
        loss_components = {
            'primary': primary_loss.item(),
            'spatial': spatial_loss.item(),
            'physics': physics_loss.item(),
            'temporal': temporal_loss.item()
        }
        
        return total_loss, loss_components
    
    def _physics_regularization(self, model: EnhancedSpatialInSARModel) -> torch.Tensor:
        """Enhanced physics-based regularization"""
        # Seasonal amplitude reasonableness
        amplitude_penalty = torch.mean(torch.relu(torch.sum(model.seasonal_amplitudes, dim=1) - 80))
        
        # Annual component dominance (geological expectation)
        annual_amp = model.seasonal_amplitudes[:, 2]  # Annual
        other_amps = torch.cat([model.seasonal_amplitudes[:, :2], model.seasonal_amplitudes[:, 3:]], dim=1)
        max_other = torch.max(other_amps, dim=1)[0]
        
        annual_dominance_penalty = torch.mean(torch.relu(max_other - 2.0 * annual_amp))
        
        # Spatial adaptation weights regularization
        adaptation_penalty = torch.mean(model.spatial_adaptation_weights**2) * 0.1
        
        return amplitude_penalty + annual_dominance_penalty + adaptation_penalty
    
    def _temporal_consistency_loss(self, predicted: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness regularization"""
        # Penalize abrupt changes in time series
        temporal_diff = predicted[:, 1:] - predicted[:, :-1]
        temporal_variance = torch.var(temporal_diff, dim=1)
        
        # Allow reasonable temporal variation but penalize excessive noise
        temporal_penalty = torch.mean(torch.relu(temporal_variance - 10.0))  # 10mm threshold
        
        return temporal_penalty

class EnhancedSpatialTaiwanInSARFitter:
    """Enhanced fitter with advanced spatial strategies"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = EnhancedSpatialLoss()
        self.training_history = {'loss_components': [], 'correlations': [], 'spatial_weights': []}
        
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
    
    def initialize_enhanced_model(self):
        """Initialize enhanced spatial model"""
        self.model = EnhancedSpatialInSARModel(
            self.n_stations, self.n_timepoints, 
            self.coordinates, self.subsidence_rates, 
            device=self.device
        )
        
        # Enhanced initialization strategy
        self._enhanced_initialization()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"🚀 Initialized Enhanced Spatial InSAR model on {self.device}")
        print(f"📊 Model parameters: {total_params} total")
        print(f"🌐 Multi-scale spatial graphs + cluster regularization")
    
    def _enhanced_initialization(self):
        """Enhanced parameter initialization with geological insights"""
        with torch.no_grad():
            # Initialize offset to station means
            station_means = torch.mean(self.displacement, dim=1)
            self.model.constant_offset.data = station_means
            
            # Enhanced seasonal initialization based on subsidence magnitude
            subsidence_magnitudes = torch.abs(self.model.linear_trend)
            
            for i in range(self.n_stations):
                signal = self.displacement[i].cpu().numpy()
                time_np = self.time_years.cpu().numpy()
                
                # Remove trend for seasonal analysis
                detrended = signal - station_means[i].item() - self.subsidence_rates[i].item() * time_np
                signal_std = np.std(detrended)
                
                # Subsidence-magnitude informed initialization
                magnitude_factor = float(1.0 + subsidence_magnitudes[i].item() / 50.0)  # Scale by subsidence
                
                # Initialize amplitudes with geological reasoning
                self.model.seasonal_amplitudes.data[i, 0] = float(signal_std * 0.2 * magnitude_factor)  # Quarterly
                self.model.seasonal_amplitudes.data[i, 1] = float(signal_std * 0.3 * magnitude_factor)  # Semi-annual
                self.model.seasonal_amplitudes.data[i, 2] = float(signal_std * 0.6 * magnitude_factor)  # Annual (dominant)
                self.model.seasonal_amplitudes.data[i, 3] = float(signal_std * 0.4 * magnitude_factor)  # Long-annual
            
            # Initialize spatial adaptation weights near zero (start with minimal adaptation)
            self.model.spatial_adaptation_weights.data.fill_(0.1)
            
            # Random phase initialization with cluster-based similarity
            cluster_labels = self.model.spatial_clusters['labels']
            n_clusters = self.model.spatial_clusters['n_clusters']
            
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                cluster_stations = torch.where(cluster_mask)[0]
                
                if len(cluster_stations) > 1:
                    # Similar phases within clusters
                    base_phases = torch.rand(4) * 2 * np.pi
                    noise = torch.randn(len(cluster_stations), 4) * 0.5  # Small cluster variation
                    
                    for i, station_idx in enumerate(cluster_stations):
                        self.model.seasonal_phases.data[station_idx] = base_phases + noise[i]
                else:
                    # Random for isolated stations
                    self.model.seasonal_phases.data[cluster_stations] = torch.rand(len(cluster_stations), 4) * 2 * np.pi
    
    def train_enhanced_spatial(self, max_epochs: int = 1500, target_correlation: float = 0.3):
        """Train with enhanced multi-stage spatial strategy"""
        
        print(f"🎯 Starting Enhanced Spatial Training (target: {target_correlation:.3f})")
        print("🌐 Multi-stage strategy: Progressive spatial regularization")
        print("="*80)
        
        # Multi-stage training strategy
        stages = [
            {'epochs': 500, 'spatial_weight': 0.05, 'lr': 0.03, 'name': 'Warm-up'},
            {'epochs': 500, 'spatial_weight': 0.15, 'lr': 0.02, 'name': 'Spatial Integration'},
            {'epochs': 500, 'spatial_weight': 0.25, 'lr': 0.01, 'name': 'Fine-tuning'}
        ]
        
        total_loss_history = []
        total_correlation_history = []
        best_correlation = -1.0
        
        for stage_idx, stage_config in enumerate(stages):
            print(f"\n🔥 STAGE {stage_idx + 1}: {stage_config['name']}")
            print(f"   Spatial weight: {stage_config['spatial_weight']}")
            print(f"   Learning rate: {stage_config['lr']}")
            print("-" * 50)
            
            # Update loss function weights
            self.loss_function.alpha_spatial = stage_config['spatial_weight']
            
            # Setup optimizer for this stage
            self.optimizer = optim.Adam(self.model.parameters(), lr=stage_config['lr'], weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=100, factor=0.8
            )
            
            # Train this stage
            stage_loss, stage_corr = self._train_stage(
                stage_config['epochs'], 
                stage_idx, 
                target_correlation
            )
            
            total_loss_history.extend(stage_loss)
            total_correlation_history.extend(stage_corr)
            
            # Check for early achievement
            current_best = max(stage_corr)
            if current_best > best_correlation:
                best_correlation = current_best
            
            if best_correlation >= target_correlation:
                print(f"🎉 Target correlation achieved in stage {stage_idx + 1}!")
                break
        
        print("="*80)
        print(f"✅ Enhanced Spatial Training Completed: Best correlation = {best_correlation:.4f}")
        
        return total_loss_history, total_correlation_history
    
    def _train_stage(self, max_epochs: int, stage_idx: int, target_correlation: float):
        """Train single stage with specific configuration"""
        
        self.model.train()
        loss_history = []
        correlation_history = []
        patience_counter = 0
        patience = 150
        
        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.time_years)
            
            # Compute enhanced loss
            total_loss, loss_components = self.loss_function(predictions, self.displacement, self.model)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Apply constraints
            self.model.apply_constraints()
            
            # Optimizer step
            self.optimizer.step()
            
            # Track progress
            loss_history.append(total_loss.item())
            self.training_history['loss_components'].append(loss_components)
            
            # Compute correlation
            with torch.no_grad():
                correlation = self._compute_correlation(predictions, self.displacement)
                correlation_history.append(correlation.item())
                self.training_history['correlations'].append(correlation.item())
                
                # Update scheduler
                self.scheduler.step(correlation)
                
                # Early stopping check
                if correlation.item() >= target_correlation:
                    print(f"🎉 Target correlation {target_correlation:.3f} achieved at epoch {epoch}!")
                    break
                
                if len(correlation_history) > 50:
                    recent_improvement = max(correlation_history[-50:]) - min(correlation_history[-50:])
                    if recent_improvement < 0.001:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                
                if patience_counter >= patience:
                    print(f"🛑 Stage {stage_idx + 1} early stopping at epoch {epoch}")
                    break
            
            # Detailed logging
            if epoch % 75 == 0:
                rmse = torch.sqrt(torch.mean((predictions - self.displacement)**2))
                print(f"S{stage_idx+1} E{epoch:3d}: Loss={total_loss.item():.3f} "
                      f"(Fit:{loss_components['primary']:.2f}, Spatial:{loss_components['spatial']:.2f}, "
                      f"Physics:{loss_components['physics']:.2f}, Temporal:{loss_components['temporal']:.2f}) "
                      f"RMSE={rmse.item():.1f}mm, Corr={correlation.item():.4f}")
        
        return loss_history, correlation_history
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute average correlation across all stations"""
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        target_centered = target - torch.mean(target, dim=1, keepdim=True)
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1))
        
        correlations = numerator / (denominator + 1e-8)
        return torch.mean(correlations)
    
    def evaluate_enhanced(self) -> Dict:
        """Comprehensive evaluation of enhanced spatial model"""
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
                'spatial_clusters': self.model.spatial_clusters,
                'training_history': self.training_history
            }
        
        # Performance analysis
        mean_rmse = np.mean(results['rmse'])
        mean_corr = np.mean(results['correlations'])
        rate_corr = np.corrcoef(results['original_rates'], results['fitted_trends'])[0, 1]
        
        print(f"📊 ENHANCED SPATIAL MODEL EVALUATION:")
        print(f"   Signal RMSE: {mean_rmse:.2f} ± {np.std(results['rmse']):.2f} mm")
        print(f"   Signal Correlation: {mean_corr:.4f} ± {np.std(results['correlations']):.3f}")
        print(f"   Rate Correlation: {rate_corr:.6f}")
        
        # Achievement analysis
        baseline_correlation = 0.065
        target_correlation = 0.3
        achievement_ratio = mean_corr / target_correlation
        improvement_ratio = mean_corr / baseline_correlation
        
        print(f"   📈 Target Achievement: {achievement_ratio:.1%}")
        print(f"   📈 Baseline Improvement: {improvement_ratio:.1f}x")
        
        if mean_corr >= target_correlation:
            print(f"   🎉 PHASE 1 TARGET ACHIEVED!")
        else:
            print(f"   🔄 Continue optimization recommended")
        
        return results

def demonstrate_enhanced_spatial_framework():
    """Demonstrate enhanced spatial framework with advanced strategies"""
    
    print("🚀 PS02C-PYTORCH ENHANCED SPATIAL FRAMEWORK")
    print("🎯 Advanced Phase 1: Multi-scale spatial + Adaptive regularization")
    print("🏆 Target: 0.3+ signal correlation (5x improvement)")
    print("="*80)
    
    # Initialize enhanced fitter
    fitter = EnhancedSpatialTaiwanInSARFitter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print(f"\n1️⃣ Loading Taiwan InSAR Data...")
    fitter.load_data()
    
    # Use subset for demonstration
    subset_size = min(100, fitter.n_stations)
    
    # Select spatially diverse subset
    total_stations = fitter.n_stations
    step = total_stations // subset_size
    selected_indices = np.arange(0, total_stations, step)[:subset_size]
    
    fitter.displacement = fitter.displacement[selected_indices]
    fitter.coordinates = fitter.coordinates[selected_indices]
    fitter.subsidence_rates = fitter.subsidence_rates[selected_indices]
    fitter.n_stations = subset_size
    
    print(f"📊 Using enhanced spatial subset: {subset_size} stations")
    
    # Initialize enhanced model
    print(f"\n2️⃣ Initializing Enhanced Spatial Model...")
    fitter.initialize_enhanced_model()
    
    # Train with enhanced spatial strategies
    print(f"\n3️⃣ Enhanced Multi-Stage Training...")
    start_time = time.time()
    loss_history, correlation_history = fitter.train_enhanced_spatial(
        max_epochs=1500, 
        target_correlation=0.3
    )
    training_time = time.time() - start_time
    
    # Comprehensive evaluation
    print(f"\n4️⃣ Comprehensive Enhanced Evaluation...")
    results = fitter.evaluate_enhanced()
    
    print(f"\n✅ ENHANCED SPATIAL FRAMEWORK COMPLETED!")
    print(f"⏱️ Training time: {training_time:.1f} seconds")
    
    # Final Phase 1 assessment
    achieved_correlation = np.mean(results['correlations'])
    target_correlation = 0.3
    baseline_correlation = 0.065
    
    print(f"\n🏆 PHASE 1 FINAL ASSESSMENT:")
    print(f"   🎯 Target: {target_correlation:.3f}")
    print(f"   ✅ Achieved: {achieved_correlation:.4f}")
    print(f"   📊 Baseline: {baseline_correlation:.3f}")
    print(f"   📈 Improvement: {achieved_correlation/baseline_correlation:.1f}x")
    
    if achieved_correlation >= target_correlation:
        print(f"   🎉 🎉 PHASE 1 SUCCESS! TARGET ACHIEVED! 🎉 🎉")
        print(f"   ✅ Ready for Phase 2: Production scaling")
    else:
        print(f"   🔄 Phase 1 progress: {achieved_correlation/target_correlation:.1%} of target")
        print(f"   💡 Recommendations for Phase 1 completion:")
        print(f"      • Integrate EMD initialization")
        print(f"      • Add borehole geological constraints") 
        print(f"      • Increase training epochs")
        print(f"      • Tune spatial regularization weights")
    
    return fitter, results

if __name__ == "__main__":
    try:
        fitter, results = demonstrate_enhanced_spatial_framework()
        
        # Save enhanced Phase 1 results
        output_file = Path("data/processed/ps02c_enhanced_spatial_phase1_results.npz")
        output_file.parent.mkdir(exist_ok=True)
        
        np.savez(output_file,
                 correlations=results['correlations'],
                 rmse=results['rmse'],
                 training_time=time.time(),
                 target_achieved=np.mean(results['correlations']) >= 0.3,
                 improvement_ratio=np.mean(results['correlations']) / 0.065)
        
        print(f"💾 Enhanced Phase 1 results saved: {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Enhanced spatial framework interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Enhanced spatial framework failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
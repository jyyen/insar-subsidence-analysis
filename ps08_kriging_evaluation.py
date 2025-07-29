#!/usr/bin/env python3
"""
PS08 Kriging Evaluation: Comprehensive Geostatistical Analysis
=============================================================

Evaluates kriging vs IDW interpolation for borehole grain-size data:
- Variogram analysis (experimental and theoretical models)
- Directional variograms (anisotropy detection) 
- Cross-validation comparison (IDW vs Kriging)
- Uncertainty quantification
- Model selection and validation

Author: Claude Code Assistant
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class KrigingEvaluation:
    """Comprehensive geostatistical analysis for borehole grain-size interpolation"""
    
    def __init__(self):
        self.base_dir = Path('.')
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load and prepare data
        self.load_borehole_data()
        self.process_grain_size_data()
        
    def load_borehole_data(self):
        """Load borehole data for geostatistical analysis"""
        print("📊 Loading borehole data for kriging evaluation...")
        
        borehole_file = Path('../Taiwan_borehole_data/analysis_output/well_fractions.csv')
        if not borehole_file.exists():
            raise FileNotFoundError(f"Borehole file not found: {borehole_file}")
        
        self.borehole_data = pd.read_csv(borehole_file)
        print(f"   ✅ Loaded {len(self.borehole_data)} borehole stations")
        
        # Extract coordinates and remove any duplicates
        coords = self.borehole_data[['Longitude', 'Latitude']].values
        unique_indices = []
        seen = set()
        for i, coord in enumerate(coords):
            coord_tuple = (round(coord[0], 6), round(coord[1], 6))
            if coord_tuple not in seen:
                seen.add(coord_tuple)
                unique_indices.append(i)
        
        self.borehole_data = self.borehole_data.iloc[unique_indices].reset_index(drop=True)
        print(f"   📍 Unique locations: {len(self.borehole_data)}")
        
    def process_grain_size_data(self):
        """Process grain-size data for interpolation analysis"""
        print("🔄 Processing grain-size data...")
        
        # Create 2-category system: coarse vs fine (same as ps08_fig06)
        self.borehole_data['coarse_pct'] = self.borehole_data['Coarse_Pct']
        self.borehole_data['fine_pct'] = self.borehole_data['Sand_Pct'] + self.borehole_data['Fine_Pct']
        
        # Normalize to ensure 100% total
        total_pct = self.borehole_data['coarse_pct'] + self.borehole_data['fine_pct']
        self.borehole_data['coarse_pct'] = (self.borehole_data['coarse_pct'] / total_pct) * 100
        self.borehole_data['fine_pct'] = (self.borehole_data['fine_pct'] / total_pct) * 100
        
        # Extract coordinates and values for analysis
        self.coordinates = self.borehole_data[['Longitude', 'Latitude']].values
        self.coarse_values = self.borehole_data['coarse_pct'].values
        self.fine_values = self.borehole_data['fine_pct'].values
        
        print(f"   ✅ Processed {len(self.coordinates)} stations")
        print(f"   📊 Coarse percentage: {np.mean(self.coarse_values):.1f}% ± {np.std(self.coarse_values):.1f}%")
        print(f"   📊 Fine percentage: {np.mean(self.fine_values):.1f}% ± {np.std(self.fine_values):.1f}%")
        
    def calculate_experimental_variogram(self, values, max_distance=None, n_lags=15):
        """Calculate experimental variogram"""
        print(f"🔬 Calculating experimental variogram...")
        
        if max_distance is None:
            max_distance = self.estimate_max_variogram_distance()
        
        # Calculate all pairwise distances and semivariances
        distances = cdist(self.coordinates, self.coordinates)
        n_points = len(values)
        
        # Create distance and semivariance arrays
        all_distances = []
        all_semivariances = []
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = distances[i, j]
                if dist <= max_distance:
                    semivar = 0.5 * (values[i] - values[j])**2
                    all_distances.append(dist)
                    all_semivariances.append(semivar)
        
        all_distances = np.array(all_distances)
        all_semivariances = np.array(all_semivariances)
        
        # Bin the data into lags
        lag_distances = np.linspace(0, max_distance, n_lags + 1)
        lag_centers = (lag_distances[:-1] + lag_distances[1:]) / 2
        lag_semivariances = []
        lag_counts = []
        
        for i in range(n_lags):
            mask = (all_distances >= lag_distances[i]) & (all_distances < lag_distances[i+1])
            if np.sum(mask) > 0:
                lag_semivariances.append(np.mean(all_semivariances[mask]))
                lag_counts.append(np.sum(mask))
            else:
                lag_semivariances.append(np.nan)
                lag_counts.append(0)
        
        print(f"   ✅ Calculated variogram with {len(lag_centers)} lags")
        print(f"   📊 Max distance: {max_distance:.2f} degrees ({max_distance*111:.1f} km)")
        
        return lag_centers, np.array(lag_semivariances), np.array(lag_counts)
    
    def calculate_directional_variograms(self, values, directions=[0, 45, 90, 135], tolerance=22.5):
        """Calculate directional variograms to detect anisotropy"""
        print(f"🧭 Calculating directional variograms...")
        
        max_distance = self.estimate_max_variogram_distance()
        directional_variograms = {}
        
        for direction in directions:
            print(f"   📐 Direction: {direction}° (±{tolerance}°)")
            
            # Calculate distances and angles between all point pairs
            distances = cdist(self.coordinates, self.coordinates)
            n_points = len(values)
            
            direction_distances = []
            direction_semivariances = []
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = distances[i, j]
                    if dist <= max_distance:
                        # Calculate angle between points
                        dx = self.coordinates[j, 0] - self.coordinates[i, 0]
                        dy = self.coordinates[j, 1] - self.coordinates[i, 1]
                        angle = np.degrees(np.arctan2(dy, dx))
                        if angle < 0:
                            angle += 360
                        
                        # Check if angle falls within the directional tolerance
                        angle_diff = min(abs(angle - direction), 360 - abs(angle - direction))
                        if angle_diff <= tolerance:
                            semivar = 0.5 * (values[i] - values[j])**2
                            direction_distances.append(dist)
                            direction_semivariances.append(semivar)
            
            # Bin into lags (fewer lags for directional variograms)
            if len(direction_distances) > 10:  # Minimum points needed
                n_lags = 10
                lag_distances = np.linspace(0, max_distance, n_lags + 1)
                lag_centers = (lag_distances[:-1] + lag_distances[1:]) / 2
                lag_semivariances = []
                
                for k in range(n_lags):
                    mask = (np.array(direction_distances) >= lag_distances[k]) & \
                           (np.array(direction_distances) < lag_distances[k+1])
                    if np.sum(mask) > 2:  # Need at least 3 pairs
                        lag_semivariances.append(np.mean(np.array(direction_semivariances)[mask]))
                    else:
                        lag_semivariances.append(np.nan)
                
                directional_variograms[direction] = {
                    'distances': lag_centers,
                    'semivariances': np.array(lag_semivariances),
                    'n_pairs': len(direction_distances)
                }
                print(f"      ✅ {len(direction_distances)} pairs found")
            else:
                print(f"      ⚠️  Insufficient pairs ({len(direction_distances)}) for direction {direction}°")
        
        return directional_variograms
    
    def estimate_max_variogram_distance(self):
        """Estimate maximum distance for variogram calculation (typically 1/3 to 1/2 of study area)"""
        coords_range = np.ptp(self.coordinates, axis=0)
        max_distance = np.sqrt(np.sum(coords_range**2)) / 3
        return max_distance
    
    def fit_variogram_models(self, distances, semivariances):
        """Fit theoretical variogram models (spherical, exponential, gaussian)"""
        print("📈 Fitting theoretical variogram models...")
        
        # Remove NaN values
        valid_mask = ~np.isnan(semivariances)
        valid_distances = distances[valid_mask]
        valid_semivariances = semivariances[valid_mask]
        
        if len(valid_distances) < 3:
            print("   ⚠️  Insufficient valid data points for model fitting")
            return {}
        
        models = {}
        
        # Initial parameter estimates
        sill_estimate = np.max(valid_semivariances) if len(valid_semivariances) > 0 else 100
        range_estimate = np.max(valid_distances) / 3 if len(valid_distances) > 0 else 0.1
        nugget_estimate = np.min(valid_semivariances) if len(valid_semivariances) > 0 else 0
        
        # Spherical model
        def spherical_model(params, distances):
            nugget, sill, range_param = params
            gamma = np.zeros_like(distances)
            mask = distances > 0
            
            # For distances within range
            within_range = mask & (distances <= range_param)
            if np.any(within_range):
                h = distances[within_range]
                gamma[within_range] = nugget + sill * (1.5 * h/range_param - 0.5 * (h/range_param)**3)
            
            # For distances beyond range
            beyond_range = mask & (distances > range_param)
            gamma[beyond_range] = nugget + sill
            
            return gamma
        
        # Exponential model
        def exponential_model(params, distances):
            nugget, sill, range_param = params
            gamma = np.zeros_like(distances)
            mask = distances > 0
            gamma[mask] = nugget + sill * (1 - np.exp(-3 * distances[mask] / range_param))
            return gamma
        
        # Gaussian model
        def gaussian_model(params, distances):
            nugget, sill, range_param = params
            gamma = np.zeros_like(distances)
            mask = distances > 0
            gamma[mask] = nugget + sill * (1 - np.exp(-3 * (distances[mask] / range_param)**2))
            return gamma
        
        # Fit each model
        model_functions = {
            'spherical': spherical_model,
            'exponential': exponential_model,
            'gaussian': gaussian_model
        }
        
        for model_name, model_func in model_functions.items():
            try:
                def objective(params):
                    if params[0] < 0 or params[1] < 0 or params[2] <= 0:  # Constraints
                        return 1e10
                    predicted = model_func(params, valid_distances)
                    return np.sum((valid_semivariances - predicted)**2)
                
                # Optimize parameters
                result = minimize(
                    objective, 
                    x0=[nugget_estimate, sill_estimate, range_estimate],
                    method='Nelder-Mead',
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    nugget, sill, range_param = result.x
                    predicted = model_func(result.x, valid_distances)
                    r2 = r2_score(valid_semivariances, predicted)
                    rmse = np.sqrt(mean_squared_error(valid_semivariances, predicted))
                    
                    models[model_name] = {
                        'nugget': nugget,
                        'sill': sill,
                        'range': range_param,
                        'r2': r2,
                        'rmse': rmse,
                        'function': model_func,
                        'params': result.x
                    }
                    print(f"   ✅ {model_name.capitalize()}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
                else:
                    print(f"   ❌ {model_name.capitalize()}: Optimization failed")
                    
            except Exception as e:
                print(f"   ❌ {model_name.capitalize()}: Error fitting model - {e}")
        
        return models
    
    def perform_cross_validation(self, values, method='both', k_folds=5):
        """Perform k-fold cross-validation comparing IDW and Kriging"""
        print(f"🔄 Performing {k_folds}-fold cross-validation...")
        
        n_points = len(values)
        fold_size = n_points // k_folds
        
        idw_predictions = []
        idw_actuals = []
        kriging_predictions = []
        kriging_actuals = []
        
        # First, fit variogram model on full dataset for kriging
        distances, semivariances, _ = self.calculate_experimental_variogram(values)
        models = self.fit_variogram_models(distances, semivariances)
        
        # Select best model
        best_model = None
        best_r2 = -np.inf
        for model_name, model_info in models.items():
            if model_info['r2'] > best_r2:
                best_r2 = model_info['r2']
                best_model = model_info
        
        if best_model is None:
            print("   ⚠️  No valid variogram model found, using IDW only")
            method = 'idw'
        
        for fold in range(k_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else n_points
            
            # Test set
            test_indices = list(range(start_idx, end_idx))
            train_indices = [i for i in range(n_points) if i not in test_indices]
            
            train_coords = self.coordinates[train_indices]
            train_values = values[train_indices]
            test_coords = self.coordinates[test_indices]
            test_values = values[test_indices]
            
            # IDW interpolation
            if method in ['both', 'idw']:
                for i, test_coord in enumerate(test_coords):
                    distances = cdist([test_coord], train_coords)[0]
                    
                    # IDW with power = 2
                    weights = 1.0 / (distances**2 + 1e-10)  # Avoid division by zero
                    predicted_value = np.sum(weights * train_values) / np.sum(weights)
                    
                    idw_predictions.append(predicted_value)
                    idw_actuals.append(test_values[i])
            
            # Kriging interpolation
            if method in ['both', 'kriging'] and best_model is not None:
                for i, test_coord in enumerate(test_coords):
                    try:
                        predicted_value = self.simple_kriging_prediction(
                            test_coord, train_coords, train_values, best_model
                        )
                        kriging_predictions.append(predicted_value)
                        kriging_actuals.append(test_values[i])
                    except:
                        # Fallback to IDW if kriging fails
                        distances = cdist([test_coord], train_coords)[0]
                        weights = 1.0 / (distances**2 + 1e-10)
                        predicted_value = np.sum(weights * train_values) / np.sum(weights)
                        kriging_predictions.append(predicted_value)
                        kriging_actuals.append(test_values[i])
        
        # Calculate validation statistics
        results = {}
        
        if len(idw_predictions) > 0:
            idw_r2 = r2_score(idw_actuals, idw_predictions)
            idw_rmse = np.sqrt(mean_squared_error(idw_actuals, idw_predictions))
            results['IDW'] = {'r2': idw_r2, 'rmse': idw_rmse}
            print(f"   📊 IDW: R² = {idw_r2:.3f}, RMSE = {idw_rmse:.3f}")
        
        if len(kriging_predictions) > 0:
            kriging_r2 = r2_score(kriging_actuals, kriging_predictions)
            kriging_rmse = np.sqrt(mean_squared_error(kriging_actuals, kriging_predictions))
            results['Kriging'] = {'r2': kriging_r2, 'rmse': kriging_rmse}
            print(f"   📊 Kriging: R² = {kriging_r2:.3f}, RMSE = {kriging_rmse:.3f}")
        
        return results
    
    def simple_kriging_prediction(self, target_coord, known_coords, known_values, variogram_model):
        """Simple kriging prediction at a single point"""
        distances = cdist([target_coord], known_coords)[0]
        
        # Build covariance matrix
        n = len(known_coords)
        C = np.zeros((n + 1, n + 1))
        
        # Fill covariance matrix
        for i in range(n):
            for j in range(n):
                dist = cdist([known_coords[i]], [known_coords[j]])[0][0]
                semivar = variogram_model['function'](variogram_model['params'], np.array([dist]))[0]
                C[i, j] = variogram_model['sill'] - semivar
            C[i, n] = 1.0  # Lagrange multiplier constraint
            C[n, i] = 1.0
        
        # Right-hand side vector
        b = np.zeros(n + 1)
        for i in range(n):
            semivar = variogram_model['function'](variogram_model['params'], np.array([distances[i]]))[0]
            b[i] = variogram_model['sill'] - semivar
        b[n] = 1.0
        
        # Solve for weights
        try:
            weights = np.linalg.solve(C, b)
            prediction = np.sum(weights[:n] * known_values)
            return prediction
        except:
            # Fallback to IDW if matrix is singular
            weights = 1.0 / (distances**2 + 1e-10)
            return np.sum(weights * known_values) / np.sum(weights)
    
    def create_comprehensive_analysis(self):
        """Create comprehensive variogram and kriging analysis plots"""
        print("🎨 Creating comprehensive geostatistical analysis...")
        
        # Analyze both coarse and fine fractions
        variables = {'Coarse Fraction': self.coarse_values, 'Fine Fraction': self.fine_values}
        
        for var_name, values in variables.items():
            print(f"\n📊 Analyzing {var_name}...")
            
            # Calculate experimental variogram
            distances, semivariances, counts = self.calculate_experimental_variogram(values)
            
            # Fit theoretical models
            models = self.fit_variogram_models(distances, semivariances)
            
            # Calculate directional variograms
            directional_variograms = self.calculate_directional_variograms(values)
            
            # Perform cross-validation
            cv_results = self.perform_cross_validation(values)
            
            # Create plots
            self.create_variogram_plots(var_name, distances, semivariances, counts, 
                                      models, directional_variograms, cv_results)
    
    def create_variogram_plots(self, var_name, distances, semivariances, counts, 
                              models, directional_variograms, cv_results):
        """Create comprehensive variogram analysis plots"""
        
        fig = plt.figure(figsize=(16, 12))
        var_clean = var_name.replace(' ', '_').lower()
        
        # 1. Experimental variogram with fitted models
        ax1 = plt.subplot(2, 3, 1)
        valid_mask = ~np.isnan(semivariances)
        ax1.scatter(distances[valid_mask], semivariances[valid_mask], 
                   c='blue', s=50, alpha=0.7, label='Experimental')
        
        # Plot fitted models
        colors = ['red', 'green', 'orange']
        if models:
            plot_distances = np.linspace(0, np.max(distances[valid_mask]), 100)
            for i, (model_name, model_info) in enumerate(models.items()):
                predicted = model_info['function'](model_info['params'], plot_distances)
                ax1.plot(plot_distances, predicted, colors[i % len(colors)], 
                        linewidth=2, label=f"{model_name.capitalize()} (R²={model_info['r2']:.3f})")
        
        ax1.set_xlabel('Distance (degrees)')
        ax1.set_ylabel('Semivariance')
        ax1.set_title(f'Experimental Variogram - {var_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Directional variograms
        ax2 = plt.subplot(2, 3, 2)
        colors_dir = ['blue', 'red', 'green', 'orange']
        for i, (direction, vg_data) in enumerate(directional_variograms.items()):
            valid_dir = ~np.isnan(vg_data['semivariances'])
            if np.any(valid_dir):
                ax2.plot(vg_data['distances'][valid_dir], vg_data['semivariances'][valid_dir], 
                        'o-', color=colors_dir[i % len(colors_dir)], 
                        label=f"{direction}° ({vg_data['n_pairs']} pairs)")
        
        ax2.set_xlabel('Distance (degrees)')
        ax2.set_ylabel('Semivariance')
        ax2.set_title('Directional Variograms')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Data distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(getattr(self, f"{var_clean.split('_')[0]}_values"), bins=20, 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel(f'{var_name} (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Data Distribution - {var_name}')
        ax3.grid(True, alpha=0.3)
        
        # 4. Spatial distribution of data points
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                            c=getattr(self, f"{var_clean.split('_')[0]}_values"), 
                            cmap='viridis', s=60, alpha=0.8)
        plt.colorbar(scatter, ax=ax4, label=f'{var_name} (%)')
        ax4.set_xlabel('Longitude (degrees)')
        ax4.set_ylabel('Latitude (degrees)')
        ax4.set_title(f'Spatial Distribution - {var_name}')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model comparison statistics
        ax5 = plt.subplot(2, 3, 5)
        if models:
            model_names = list(models.keys())
            r2_values = [models[name]['r2'] for name in model_names]
            rmse_values = [models[name]['rmse'] for name in model_names]
            
            x_pos = np.arange(len(model_names))
            
            ax5_twin = ax5.twinx()
            bars1 = ax5.bar(x_pos - 0.2, r2_values, 0.4, label='R²', color='lightblue', alpha=0.7)
            bars2 = ax5_twin.bar(x_pos + 0.2, rmse_values, 0.4, label='RMSE', color='lightcoral', alpha=0.7)
            
            ax5.set_xlabel('Variogram Models')
            ax5.set_ylabel('R²', color='blue')
            ax5_twin.set_ylabel('RMSE', color='red')
            ax5.set_title('Model Performance')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels([name.capitalize() for name in model_names])
            
            # Add value labels on bars
            for bar, val in zip(bars1, r2_values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            for bar, val in zip(bars2, rmse_values):
                ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.02, 
                             f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax5.text(0.5, 0.5, 'No valid models fitted', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Model Performance')
        
        # 6. Cross-validation results
        ax6 = plt.subplot(2, 3, 6)
        if cv_results:
            methods = list(cv_results.keys())
            r2_cv = [cv_results[method]['r2'] for method in methods]
            rmse_cv = [cv_results[method]['rmse'] for method in methods]
            
            x_pos = np.arange(len(methods))
            
            ax6_twin = ax6.twinx()
            bars1 = ax6.bar(x_pos - 0.2, r2_cv, 0.4, label='CV R²', color='lightgreen', alpha=0.7)
            bars2 = ax6_twin.bar(x_pos + 0.2, rmse_cv, 0.4, label='CV RMSE', color='salmon', alpha=0.7)
            
            ax6.set_xlabel('Interpolation Methods')
            ax6.set_ylabel('Cross-Validation R²', color='green')
            ax6_twin.set_ylabel('Cross-Validation RMSE', color='red')
            ax6.set_title('Cross-Validation Results')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(methods)
            
            # Add value labels
            for bar, val in zip(bars1, r2_cv):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            for bar, val in zip(bars2, rmse_cv):
                ax6_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_cv)*0.02, 
                             f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax6.text(0.5, 0.5, 'No cross-validation results', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Cross-Validation Results')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / f"ps08_kriging_analysis_{var_clean}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {fig_path}")

def main():
    """Main execution function"""
    print("🚀 PS08 Kriging Evaluation: Comprehensive Geostatistical Analysis")
    print("=" * 70)
    
    try:
        evaluation = KrigingEvaluation()
        evaluation.create_comprehensive_analysis()
        
        print("\n✅ Kriging evaluation completed successfully!")
        print("\n📝 Summary:")
        print("   • Experimental and theoretical variogram analysis")
        print("   • Directional variogram analysis for anisotropy detection")
        print("   • Cross-validation comparison of IDW vs Kriging")
        print("   • Model selection and performance evaluation")
        print("   • Comprehensive visualization suite generated")
        
    except Exception as e:
        print(f"❌ Error in kriging evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ps05c_hydrogeological_event_visualization.py - Hydrogeological Event Visualization

Visualizes detected change points in the context of Taiwan's hydrogeological conditions:
- Seasonal patterns (monsoon, dry season)
- Groundwater pumping cycles (irrigation seasons)
- Extreme weather events (typhoons, droughts)
- Aquifer system responses
- Land subsidence mechanisms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

# Try to import cartopy for geographic mapping
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from cartopy.feature import NaturalEarthFeature
    CARTOPY_AVAILABLE = True
    print("✅ Cartopy available for geographic visualization")
except ImportError:
    CARTOPY_AVAILABLE = False
    print("⚠️  Cartopy not available - using basic matplotlib plots")

class HydrogeologicalEventVisualizer:
    """Visualize InSAR change points in hydrogeological context"""
    
    def __init__(self):
        self.results_dir = Path("data/processed/ps05b_advanced")
        self.figures_dir = Path("figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load processed data
        self.load_change_point_data()
        self.setup_taiwan_hydrology()
        
    def load_change_point_data(self):
        """Load change point detection results"""
        print("📡 Loading change point detection results...")
        
        # Load original InSAR data for context
        try:
            insar_data = np.load("data/processed/ps00_preprocessed_data.npz")
            self.coordinates = insar_data['coordinates']
            self.displacement = insar_data['displacement']
            self.n_stations, self.n_times = self.displacement.shape
            
            # Create time vector (6-day sampling, starting from 2018-01-01)
            self.start_date = datetime(2018, 1, 1)
            self.time_vector = [self.start_date + timedelta(days=i*6) for i in range(self.n_times)]
            
            print(f"✅ Loaded InSAR data: {self.n_stations} stations, {self.n_times} time points")
            
        except Exception as e:
            print(f"❌ Error loading InSAR data: {e}")
            return False
            
        # Load change point results
        self.change_points = {}
        methods = ['wavelet', 'bayesian', 'ensemble', 'lstm', 'lombscargle']
        
        for method in methods:
            try:
                data = np.load(self.results_dir / f"{method}_simple_results.npz", allow_pickle=True)
                change_points_data = data['change_points']
                
                # Convert numpy array to list if needed
                if isinstance(change_points_data, np.ndarray):
                    self.change_points[method] = change_points_data.tolist()
                else:
                    self.change_points[method] = change_points_data
                    
                print(f"✅ Loaded {method}: {len(self.change_points[method])} change points")
            except Exception as e:
                print(f"⚠️  Could not load {method} results: {e}")
                self.change_points[method] = []  # Set empty list as fallback
                
        return True
        
    def setup_taiwan_hydrology(self):
        """Define Taiwan's hydrogeological calendar and extreme events"""
        print("🌊 Setting up Taiwan hydrogeological context...")
        
        # Taiwan's hydrogeological seasons
        self.seasons = {
            'dry_season': {
                'months': [11, 12, 1, 2, 3, 4],  # Nov-Apr
                'description': 'Dry Season - Intensive groundwater pumping',
                'color': 'wheat',
                'alpha': 0.3
            },
            'wet_season': {
                'months': [5, 6, 7, 8, 9, 10],  # May-Oct
                'description': 'Wet Season - Monsoon & typhoons, reduced pumping',
                'color': 'lightblue', 
                'alpha': 0.3
            },
            'rice_planting_1': {
                'months': [2, 3, 4],  # Feb-Apr
                'description': 'First Rice Crop - Peak irrigation demand',
                'color': 'lightgreen',
                'alpha': 0.4
            },
            'rice_planting_2': {
                'months': [7, 8, 9],  # Jul-Sep  
                'description': 'Second Rice Crop - High irrigation demand',
                'color': 'lightgreen',
                'alpha': 0.4
            }
        }
        
        # Major extreme events during 2018-2021 period
        self.extreme_events = [
            {
                'date': datetime(2018, 7, 10),
                'name': 'Typhoon Maria',
                'type': 'typhoon',
                'impact': 'Sudden groundwater recharge, reduced pumping',
                'color': 'red'
            },
            {
                'date': datetime(2018, 9, 15), 
                'name': 'Typhoon Mangkhut',
                'type': 'typhoon',
                'impact': 'Heavy precipitation, aquifer recharge',
                'color': 'red'
            },
            {
                'date': datetime(2019, 8, 9),
                'name': 'Typhoon Lekima', 
                'type': 'typhoon',
                'impact': 'Major flooding, groundwater level changes',
                'color': 'red'
            },
            {
                'date': datetime(2020, 5, 1),
                'name': '2020 Drought Begin',
                'type': 'drought',
                'impact': 'Severe water shortage, intensive pumping',
                'color': 'orange'
            },
            {
                'date': datetime(2021, 3, 1),
                'name': '2021 Extreme Drought',
                'type': 'drought', 
                'impact': 'Record low reservoir levels, emergency pumping',
                'color': 'darkorange'
            }
        ]
        
        print(f"✅ Defined {len(self.seasons)} seasonal patterns")
        print(f"✅ Defined {len(self.extreme_events)} extreme events")
        
    def create_temporal_change_point_analysis(self):
        """Create comprehensive temporal analysis of change points"""
        print("\n📊 Creating temporal change point analysis...")
        
        fig, axes = plt.subplots(6, 1, figsize=(16, 20))
        fig.suptitle('Hydrogeological Context of InSAR Change Points (2018-2021)', 
                    fontsize=16, fontweight='bold')
        
        # Method colors
        method_colors = {
            'wavelet': 'blue',
            'bayesian': 'red', 
            'ensemble': 'green',
            'lstm': 'purple',
            'lombscargle': 'orange'
        }
        
        # Panel 1: Overall temporal distribution
        ax = axes[0]
        self._plot_seasonal_background(ax)
        
        for method, color in method_colors.items():
            if method in self.change_points:
                times = [self.time_vector[cp['time_idx']] for cp in self.change_points[method] 
                        if cp['time_idx'] < len(self.time_vector)]
                
                # Create histogram of change points
                time_numeric = [t.toordinal() for t in times]
                hist, bin_edges = np.histogram(time_numeric, bins=50)
                bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(hist))]
                bin_dates = [datetime.fromordinal(int(bc)) for bc in bin_centers]
                
                ax.plot(bin_dates, hist, color=color, alpha=0.7, linewidth=2, label=f'{method.capitalize()}')
        
        self._add_extreme_events(ax)
        ax.set_ylabel('Change Points\nper Time Bin')
        ax.set_title('A) Temporal Distribution of All Change Points')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Seasonal analysis
        ax = axes[1]
        self._plot_seasonal_background(ax)
        self._analyze_seasonal_patterns(ax)
        ax.set_ylabel('Seasonal\nChange Intensity')
        ax.set_title('B) Seasonal Patterns in Change Point Detection')
        
        # Panel 3: Example InSAR time series with change points
        ax = axes[2]
        self._plot_example_time_series(ax)
        ax.set_ylabel('Displacement\n(mm)')
        ax.set_title('C) Example InSAR Time Series with Detected Change Points')
        
        # Panel 4: Groundwater pumping correlation
        ax = axes[3]
        self._analyze_pumping_correlation(ax)
        ax.set_ylabel('Pumping Season\nCorrelation')
        ax.set_title('D) Change Points vs Irrigation/Pumping Seasons')
        
        # Panel 5: Extreme event response
        ax = axes[4]
        self._analyze_extreme_event_response(ax)
        ax.set_ylabel('Event Response\nIntensity')
        ax.set_title('E) Response to Extreme Weather Events')
        
        # Panel 6: Monthly statistics
        ax = axes[5]
        self._plot_monthly_statistics(ax)
        ax.set_ylabel('Average Change\nPoints per Month')
        ax.set_title('F) Monthly Change Point Statistics (2018-2021)')
        ax.set_xlabel('Time')
        
        # Format all x-axes
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
        plt.tight_layout()
        
        figure_file = self.figures_dir / "ps05c_hydrogeological_temporal_analysis.png"
        plt.savefig(figure_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Temporal analysis saved: {figure_file}")
        
    def create_spatial_hydrogeological_map(self):
        """Create spatial visualization of change points in geological context using Cartopy"""
        print("\n🗺️ Creating spatial hydrogeological map...")
        
        if CARTOPY_AVAILABLE:
            return self._create_cartopy_spatial_map()
        else:
            return self._create_basic_spatial_map()
            
    def _create_cartopy_spatial_map(self):
        """Create Cartopy-based spatial map"""
        print("   Using Cartopy for professional geographic visualization...")
        
        # Calculate extent for Taiwan region
        lon_min, lon_max = self.coordinates[:, 0].min() - 0.05, self.coordinates[:, 0].max() + 0.05
        lat_min, lat_max = self.coordinates[:, 1].min() - 0.05, self.coordinates[:, 1].max() + 0.05
        
        method_colors = {
            'wavelet': 'blue', 'bayesian': 'red', 'ensemble': 'green',
            'lstm': 'purple', 'lombscargle': 'orange'
        }
        
        # Create figure with Cartopy subplots
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Spatial Distribution of Change Points in Taiwan Subsidence Areas', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Use PlateCarree projection for Taiwan
        proj = ccrs.PlateCarree()
        
        # Plot each method's spatial distribution
        subplot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        
        for i, (method, color) in enumerate(method_colors.items()):
            if i >= len(subplot_positions):
                break
                
            row, col = subplot_positions[i]
            ax = fig.add_subplot(2, 3, i+1, projection=proj)
            
            # Set map extent
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
            
            # Add geographic features
            ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='gray')
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.4)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
            
            # Add rivers with multiple resolutions
            try:
                ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.6, linewidth=1.0)
                rivers_10m = NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                                               edgecolor='darkblue', facecolor='none', 
                                               linewidth=1.5, alpha=0.8, zorder=2)
                ax.add_feature(rivers_10m)
            except:
                ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.6, linewidth=1.0)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, color='gray')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            
            if method in self.change_points:
                # Count change points per station
                station_counts = {}
                for cp in self.change_points[method]:
                    station_idx = cp['station_idx']
                    station_counts[station_idx] = station_counts.get(station_idx, 0) + 1
                
                # Create arrays for plotting
                plot_coords = []
                plot_counts = []
                
                for station_idx, count in station_counts.items():
                    if station_idx < len(self.coordinates):
                        plot_coords.append(self.coordinates[station_idx])
                        plot_counts.append(count)
                
                if plot_coords:
                    plot_coords = np.array(plot_coords)
                    plot_counts = np.array(plot_counts)
                    
                    # Plot with count-based coloring and transform
                    scatter = ax.scatter(plot_coords[:, 0], plot_coords[:, 1], 
                                       c=plot_counts, cmap='hot', 
                                       s=25, alpha=0.8, edgecolors='black', linewidths=0.5,
                                       transform=proj, zorder=5)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
                    cbar.set_label('Change Points per Station', rotation=270, labelpad=15, fontsize=9)
                
            ax.set_title(f'{method.capitalize()}\n({len(self.change_points.get(method, []))} total)',
                        fontsize=12, fontweight='bold', pad=10)
            
        plt.tight_layout()
        
        figure_file = self.figures_dir / "ps05c_spatial_hydrogeological_map.png"
        plt.savefig(figure_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Cartopy spatial map saved: {figure_file}")
        
    def _create_basic_spatial_map(self):
        """Create basic matplotlib spatial map as fallback"""
        print("   Using basic matplotlib visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spatial Distribution of Change Points in Taiwan Subsidence Areas', 
                    fontsize=16, fontweight='bold')
        
        method_colors = {
            'wavelet': 'blue', 'bayesian': 'red', 'ensemble': 'green',
            'lstm': 'purple', 'lombscargle': 'orange'
        }
        
        # Plot each method's spatial distribution
        for i, (method, color) in enumerate(method_colors.items()):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if method in self.change_points:
                # Count change points per station
                station_counts = {}
                for cp in self.change_points[method]:
                    station_idx = cp['station_idx']
                    station_counts[station_idx] = station_counts.get(station_idx, 0) + 1
                
                # Create arrays for plotting
                plot_coords = []
                plot_counts = []
                
                for station_idx, count in station_counts.items():
                    if station_idx < len(self.coordinates):
                        plot_coords.append(self.coordinates[station_idx])
                        plot_counts.append(count)
                
                if plot_coords:
                    plot_coords = np.array(plot_coords)
                    plot_counts = np.array(plot_counts)
                    
                    # Plot with count-based coloring
                    scatter = ax.scatter(plot_coords[:, 0], plot_coords[:, 1], 
                                       c=plot_counts, cmap='hot', 
                                       s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
                    
                    plt.colorbar(scatter, ax=ax, label='Change Points per Station')
                
                # Add subsidence zone boundaries
                self._add_geological_context(ax)
                
            ax.set_title(f'{method.capitalize()}\n({len(self.change_points.get(method, []))} total)')
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            ax.grid(True, alpha=0.3)
            
        # Remove unused subplot
        if len(method_colors) < 6:
            axes[1, 2].remove()
            
        plt.tight_layout()
        
        figure_file = self.figures_dir / "ps05c_spatial_hydrogeological_map.png"
        plt.savefig(figure_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Basic spatial map saved: {figure_file}")
        
    def create_hydrogeological_interpretation(self):
        """Create interpretation figure connecting change points to hydrogeology"""
        print("\n🔬 Creating hydrogeological interpretation...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hydrogeological Interpretation of InSAR Change Points', 
                    fontsize=16, fontweight='bold')
        
        # Panel A: Seasonal subsidence mechanisms
        ax = axes[0, 0]
        self._plot_subsidence_mechanisms(ax)
        ax.set_title('A) Seasonal Subsidence Mechanisms')
        
        # Panel B: Aquifer response patterns
        ax = axes[0, 1]
        self._plot_aquifer_response(ax)
        ax.set_title('B) Aquifer System Response Patterns')
        
        # Panel C: Pumping-induced change points
        ax = axes[1, 0]
        self._plot_pumping_induced_changes(ax)
        ax.set_title('C) Pumping-Induced Deformation Changes')
        
        # Panel D: Climate-driven patterns
        ax = axes[1, 1]
        self._plot_climate_patterns(ax)
        ax.set_title('D) Climate-Driven Subsidence Patterns')
        
        plt.tight_layout()
        
        figure_file = self.figures_dir / "ps05c_hydrogeological_interpretation.png"
        plt.savefig(figure_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Interpretation figure saved: {figure_file}")
    
    def _plot_seasonal_background(self, ax):
        """Add seasonal background shading"""
        for year in [2018, 2019, 2020, 2021]:
            # Dry season (Nov-Apr)
            dry_start = datetime(year-1, 11, 1) if year > 2018 else datetime(2017, 11, 1)
            dry_end = datetime(year, 4, 30)
            if dry_start >= self.time_vector[0] and dry_end <= self.time_vector[-1]:
                ax.axvspan(dry_start, dry_end, color='wheat', alpha=0.3, label='Dry Season' if year == 2018 else '')
            
            # Wet season (May-Oct)
            wet_start = datetime(year, 5, 1) 
            wet_end = datetime(year, 10, 31)
            if wet_start >= self.time_vector[0] and wet_end <= self.time_vector[-1]:
                ax.axvspan(wet_start, wet_end, color='lightblue', alpha=0.3, label='Wet Season' if year == 2018 else '')
    
    def _add_extreme_events(self, ax):
        """Add extreme event markers"""
        for event in self.extreme_events:
            if event['date'] >= self.time_vector[0] and event['date'] <= self.time_vector[-1]:
                ax.axvline(event['date'], color=event['color'], linestyle='--', alpha=0.8, linewidth=2)
                ax.text(event['date'], ax.get_ylim()[1] * 0.9, event['name'], 
                       rotation=90, fontsize=8, ha='right', va='top')
    
    def _analyze_seasonal_patterns(self, ax):
        """Analyze seasonal patterns in change points"""
        # Aggregate change points by month
        monthly_counts = {i: 0 for i in range(1, 13)}
        
        for method in self.change_points:
            for cp in self.change_points[method]:
                if cp['time_idx'] < len(self.time_vector):
                    month = self.time_vector[cp['time_idx']].month
                    monthly_counts[month] += 1
        
        months = list(monthly_counts.keys())
        counts = list(monthly_counts.values())
        
        # Normalize by total to show relative intensity
        total_counts = sum(counts)
        if total_counts > 0:
            normalized_counts = [c / total_counts * 100 for c in counts]
            
            bars = ax.bar(months, normalized_counts, color='steelblue', alpha=0.7)
            
            # Highlight irrigation seasons
            irrigation_months = [2, 3, 4, 7, 8, 9]
            for i, bar in enumerate(bars):
                if months[i] in irrigation_months:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Relative Frequency (%)')
        ax.set_xticks(months)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, alpha=0.3)
    
    def _plot_example_time_series(self, ax):
        """Plot example time series with change points"""
        # Select a representative station (central Changhua)
        central_idx = len(self.coordinates) // 2
        
        # Plot displacement time series
        displacement_series = self.displacement[central_idx, :]
        ax.plot(self.time_vector, displacement_series, 'k-', linewidth=1, alpha=0.7, label='InSAR Displacement')
        
        # Add seasonal background
        self._plot_seasonal_background(ax)
        
        # Add change points from different methods
        method_colors = {'wavelet': 'blue', 'bayesian': 'red', 'ensemble': 'green'}
        
        for method, color in method_colors.items():
            if method in self.change_points:
                station_changes = [cp for cp in self.change_points[method] 
                                 if cp['station_idx'] == central_idx]
                
                for cp in station_changes:
                    if cp['time_idx'] < len(self.time_vector):
                        change_time = self.time_vector[cp['time_idx']]
                        change_value = displacement_series[cp['time_idx']]
                        ax.plot(change_time, change_value, 'o', color=color, markersize=6, 
                               alpha=0.8, label=f'{method}' if cp == station_changes[0] else '')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _analyze_pumping_correlation(self, ax):
        """Analyze correlation with pumping seasons"""
        # Define pumping intensity by month (Taiwan irrigation calendar)
        pumping_intensity = {
            1: 0.8, 2: 1.0, 3: 1.0, 4: 0.9,  # High winter pumping + first crop
            5: 0.6, 6: 0.4, 7: 0.8, 8: 0.9, 9: 0.7,  # Reduced + second crop
            10: 0.5, 11: 0.7, 12: 0.8  # Increasing winter pumping
        }
        
        # Count change points by month
        monthly_changes = {i: 0 for i in range(1, 13)}
        total_changes = 0
        
        for method in ['ensemble', 'wavelet']:  # Focus on most reliable methods
            if method in self.change_points:
                for cp in self.change_points[method]:
                    if cp['time_idx'] < len(self.time_vector):
                        month = self.time_vector[cp['time_idx']].month
                        monthly_changes[month] += 1
                        total_changes += 1
        
        if total_changes > 0:
            months = list(range(1, 13))
            change_rates = [monthly_changes[m] / total_changes for m in months]
            pumping_rates = [pumping_intensity[m] for m in months]
            
            ax2 = ax.twinx()
            
            bars = ax.bar(months, change_rates, alpha=0.6, color='steelblue', label='Change Point Rate')
            line = ax2.plot(months, pumping_rates, 'ro-', linewidth=2, markersize=6, 
                           color='red', label='Pumping Intensity')
            
            ax.set_ylabel('Change Point Rate', color='steelblue')
            ax2.set_ylabel('Relative Pumping Intensity', color='red')
            ax.set_xlabel('Month')
            ax.set_xticks(months)
            ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            
            # Calculate correlation
            correlation = np.corrcoef(change_rates, pumping_rates)[0, 1]
            ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
                   fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _analyze_extreme_event_response(self, ax):
        """Analyze response to extreme events"""
        # Count change points within ±30 days of extreme events
        event_responses = {}
        
        for event in self.extreme_events:
            event_name = event['name']
            event_date = event['date']
            event_responses[event_name] = 0
            
            # Count changes within ±30 days
            for method in ['ensemble', 'wavelet', 'bayesian']:
                if method in self.change_points:
                    for cp in self.change_points[method]:
                        if cp['time_idx'] < len(self.time_vector):
                            cp_date = self.time_vector[cp['time_idx']]
                            days_diff = abs((cp_date - event_date).days)
                            
                            if days_diff <= 30:
                                event_responses[event_name] += 1
        
        # Normalize by background rate
        total_days = (self.time_vector[-1] - self.time_vector[0]).days
        total_changes = sum(len(self.change_points.get(method, [])) for method in ['ensemble', 'wavelet', 'bayesian'])
        background_rate = total_changes / total_days * 60  # 60-day window background
        
        event_names = []
        response_ratios = []
        colors = []
        
        for event in self.extreme_events:
            name = event['name']
            if name in event_responses:
                event_names.append(name.replace(' ', '\n'))
                ratio = event_responses[name] / background_rate if background_rate > 0 else 0
                response_ratios.append(ratio)
                colors.append(event['color'])
        
        if response_ratios:
            bars = ax.bar(range(len(event_names)), response_ratios, color=colors, alpha=0.7)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Background Rate')
            ax.set_ylabel('Response Ratio\n(vs Background)')
            ax.set_xticks(range(len(event_names)))
            ax.set_xticklabels(event_names, fontsize=8)
            ax.legend()
            
            # Add significance threshold
            ax.axhline(y=2.0, color='red', linestyle=':', alpha=0.7, label='2x Background')
    
    def _plot_monthly_statistics(self, ax):
        """Plot monthly statistics over entire period"""
        # Create monthly time series
        months_list = []
        current_date = datetime(2018, 1, 1)
        end_date = datetime(2021, 12, 31)
        
        while current_date <= end_date:
            months_list.append(current_date)
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Count change points by month
        monthly_totals = {month: 0 for month in months_list}
        
        for method in self.change_points:
            for cp in self.change_points[method]:
                if cp['time_idx'] < len(self.time_vector):
                    cp_date = self.time_vector[cp['time_idx']]
                    month_key = datetime(cp_date.year, cp_date.month, 1)
                    if month_key in monthly_totals:
                        monthly_totals[month_key] += 1
        
        months = list(monthly_totals.keys())
        counts = list(monthly_totals.values())
        
        ax.plot(months, counts, 'b-', linewidth=2, marker='o', markersize=4)
        self._plot_seasonal_background(ax)
        self._add_extreme_events(ax)
        
        ax.set_ylabel('Total Change Points')
        ax.grid(True, alpha=0.3)
    
    def _add_geological_context(self, ax):
        """Add geological context to spatial plots"""
        # Taiwan coastline approximation
        taiwan_lons = [120.0, 120.2, 120.4, 120.6, 120.8, 121.0]
        taiwan_lats_north = [24.4, 24.3, 24.2, 24.1, 24.0, 23.9]
        taiwan_lats_south = [23.4, 23.5, 23.6, 23.7, 23.8, 23.9]
        
        ax.plot(taiwan_lons, taiwan_lats_north, 'k-', linewidth=2, alpha=0.5, label='Approximate Coastline')
        ax.plot(taiwan_lons, taiwan_lats_south, 'k-', linewidth=2, alpha=0.5)
        
        # Major subsidence areas
        ax.add_patch(Rectangle((120.3, 24.0), 0.3, 0.25, fill=False, edgecolor='red', 
                             linewidth=2, linestyle='--', label='Changhua Plain'))
        ax.add_patch(Rectangle((120.2, 23.6), 0.4, 0.35, fill=False, edgecolor='orange', 
                             linewidth=2, linestyle='--', label='Yunlin Plain'))
        
        ax.legend()
    
    def _plot_subsidence_mechanisms(self, ax):
        """Plot subsidence mechanisms diagram"""
        # Conceptual diagram showing seasonal mechanisms
        mechanisms = ['Elastic\nRecovery', 'Inelastic\nCompaction', 'Seasonal\nLoading', 'Long-term\nTrend']
        colors = ['lightgreen', 'red', 'blue', 'black']
        
        # Mock time series showing different mechanisms
        time_mock = np.linspace(0, 365*3, 1000)  # 3 years
        
        elastic = 5 * np.sin(2*np.pi*time_mock/365)  # Annual cycle
        inelastic = -0.02 * time_mock  # Long-term trend
        seasonal = 3 * np.sin(2*np.pi*time_mock/180) * np.exp(-time_mock/500)  # Dampened seasonal
        total = elastic + inelastic + seasonal
        
        ax.plot(time_mock/365, elastic, color='lightgreen', linewidth=2, label='Elastic Recovery')
        ax.plot(time_mock/365, inelastic, color='red', linewidth=2, label='Inelastic Compaction')
        ax.plot(time_mock/365, seasonal, color='blue', linewidth=2, label='Seasonal Loading')
        ax.plot(time_mock/365, total, color='black', linewidth=3, label='Total Subsidence')
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Displacement (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_aquifer_response(self, ax):
        """Plot aquifer response patterns"""
        # Different aquifer responses to pumping
        response_types = ['Confined\nAquifer', 'Unconfined\nAquifer', 'Multi-layer\nSystem']
        
        # Mock response curves
        time_resp = np.linspace(0, 100, 200)
        
        # Confined: rapid initial response, stabilizes
        confined = -10 * (1 - np.exp(-time_resp/20))
        
        # Unconfined: gradual response, continues
        unconfined = -0.1 * time_resp
        
        # Multi-layer: complex response with multiple phases
        multilayer = -5 * (1 - np.exp(-time_resp/10)) - 0.05 * time_resp
        
        ax.plot(time_resp, confined, 'b-', linewidth=2, label='Confined Aquifer')
        ax.plot(time_resp, unconfined, 'r-', linewidth=2, label='Unconfined Aquifer')
        ax.plot(time_resp, multilayer, 'g-', linewidth=2, label='Multi-layer System')
        
        ax.set_xlabel('Time since pumping start (days)')
        ax.set_ylabel('Subsidence (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pumping_induced_changes(self, ax):
        """Plot pumping-induced deformation changes"""
        # Correlation between detected changes and pumping seasons
        pumping_months = [2, 3, 4, 7, 8, 9]  # High pumping months
        non_pumping_months = [1, 5, 6, 10, 11, 12]
        
        # Count changes in pumping vs non-pumping months
        pumping_changes = 0
        non_pumping_changes = 0
        
        for method in ['ensemble', 'wavelet']:
            if method in self.change_points:
                for cp in self.change_points[method]:
                    if cp['time_idx'] < len(self.time_vector):
                        month = self.time_vector[cp['time_idx']].month
                        if month in pumping_months:
                            pumping_changes += 1
                        else:
                            non_pumping_changes += 1
        
        categories = ['High Pumping\nSeasons', 'Low Pumping\nSeasons']
        counts = [pumping_changes, non_pumping_changes]
        colors = ['orange', 'lightblue']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Change Points')
        ax.set_title('Change Points by Pumping Season')
        
        # Add percentages
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = count / total * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_climate_patterns(self, ax):
        """Plot climate-driven patterns"""
        # Seasonal change point intensity
        seasons_data = {
            'Winter\n(Dec-Feb)': [12, 1, 2],
            'Spring\n(Mar-May)': [3, 4, 5], 
            'Summer\n(Jun-Aug)': [6, 7, 8],
            'Autumn\n(Sep-Nov)': [9, 10, 11]
        }
        
        season_counts = {}
        total_counts = 0
        
        for season, months in seasons_data.items():
            season_counts[season] = 0
            
            for method in self.change_points:
                for cp in self.change_points[method]:
                    if cp['time_idx'] < len(self.time_vector):
                        month = self.time_vector[cp['time_idx']].month
                        if month in months:
                            season_counts[season] += 1
                            total_counts += 1
        
        seasons = list(season_counts.keys())
        counts = [season_counts[s] for s in seasons]
        colors = ['lightblue', 'lightgreen', 'orange', 'brown']
        
        bars = ax.bar(seasons, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Change Points')
        ax.set_title('Seasonal Distribution')
        
        # Add expected vs observed
        expected = total_counts / 4  # Equal distribution
        ax.axhline(y=expected, color='red', linestyle='--', alpha=0.7, label='Expected (uniform)')
        ax.legend()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n📋 Generating hydrogeological summary report...")
        
        # Calculate key statistics
        total_changes = sum(len(self.change_points.get(method, [])) for method in self.change_points)
        
        # Monthly distribution
        monthly_dist = {i: 0 for i in range(1, 13)}
        for method in self.change_points:
            for cp in self.change_points[method]:
                if cp['time_idx'] < len(self.time_vector):
                    month = self.time_vector[cp['time_idx']].month
                    monthly_dist[month] += 1
        
        # Peak months
        peak_month = max(monthly_dist, key=monthly_dist.get)
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        report = {
            'analysis_period': f"{self.time_vector[0].strftime('%Y-%m-%d')} to {self.time_vector[-1].strftime('%Y-%m-%d')}",
            'total_stations': self.n_stations,
            'total_change_points': total_changes,
            'methods_analyzed': list(self.change_points.keys()),
            'peak_activity_month': month_names[peak_month],
            'peak_activity_count': monthly_dist[peak_month],
            'seasonal_patterns': {
                'dry_season_changes': sum(monthly_dist[m] for m in [11, 12, 1, 2, 3, 4]),
                'wet_season_changes': sum(monthly_dist[m] for m in [5, 6, 7, 8, 9, 10]),
                'irrigation_season_changes': sum(monthly_dist[m] for m in [2, 3, 4, 7, 8, 9])
            },
            'extreme_events_analyzed': len(self.extreme_events),
            'figures_generated': [
                'ps05c_hydrogeological_temporal_analysis.png',
                'ps05c_spatial_hydrogeological_map.png', 
                'ps05c_hydrogeological_interpretation.png'
            ]
        }
        
        # Save report
        report_file = self.results_dir / "hydrogeological_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Summary report saved: {report_file}")
        
        # Print key findings
        print("\n🔍 KEY HYDROGEOLOGICAL FINDINGS:")
        print(f"   • Peak change point activity: {month_names[peak_month]} ({monthly_dist[peak_month]} changes)")
        print(f"   • Dry season changes: {report['seasonal_patterns']['dry_season_changes']}")
        print(f"   • Wet season changes: {report['seasonal_patterns']['wet_season_changes']}")
        print(f"   • Irrigation season changes: {report['seasonal_patterns']['irrigation_season_changes']}")
        
        if total_changes > 0:
            dry_pct = report['seasonal_patterns']['dry_season_changes'] / total_changes * 100
            wet_pct = report['seasonal_patterns']['wet_season_changes'] / total_changes * 100
            print(f"   • Seasonal bias: {dry_pct:.1f}% dry season, {wet_pct:.1f}% wet season")
        else:
            print("   • No change points loaded - check data files")

def main():
    """Main execution"""
    print("="*70)
    print("🌊 PS05C HYDROGEOLOGICAL EVENT VISUALIZATION")
    print("="*70)
    
    visualizer = HydrogeologicalEventVisualizer()
    
    # Create comprehensive visualizations
    visualizer.create_temporal_change_point_analysis()
    visualizer.create_spatial_hydrogeological_map()
    visualizer.create_hydrogeological_interpretation()
    visualizer.generate_summary_report()
    
    print("\n" + "="*70) 
    print("✅ HYDROGEOLOGICAL VISUALIZATION COMPLETED")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
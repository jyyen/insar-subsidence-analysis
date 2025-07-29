#!/usr/bin/env python3
"""
ps04f_synchronized_anomaly_analysis.py - Synchronized Anomaly Detection & Visualization

Purpose: Detect anomalies that occur simultaneously across multiple InSAR time series
         and visualize them both temporally and geographically

Key Features:
- Multi-station synchronized anomaly detection
- Time series visualization with anomaly markup
- Geographic distribution of anomalous events
- Statistical analysis of synchronized anomalies
- Taiwan-specific temporal and spatial analysis

Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path
import json
import argparse
from datetime import datetime, timedelta
from scipy import stats, signal
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("✅ Cartopy available for geographic visualization")
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not available. Using matplotlib for geographic plots.")

class SynchronizedAnomalyAnalysis:
    """
    Detect and visualize anomalies that occur simultaneously across multiple InSAR stations
    """
    
    def __init__(self, min_station_threshold=0.3, anomaly_zscore=2.5, time_window_days=12):
        """
        Initialize synchronized anomaly detection
        
        Parameters:
        -----------
        min_station_threshold : float
            Minimum fraction of stations that must show anomaly for synchronization (0.3 = 30%)
        anomaly_zscore : float
            Z-score threshold for individual station anomaly detection
        time_window_days : int
            Time window in days for considering anomalies as synchronized
        """
        self.min_station_threshold = min_station_threshold
        self.anomaly_zscore = anomaly_zscore
        self.time_window_days = time_window_days
        
        # Results storage
        self.time_series = None
        self.coordinates = None
        self.dates = None
        self.synchronized_anomalies = []
        self.station_anomalies = {}
        
        # Ensure output directories exist
        self.results_dir = Path("data/processed/ps04f_synchronized")
        self.figures_dir = Path("figures")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Synchronized Anomaly Analysis Initialized")
        print(f"   Min station threshold: {min_station_threshold:.1%}")
        print(f"   Anomaly Z-score: {anomaly_zscore}")
        print(f"   Time window: {time_window_days} days")
    
    def load_time_series_data(self, n_stations=500):
        """Load InSAR time series data for anomaly analysis"""
        
        print("📡 Loading time series data for synchronized anomaly analysis...")
        
        # Load preprocessed data
        data = np.load('data/processed/ps00_preprocessed_data.npz')
        
        # Get full dataset
        coordinates = data['coordinates']  # (7154, 2)
        displacement = data['displacement']  # (7154, 215)
        subsidence_rates = data['subsidence_rates']  # (7154,)
        
        print(f"✅ Loaded full dataset: {displacement.shape[0]} stations x {displacement.shape[1]} time points")
        
        # Select subset of stations (highest subsidence rates for better anomaly visibility)
        significant_mask = np.abs(subsidence_rates) >= 10.0  # 10 mm/year threshold
        significant_indices = np.where(significant_mask)[0]
        
        if len(significant_indices) >= n_stations:
            # Sort by absolute subsidence rate and take top N
            sorted_indices = significant_indices[np.argsort(-np.abs(subsidence_rates[significant_indices]))]
            selected_indices = sorted_indices[:n_stations]
        else:
            # Take all significant + random fill
            remaining = n_stations - len(significant_indices)
            other_indices = np.where(~significant_mask)[0]
            additional = np.random.choice(other_indices, remaining, replace=False)
            selected_indices = np.concatenate([significant_indices, additional])
        
        # Apply selection
        self.coordinates = coordinates[selected_indices]
        self.time_series = displacement[selected_indices]
        
        # Create time array (6-day sampling, starting from 2018)
        start_date = datetime(2018, 1, 1)
        self.dates = [start_date + timedelta(days=i*6) for i in range(displacement.shape[1])]
        
        print(f"✅ Selected {len(selected_indices)} stations for analysis")
        print(f"   Geographic bounds: {np.min(self.coordinates[:, 0]):.3f}°E to {np.max(self.coordinates[:, 0]):.3f}°E")
        print(f"                      {np.min(self.coordinates[:, 1]):.3f}°N to {np.max(self.coordinates[:, 1]):.3f}°N")
        print(f"   Time span: {self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')}")
        
        return True
    
    def detect_station_anomalies(self):
        """Detect anomalies for each individual station"""
        
        print("🔍 Detecting anomalies for individual stations...")
        
        n_stations, n_time = self.time_series.shape
        
        for station_idx in range(n_stations):
            ts = self.time_series[station_idx]
            
            # Method 1: Statistical outliers (Z-score)
            z_scores = np.abs(zscore(ts))
            statistical_anomalies = np.where(z_scores > self.anomaly_zscore)[0]
            
            # Method 2: Isolation Forest for complex anomalies
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            ts_reshaped = ts.reshape(-1, 1)
            isolation_anomalies = np.where(iso_forest.fit_predict(ts_reshaped) == -1)[0]
            
            # Method 3: Rate of change anomalies
            velocity = np.gradient(ts)
            velocity_z = np.abs(zscore(velocity))
            velocity_anomalies = np.where(velocity_z > self.anomaly_zscore)[0]
            
            # Combine all anomaly types
            all_anomalies = np.unique(np.concatenate([
                statistical_anomalies, 
                isolation_anomalies, 
                velocity_anomalies
            ]))
            
            self.station_anomalies[station_idx] = {
                'statistical': statistical_anomalies,
                'isolation': isolation_anomalies,
                'velocity': velocity_anomalies,
                'combined': all_anomalies,
                'z_scores': z_scores,
                'velocity': velocity
            }
        
        print(f"✅ Anomaly detection completed for {n_stations} stations")
        
        # Statistics
        total_anomalies = sum(len(data['combined']) for data in self.station_anomalies.values())
        avg_anomalies = total_anomalies / n_stations
        print(f"   Total anomalies detected: {total_anomalies}")
        print(f"   Average per station: {avg_anomalies:.1f}")
        
        return True
    
    def detect_synchronized_anomalies(self):
        """Detect anomalies that occur simultaneously across multiple stations"""
        
        print("🔗 Detecting synchronized anomalies across stations...")
        
        n_stations, n_time = self.time_series.shape
        min_stations = int(n_stations * self.min_station_threshold)
        
        print(f"   Minimum {min_stations} stations ({self.min_station_threshold:.1%}) must show anomaly for synchronization")
        
        # Create anomaly matrix (stations x time)
        anomaly_matrix = np.zeros((n_stations, n_time), dtype=bool)
        
        for station_idx, anomaly_data in self.station_anomalies.items():
            anomaly_indices = anomaly_data['combined']
            anomaly_matrix[station_idx, anomaly_indices] = True
        
        # Find time points with sufficient station anomalies
        stations_with_anomalies = np.sum(anomaly_matrix, axis=0)
        synchronized_times = np.where(stations_with_anomalies >= min_stations)[0]
        
        print(f"   Found {len(synchronized_times)} time points with synchronized anomalies")
        
        # Group nearby synchronized times into events
        if len(synchronized_times) > 0:
            time_window_samples = self.time_window_days // 6  # Convert days to samples (6-day sampling)
            
            # Group consecutive or nearby time indices
            event_groups = []
            current_group = [synchronized_times[0]]
            
            for i in range(1, len(synchronized_times)):
                if synchronized_times[i] - synchronized_times[i-1] <= time_window_samples:
                    current_group.append(synchronized_times[i])
                else:
                    event_groups.append(current_group)
                    current_group = [synchronized_times[i]]
            
            event_groups.append(current_group)  # Add the last group
            
            # Process each synchronized event
            for event_id, event_times in enumerate(event_groups):
                event_start = min(event_times)
                event_end = max(event_times)
                event_duration = (event_end - event_start + 1) * 6  # Convert to days
                
                # Find stations involved in this event
                involved_stations = []
                for time_idx in event_times:
                    station_indices = np.where(anomaly_matrix[:, time_idx])[0]
                    involved_stations.extend(station_indices)
                
                involved_stations = np.unique(involved_stations)
                
                # Calculate event statistics
                event_coords = self.coordinates[involved_stations]
                event_center = np.mean(event_coords, axis=0)
                
                # Spatial extent (maximum distance from center)
                distances = np.sqrt(np.sum((event_coords - event_center)**2, axis=1)) * 111  # Convert degrees to km
                spatial_extent = np.max(distances) if len(distances) > 0 else 0
                
                # Temporal characteristics
                event_displacement_change = []
                for station_idx in involved_stations:
                    ts = self.time_series[station_idx]
                    if event_end < len(ts) and event_start >= 0:
                        change = ts[event_end] - ts[event_start]
                        event_displacement_change.append(change)
                
                synchronized_event = {
                    'event_id': event_id,
                    'start_time_idx': event_start,
                    'end_time_idx': event_end,
                    'start_date': self.dates[event_start],
                    'end_date': self.dates[event_end],
                    'duration_days': event_duration,
                    'involved_stations': involved_stations.tolist(),
                    'n_stations': len(involved_stations),
                    'station_fraction': len(involved_stations) / n_stations,
                    'center_coordinate': event_center.tolist(),
                    'spatial_extent_km': spatial_extent,
                    'displacement_changes': event_displacement_change,
                    'mean_displacement_change': np.mean(event_displacement_change) if event_displacement_change else 0,
                    'time_indices': event_times
                }
                
                self.synchronized_anomalies.append(synchronized_event)
        
        print(f"✅ Synchronized anomaly detection completed")
        print(f"   {len(self.synchronized_anomalies)} synchronized events detected")
        
        if self.synchronized_anomalies:
            durations = [event['duration_days'] for event in self.synchronized_anomalies]
            station_counts = [event['n_stations'] for event in self.synchronized_anomalies]
            extents = [event['spatial_extent_km'] for event in self.synchronized_anomalies]
            
            print(f"   Event duration: {np.mean(durations):.1f} ± {np.std(durations):.1f} days")
            print(f"   Stations per event: {np.mean(station_counts):.1f} ± {np.std(station_counts):.1f}")
            print(f"   Spatial extent: {np.mean(extents):.1f} ± {np.std(extents):.1f} km")
        
        return True
    
    def create_time_series_visualization(self):
        """Create time series plots with synchronized anomalies marked"""
        
        print("📊 Creating time series visualization with anomaly markup...")
        
        if not self.synchronized_anomalies:
            print("⚠️  No synchronized anomalies to visualize")
            return None
        
        # Select representative stations for visualization
        n_display_stations = min(20, len(self.coordinates))
        display_indices = np.linspace(0, len(self.coordinates)-1, n_display_stations, dtype=int)
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        fig.suptitle('InSAR Time Series with Synchronized Anomalies\nTaiwan Subsidence Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Create date array for x-axis
        date_array = np.array(self.dates)
        
        for i, station_idx in enumerate(display_indices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ts = self.time_series[station_idx]
            coords = self.coordinates[station_idx]
            
            # Plot time series
            ax.plot(date_array, ts, 'b-', linewidth=1, alpha=0.7, label='Displacement')
            
            # Mark individual station anomalies (light background)
            if station_idx in self.station_anomalies:
                individual_anomalies = self.station_anomalies[station_idx]['combined']
                for anomaly_idx in individual_anomalies:
                    ax.axvline(date_array[anomaly_idx], color='lightcoral', alpha=0.3, linewidth=1)
            
            # Mark synchronized anomalies (prominent highlighting)
            for event in self.synchronized_anomalies:
                if station_idx in event['involved_stations']:
                    # Highlight the entire event duration
                    start_date = event['start_date']
                    end_date = event['end_date']
                    
                    # Add colored background for event duration
                    ax.axvspan(start_date, end_date, alpha=0.3, color='red', 
                              label=f"Event {event['event_id']}" if i == 0 else "")
                    
                    # Mark start and end with vertical lines
                    ax.axvline(start_date, color='red', linewidth=2, alpha=0.8)
                    ax.axvline(end_date, color='darkred', linewidth=2, alpha=0.8)
                    
                    # Add text annotation for significant events
                    if event['n_stations'] > len(self.coordinates) * 0.5:  # Major event
                        mid_date = start_date + (end_date - start_date) / 2
                        y_pos = np.max(ts) * 0.9
                        ax.text(mid_date, y_pos, f"E{event['event_id']}", 
                               ha='center', va='center', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Formatting
            ax.set_title(f"Station {station_idx}\n({coords[0]:.3f}°E, {coords[1]:.3f}°N)", 
                        fontsize=10)
            ax.set_ylabel('Displacement (mm)', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        if self.synchronized_anomalies:
            axes[0].legend(loc='upper left', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(display_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps04f_fig01_synchronized_anomalies_timeseries.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Time series visualization saved: {fig_path}")
        return str(fig_path)
    
    def create_geographic_visualization(self):
        """Create geographic map showing synchronized anomaly locations"""
        
        print("🗺️  Creating geographic visualization of synchronized anomalies...")
        
        if not self.synchronized_anomalies:
            print("⚠️  No synchronized anomalies to visualize")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        fig.suptitle('Geographic Distribution of Synchronized Anomalies\nTaiwan InSAR Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Get coordinate bounds
        lon_min, lon_max = np.min(self.coordinates[:, 0]), np.max(self.coordinates[:, 0])
        lat_min, lat_max = np.min(self.coordinates[:, 1]), np.max(self.coordinates[:, 1])
        
        # Plot 1: All stations with anomaly frequency
        ax1 = axes[0, 0]
        ax1.set_title('Station Anomaly Frequency', fontsize=12, fontweight='bold')
        
        # Calculate anomaly frequency per station
        anomaly_counts = []
        for i in range(len(self.coordinates)):
            if i in self.station_anomalies:
                count = len(self.station_anomalies[i]['combined'])
            else:
                count = 0
            anomaly_counts.append(count)
        
        scatter1 = ax1.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                              c=anomaly_counts, cmap='Reds', s=30, alpha=0.7)
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label('Individual Anomaly Count')
        
        # Plot 2: Synchronized event locations
        ax2 = axes[0, 1]
        ax2.set_title('Synchronized Event Centers', fontsize=12, fontweight='bold')
        
        # Plot all stations as background
        ax2.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                   c='lightgray', s=10, alpha=0.5, label='All Stations')
        
        # Plot event centers
        for i, event in enumerate(self.synchronized_anomalies):
            center = event['center_coordinate']
            extent = event['spatial_extent_km']
            n_stations = event['n_stations']
            
            # Size by number of involved stations, color by spatial extent
            scatter2 = ax2.scatter(center[0], center[1], 
                                  s=n_stations*2, c=extent, cmap='viridis',
                                  alpha=0.8, edgecolors='black', linewidth=1,
                                  label=f"Event {i}" if i < 5 else "")
            
            # Add event ID annotation
            ax2.annotate(f'E{i}', (center[0], center[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='white')
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Event temporal distribution
        ax3 = axes[1, 0]
        ax3.set_title('Event Temporal Distribution', fontsize=12, fontweight='bold')
        
        event_dates = [event['start_date'] for event in self.synchronized_anomalies]
        event_durations = [event['duration_days'] for event in self.synchronized_anomalies]
        
        scatter3 = ax3.scatter(event_dates, range(len(event_dates)), 
                              c=event_durations, cmap='plasma', s=60, alpha=0.8)
        cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
        cbar3.set_label('Duration (days)')
        
        ax3.set_ylabel('Event ID')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        # Format dates
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Event characteristics
        ax4 = axes[1, 1]
        ax4.set_title('Event Characteristics', fontsize=12, fontweight='bold')
        
        extents = [event['spatial_extent_km'] for event in self.synchronized_anomalies]
        station_fractions = [event['station_fraction'] * 100 for event in self.synchronized_anomalies]
        
        scatter4 = ax4.scatter(extents, station_fractions, 
                              c=event_durations, cmap='coolwarm', s=80, alpha=0.8)
        cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.8)
        cbar4.set_label('Duration (days)')
        
        ax4.set_xlabel('Spatial Extent (km)')
        ax4.set_ylabel('Station Involvement (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add annotations for major events
        for i, event in enumerate(self.synchronized_anomalies):
            if event['station_fraction'] > 0.3:  # Major events
                ax4.annotate(f'E{i}', (extents[i], station_fractions[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
        
        # Set geographic extents for spatial plots
        for ax in [ax1, ax2]:
            ax.set_xlim(lon_min - 0.01, lon_max + 0.01)
            ax.set_ylim(lat_min - 0.01, lat_max + 0.01)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / "ps04f_fig02_synchronized_anomalies_geographic.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Geographic visualization saved: {fig_path}")
        return str(fig_path)
    
    def create_summary_report(self):
        """Create summary report of synchronized anomalies"""
        
        print("📋 Creating synchronized anomaly summary report...")
        
        summary = {
            'analysis_parameters': {
                'min_station_threshold': self.min_station_threshold,
                'anomaly_zscore': self.anomaly_zscore,
                'time_window_days': self.time_window_days,
                'n_stations_analyzed': len(self.coordinates),
                'time_span_days': len(self.dates) * 6,
                'analysis_date': datetime.now().isoformat()
            },
            'detection_summary': {
                'total_synchronized_events': len(self.synchronized_anomalies),
                'total_individual_anomalies': sum(len(data['combined']) for data in self.station_anomalies.values()),
                'stations_with_anomalies': len([s for s, data in self.station_anomalies.items() if len(data['combined']) > 0])
            },
            'synchronized_events': []
        }
        
        for event in self.synchronized_anomalies:
            event_summary = {
                'event_id': event['event_id'],
                'start_date': event['start_date'].isoformat(),
                'end_date': event['end_date'].isoformat(),
                'duration_days': event['duration_days'],
                'involved_stations': len(event['involved_stations']),
                'station_fraction': event['station_fraction'],
                'center_coordinate': event['center_coordinate'],
                'spatial_extent_km': event['spatial_extent_km'],
                'mean_displacement_change': event['mean_displacement_change'],
                'significance': 'Major' if event['station_fraction'] > 0.5 else 'Moderate' if event['station_fraction'] > 0.3 else 'Minor'
            }
            summary['synchronized_events'].append(event_summary)
        
        # Save summary
        summary_path = self.results_dir / "synchronized_anomaly_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Summary report saved: {summary_path}")
        
        # Print key findings
        if self.synchronized_anomalies:
            major_events = [e for e in self.synchronized_anomalies if e['station_fraction'] > 0.5]
            moderate_events = [e for e in self.synchronized_anomalies if 0.3 < e['station_fraction'] <= 0.5]
            
            print(f"\n📊 KEY FINDINGS:")
            print(f"   🔴 Major events (>50% stations): {len(major_events)}")
            print(f"   🟡 Moderate events (30-50% stations): {len(moderate_events)}")
            print(f"   🟢 Minor events (<30% stations): {len(self.synchronized_anomalies) - len(major_events) - len(moderate_events)}")
            
            if major_events:
                print(f"\n   📅 Major Event Dates:")
                for event in major_events:
                    print(f"      Event {event['event_id']}: {event['start_date'].strftime('%Y-%m-%d')} "
                          f"({event['n_stations']} stations, {event['spatial_extent_km']:.1f} km extent)")
        
        return str(summary_path)
    
    def run_analysis(self, n_stations=500):
        """Run complete synchronized anomaly analysis"""
        
        print("🚀 Starting synchronized anomaly analysis...")
        print("="*80)
        
        try:
            # Load data
            self.load_time_series_data(n_stations)
            
            # Detect individual station anomalies
            self.detect_station_anomalies()
            
            # Detect synchronized anomalies
            self.detect_synchronized_anomalies()
            
            # Create visualizations
            ts_fig = self.create_time_series_visualization()
            geo_fig = self.create_geographic_visualization()
            
            # Create summary report
            summary_report = self.create_summary_report()
            
            print("\n" + "="*80)
            print("✅ Synchronized anomaly analysis completed successfully!")
            print(f"📊 Generated visualizations:")
            if ts_fig:
                print(f"   1. Time series with anomalies: {ts_fig}")
            if geo_fig:
                print(f"   2. Geographic anomaly distribution: {geo_fig}")
            if summary_report:
                print(f"   3. Summary report: {summary_report}")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in synchronized anomaly analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main analysis function"""
    
    parser = argparse.ArgumentParser(description='Synchronized Anomaly Analysis')
    parser.add_argument('--n-stations', type=int, default=300, 
                       help='Number of stations to analyze (default: 300)')
    parser.add_argument('--min-threshold', type=float, default=0.3, 
                       help='Minimum station fraction for synchronization (default: 0.3)')
    parser.add_argument('--zscore', type=float, default=2.5, 
                       help='Z-score threshold for anomaly detection (default: 2.5)')
    parser.add_argument('--time-window', type=int, default=12, 
                       help='Time window in days for synchronization (default: 12)')
    
    args = parser.parse_args()
    
    print("🔍 Synchronized Anomaly Analysis for Taiwan InSAR Data")
    print(f"📊 Configuration: {args.n_stations} stations, {args.min_threshold:.1%} threshold, "
          f"Z-score {args.zscore}, {args.time_window}d window")
    
    # Initialize and run analysis
    analyzer = SynchronizedAnomalyAnalysis(
        min_station_threshold=args.min_threshold,
        anomaly_zscore=args.zscore,
        time_window_days=args.time_window
    )
    
    success = analyzer.run_analysis(n_stations=args.n_stations)
    
    if success:
        print("\n✅ Analysis completed! Check the figures directory for visualizations.")
    else:
        print("\n❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()
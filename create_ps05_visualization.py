#!/usr/bin/env python3
"""
create_ps05_visualization.py - Quick visualization for ps05 event detection results
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print("📊 Creating ps05 event detection visualization...")
    
    # Load existing ps05 results
    results_dir = Path("data/processed/ps05_events")
    figures_dir = Path("figures")
    
    # Load event catalog
    event_file = results_dir / "event_catalog_emd.json"
    if not event_file.exists():
        print(f"❌ Event catalog not found: {event_file}")
        return
    
    with open(event_file, 'r') as f:
        event_data = json.load(f)
    
    print(f"✅ Loaded {event_data['total_events']} events from EMD analysis")
    
    # Extract event information
    events = event_data['event_details']
    
    # Extract coordinates and event types
    lons = [event['coordinates'][0] for event in events]
    lats = [event['coordinates'][1] for event in events]
    event_types = [event['detection_type'] for event in events]
    time_days = [int(event['time_days']) for event in events]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Taiwan InSAR Event Detection Results (EMD Method)', fontsize=16, fontweight='bold')
    
    # Subplot 1: Geographic distribution of events
    ax1 = axes[0, 0]
    scatter = ax1.scatter(lons, lats, c=time_days, cmap='viridis', s=20, alpha=0.6)
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.set_title(f'Geographic Distribution of Events (n={len(events)})')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Event Time (days)', rotation=270, labelpad=15)
    
    # Subplot 2: Event types
    ax2 = axes[0, 1]
    event_type_counts = {}
    for event_type in event_types:
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
    
    bars = ax2.bar(event_type_counts.keys(), event_type_counts.values(), 
                   color=['lightblue', 'lightcoral', 'lightgreen'][:len(event_type_counts)])
    ax2.set_xlabel('Event Type')
    ax2.set_ylabel('Number of Events')
    ax2.set_title('Event Types Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Subplot 3: Temporal distribution
    ax3 = axes[1, 0]
    ax3.hist(time_days, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Number of Events')
    ax3.set_title('Temporal Distribution of Events')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Event summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    total_events = len(events)
    unique_stations = len(set(event['station_idx'] for event in events))
    time_span = max(time_days) - min(time_days)
    
    # Get z-scores and rate magnitudes if available
    z_scores = []
    rate_magnitudes = []
    for event in events:
        if 'z_score' in event:
            z_scores.append(abs(event['z_score']))
        if 'rate_magnitude' in event:
            rate_magnitudes.append(abs(event['rate_magnitude']))
    
    summary_text = f"""
Event Detection Summary (EMD Method)

📊 Total Events: {total_events:,}
🎯 Affected Stations: {unique_stations:,}
⏱️  Time Span: {time_span:.0f} days
📍 Station Coverage: {unique_stations/7154*100:.1f}% of dataset

Event Type Breakdown:
"""
    
    for event_type, count in event_type_counts.items():
        percentage = count/total_events*100
        summary_text += f"• {event_type.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)\n"
    
    if z_scores:
        summary_text += f"\n📈 Average Z-Score: {np.mean(z_scores):.2f}"
    if rate_magnitudes:
        summary_text += f"\n📉 Average Rate Change: {np.mean(rate_magnitudes):.1f} mm/year"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
             facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_file = figures_dir / "ps05_fig01_event_detection_overview.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_file}")
    
    plt.show()
    
    # Create a second figure for anomalies if available
    anomaly_file = results_dir / "anomalies_emd.json"
    if anomaly_file.exists():
        with open(anomaly_file, 'r') as f:
            anomaly_data = json.load(f)
        
        print(f"✅ Found {anomaly_data['total_anomalies']} anomalies")
        
        # Simple anomaly visualization
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        anomalies = anomaly_data['anomaly_details']
        anomaly_lons = [anom['coordinates'][0] for anom in anomalies]
        anomaly_lats = [anom['coordinates'][1] for anom in anomalies]
        anomaly_scores = [anom['anomaly_score'] for anom in anomalies]
        
        scatter = ax.scatter(anomaly_lons, anomaly_lats, c=anomaly_scores, 
                           cmap='Reds', s=30, alpha=0.7)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_title(f'Anomalous Stations (n={len(anomalies)})')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Anomaly Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Save anomaly figure
        anomaly_output = figures_dir / "ps05_fig02_anomaly_detection.png"
        plt.savefig(anomaly_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved anomaly visualization: {anomaly_output}")
        
        plt.show()

if __name__ == "__main__":
    main()
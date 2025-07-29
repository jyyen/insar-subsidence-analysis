#!/usr/bin/env python3
"""Quick test of 100-station analysis"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load data
try:
    data_file = Path("data/processed/ps00_preprocessed_data.npz")
    data = np.load(data_file)
    displacement = data['displacement']
    coordinates = data['coordinates']
    
    print(f"✅ Loaded {displacement.shape[0]} stations, {displacement.shape[1]} time points")
    
    # Create time vector
    time_vector = np.arange(0, displacement.shape[1] * 6, 6)
    
    # Select 10 random stations for quick test
    np.random.seed(42)
    valid_stations = []
    for i in range(displacement.shape[0]):
        if np.sum(~np.isnan(displacement[i, :])) > 150:  # At least 150 valid points
            valid_stations.append(i)
    
    if len(valid_stations) < 10:
        print(f"Only {len(valid_stations)} valid stations found")
        selected = valid_stations
    else:
        selected = np.random.choice(valid_stations, 10, replace=False)
    
    print(f"📍 Selected {len(selected)} stations for test")
    
    # Quick visualization of selected signals
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    
    for i, station_idx in enumerate(selected):
        signal = displacement[station_idx, :]
        
        # Interpolate NaN values
        valid_mask = ~np.isnan(signal)
        if np.any(~valid_mask):
            signal = np.interp(np.arange(len(signal)), 
                             np.where(valid_mask)[0], 
                             signal[valid_mask])
        
        ax.plot(time_vector, signal, color=colors[i], alpha=0.7, 
               label=f'Station {station_idx}', linewidth=1.5)
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Displacement (mm)')
    ax.set_title(f'Sample InSAR Time Series - {len(selected)} Stations')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save test figure
    save_path = Path("figures") / "test_selected_stations.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved test figure: {save_path}")
    print("✅ Data loading and basic processing working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
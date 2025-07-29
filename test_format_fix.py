#!/usr/bin/env python3
"""
Test script to verify text formatting in matplotlib plots
"""

import matplotlib.pyplot as plt
import numpy as np

# Create test data
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'b-', linewidth=2, label='Test Signal')

# Test text formatting with newlines
rmse = 1.98
corr = 0.998
snr = 10.7

# Method 1: f-string with \n
stats_text1 = f'RMSE: {rmse:.2f} mm\nCorr: {corr:.3f}\nSNR: {snr:.1f}'

# Method 2: Using raw string 
stats_text2 = f'RMSE: {rmse:.2f} mm\nCorr: {corr:.3f}\nSNR: {snr:.1f}'

# Method 3: Multi-line string
stats_text3 = f"""RMSE: {rmse:.2f} mm
Corr: {corr:.3f}
SNR: {snr:.1f}"""

print("Testing text formatting:")
print("Method 1 (f-string):", repr(stats_text1))
print("Method 2 (raw):", repr(stats_text2))
print("Method 3 (multi-line):", repr(stats_text3))

# Add text to plot
ax.text(0.02, 0.98, stats_text3, transform=ax.transAxes,
        verticalalignment='top', fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Text Formatting Test')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('insar_subsidence_analysis/figures/text_format_test.png', dpi=300, bbox_inches='tight')
plt.show()

print("Test plot saved to figures/text_format_test.png")
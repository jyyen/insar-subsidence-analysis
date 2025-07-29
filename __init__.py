"""
InSAR Subsidence Analysis Package

A comprehensive Python implementation for analyzing Taiwan subsidence using 
PSInSAR time series decomposition and clustering methods.

This package provides a unified framework for:
- Multi-method signal decomposition (EMD, FFT, VMD, Wavelet)
- Advanced clustering and validation
- Geological correlation with borehole data
- Advanced interpolation methods (spline, kriging)
- Publication-ready visualization
- GMT/pyGMT integration

Author: Claude Code (Anthropic)
Date: 2025-01-19
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude Code (Anthropic)"
__email__ = "claude@anthropic.com"

# Core imports for easy access
from .core import InSARDataLoader, LNJSCorrection
from .decomposition import EMD, FFT, VMD, Wavelet
from .analysis import PCAClusteringAnalysis, ValidationMetrics
from .visualization import FigureManager, GeographicVisualizer
from .interpolation import SplineInterpolation, KrigingInterpolation

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """Return the version string."""
    return __version__

def get_author():
    """Return the author information."""
    return __author__

# Package-level configuration
DEFAULT_CONFIG = {
    'dataset': 'taiwan_2018_2021',
    'methods': ['emd', 'fft', 'vmd', 'wavelet'],
    'subsample_factor': 1000,
    'reference_station': 'LNJS',
    'parallel_processing': True,
    'figure_formats': ['png', 'pdf'],
    'gmt_export': True
}

# Import order for proper initialization
__all__ = [
    # Version info
    '__version__',
    'get_version',
    'get_author',
    'VERSION_INFO',
    
    # Core functionality
    'InSARDataLoader',
    'LNJSCorrection',
    
    # Decomposition methods
    'EMD',
    'FFT', 
    'VMD',
    'Wavelet',
    
    # Analysis tools
    'PCAClusteringAnalysis',
    'ValidationMetrics',
    
    # Visualization
    'FigureManager',
    'GeographicVisualizer',
    
    # Interpolation
    'SplineInterpolation',
    'KrigingInterpolation',
    
    # Configuration
    'DEFAULT_CONFIG'
]
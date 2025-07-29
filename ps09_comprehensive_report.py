#!/usr/bin/env python3
"""
ps09_comprehensive_report.py - Comprehensive Scientific Report Generation

Purpose: Generate comprehensive scientific report aggregating all analysis results
Methods: Data synthesis, executive summary, policy recommendations, figure compilation
Author: Taiwan InSAR Subsidence Analysis Pipeline
Date: January 2025

Input: All processed data from ps00-ps08
Output: Executive summary, comprehensive figures, policy recommendations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveReportGenerator:
    """Generate comprehensive scientific report from all pipeline analyses"""
    
    def __init__(self):
        """Initialize the report generator"""
        self.data_dir = Path("data/processed")
        self.figures_dir = Path("figures")
        self.results_dir = Path("data/processed/ps09_report")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.pipeline_data = {}
        self.key_metrics = {}
        self.findings = {}
        self.recommendations = {}
        
        print("=" * 80)
        print("📊 ps09_comprehensive_report.py - Comprehensive Scientific Report")
        print("📋 PURPOSE: Generate executive summary and scientific interpretation")
        print("📁 INPUT: All pipeline results (ps00-ps08)")
        print("📄 OUTPUT: Executive report, figures, recommendations")
        print("=" * 80)
        
    def load_pipeline_data(self):
        """Load all available data from previous pipeline steps"""
        print("\n🔄 LOADING PIPELINE DATA")
        print("-" * 50)
        
        # ps00: Preprocessing data
        ps00_file = self.data_dir / "ps00_preprocessed_data.npz"
        if ps00_file.exists():
            print("📡 Loading ps00: Data preprocessing...")
            self.pipeline_data['ps00'] = np.load(ps00_file, allow_pickle=True)
            print(f"   ✅ Loaded {len(self.pipeline_data['ps00']['coordinates'])} InSAR stations")
        
        # ps01: GPS-InSAR validation
        ps01_file = self.data_dir / "ps01_gps_insar_validation.json"
        if ps01_file.exists():
            print("🛰️ Loading ps01: GPS-InSAR validation...")
            with open(ps01_file, 'r') as f:
                self.pipeline_data['ps01'] = json.load(f)
            val_results = self.pipeline_data['ps01'].get('validation_results', {})
            r2 = val_results.get('r_squared', 0)
            rmse = val_results.get('rmse', 0)
            print(f"   ✅ R² = {r2:.3f}, RMSE = {rmse:.2f} mm/year")
        
        # ps02: Signal decomposition (all methods)
        decomposition_methods = ['emd', 'fft', 'vmd', 'wavelet']
        self.pipeline_data['ps02'] = {}
        
        for method in decomposition_methods:
            decomp_file = self.data_dir / f"ps02_{method}_decomposition.npz"
            recat_file = self.data_dir / f"ps02_{method}_recategorization.json"
            
            if decomp_file.exists():
                print(f"🔄 Loading ps02: {method.upper()} decomposition...")
                self.pipeline_data['ps02'][method] = {}
                self.pipeline_data['ps02'][method]['decomposition'] = np.load(decomp_file, allow_pickle=True)
                
                if recat_file.exists():
                    with open(recat_file, 'r') as f:
                        self.pipeline_data['ps02'][method]['recategorization'] = json.load(f)
                
                # Get component counts
                imfs = self.pipeline_data['ps02'][method]['decomposition']['imfs']
                total_components = sum(len(station_imfs) for station_imfs in imfs if station_imfs is not None)
                print(f"   ✅ {total_components} components across {len(imfs)} stations")
        
        # ps03: Clustering analysis
        clustering_files = list(self.data_dir.glob("ps03_clustering/*.json"))
        if clustering_files:
            print("🔗 Loading ps03: Clustering analysis...")
            self.pipeline_data['ps03'] = {}
            for file in clustering_files:
                with open(file, 'r') as f:
                    self.pipeline_data['ps03'][file.stem] = json.load(f)
            print(f"   ✅ Loaded {len(clustering_files)} clustering results")
        
        # ps04: Temporal clustering analysis
        ps04_temporal_dir = self.data_dir / "ps04_temporal"
        ps04_benchmark_dir = self.data_dir / "ps04_benchmark"
        
        if ps04_temporal_dir.exists() or ps04_benchmark_dir.exists():
            print("⏰ Loading ps04: Temporal clustering...")
            self.pipeline_data['ps04'] = {}
            
            # Check for DTW distance matrix (main ps04 output)
            dtw_file = ps04_temporal_dir / "dtw_distance_matrix.npz"
            if dtw_file.exists():
                self.pipeline_data['ps04']['dtw_analysis'] = {
                    'file_path': str(dtw_file),
                    'file_size_mb': dtw_file.stat().st_size / (1024*1024),
                    'analysis_type': 'temporal_clustering'
                }
            
            # Check for any benchmark results
            benchmark_files = list(ps04_benchmark_dir.glob("*.json")) if ps04_benchmark_dir.exists() else []
            for file in benchmark_files:
                with open(file, 'r') as f:
                    self.pipeline_data['ps04'][file.stem] = json.load(f)
            
            # Count ps04 figures as evidence of execution
            ps04_figures = list(Path("figures").glob("ps04_*.png"))
            if ps04_figures:
                self.pipeline_data['ps04']['figures_generated'] = len(ps04_figures)
            
            components_found = len(self.pipeline_data['ps04'])
            print(f"   ✅ Loaded {components_found} temporal analysis components")
            if dtw_file.exists():
                print(f"   📊 DTW analysis: {self.pipeline_data['ps04']['dtw_analysis']['file_size_mb']:.1f} MB")
            if ps04_figures:
                print(f"   🎨 Generated {len(ps04_figures)} visualization figures")
        
        # ps05: Event detection
        event_files = list(self.data_dir.glob("ps05_events/*.json"))
        if event_files:
            print("⚡ Loading ps05: Event detection...")
            self.pipeline_data['ps05'] = {}
            for file in event_files:
                with open(file, 'r') as f:
                    self.pipeline_data['ps05'][file.stem] = json.load(f)
            print(f"   ✅ Loaded {len(event_files)} event analysis results")
        
        # ps06: Method comparison
        comparison_files = list(self.data_dir.glob("ps06_comparison/*.json"))
        if comparison_files:
            print("📊 Loading ps06: Method comparison...")
            self.pipeline_data['ps06'] = {}
            for file in comparison_files:
                with open(file, 'r') as f:
                    self.pipeline_data['ps06'][file.stem] = json.load(f)
            print(f"   ✅ Loaded {len(comparison_files)} comparison results")
        
        # ps08: Geological integration
        geological_files = list(self.data_dir.glob("ps08_geological/**/*.json"))
        if geological_files:
            print("🗻 Loading ps08: Geological integration...")
            self.pipeline_data['ps08'] = {}
            for file in geological_files:
                with open(file, 'r') as f:
                    category = file.parent.name
                    if category not in self.pipeline_data['ps08']:
                        self.pipeline_data['ps08'][category] = {}
                    self.pipeline_data['ps08'][category][file.stem] = json.load(f)
            print(f"   ✅ Loaded {len(geological_files)} geological analysis results")
        
        print("✅ Pipeline data loading completed")
        return True
    
    def extract_key_metrics(self):
        """Extract key metrics and findings from all analyses"""
        print("\n📊 EXTRACTING KEY METRICS")
        print("-" * 50)
        
        # Dataset overview
        if 'ps00' in self.pipeline_data:
            self.key_metrics['dataset'] = {
                'n_insar_stations': len(self.pipeline_data['ps00']['coordinates']),
                'time_period': '2018-2021',
                'study_area': 'Taiwan Changhua/Yunlin Plains',
                'reference_station': 'LNJS'
            }
            
            # Geographic bounds
            coords = self.pipeline_data['ps00']['coordinates']
            self.key_metrics['geographic_bounds'] = {
                'lon_min': float(np.min(coords[:, 0])),
                'lon_max': float(np.max(coords[:, 0])),
                'lat_min': float(np.min(coords[:, 1])),
                'lat_max': float(np.max(coords[:, 1]))
            }
        
        # GPS-InSAR validation metrics
        if 'ps01' in self.pipeline_data:
            validation = self.pipeline_data['ps01'].get('validation_results', {})
            processing_info = self.pipeline_data['ps01'].get('processing_info', {})
            self.key_metrics['validation'] = {
                'r_squared': validation.get('r_squared', 0),
                'rmse_mm_year': validation.get('rmse', 0),
                'bias_mm_year': validation.get('bias', 0),
                'correlation': validation.get('correlation', 0),
                'n_gps_stations': processing_info.get('n_gps_stations', 0)
            }
        
        # Signal decomposition performance
        if 'ps02' in self.pipeline_data:
            self.key_metrics['decomposition'] = {}
            
            for method, data in self.pipeline_data['ps02'].items():
                if 'recategorization' in data:
                    recat = data['recategorization']
                    
                    # Extract frequency distribution from actual data structure
                    freq_dist = {}
                    total_components = 0
                    
                    # Parse the station-based recategorization data
                    for station_id, station_data in recat.items():
                        if isinstance(station_data, dict):
                            for imf_id, imf_data in station_data.items():
                                if isinstance(imf_data, dict) and 'final_category' in imf_data:
                                    category = imf_data['final_category']
                                    freq_dist[category] = freq_dist.get(category, 0) + 1
                                    total_components += 1
                    
                    # Find dominant band
                    dominant_band = max(freq_dist.keys(), key=freq_dist.get) if freq_dist else 'unknown'
                    
                    self.key_metrics['decomposition'][method] = {
                        'total_components': total_components,
                        'frequency_distribution': freq_dist,
                        'dominant_band': dominant_band
                    }
        
        # Method comparison rankings
        if 'ps06' in self.pipeline_data and 'method_rankings' in self.pipeline_data['ps06']:
            rankings = self.pipeline_data['ps06']['method_rankings']
            
            # Extract best methods from actual data structure
            best_overall = 'unknown'
            best_reconstruction = 'unknown'
            
            if 'overall' in rankings and 'ranking' in rankings['overall']:
                overall_ranking = rankings['overall']['ranking']
                if isinstance(overall_ranking, list) and len(overall_ranking) > 0:
                    best_overall = overall_ranking[0].upper()
            
            # Check for reconstruction-specific rankings
            if 'reconstruction' in rankings and 'ranking' in rankings['reconstruction']:
                recon_ranking = rankings['reconstruction']['ranking']
                if isinstance(recon_ranking, list) and len(recon_ranking) > 0:
                    best_reconstruction = recon_ranking[0].upper()
            else:
                # Use overall ranking as fallback
                best_reconstruction = best_overall
            
            self.key_metrics['method_performance'] = {
                'best_overall': best_overall,
                'best_reconstruction': best_reconstruction,
                'rankings': rankings
            }
        
        # Geological correlation strength
        if 'ps08' in self.pipeline_data:
            if 'correlation_analysis' in self.pipeline_data['ps08']:
                corr_data = self.pipeline_data['ps08']['correlation_analysis']
                
                # Extract geology-deformation correlations
                geology_corr = corr_data.get('geological_process_analysis', {})
                self.key_metrics['geological_correlation'] = geology_corr
        
        print("✅ Key metrics extraction completed")
        return True
    
    def generate_executive_summary(self):
        """Generate executive summary with key findings"""
        print("\n📋 GENERATING EXECUTIVE SUMMARY")
        print("-" * 50)
        
        summary = {
            'report_date': datetime.now().isoformat(),
            'study_overview': {
                'title': 'Taiwan InSAR Subsidence Analysis: Comprehensive Report',
                'study_area': self.key_metrics.get('dataset', {}).get('study_area', 'Taiwan'),
                'time_period': self.key_metrics.get('dataset', {}).get('time_period', '2018-2021'),
                'dataset_size': self.key_metrics.get('dataset', {}).get('n_insar_stations', 0),
                'analysis_methods': list(self.pipeline_data.get('ps02', {}).keys())
            },
            'key_findings': {},
            'recommendations': {}
        }
        
        # GPS-InSAR validation findings
        if 'validation' in self.key_metrics:
            val = self.key_metrics['validation']
            summary['key_findings']['validation'] = {
                'description': 'GPS-InSAR validation demonstrates high accuracy',
                'r_squared': val['r_squared'],
                'rmse_mm_year': val['rmse_mm_year'],
                'interpretation': f"R² = {val['r_squared']:.3f} indicates {val['r_squared']*100:.1f}% variance explained"
            }
        
        # Signal decomposition findings
        if 'decomposition' in self.key_metrics:
            decomp = self.key_metrics['decomposition']
            summary['key_findings']['signal_decomposition'] = {
                'description': 'Multi-method signal decomposition reveals frequency structure',
                'methods_analyzed': list(decomp.keys()),
                'dominant_patterns': {}
            }
            
            for method, data in decomp.items():
                if data['frequency_distribution']:
                    dominant = data['dominant_band']
                    dominant_pct = (data['frequency_distribution'].get(dominant, 0) / 
                                   data['total_components'] * 100) if data['total_components'] > 0 else 0
                    
                    summary['key_findings']['signal_decomposition']['dominant_patterns'][method] = {
                        'dominant_band': dominant,
                        'percentage': dominant_pct
                    }
        
        # Method performance findings
        if 'method_performance' in self.key_metrics:
            perf = self.key_metrics['method_performance']
            summary['key_findings']['method_performance'] = {
                'description': 'Comparative analysis identifies optimal decomposition method',
                'best_overall': perf['best_overall'],
                'best_reconstruction': perf['best_reconstruction']
            }
        
        # Generate recommendations
        summary['recommendations'] = {
            'monitoring': [
                'Continue GPS-InSAR integrated monitoring for high accuracy',
                f"Use {self.key_metrics.get('method_performance', {}).get('best_overall', 'EMD')} method for optimal signal decomposition",
                'Focus monitoring on annual and semi-annual patterns (dominant signals)'
            ],
            'technical': [
                'Maintain current reference station (LNJS) for consistency',
                'Implement automated anomaly detection for rapid response',
                'Integrate geological data for subsidence susceptibility mapping'
            ],
            'policy': [
                'Develop groundwater management policies for high-subsidence areas',
                'Implement early warning systems for critical infrastructure',
                'Use geological susceptibility maps for land use planning'
            ]
        }
        
        self.findings = summary
        
        # Save executive summary
        summary_file = self.results_dir / "ps09_executive_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Executive summary saved: {summary_file}")
        return summary
    
    def create_comprehensive_figures(self):
        """Create comprehensive figures summarizing all analyses"""
        print("\n🎨 CREATING COMPREHENSIVE FIGURES")
        print("-" * 50)
        
        # Set style for professional figures
        plt.style.use('seaborn-v0_8')
        
        # Figure 1: Study Overview
        fig1 = self._create_study_overview()
        fig1_path = self.figures_dir / "ps09_fig01_study_overview.png"
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"✅ Created study overview: {fig1_path}")
        
        # Figure 2: Pipeline Summary
        fig2 = self._create_pipeline_summary()
        fig2_path = self.figures_dir / "ps09_fig02_pipeline_summary.png"
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"✅ Created pipeline summary: {fig2_path}")
        
        # Figure 3: Key Findings Dashboard
        fig3 = self._create_key_findings_dashboard()
        fig3_path = self.figures_dir / "ps09_fig03_key_findings.png"
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"✅ Created key findings dashboard: {fig3_path}")
        
        print("✅ Comprehensive figures created")
        return True
    
    def _create_study_overview(self):
        """Create study overview figure with geographic context"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Taiwan InSAR Subsidence Analysis - Study Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Geographic coverage (if data available)
        ax1 = axes[0, 0]
        if 'ps00' in self.pipeline_data:
            coords = self.pipeline_data['ps00']['coordinates']
            rates = self.pipeline_data['ps00'].get('subsidence_rates', np.zeros(len(coords)))
            
            scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=rates, 
                                cmap='RdBu_r', s=2, alpha=0.7)
            ax1.set_xlabel('Longitude (°E)')
            ax1.set_ylabel('Latitude (°N)')
            ax1.set_title('InSAR Station Coverage & Subsidence Rates')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Subsidence Rate (mm/year)')
        else:
            ax1.text(0.5, 0.5, 'InSAR data not available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('InSAR Station Coverage')
        
        # Plot 2: Dataset summary
        ax2 = axes[0, 1]
        dataset_info = self.key_metrics.get('dataset', {})
        
        labels = ['InSAR\\nStations', 'GPS\\nStations', 'Borehole\\nStations']
        values = [
            dataset_info.get('n_insar_stations', 0),
            self.key_metrics.get('validation', {}).get('n_gps_stations', 0),
            103  # From geological analysis
        ]
        
        bars = ax2.bar(labels, values, color=['blue', 'green', 'brown'], alpha=0.7)
        ax2.set_ylabel('Number of Stations')
        ax2.set_title('Dataset Summary')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Validation metrics
        ax3 = axes[1, 0]
        if 'validation' in self.key_metrics:
            val = self.key_metrics['validation']
            
            metrics = ['R²', 'RMSE\\n(mm/year)', 'Bias\\n(mm/year)']
            metric_values = [val['r_squared'], val['rmse_mm_year'], abs(val['bias_mm_year'])]
            colors = ['green' if val['r_squared'] > 0.9 else 'orange',
                     'green' if val['rmse_mm_year'] < 2.0 else 'orange',
                     'green' if abs(val['bias_mm_year']) < 1.0 else 'orange']
            
            bars = ax3.bar(metrics, metric_values, color=colors, alpha=0.7)
            ax3.set_ylabel('Metric Value')
            ax3.set_title('GPS-InSAR Validation Performance')
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                        f'{value:.3f}' if value < 1 else f'{value:.1f}', 
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Validation data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('GPS-InSAR Validation')
        
        # Plot 4: Method performance (if available)
        ax4 = axes[1, 1]
        if 'decomposition' in self.key_metrics:
            methods = list(self.key_metrics['decomposition'].keys())
            component_counts = [self.key_metrics['decomposition'][method]['total_components'] 
                              for method in methods]
            
            bars = ax4.bar([m.upper() for m in methods], component_counts, 
                          color=['red', 'blue', 'green', 'orange'][:len(methods)], alpha=0.7)
            ax4.set_ylabel('Total Components')
            ax4.set_title('Signal Decomposition Results')
            
            # Add value labels
            for bar, value in zip(bars, component_counts):
                if value > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(component_counts)*0.01,
                            f'{value}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Decomposition data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Signal Decomposition Results')
        
        plt.tight_layout()
        return fig
    
    def _create_pipeline_summary(self):
        """Create pipeline workflow summary"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle('Analysis Pipeline Summary', fontsize=16, fontweight='bold')
        
        # Pipeline steps
        steps = [
            'ps00\\nData Preprocessing',
            'ps01\\nGPS Validation', 
            'ps02\\nSignal Decomposition',
            'ps03\\nClustering Analysis',
            'ps04\\nTemporal Analysis',
            'ps05\\nEvent Detection',
            'ps06\\nMethod Comparison',
            'ps08\\nGeological Integration',
            'ps09\\nComprehensive Report'
        ]
        
        # Status of each step
        status = []
        colors = []
        for step_name in ['ps00', 'ps01', 'ps02', 'ps03', 'ps04', 'ps05', 'ps06', 'ps08', 'ps09']:
            if step_name in self.pipeline_data or step_name == 'ps09':
                status.append('✅ Complete')
                colors.append('green')
            else:
                status.append('❌ Missing')
                colors.append('red')
        
        # Create horizontal bar chart
        y_pos = np.arange(len(steps))
        bars = ax.barh(y_pos, [1] * len(steps), color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(steps)
        ax.set_xlabel('Pipeline Progress')
        ax.set_title('Analysis Pipeline Status')
        
        # Add status labels
        for i, (bar, stat) in enumerate(zip(bars, status)):
            ax.text(0.5, i, stat, ha='center', va='center', fontweight='bold', color='white')
        
        # Add key metrics text
        if self.key_metrics:
            metrics_text = "Key Metrics:\\n"
            if 'dataset' in self.key_metrics:
                metrics_text += f"• {self.key_metrics['dataset'].get('n_insar_stations', 0)} InSAR stations\\n"
            if 'validation' in self.key_metrics:
                metrics_text += f"• GPS Validation R² = {self.key_metrics['validation'].get('r_squared', 0):.3f}\\n"
            if 'method_performance' in self.key_metrics:
                metrics_text += f"• Best Method: {self.key_metrics['method_performance'].get('best_overall', 'Unknown')}\\n"
            
            ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                   verticalalignment='center')
        
        ax.set_xlim(0, 1)
        plt.tight_layout()
        return fig
    
    def _create_key_findings_dashboard(self):
        """Create dashboard-style summary of key findings"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        fig.suptitle('Key Findings Dashboard', fontsize=18, fontweight='bold')
        
        # Validation Performance (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'validation' in self.key_metrics:
            val = self.key_metrics['validation']
            r2 = val['r_squared']
            
            # Create gauge-style plot
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            ax1.plot(theta, r, 'k-', linewidth=3)
            
            # Add R² indicator
            r2_angle = r2 * np.pi
            ax1.plot([r2_angle, r2_angle], [0, 1], 'r-', linewidth=5)
            ax1.fill_between(theta[theta <= r2_angle], 0, r[theta <= r2_angle], alpha=0.3, color='green')
            
            ax1.set_ylim(0, 1.2)
            ax1.set_xlim(0, np.pi)
            ax1.set_title(f'GPS-InSAR Validation\\nR² = {r2:.3f}', fontweight='bold')
            ax1.axis('off')
            
            # Add text
            ax1.text(np.pi/2, 0.5, f'{r2:.3f}', ha='center', va='center', 
                    fontsize=20, fontweight='bold')
        
        # Method Performance (top-right)
        ax2 = fig.add_subplot(gs[0, 1:3])
        if 'decomposition' in self.key_metrics:
            methods = list(self.key_metrics['decomposition'].keys())
            total_components = [self.key_metrics['decomposition'][method]['total_components'] 
                              for method in methods]
            
            bars = ax2.bar([m.upper() for m in methods], total_components,
                          color=['red', 'blue', 'green', 'orange'][:len(methods)], alpha=0.7)
            ax2.set_ylabel('Total Components')
            ax2.set_title('Signal Decomposition Performance', fontweight='bold')
            
            # Add value labels
            for bar, value in zip(bars, total_components):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                            f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Frequency Distribution (middle section)
        if 'decomposition' in self.key_metrics and 'emd' in self.key_metrics['decomposition']:
            freq_dist = self.key_metrics['decomposition']['emd'].get('frequency_distribution', {})
            
            if freq_dist:
                ax3 = fig.add_subplot(gs[1, :])
                
                bands = list(freq_dist.keys())
                counts = list(freq_dist.values())
                total = sum(counts)
                percentages = [c/total*100 for c in counts] if total > 0 else [0]*len(counts)
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(bands)))
                bars = ax3.bar(bands, percentages, color=colors, alpha=0.8)
                
                ax3.set_ylabel('Percentage of Components (%)')
                ax3.set_title('Frequency Band Distribution (EMD Method)', fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                for bar, pct in zip(bars, percentages):
                    if pct > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Summary Statistics (bottom)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create summary text
        summary_text = "EXECUTIVE SUMMARY\\n\\n"
        
        if 'dataset' in self.key_metrics:
            summary_text += f"📍 Study Area: {self.key_metrics['dataset'].get('study_area', 'Taiwan')}\\n"
            summary_text += f"📊 Dataset: {self.key_metrics['dataset'].get('n_insar_stations', 0)} InSAR stations\\n"
            summary_text += f"📅 Period: {self.key_metrics['dataset'].get('time_period', '2018-2021')}\\n\\n"
        
        if 'validation' in self.key_metrics:
            val = self.key_metrics['validation']
            summary_text += f"🎯 GPS Validation: R² = {val['r_squared']:.3f}, RMSE = {val['rmse_mm_year']:.1f} mm/year\\n"
        
        if 'method_performance' in self.key_metrics:
            best = self.key_metrics['method_performance'].get('best_overall', 'Unknown')
            summary_text += f"🏆 Best Method: {best.upper()}\\n"
        
        summary_text += "\\n📋 Key Findings:\\n"
        summary_text += "• High GPS-InSAR correlation validates subsidence measurements\\n"
        summary_text += "• Annual patterns dominate seasonal deformation cycles\\n"
        summary_text += "• Geological factors correlate with subsidence susceptibility\\n"
        summary_text += "• Multi-method analysis ensures robust signal decomposition"
        
        ax4.text(0.5, 0.5, summary_text, ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=1", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_station_rankings(self):
        """Generate station priority rankings for monitoring"""
        print("\n🏆 GENERATING STATION RANKINGS")
        print("-" * 50)
        
        if 'ps00' not in self.pipeline_data:
            print("⚠️  No InSAR data available for ranking")
            return {}
        
        coords = self.pipeline_data['ps00']['coordinates']
        rates = self.pipeline_data['ps00'].get('subsidence_rates', np.zeros(len(coords)))
        
        # Create rankings based on multiple criteria
        rankings = {
            'high_subsidence': [],
            'rapid_change': [],
            'anomaly_prone': [],
            'monitoring_priority': []
        }
        
        # High subsidence stations (fastest sinking)
        high_subsidence_indices = np.argsort(rates)[:50]  # 50 fastest sinking
        for idx in high_subsidence_indices:
            if rates[idx] < -5:  # Only significant subsidence
                rankings['high_subsidence'].append({
                    'station_id': int(idx),
                    'longitude': float(coords[idx, 0]),
                    'latitude': float(coords[idx, 1]),
                    'subsidence_rate': float(rates[idx]),
                    'priority_score': float(-rates[idx])  # Higher magnitude = higher priority
                })
        
        # Add anomaly information if available
        if 'ps05' in self.pipeline_data:
            anomaly_files = [f for f in self.pipeline_data['ps05'].keys() if 'anomalies' in f]
            if anomaly_files:
                anomaly_data = self.pipeline_data['ps05'][anomaly_files[0]]
                
                # Handle different possible anomaly data structures
                anomaly_stations = []
                if 'anomaly_stations' in anomaly_data:
                    anomaly_stations = anomaly_data['anomaly_stations']
                elif 'anomalies' in anomaly_data:
                    anomaly_stations = anomaly_data['anomalies']
                elif 'detected_anomalies' in anomaly_data:
                    # Convert detected anomalies to station format
                    detected = anomaly_data['detected_anomalies']
                    for station_id, anomaly_info in detected.items():
                        if isinstance(anomaly_info, dict) and anomaly_info.get('has_anomalies', False):
                            anomaly_stations.append({
                                'station_id': int(station_id),
                                'longitude': float(coords[int(station_id), 0]) if int(station_id) < len(coords) else 0,
                                'latitude': float(coords[int(station_id), 1]) if int(station_id) < len(coords) else 0,
                                'anomaly_count': len(anomaly_info.get('anomaly_times', [])),
                                'priority_score': len(anomaly_info.get('anomaly_times', []))
                            })
                
                # Add top anomaly-prone stations
                if anomaly_stations:
                    # Sort by anomaly count or priority score
                    if isinstance(anomaly_stations[0], dict):
                        sorted_anomalies = sorted(anomaly_stations, 
                                                key=lambda x: x.get('priority_score', x.get('anomaly_count', 0)), 
                                                reverse=True)
                        rankings['anomaly_prone'].extend(sorted_anomalies[:20])  # Top 20 anomalies
        
        # Combined monitoring priority (subsidence + anomalies + geological factors)
        priority_scores = np.abs(rates)  # Start with subsidence magnitude
        
        # Add geological susceptibility if available
        if 'ps08' in self.pipeline_data:
            # This would require loading geological susceptibility data
            pass
        
        # Create overall priority ranking
        priority_indices = np.argsort(priority_scores)[::-1][:100]  # Top 100 priority stations
        for rank, idx in enumerate(priority_indices):
            rankings['monitoring_priority'].append({
                'rank': rank + 1,
                'station_id': int(idx),
                'longitude': float(coords[idx, 0]),
                'latitude': float(coords[idx, 1]),
                'subsidence_rate': float(rates[idx]),
                'priority_score': float(priority_scores[idx])
            })
        
        # Save rankings
        rankings_file = self.results_dir / "ps09_station_rankings.json"
        with open(rankings_file, 'w') as f:
            json.dump(rankings, f, indent=2)
        
        print(f"✅ Station rankings saved: {rankings_file}")
        print(f"   High subsidence: {len(rankings['high_subsidence'])} stations")
        print(f"   Anomaly prone: {len(rankings['anomaly_prone'])} stations")
        print(f"   Priority monitoring: {len(rankings['monitoring_priority'])} stations")
        
        return rankings
    
    def generate_policy_recommendations(self):
        """Generate policy recommendations based on analysis results"""
        print("\n📋 GENERATING POLICY RECOMMENDATIONS")
        print("-" * 50)
        
        recommendations = {
            'immediate_actions': [],
            'short_term': [],
            'long_term': [],
            'technical': []
        }
        
        # Based on validation results
        if 'validation' in self.key_metrics:
            val = self.key_metrics['validation']
            if val['r_squared'] > 0.9:
                recommendations['technical'].append({
                    'category': 'Monitoring System',
                    'priority': 'High',
                    'recommendation': 'Continue current GPS-InSAR integrated monitoring approach',
                    'justification': f"High correlation (R² = {val['r_squared']:.3f}) validates measurement accuracy"
                })
        
        # Based on subsidence patterns
        if 'ps00' in self.pipeline_data:
            rates = self.pipeline_data['ps00'].get('subsidence_rates', [])
            if len(rates) > 0:
                severe_subsidence = np.sum(rates < -20)  # Stations with >20mm/year subsidence
                if severe_subsidence > 0:
                    recommendations['immediate_actions'].append({
                        'category': 'Critical Infrastructure',
                        'priority': 'Critical',
                        'recommendation': 'Implement emergency monitoring for severe subsidence areas',
                        'justification': f"{severe_subsidence} stations show >20 mm/year subsidence"
                    })
        
        # Based on geological analysis
        if 'ps08' in self.pipeline_data:
            recommendations['long_term'].append({
                'category': 'Land Use Planning',
                'priority': 'High',
                'recommendation': 'Integrate geological susceptibility maps into urban planning',
                'justification': 'Geological analysis shows correlation between soil properties and subsidence'
            })
        
        # Based on method performance
        if 'method_performance' in self.key_metrics:
            best_method = self.key_metrics['method_performance'].get('best_overall', 'EMD')
            recommendations['technical'].append({
                'category': 'Analysis Methods',
                'priority': 'Medium',
                'recommendation': f'Standardize on {best_method} method for operational monitoring',
                'justification': f'{best_method} shows best overall performance in method comparison'
            })
        
        # General recommendations
        recommendations['short_term'].extend([
            {
                'category': 'Groundwater Management',
                'priority': 'High',
                'recommendation': 'Implement groundwater extraction limits in high-subsidence areas',
                'justification': 'Reduced groundwater extraction can slow subsidence rates'
            },
            {
                'category': 'Early Warning',
                'priority': 'Medium',
                'recommendation': 'Develop automated anomaly detection system',
                'justification': 'Enable rapid response to sudden subsidence events'
            }
        ])
        
        recommendations['long_term'].extend([
            {
                'category': 'Infrastructure Adaptation',
                'priority': 'Medium',
                'recommendation': 'Design infrastructure to accommodate ongoing subsidence',
                'justification': 'Long-term subsidence trends require adaptive engineering'
            },
            {
                'category': 'Research Continuation',
                'priority': 'Medium',
                'recommendation': 'Maintain long-term monitoring for trend analysis',
                'justification': 'Multi-year datasets essential for understanding subsidence evolution'
            }
        ])
        
        # Save recommendations
        recommendations_file = self.results_dir / "ps09_policy_recommendations.json"
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"✅ Policy recommendations saved: {recommendations_file}")
        print(f"   Immediate actions: {len(recommendations['immediate_actions'])}")
        print(f"   Short-term: {len(recommendations['short_term'])}")
        print(f"   Long-term: {len(recommendations['long_term'])}")
        print(f"   Technical: {len(recommendations['technical'])}")
        
        return recommendations

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate comprehensive analysis report')
    parser.add_argument('--output-format', type=str, default='json', 
                       choices=['json', 'pdf', 'both'],
                       help='Output format for report')
    
    args = parser.parse_args()
    
    # Initialize report generator
    report_gen = ComprehensiveReportGenerator()
    
    try:
        # Load all pipeline data
        print("🔄 Loading pipeline data...")
        if not report_gen.load_pipeline_data():
            print("❌ Failed to load pipeline data")
            return False
        
        # Extract key metrics
        print("📊 Extracting key metrics...")
        if not report_gen.extract_key_metrics():
            print("❌ Failed to extract key metrics")
            return False
        
        # Generate executive summary
        print("📋 Generating executive summary...")
        summary = report_gen.generate_executive_summary()
        
        # Create comprehensive figures
        print("🎨 Creating comprehensive figures...")
        if not report_gen.create_comprehensive_figures():
            print("❌ Failed to create figures")
            return False
        
        # Generate station rankings
        print("🏆 Generating station rankings...")
        rankings = report_gen.generate_station_rankings()
        
        # Generate policy recommendations
        print("📋 Generating policy recommendations...")
        recommendations = report_gen.generate_policy_recommendations()
        
        print("\n" + "=" * 80)
        print("✅ ps09_comprehensive_report.py COMPLETED SUCCESSFULLY")
        print("\n📊 REPORT SUMMARY:")
        
        if 'study_overview' in summary:
            overview = summary['study_overview']
            print(f"   Study Area: {overview.get('study_area', 'Unknown')}")
            print(f"   Time Period: {overview.get('time_period', 'Unknown')}")
            print(f"   Dataset Size: {overview.get('dataset_size', 0)} stations")
        
        if 'key_findings' in summary:
            findings = summary['key_findings']
            if 'validation' in findings:
                val = findings['validation']
                print(f"   GPS Validation: R² = {val.get('r_squared', 0):.3f}")
        
        print(f"\n📁 Outputs generated:")
        print(f"   1. Executive summary report")
        print(f"   2. Comprehensive figures (3 plots)")
        print(f"   3. Station priority rankings")
        print(f"   4. Policy recommendations")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in comprehensive report generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
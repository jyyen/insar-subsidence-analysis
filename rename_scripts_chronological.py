#!/usr/bin/env python3
"""
Rename PS02 scripts in chronological order with descriptions
"""

import os
import shutil
from pathlib import Path

def rename_scripts_chronologically():
    """Rename scripts and figures in chronological order"""
    
    # Define chronological mapping with descriptions
    script_mapping = [
        # Early development (Jul 26-27)
        ("ps02_signal_decomposition.py", "ps02_01_signal_decomposition.py", "Original signal decomposition framework"),
        ("ps02a_MCSSA_decomposition_test.py", "ps02_02_mcssa_decomposition_test.py", "MCSSA decomposition testing"),
        ("ps02b_real_data_fitting.py", "ps02_03_real_data_fitting.py", "Real data fitting experiments"),
        ("ps02b_InSAR_signal_simulator.py", "ps02_04_insar_signal_simulator.py", "InSAR signal simulation tools"),
        ("ps02d_optimization_convergence.py", "ps02_05_optimization_convergence.py", "Optimization convergence analysis"),
        ("ps02e_noise_validation.py", "ps02_06_noise_validation.py", "Noise validation studies"),
        
        # PyTorch development (Jul 27)
        ("ps02c_1_algorithmic_optimization.py", "ps02_07_algorithmic_optimization.py", "Algorithmic optimization experiments"),
        ("ps02c_2_iterative_improvement.py", "ps02_08_iterative_improvement.py", "Iterative improvement methods"),
        ("ps02c_3_visualize_results.py", "ps02_09_visualize_results.py", "Results visualization"),
        ("ps02c_pytorch_demo.py", "ps02_10_pytorch_demo.py", "PyTorch framework demonstration"),
        ("ps02c_pytorch_framework.py", "ps02_11_pytorch_framework.py", "Basic PyTorch framework"),
        ("ps02c_pytorch_enhanced.py", "ps02_12_pytorch_enhanced.py", "Enhanced PyTorch model"),
        ("ps02c_pytorch_optimal.py", "ps02_13_pytorch_optimal.py", "Optimal PyTorch configuration"),
        ("ps02c_pytorch_geographic_analysis.py", "ps02_14_pytorch_geographic_analysis.py", "Geographic analysis tools"),
        ("ps02c_pytorch_corrected.py", "ps02_15_pytorch_corrected.py", "Sign convention corrections"),
        
        # Spatial optimization (Jul 28 early)
        ("ps02c_pytorch_spatial.py", "ps02_16_pytorch_spatial.py", "Spatial regularization"),
        ("ps02c_pytorch_enhanced_spatial.py", "ps02_17_pytorch_enhanced_spatial.py", "Enhanced spatial modeling"),
        ("ps02c_pytorch_spatial_optimized.py", "ps02_18_pytorch_spatial_optimized.py", "Optimized spatial framework"),
        ("ps02c_pytorch_improved_spatial.py", "ps02_19_pytorch_improved_spatial.py", "Improved spatial algorithms"),
        
        # EMD integration (Jul 28)
        ("ps02c_pytorch_emd_informed.py", "ps02_20_pytorch_emd_informed.py", "EMD-informed PyTorch"),
        ("ps02c_pytorch_emd_hybrid.py", "ps02_21_pytorch_emd_hybrid.py", "EMD-PyTorch hybrid approach"),
        ("ps02c_pytorch_emd_optimized.py", "ps02_22_pytorch_emd_optimized.py", "Phase 1 SUCCESS - EMD hybrid optimized"),
        
        # Phase 2 development
        ("ps02_phase2_production_emd_hybrid.py", "ps02_23_phase2_production_emd_hybrid.py", "Phase 2 production framework"),
        ("ps02_phase2_emd_denoised_pytorch.py", "ps02_24_phase2_emd_denoised_pytorch.py", "Phase 2 EMD denoising"),
        ("ps02c_phase2_visualize_results.py", "ps02_25_phase2_visualize_results.py", "Phase 2 results visualization"),
        
        # Evaluation and metrics
        ("ps02c_evaluate_metrics.py", "ps02_26_evaluate_metrics.py", "Metrics evaluation study"),
        ("ps02c_dtw_evaluation.py", "ps02_27_dtw_evaluation.py", "DTW metric analysis"),
        ("ps02c_comprehensive_evaluation.py", "ps02_28_comprehensive_evaluation.py", "Comprehensive phase comparison"),
        
        # Phase 2 optimized
        ("ps02_phase2_optimized_implementation.py", "ps02_29_phase2_optimized_implementation.py", "Phase 2 OPTIMIZED - Final implementation"),
        ("ps02_phase2_cartopy_visualization.py", "ps02_30_phase2_cartopy_visualization.py", "Geographic visualization with Cartopy"),
        ("ps02_phase2_full_dataset_implementation.py", "ps02_31_phase2_full_dataset_implementation.py", "Full dataset sequential processing"),
    ]
    
    # Figure mapping (match script names)
    figure_mapping = [
        ("ps02c_pytorch_demonstration.png", "ps02_10_pytorch_demo.png"),
        ("ps02c_pytorch_enhanced_results.png", "ps02_12_pytorch_enhanced.png"),
        ("ps02c_pytorch_optimal_results.png", "ps02_13_pytorch_optimal.png"),
        ("ps02c_pytorch_focused_subsidence_analysis.png", "ps02_14_pytorch_geographic_analysis.png"),
        ("ps02c_pytorch_comprehensive_geographic_analysis.png", "ps02_15_pytorch_corrected.png"),
        ("ps02c_pytorch_training_progress.png", "ps02_16_pytorch_spatial.png"),
        ("ps02c_time_series_comparison.png", "ps02_17_pytorch_enhanced_spatial.png"),
        ("ps02c_phase2_comprehensive_results.png", "ps02_25_phase2_visualize_results.png"),
        ("ps02c_phase2_time_series_comparison.png", "ps02_25_phase2_visualize_results_timeseries.png"),
        ("ps02c_metrics_evaluation.png", "ps02_26_evaluate_metrics.png"),
        ("ps02c_metrics_importance.png", "ps02_26_evaluate_metrics_importance.png"),
        ("ps02c_dtw_evaluation.png", "ps02_27_dtw_evaluation.png"),
        ("ps02c_dtw_analysis_summary.png", "ps02_27_dtw_evaluation_summary.png"),
        ("ps02c_comprehensive_evaluation.png", "ps02_28_comprehensive_evaluation.png"),
        ("ps02c_pytorch_training_plan.png", "ps02_28_comprehensive_evaluation_training_plan.png"),
        ("ps02_phase2_cartopy_subsidence_maps.png", "ps02_30_phase2_cartopy_visualization.png"),
    ]
    
    print("📝 Renaming PS02 scripts and figures in chronological order...")
    
    # Rename scripts
    for old_name, new_name, description in script_mapping:
        if Path(old_name).exists():
            # Add description as comment at top
            with open(old_name, 'r') as f:
                content = f.read()
            
            # Add description header
            new_content = f'''#!/usr/bin/env python3
"""
{new_name}: {description}
Chronological order script from Taiwan InSAR Subsidence Analysis Project
"""

''' + content.split('"""', 2)[-1] if '"""' in content else content
            
            # Write to new file
            with open(new_name, 'w') as f:
                f.write(new_content)
            
            print(f"✅ {old_name} → {new_name}")
        else:
            print(f"⚠️ {old_name} not found")
    
    # Rename figures
    figures_dir = Path("figures")
    if figures_dir.exists():
        for old_name, new_name in figure_mapping:
            old_path = figures_dir / old_name
            new_path = figures_dir / new_name
            if old_path.exists():
                shutil.copy2(old_path, new_path)
                print(f"✅ figures/{old_name} → figures/{new_name}")
            else:
                print(f"⚠️ figures/{old_name} not found")
    
    # Create index file
    with open("PS02_SCRIPT_INDEX.md", 'w') as f:
        f.write("# PS02 Script Index - Chronological Order\n\n")
        f.write("Taiwan InSAR Subsidence Analysis Project Development Timeline\n\n")
        
        current_phase = ""
        for old_name, new_name, description in script_mapping:
            # Determine phase
            if "phase2" in new_name.lower():
                phase = "## Phase 2 Development\n"
            elif "emd" in new_name.lower():
                phase = "## EMD Integration\n"
            elif "spatial" in new_name.lower():
                phase = "## Spatial Optimization\n"
            elif "pytorch" in new_name.lower():
                phase = "## PyTorch Development\n"
            else:
                phase = "## Early Development\n"
            
            if phase != current_phase:
                f.write(f"\n{phase}\n")
                current_phase = phase
            
            f.write(f"- `{new_name}`: {description}\n")
    
    print("\n📋 Created PS02_SCRIPT_INDEX.md with complete timeline")
    print("🎉 Renaming complete!")

if __name__ == "__main__":
    rename_scripts_chronologically()
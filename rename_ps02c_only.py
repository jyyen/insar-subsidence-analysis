#!/usr/bin/env python3
"""
Rename only PS02C scripts in chronological order with descriptions
"""

import os
import shutil
from pathlib import Path

def rename_ps02c_scripts():
    """Rename PS02C scripts in chronological order"""
    
    # PS02C series mapping in chronological order (oldest to newest)
    ps02c_mapping = [
        # Early algorithmic work
        ("ps02c_1_algorithmic_optimization.py", "ps02c_01_algorithmic_optimization.py", "Initial algorithmic optimization approach"),
        ("ps02c_2_iterative_improvement.py", "ps02c_02_iterative_improvement.py", "Iterative improvement methods"),
        ("ps02c_3_visualize_results.py", "ps02c_03_visualize_results.py", "Early results visualization"),
        
        # PyTorch framework development
        ("ps02c_pytorch_demo.py", "ps02c_04_pytorch_demo.py", "PyTorch framework demonstration"),
        ("ps02c_pytorch_framework.py", "ps02c_05_pytorch_framework.py", "Basic PyTorch framework implementation"),
        ("ps02c_pytorch_enhanced.py", "ps02c_06_pytorch_enhanced.py", "Enhanced PyTorch model with improvements"),
        ("ps02c_pytorch_optimal.py", "ps02c_07_pytorch_optimal.py", "Optimal PyTorch configuration"),
        ("ps02c_pytorch_geographic_analysis.py", "ps02c_08_pytorch_geographic_analysis.py", "Geographic analysis and visualization"),
        ("ps02c_pytorch_corrected.py", "ps02c_09_pytorch_corrected.py", "Sign convention corrections for subsidence"),
        
        # Spatial optimization phase
        ("ps02c_pytorch_spatial.py", "ps02c_10_pytorch_spatial.py", "Spatial regularization introduction"),
        ("ps02c_pytorch_enhanced_spatial.py", "ps02c_11_pytorch_enhanced_spatial.py", "Enhanced spatial modeling"),
        ("ps02c_pytorch_spatial_optimized.py", "ps02c_12_pytorch_spatial_optimized.py", "Optimized spatial framework"),
        ("ps02c_pytorch_improved_spatial.py", "ps02c_13_pytorch_improved_spatial.py", "Improved spatial algorithms"),
        
        # EMD integration (Phase 1)
        ("ps02c_pytorch_emd_informed.py", "ps02c_14_pytorch_emd_informed.py", "EMD-informed PyTorch initialization"),
        ("ps02c_pytorch_emd_hybrid.py", "ps02c_15_pytorch_emd_hybrid.py", "EMD-PyTorch hybrid approach"),
        ("ps02c_pytorch_emd_optimized.py", "ps02c_16_pytorch_emd_optimized.py", "PHASE 1 SUCCESS: EMD-hybrid optimized (0.3238 correlation)"),
        
        # Phase 2 and evaluation
        ("ps02c_phase2_visualize_results.py", "ps02c_17_phase2_visualize_results.py", "Phase 2 results visualization"),
        ("ps02c_evaluate_metrics.py", "ps02c_18_evaluate_metrics.py", "Metrics evaluation study (RMSE, DTW vs correlation)"),
        ("ps02c_dtw_evaluation.py", "ps02c_19_dtw_evaluation.py", "DTW (Dynamic Time Warping) analysis"),
        ("ps02c_comprehensive_evaluation.py", "ps02c_20_comprehensive_evaluation.py", "Comprehensive phase comparison and training plan"),
    ]
    
    # Corresponding figure mappings
    figure_mapping = [
        ("ps02c_phase2_comprehensive_results.png", "ps02c_17_phase2_visualize_results.png"),
        ("ps02c_phase2_time_series_comparison.png", "ps02c_17_phase2_visualize_results_timeseries.png"),
        ("ps02c_metrics_evaluation.png", "ps02c_18_evaluate_metrics.png"),
        ("ps02c_metrics_importance.png", "ps02c_18_evaluate_metrics_importance.png"),
        ("ps02c_dtw_evaluation.png", "ps02c_19_dtw_evaluation.png"),
        ("ps02c_dtw_analysis_summary.png", "ps02c_19_dtw_evaluation_summary.png"),
        ("ps02c_comprehensive_evaluation.png", "ps02c_20_comprehensive_evaluation.png"),
        ("ps02c_pytorch_training_plan.png", "ps02c_20_comprehensive_evaluation_training_plan.png"),
    ]
    
    print("📝 Renaming PS02C scripts in chronological order...")
    
    # Rename scripts
    renamed_count = 0
    for old_name, new_name, description in ps02c_mapping:
        if Path(old_name).exists():
            # Read original content
            with open(old_name, 'r') as f:
                content = f.read()
            
            # Add description as header comment
            header = f'''#!/usr/bin/env python3
"""
{new_name}: {description}
Chronological order: PS02C development timeline
Taiwan InSAR Subsidence Analysis Project
"""

'''
            
            # Remove old shebang/header if exists and add new one
            lines = content.split('\n')
            start_idx = 0
            if lines[0].startswith('#!'):
                start_idx = 1
            if len(lines) > start_idx and lines[start_idx].startswith('"""'):
                # Find end of docstring
                for i in range(start_idx + 1, len(lines)):
                    if lines[i].strip().endswith('"""'):
                        start_idx = i + 1
                        break
            
            new_content = header + '\n'.join(lines[start_idx:])
            
            # Write to new file
            with open(new_name, 'w') as f:
                f.write(new_content)
            
            print(f"✅ {old_name} → {new_name}")
            renamed_count += 1
        else:
            print(f"⚠️ {old_name} not found")
    
    # Rename corresponding figures
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
    
    # Create PS02C index
    with open("PS02C_CHRONOLOGICAL_INDEX.md", 'w') as f:
        f.write("# PS02C Chronological Index\n\n")
        f.write("Taiwan InSAR Subsidence Analysis - PS02C Development Timeline\n\n")
        
        # Group by phases
        phases = [
            ("## Early Development", [0, 1, 2]),
            ("## PyTorch Framework Development", [3, 4, 5, 6, 7, 8]),
            ("## Spatial Optimization Phase", [9, 10, 11, 12]),
            ("## EMD Integration (Phase 1)", [13, 14, 15]),
            ("## Phase 2 & Evaluation", [16, 17, 18, 19])
        ]
        
        for phase_name, indices in phases:
            f.write(f"{phase_name}\n\n")
            for i in indices:
                if i < len(ps02c_mapping):
                    old_name, new_name, description = ps02c_mapping[i]
                    f.write(f"- **`{new_name}`**: {description}\n")
            f.write("\n")
        
        f.write("## Key Milestones\n\n")
        f.write("- **ps02c_16**: 🏆 Phase 1 Success (0.3238 correlation, 5.0x improvement)\n")
        f.write("- **ps02c_19**: 📊 DTW analysis (better metric than correlation)\n")
        f.write("- **ps02c_20**: 🎯 Comprehensive evaluation and training optimization\n")
    
    print(f"\n📋 Created PS02C_CHRONOLOGICAL_INDEX.md")
    print(f"🎉 Renamed {renamed_count} PS02C scripts chronologically!")
    
    # Check for parallelization in latest scripts
    print(f"\n🔍 Checking latest scripts for parallelization...")
    latest_scripts = ["ps02_phase2_full_dataset_implementation.py", "ps02_29_phase2_optimized_implementation.py"]
    
    for script in latest_scripts:
        if Path(script).exists():
            has_parallel = False
            with open(script, 'r') as f:
                content = f.read()
                parallel_keywords = ['multiprocessing', 'concurrent.futures', 'joblib', 'torch.multiprocessing', 'parallel']
                has_parallel = any(keyword in content.lower() for keyword in parallel_keywords)
            
            status = "✅ PARALLELIZED" if has_parallel else "❌ SEQUENTIAL"
            print(f"   {script}: {status}")

if __name__ == "__main__":
    rename_ps02c_scripts()
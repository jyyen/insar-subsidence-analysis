#!/usr/bin/env python3
"""
Install Dependencies for PS02 Phase 2 Implementation
Checks and installs required packages for M1 Mac compatibility
"""

import subprocess
import sys
import platform

def check_and_install_dependencies():
    """Check and install required dependencies for M1 Mac"""
    
    print("🔧 Checking dependencies for PS02 Phase 2...")
    print(f"🖥️ System: {platform.system()} {platform.machine()}")
    print(f"🐍 Python: {sys.version}")
    
    # Required packages
    required_packages = [
        ('torch', 'PyTorch for neural networks'),
        ('numpy', 'Numerical computing'),
        ('matplotlib', 'Plotting and visualization'),
        ('scipy', 'Scientific computing'),
        ('scikit-learn', 'Machine learning utilities'),
        ('tslearn', 'Time series analysis'),
        ('cartopy', 'Geographic mapping'),
        ('psutil', 'System monitoring')
    ]
    
    missing_packages = []
    
    # Check each package
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Found")
        except ImportError:
            print(f"❌ {package}: Missing - {description}")
            missing_packages.append(package)
    
    if not missing_packages:
        print("\n🎉 All dependencies are installed!")
        return True
    
    print(f"\n📦 Missing packages: {missing_packages}")
    
    # M1 Mac specific installation commands
    if platform.machine() == 'arm64':  # M1/M2 Mac
        print("\n🍎 Detected Apple Silicon Mac - using optimized installation...")
        
        install_commands = {
            'torch': 'pip install torch torchvision torchaudio',
            'numpy': 'pip install numpy',
            'matplotlib': 'pip install matplotlib',
            'scipy': 'pip install scipy',
            'scikit-learn': 'pip install scikit-learn',
            'tslearn': 'pip install tslearn',
            'cartopy': 'pip install cartopy',
            'psutil': 'pip install psutil'
        }
        
        print("\n🚀 Installing missing packages...")
        for package in missing_packages:
            if package in install_commands:
                cmd = install_commands[package]
                print(f"\n📥 Installing {package}...")
                print(f"Command: {cmd}")
                
                try:
                    result = subprocess.run(cmd, shell=True, check=True, 
                                          capture_output=True, text=True)
                    print(f"✅ {package} installed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {package}: {e}")
                    print(f"Error output: {e.stderr}")
    
    else:
        print("\n💻 For Intel Mac or other systems:")
        print("Run these commands manually:")
        for package in missing_packages:
            print(f"  pip install {package}")
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    success = True
    for package, _ in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: OK")
        except ImportError:
            print(f"❌ {package}: Still missing")
            success = False
    
    return success

if __name__ == "__main__":
    print("🔧 PS02 Phase 2 Dependency Installer")
    print("="*50)
    
    success = check_and_install_dependencies()
    
    if success:
        print("\n🎉 All dependencies installed successfully!")
        print("✅ Ready to run PS02 Phase 2 implementation")
        print("\nNext steps:")
        print("python ps02_32_phase2_parallelized_full_dataset.py")
    else:
        print("\n⚠️ Some dependencies are still missing")
        print("Please install them manually and try again")
        
        print("\n🍎 For M1 Mac, try these specific commands:")
        print("# Create conda environment (recommended)")
        print("conda create -n insar python=3.9")
        print("conda activate insar")
        print("# Install PyTorch for M1")
        print("pip install torch torchvision torchaudio")
        print("# Install other packages")
        print("pip install numpy matplotlib scipy scikit-learn tslearn cartopy psutil")
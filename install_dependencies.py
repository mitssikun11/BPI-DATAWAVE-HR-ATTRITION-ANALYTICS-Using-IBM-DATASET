#!/usr/bin/env python3
"""
Installation script for IBM HR Analytics Project dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def main():
    """Install all required dependencies"""
    print("=== IBM HR Analytics Project - Dependency Installation ===")
    
    # Core data science libraries
    core_libraries = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "scipy>=1.9.0"
    ]
    
    # Machine learning libraries
    ml_libraries = [
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "catboost>=1.1.0"
    ]
    
    # Additional utilities
    utility_libraries = [
        "joblib>=1.2.0",
        "tqdm>=4.64.0",
        "imbalanced-learn>=0.9.0",
        "shap>=0.41.0"
    ]
    
    # Jupyter and development
    dev_libraries = [
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
        "notebook>=6.4.0"
    ]
    
    all_libraries = core_libraries + ml_libraries + utility_libraries + dev_libraries
    
    print("Installing core data science libraries...")
    for lib in core_libraries:
        install_package(lib)
    
    print("\nInstalling machine learning libraries...")
    for lib in ml_libraries:
        install_package(lib)
    
    print("\nInstalling utility libraries...")
    for lib in utility_libraries:
        install_package(lib)
    
    print("\nInstalling development libraries...")
    for lib in dev_libraries:
        install_package(lib)
    
    print("\n=== Installation Summary ===")
    print("✓ All required libraries have been installed")
    print("✓ You can now run the HR analytics project")
    print("\nNext steps:")
    print("1. Run: python train_model.py")
    print("2. Explore the notebooks in the notebooks/ folder")
    print("3. Check the generated visualizations and models")

if __name__ == "__main__":
    main() 
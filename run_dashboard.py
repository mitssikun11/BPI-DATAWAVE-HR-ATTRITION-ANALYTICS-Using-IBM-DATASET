#!/usr/bin/env python3
"""
ATTRISEnse Dashboard Launcher
Automatically installs dependencies and runs the Streamlit dashboard
"""
import subprocess
import sys
import os
import time

def install_streamlit():
    """Install Streamlit if not already installed"""
    try:
        import streamlit
        print("✅ Streamlit is already installed!")
        return True
    except ImportError:
        print("📦 Installing Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit. Please install manually:")
            print("   pip install streamlit")
            return False

def install_other_dependencies():
    """Install other required dependencies"""
    print("📦 Installing other dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "plotly", "joblib"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Some dependencies may not be installed. Continuing anyway...")
        return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting ATTRISEnse Dashboard...")
    print("📊 Opening in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run the streamlit dashboard
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_dashboard_simple.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main launcher function"""
    print("🎯 ATTRISEnse Dashboard Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_dashboard_simple.py"):
        print("❌ Error: streamlit_dashboard_simple.py not found!")
        print("   Please run this script from the project directory.")
        return
    
    # Install dependencies
    if not install_streamlit():
        return
    
    install_other_dependencies()
    
    # Run the dashboard
    run_dashboard()

if __name__ == "__main__":
    main()

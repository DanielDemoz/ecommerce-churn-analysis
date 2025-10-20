#!/usr/bin/env python3
"""
E-commerce Churn Analysis Dashboard Launcher
This script launches the Streamlit dashboard with proper configuration.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'scikit-learn', 'plotly', 'openpyxl'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ All packages installed successfully!")
    else:
        print("✅ All required packages are available!")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching E-commerce Churn Analysis Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    try:
        # Launch Streamlit with custom configuration
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#1f77b4",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thank you for using the E-commerce Churn Analysis Dashboard!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Make sure you have Streamlit installed: pip install streamlit")

if __name__ == "__main__":
    print("=" * 60)
    print("📊 E-commerce Churn Analysis Dashboard")
    print("=" * 60)
    
    # Check if dashboard.py exists
    if not os.path.exists("dashboard.py"):
        print("❌ dashboard.py not found in current directory!")
        print("💡 Make sure you're running this script from the project directory")
        sys.exit(1)
    
    # Check dependencies
    check_dependencies()
    
    # Launch dashboard
    launch_dashboard()

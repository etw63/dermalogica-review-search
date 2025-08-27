#!/usr/bin/env python3
"""
Dermalogica Review Search Application
Run this script to start the vector database search application.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_data_file():
    """Check if the data file exists"""
    if not os.path.exists("dermalogica_aggregated_reviews.csv"):
        print("❌ Error: dermalogica_aggregated_reviews.csv not found!")
        print("Please make sure the aggregated reviews file is in the current directory.")
        return False
    return True

def start_app():
    """Start the FastAPI application"""
    print("Starting the Dermalogica Review Search application...")
    print("🚀 The app will be available at: http://localhost:8000")
    print("📊 Initial setup may take a few minutes to create the vector database...")
    print("\nPress Ctrl+C to stop the application\n")
    
    try:
        import uvicorn
        uvicorn.run("vector_search_app:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("❌ Error: uvicorn not installed. Please install requirements first.")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("🔍 DERMALOGICA REVIEW SEARCH APPLICATION")
    print("=" * 60)
    
    # Check if data file exists
    if not check_data_file():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Start the application
    start_app()

if __name__ == "__main__":
    main()

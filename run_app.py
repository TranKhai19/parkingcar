#!/usr/bin/env python3
"""
Optimized startup script for the parking management Flask application.
This script helps reduce startup time and prevents frequent reloading.
"""

import os
import sys
import logging
from pathlib import Path

# Set environment variables before importing other modules
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'

# Reduce logging from various libraries
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('torchvision').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def check_requirements():
    """Check if all required files exist"""
    current_dir = Path(__file__).parent
    
    required_files = [
        'app.py',
        'bouding_box.json',
        'runs/detect/train/weights/best.pt',
        'templates/index.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    """Main function to run the application"""
    print("🚗 Parking Management System")
    print("=" * 40)
    
    if not check_requirements():
        print("\n❌ Cannot start application due to missing files")
        sys.exit(1)
    
    print("🔄 Loading model and initializing application...")
    
    try:
        # Import the Flask app
        from app import app
        
        print("✅ Application initialized successfully")
        print("🌐 Starting web server...")
        print("📱 Access the application at: http://127.0.0.1:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 40)
        
        # Run the Flask application
        app.run(
            debug=False,           # Disable debug mode for production
            use_reloader=False,    # Disable auto-reloader
            host='127.0.0.1',
            port=5000,
            threaded=True          # Enable threading for better performance
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

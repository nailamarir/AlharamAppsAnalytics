#!/usr/bin/env python3
"""
Launch the AlHaram Analytics Web Application

Usage:
    python run_webapp.py

Then open http://localhost:5000 in your browser
"""

import os
import sys
import webbrowser
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

# Add src to path for alharam_analytics imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Add webapp to path for app import
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

# Change to webapp directory for proper template/static file resolution
os.chdir(PROJECT_ROOT / "webapp")

from app import app

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🕌  AlHaram Analytics - Preprocessing Pipeline          ║
    ║                                                           ║
    ║   Starting web application...                             ║
    ║   Open your browser at: http://localhost:5000             ║
    ║                                                           ║
    ║   Press Ctrl+C to stop the server                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Try to open browser automatically
    webbrowser.open('http://localhost:5000')

    # Run the Flask app
    app.run(host='127.0.0.1', debug=True, port=5000, use_reloader=False)

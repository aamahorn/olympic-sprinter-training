#!/usr/bin/env python3
"""
Railway-optimized entry point for Olympic Sprinter Training App
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simplified entry point for deployment that ensures proper startup
"""
import streamlit as st
import os
import sys

# Set page config first
st.set_page_config(
    page_title="Olympic Sprinter Training",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import and run main app
    from app import main
    main()
except Exception as e:
    st.error(f"Application startup error: {str(e)}")
    st.write("Please check the deployment logs for more details.")
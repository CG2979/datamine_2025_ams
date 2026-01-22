"""
Main Streamlit application entry point.
This file should be thin - it just orchestrates components.
"""

import streamlit as st
from utils.session_state import initialize_session_state
from components.sidebar import render_sidebar
from components.title_cleaning_tab import render_title_cleaning_tab
from components.data_overview_tab import render_data_overview_tab
from components.export_tab import render_export_tab

# App setup
st.set_page_config(page_title="Survey Data Cleaning App", layout="wide")

# Initialize session state
initialize_session_state()

# Render sidebar
render_sidebar()

# Check if data is loaded
if st.session_state["df"] is None:
    st.title("Survey Data Cleaner")
    st.info("Upload a CSV file to get started")
    st.stop()

# Main content
st.title("Survey Data Cleaner")

# Render tabs
tab1, tab2, tab3 = st.tabs(["Title Cleaning", "Data Overview", "Export"])

with tab1:
    render_title_cleaning_tab()

with tab2:
    render_data_overview_tab()

with tab3:
    render_export_tab()

"""Session state initialization and management."""

import streamlit as st

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "history": [],
        "mapping": {},
        "clusters": None,
        "df": None,
        "original_df": None,
        "auto_cleaned": False,
        "selected_cluster": None,
        "similarity_threshold": 90,
        "last_selected_col": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

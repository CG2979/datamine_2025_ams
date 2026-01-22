"""Data loading utilities."""

import pandas as pd
import streamlit as st
from core.data_cleaning import auto_clean_dataframe

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """Load CSV file with caching."""
    return pd.read_csv(file, low_memory=False)

def handle_file_upload(uploaded_file):
    """Handle file upload and auto-cleaning."""
    if uploaded_file is not None and st.session_state["df"] is None:
        with st.spinner("Loading and auto-cleaning..."):
            raw_df = load_csv(uploaded_file)
            st.session_state["original_df"] = raw_df.copy()
            
            cleaned_df = auto_clean_dataframe(raw_df)
            st.session_state["df"] = cleaned_df
            st.session_state["auto_cleaned"] = True
        
        st.success("File loaded & cleaned!")

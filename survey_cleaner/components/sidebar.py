"""Sidebar component."""

import streamlit as st
from utils.data_loader import handle_file_upload

def render_sidebar():
    """Render the sidebar with file upload and settings."""
    with st.sidebar:
        st.title("Survey Data Cleaner")
        st.markdown("*Automatically combines equivalent job titles and cleans numerical data*")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        handle_file_upload(uploaded_file)
        
        if st.session_state["df"] is not None:
            _render_dataset_stats()
            st.markdown("---")
            _render_advanced_settings()
            st.markdown("---")
            _render_reset_button()

def _render_dataset_stats():
    """Render dataset statistics."""
    st.subheader("Dataset Stats")
    df = st.session_state["df"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", len(df))
        st.metric("Missing", df.isnull().sum().sum())
    with col2:
        st.metric("Columns", len(df.columns))
        st.metric("Dupes", df.duplicated().sum())

def _render_advanced_settings():
    """Render advanced settings."""
    with st.expander("Advanced Settings"):
        st.session_state["similarity_threshold"] = st.slider(
            "Clustering threshold:", 80, 100, 90,
            help="Higher = stricter matching"
        )

def _render_reset_button():
    """Render reset button."""
    if st.button("Reset to Original"):
        st.session_state["df"] = st.session_state["original_df"].copy()
        st.session_state["mapping"] = {}
        st.session_state["clusters"] = None
        st.session_state["history"] = []
        st.session_state["auto_cleaned"] = False
        st.session_state["selected_cluster"] = None
        st.rerun()

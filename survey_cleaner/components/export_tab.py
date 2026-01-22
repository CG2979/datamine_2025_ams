"""Export tab component."""

import streamlit as st


def render_export_tab():
    """Render the export tab."""
    st.subheader("Export Cleaned Data")

    if st.session_state.get("mapping"):
        df_export = st.session_state["df"].copy()
        job_col = st.session_state["last_selected_col"]

        # Apply the canonical mappings to the job titles
        df_export[job_col] = df_export[job_col].replace(st.session_state["mapping"])

        # Show preview
        st.markdown("#### Preview of cleaned data:")
        st.dataframe(df_export.head(20), use_container_width=True, height=300)
        
        st.markdown("---")
        
        # Show mapping summary
        _render_cleaning_summary(df_export, job_col)
        
        st.markdown("---")

        # Allow user to download the cleaned file
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
            type="primary"
        )
        st.success("Cleaned data ready for download.")
    else:
        st.info("No cleaned mappings found. Try running Auto-Cluster first in the Title Cleaning tab.")


def _render_cleaning_summary(df_export, job_col):
    """Render cleaning summary metrics."""
    st.markdown("#### Cleaning Summary:")
    col1, col2, col3 = st.columns(3)
    
    df = st.session_state["df"]
    
    with col1:
        st.metric("Original Unique Titles", len(df[job_col].unique()))
    with col2:
        st.metric("Cleaned Unique Titles", len(df_export[job_col].unique()))
    with col3:
        reduction = len(df[job_col].unique()) - len(df_export[job_col].unique())
        st.metric("Titles Consolidated", reduction)

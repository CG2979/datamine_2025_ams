"""Data overview tab component."""

import streamlit as st
import pandas as pd
from core.title_clustering import detect_job_title_column


def render_data_overview_tab():
    """Render the data overview tab."""
    st.subheader("Dataset Overview")
    
    df = st.session_state["df"]
    
    # Top 200 Job Titles Section
    _render_top_job_titles(df)
    
    st.markdown("---")
    
    # Age and Gender Analysis Section
    _render_age_gender_analysis(df)


def _render_top_job_titles(df):
    """Render top 200 job titles section."""
    st.markdown("### Top 200 Job Titles")
    
    # Auto-detect or use selected job title column
    if st.session_state.get("last_selected_col") and st.session_state["last_selected_col"] in df.columns:
        job_col = st.session_state["last_selected_col"]
    else:
        job_col = detect_job_title_column(df)
    
    if job_col and job_col in df.columns:
        # Check if we have mappings (canonical titles)
        if st.session_state.get("mapping"):
            # Use canonical titles
            df_with_canonical = df.copy()
            df_with_canonical[job_col] = df_with_canonical[job_col].replace(st.session_state["mapping"])
            title_counts = df_with_canonical[job_col].value_counts().head(200)
            title_type = "Canonical"
        else:
            # Use original titles
            title_counts = df[job_col].value_counts().head(200)
            title_type = "Original"
        
        # Create a dataframe for display
        top_titles_df = pd.DataFrame({
            'Rank': range(1, len(title_counts) + 1),
            'Job Title': title_counts.index,
            'Count': title_counts.values
        })
        
        st.markdown(f"**Showing top {title_type.lower()} titles from column:** `{job_col}`")
        if title_type == "Original":
            st.info("💡 Run Auto-Cluster in the Title Cleaning tab to see canonical titles here")
        
        st.dataframe(
            top_titles_df,
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.get("mapping"):
                unique_count = len(set(st.session_state["mapping"].values()))
                st.metric("Total Unique Canonical Titles", unique_count)
            else:
                st.metric("Total Unique Titles", len(df[job_col].unique()))
        with col2:
            st.metric("Top 200 Total Records", title_counts.sum())
        with col3:
            coverage = (title_counts.sum() / len(df)) * 100
            st.metric("Coverage", f"{coverage:.1f}%")
    else:
        st.info("No job title column detected. Please run Auto-Cluster in the Title Cleaning tab first.")


def _render_age_gender_analysis(df):
    """Render age and gender analysis section."""
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    gender_cols = [col for col in df.columns if 'gender' in col.lower()]
    
    if not age_cols and not gender_cols:
        return
    
    st.markdown("### Age & Gender Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if age_cols:
            _render_age_distribution(df, age_cols)
    
    with col2:
        if gender_cols:
            _render_gender_distribution(df, gender_cols)


def _render_age_distribution(df, age_cols):
    """Render age distribution analysis."""
    st.markdown("#### Age Distribution")
    age_col = st.selectbox("Select age column:", age_cols)
    
    # Show value counts
    age_counts = df[age_col].value_counts().sort_index()
    st.dataframe(
        age_counts.reset_index().rename(columns={'index': 'Age', age_col: 'Count'}),
        use_container_width=True,
        height=300,
        hide_index=True
    )
    
    # Show statistics
    valid_ages = df[age_col][df[age_col] > 0]
    if len(valid_ages) > 0:
        st.metric("Valid Ages", len(valid_ages))
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Min", int(valid_ages.min()))
        with col_b:
            st.metric("Mean", f"{valid_ages.mean():.1f}")
        with col_c:
            st.metric("Max", int(valid_ages.max()))


def _render_gender_distribution(df, gender_cols):
    """Render gender distribution analysis."""
    st.markdown("#### Gender Distribution")
    gender_col = st.selectbox("Select gender column:", gender_cols)
    
    # Show value counts
    gender_counts = df[gender_col].value_counts()
    st.dataframe(
        gender_counts.reset_index().rename(columns={'index': 'Gender', gender_col: 'Count'}),
        use_container_width=True,
        height=300,
        hide_index=True
    )
    
    # Show statistics
    st.metric("Total Entries", len(df))
    empty_count = (df[gender_col] == '').sum()
    st.metric("Empty Values", empty_count)

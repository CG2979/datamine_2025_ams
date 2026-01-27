"""Merge files tab component."""

import streamlit as st
import pandas as pd
from core.data_cleaning import auto_clean_dataframe
from core.column_mapping import (
    suggest_column_mappings,
    suggest_semantic_mappings,
    apply_column_mapping,
    get_unmapped_columns,
    validate_mapping
)


def render_merge_files_tab():
    """Render the merge files tab."""
    st.subheader("Merge Two Cleaned Files")
    st.markdown("Upload two CSV files to merge them together. Both files will be auto-cleaned before merging.")
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### File 1")
        file1 = st.file_uploader("Upload first CSV", type=["csv"], key="merge_file1")
        if file1:
            st.success(f"✓ {file1.name}")
    
    with col2:
        st.markdown("#### File 2")
        file2 = st.file_uploader("Upload second CSV", type=["csv"], key="merge_file2")
        if file2:
            st.success(f"✓ {file2.name}")
    
    if file1 and file2:
        _render_merge_options(file1, file2)


def _render_merge_options(file1, file2):
    """Render merge options and preview."""
    st.markdown("---")
    st.subheader("Merge Settings")
    
    # Load both files
    try:
        df1 = pd.read_csv(file1, low_memory=False)
        df2 = pd.read_csv(file2, low_memory=False)
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return
    
    # Show file info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**File 1:** {file1.name}")
        st.write(f"Rows: {len(df1)}, Columns: {len(df1.columns)}")
    
    with col2:
        st.markdown(f"**File 2:** {file2.name}")
        st.write(f"Rows: {len(df2)}, Columns: {len(df2.columns)}")
    
    # Merge type selection
    st.markdown("### Merge Type")
    merge_type = st.radio(
        "Select merge method:",
        ["Vertical Stack (Append Rows)", "Horizontal Join (Match on Column)"],
        help="Vertical stacking combines rows, horizontal joining matches on a key column"
    )
    
    if merge_type == "Vertical Stack (Append Rows)":
        _render_vertical_merge(df1, df2, file1.name, file2.name)
    else:
        _render_horizontal_merge(df1, df2, file1.name, file2.name)


def _render_vertical_merge(df1, df2, name1, name2):
    """Render vertical merge (stack rows)."""
    st.markdown("---")
    st.subheader("Vertical Stack Settings")
    
    # Column alignment check
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    only_in_file1 = cols1 - cols2
    only_in_file2 = cols2 - cols1
    
    # Column Mapping Section (NEW)
    if only_in_file2:
        st.markdown("#### Column Mapping")
        st.info("📝 File 2 has columns not in File 1. Map them to align the data correctly.")
        
        # Initialize mapping in session state
        if "column_mapping" not in st.session_state:
            st.session_state["column_mapping"] = {}
        
        # Auto-suggest mappings
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Auto-Suggest (Fuzzy Match)", help="Use fuzzy matching to suggest mappings"):
                fuzzy_suggestions = suggest_column_mappings(list(cols1), list(only_in_file2), threshold=75)
                st.session_state["column_mapping"].update(fuzzy_suggestions)
                st.success(f"Suggested {len(fuzzy_suggestions)} mappings")
                st.rerun()
        
        with col2:
            if st.button("🧠 Auto-Suggest (Semantic)", help="Use common name equivalents"):
                semantic_suggestions = suggest_semantic_mappings(list(cols1), list(only_in_file2))
                st.session_state["column_mapping"].update(semantic_suggestions)
                st.success(f"Suggested {len(semantic_suggestions)} mappings")
                st.rerun()
        
        with col3:
            if st.button("📅 Auto-Suggest (Smart)", help="Detect year patterns and code/name pairs", key="smart_suggest"):
                from core.column_mapping import get_year_pattern_mappings
                # Combine semantic + year patterns
                semantic_suggestions = suggest_semantic_mappings(list(cols1), list(only_in_file2))
                year_suggestions = get_year_pattern_mappings(list(cols1), list(only_in_file2))
                smart_suggestions = {**semantic_suggestions, **year_suggestions}
                st.session_state["column_mapping"].update(smart_suggestions)
                st.success(f"Suggested {len(smart_suggestions)} mappings")
                st.rerun()
        
        # Manual mapping interface
        st.markdown("**Map File 2 columns to File 1 columns:**")
        
        mapping_changed = False
        current_mapping = st.session_state["column_mapping"].copy()
        
        # Create mapping rows
        for col2 in sorted(only_in_file2):
            col_a, col_b, col_c = st.columns([2, 2, 1])
            
            with col_a:
                st.text_input(
                    "From File 2:",
                    value=col2,
                    disabled=True,
                    key=f"from_{col2}"
                )
            
            with col_b:
                # Dropdown with all cols1 + "Don't map" option
                options = ["(Don't map)"] + sorted(cols1)
                current_value = current_mapping.get(col2, "(Don't map)")
                
                if current_value not in options:
                    current_value = "(Don't map)"
                
                new_value = st.selectbox(
                    "To File 1:",
                    options=options,
                    index=options.index(current_value) if current_value in options else 0,
                    key=f"map_{col2}",
                    label_visibility="collapsed"
                )
                
                if new_value == "(Don't map)":
                    if col2 in current_mapping:
                        del current_mapping[col2]
                        mapping_changed = True
                elif new_value != current_mapping.get(col2):
                    current_mapping[col2] = new_value
                    mapping_changed = True
            
            with col_c:
                # Show similarity score if auto-suggested
                if col2 in st.session_state["column_mapping"]:
                    from rapidfuzz import fuzz
                    target = st.session_state["column_mapping"][col2]
                    score = fuzz.ratio(col2.lower(), target.lower())
                    st.metric("Match", f"{score}%", label_visibility="collapsed")
        
        # Update session state if changed
        if mapping_changed:
            st.session_state["column_mapping"] = current_mapping
        
        # Validate mapping
        if current_mapping:
            errors = validate_mapping(current_mapping, list(cols1), list(cols2))
            if errors:
                st.error("❌ Mapping errors:")
                for error in errors:
                    st.write(f"- {error}")
            else:
                st.success(f"✅ {len(current_mapping)} columns mapped successfully")
        
        # Clear mapping button
        if st.button("Clear All Mappings"):
            st.session_state["column_mapping"] = {}
            st.rerun()
        
        st.markdown("---")
    
    # Show updated column comparison with mappings applied
    st.markdown("#### Column Comparison (After Mapping)")
    
    # Calculate what columns will look like after mapping
    mapped_cols2 = set(df2.columns)
    if "column_mapping" in st.session_state and st.session_state["column_mapping"]:
        # Simulate applying mapping
        for old_col, new_col in st.session_state["column_mapping"].items():
            if old_col in mapped_cols2:
                mapped_cols2.remove(old_col)
                mapped_cols2.add(new_col)
    
    common_after_mapping = cols1.intersection(mapped_cols2)
    only_in_file1_after = cols1 - mapped_cols2
    only_in_file2_after = mapped_cols2 - cols1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Common Columns", len(common_after_mapping))
        if common_after_mapping:
            with st.expander("View common columns"):
                st.write(sorted(common_after_mapping))
    
    with col2:
        st.metric(f"Only in {name1}", len(only_in_file1_after))
        if only_in_file1_after:
            with st.expander(f"View columns only in {name1}"):
                st.write(sorted(only_in_file1_after))
    
    with col3:
        st.metric(f"Only in {name2}", len(only_in_file2_after))
        if only_in_file2_after:
            with st.expander(f"View columns only in {name2}"):
                st.write(sorted(only_in_file2_after))
    
    # Options
    st.markdown("#### Merge Options")
    
    auto_clean = st.checkbox("Auto-clean both files before merging", value=True)
    
    handle_missing_cols = st.radio(
        "How to handle unmapped columns:",
        ["Keep all columns (fill missing with NaN)", "Keep only common columns"],
        help="Choose how to handle columns that don't exist in both files"
    )
    
    add_source_column = st.checkbox(
        "Add source file column",
        value=True,
        help="Add a column indicating which file each row came from"
    )
    
    # Preview button
    if st.button("Preview Merge", type="primary"):
        # Validate mapping first
        if "column_mapping" in st.session_state and st.session_state["column_mapping"]:
            errors = validate_mapping(
                st.session_state["column_mapping"], 
                list(cols1), 
                list(cols2)
            )
            if errors:
                st.error("Cannot merge: Fix mapping errors first")
                return
        
        with st.spinner("Merging files..."):
            # Clean if requested
            if auto_clean:
                df1_clean = auto_clean_dataframe(df1.copy())
                df2_clean = auto_clean_dataframe(df2.copy())
            else:
                df1_clean = df1.copy()
                df2_clean = df2.copy()
            
            # Apply column mapping to df2
            if "column_mapping" in st.session_state and st.session_state["column_mapping"]:
                df2_clean = apply_column_mapping(df2_clean, st.session_state["column_mapping"])
                st.success(f"Applied {len(st.session_state['column_mapping'])} column mappings to File 2")
            
            # Add source column if requested
            if add_source_column:
                df1_clean['_source_file'] = name1
                df2_clean['_source_file'] = name2
            
            # Merge
            if handle_missing_cols == "Keep only common columns":
                # Recalculate common columns after mapping
                common_after = set(df1_clean.columns).intersection(set(df2_clean.columns))
                df1_clean = df1_clean[list(common_after)]
                df2_clean = df2_clean[list(common_after)]
                if add_source_column:
                    # Re-add source column after filtering
                    df1_clean['_source_file'] = name1
                    df2_clean['_source_file'] = name2
            
            merged_df = pd.concat([df1_clean, df2_clean], ignore_index=True)
            
            # Store in session state
            st.session_state["merged_df"] = merged_df
            st.session_state["merge_name"] = f"{name1}_and_{name2}_merged"
        
        st.success(f"✓ Merged successfully! Total rows: {len(merged_df)}")
    
    # Show preview if available
    if "merged_df" in st.session_state and st.session_state["merged_df"] is not None:
        _render_merge_preview(st.session_state["merged_df"])


def _render_horizontal_merge(df1, df2, name1, name2):
    """Render horizontal merge (join on key)."""
    st.markdown("---")
    st.subheader("Horizontal Join Settings")
    
    # Find common columns for join key
    common_cols = list(set(df1.columns).intersection(set(df2.columns)))
    
    if not common_cols:
        st.error("No common columns found between files. Cannot perform horizontal join.")
        return
    
    st.markdown("#### Join Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        join_key = st.selectbox(
            "Select join key column:",
            common_cols,
            help="Column to match rows on"
        )
    
    with col2:
        join_type = st.selectbox(
            "Select join type:",
            ["inner", "left", "right", "outer"],
            help="inner: only matching rows | left: all from file 1 | right: all from file 2 | outer: all rows"
        )
    
    # Options
    auto_clean = st.checkbox("Auto-clean both files before joining", value=True, key="join_auto_clean")
    
    handle_duplicate_cols = st.radio(
        "Handle duplicate column names:",
        ["Add suffix (_file1, _file2)", "Keep only from File 1", "Keep only from File 2"],
        help="Choose how to handle columns with the same name in both files"
    )
    
    # Preview button
    if st.button("Preview Join", type="primary"):
        with st.spinner("Joining files..."):
            # Clean if requested
            if auto_clean:
                df1_clean = auto_clean_dataframe(df1.copy())
                df2_clean = auto_clean_dataframe(df2.copy())
            else:
                df1_clean = df1.copy()
                df2_clean = df2.copy()
            
            # Determine suffixes
            if handle_duplicate_cols == "Add suffix (_file1, _file2)":
                suffixes = ('_file1', '_file2')
            elif handle_duplicate_cols == "Keep only from File 1":
                # Remove duplicate columns from df2 (except join key)
                duplicate_cols = [col for col in df2_clean.columns if col in df1_clean.columns and col != join_key]
                df2_clean = df2_clean.drop(columns=duplicate_cols)
                suffixes = ('', '_file2')
            else:  # Keep only from File 2
                # Remove duplicate columns from df1 (except join key)
                duplicate_cols = [col for col in df1_clean.columns if col in df2_clean.columns and col != join_key]
                df1_clean = df1_clean.drop(columns=duplicate_cols)
                suffixes = ('_file1', '')
            
            # Perform join
            try:
                merged_df = pd.merge(
                    df1_clean, 
                    df2_clean, 
                    on=join_key, 
                    how=join_type,
                    suffixes=suffixes
                )
                
                # Store in session state
                st.session_state["merged_df"] = merged_df
                st.session_state["merge_name"] = f"{name1}_and_{name2}_joined"
                
                st.success(f"✓ Joined successfully! Total rows: {len(merged_df)}")
                
                # Show join statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File 1 rows", len(df1))
                with col2:
                    st.metric("File 2 rows", len(df2))
                with col3:
                    st.metric("Result rows", len(merged_df))
                
            except Exception as e:
                st.error(f"Error during join: {str(e)}")
                return
        
        # Show preview if available
        if "merged_df" in st.session_state and st.session_state["merged_df"] is not None:
            _render_merge_preview(st.session_state["merged_df"])


def _render_merge_preview(merged_df):
    """Render preview of merged data."""
    st.markdown("---")
    st.subheader("Merged Data Preview")
    
    # Show stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(merged_df))
    with col2:
        st.metric("Total Columns", len(merged_df.columns))
    with col3:
        missing_pct = (merged_df.isnull().sum().sum() / (len(merged_df) * len(merged_df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Show preview
    st.markdown("#### First 50 rows:")
    st.dataframe(merged_df.head(50), use_container_width=True, height=400)
    
    # Column info
    with st.expander("View column details"):
        col_info = pd.DataFrame({
            'Column': merged_df.columns,
            'Non-Null Count': merged_df.count().values,
            'Null Count': merged_df.isnull().sum().values,
            'Data Type': merged_df.dtypes.values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    # Download merged file
    st.markdown("---")
    st.subheader("Download Merged File")
    
    filename = st.text_input(
        "Filename:",
        value=st.session_state.get("merge_name", "merged_data"),
        help="Enter filename (without .csv extension)"
    )
    
    csv = merged_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Merged CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
        type="primary"
    )
    
    # Option to use merged file as main dataset
    st.markdown("---")
    if st.button("Use this merged file as main dataset", help="Replace current dataset with merged data"):
        st.session_state["df"] = merged_df.copy()
        st.session_state["original_df"] = merged_df.copy()
        st.session_state["mapping"] = {}
        st.session_state["clusters"] = None
        st.session_state["auto_cleaned"] = True
        st.success("✓ Merged file is now your main dataset! Go to Title Cleaning tab to work with it.")
        st.rerun()

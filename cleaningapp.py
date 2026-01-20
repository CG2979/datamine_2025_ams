# --- TAB 3: Export (supports merging multiple cleaned files) ---
with tab3:
    st.subheader("Export Cleaned Data")

    # If there are multiple files, offer merge UI first
    file_names = list(st.session_state["files"].keys())
    if len(file_names) > 1:
        st.markdown("### Merge multiple cleaned files")
        st.markdown("Select which cleaned files to include in the merged output, and optionally map/rename columns so mismatched names can be combined or excluded.")
        
        to_merge = st.multiselect("Files to merge (preserve order):", options=file_names, default=file_names, key="merge_select")
        add_source_col = st.checkbox("Add source filename column to merged rows", value=True, key="merge_add_source")
        
        if to_merge:
            # Build union of columns and source tokens
            union_cols = set()
            source_tokens = []
            for fname in to_merge:
                cols = st.session_state["files"][fname]["cleaned"].columns.tolist()
                union_cols.update(cols)
                source_tokens += [f"{fname}::{c}" for c in cols]
            union_cols = sorted(union_cols)
            source_tokens = sorted(source_tokens)
            
            st.markdown("**Column presence across selected files (True = present):**")
            presence = pd.DataFrame(index=to_merge, columns=union_cols)
            for fname in to_merge:
                cols = set(st.session_state["files"][fname]["cleaned"].columns)
                for c in union_cols:
                    presence.loc[fname, c] = c in cols
            st.dataframe(presence, use_container_width=True, height=200)
            
            st.markdown("---")
            st.markdown("**Column mapping for merge**")
            st.markdown("By default, columns with identical names across files are auto-mapped to a single target column. Toggle 'Adjust mapping' to change which source columns combine into which target column.")
            
            # default mapping: identical column names mapped to themselves
            default_mapping = {}
            for c in union_cols:
                default_mapping[c] = [f"{fname}::{c}" for fname in to_merge if c in st.session_state["files"][fname]["cleaned"].columns]
            
            adjust = st.checkbox("Adjust mapping manually", value=False, key="adjust_mapping_checkbox")
            mapping = {}
            if adjust:
                st.write("For each desired target column, select which source columns (file::column) should be combined into it.")
                for target in union_cols:
                    default = default_mapping.get(target, [])
                    mapping[target] = st.multiselect(f"Target column '{target}' <-", options=source_tokens, default=default, key=f"map_{target}")
            else:
                mapping = default_mapping
            
            st.write("Columns with zero mapped sources will be excluded.")
            
            if st.button("Preview merged dataframe", key="preview_merged"):
                merged_parts = []
                for fname in to_merge:
                    df_part = st.session_state["files"][fname]["cleaned"].copy()
                    # Build rename map for this file
                    rename_map = {}
                    for tgt, sources in mapping.items():
                        for src in sources:
                            src_fname, src_col = src.split("::", 1)
                            if src_fname == fname:
                                rename_map[src_col] = tgt
                    if rename_map:
                        df_part = df_part.rename(columns=rename_map)
                    # Keep only columns that are target names (and present in this part)
                    keep_cols = [c for c in df_part.columns if c in mapping.keys() and any((f"{fname}::{orig_col}") in sum(mapping.values(), []) for orig_col in [c])]
                    # Simpler: select columns that are targets after rename and present
                    after_cols = [c for c in df_part.columns if c in mapping.keys()]
                    out_df = df_part[after_cols].copy() if after_cols else pd.DataFrame(columns=sorted(mapping.keys()))
                    if add_source_col:
                        out_df["_source_file"] = fname
                    merged_parts.append(out_df)
                
                if merged_parts:
                    merged_df = pd.concat(merged_parts, ignore_index=True, sort=False)
                else:
                    merged_df = pd.DataFrame(columns=list(mapping.keys()))
                
                st.markdown("Merged preview (first 200 rows):")
                st.dataframe(merged_df.head(200), use_container_width=True)
                
                csv_bytes = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download merged CSV", data=csv_bytes, file_name="merged_cleaned.csv", mime="text/csv", key="download_merged")
    else:
        st.info("Upload multiple files in the sidebar to enable merging options.")
    
    st.markdown("---")
    # Keep original single-file cleaned export functionality (works on the active file)
    if st.session_state.get("mapping"):
        df_export = st.session_state["df"].copy()
        job_col = st.session_state["last_selected_col"] if "last_selected_col" in st.session_state else auto_detect_job_title_column(df_export)
        if job_col not in df_export.columns:
            job_col = df_export.columns[0] if len(df_export.columns) > 0 else None

        if job_col:
            df_export[job_col] = df_export[job_col].replace(st.session_state["mapping"])

        st.markdown("#### Preview of cleaned data (active file):")
        st.dataframe(df_export.head(20), use_container_width=True, height=300)
        
        st.markdown("---")
        
        # Show mapping summary
        st.markdown("#### Cleaning Summary (active file):")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Unique Titles", len(st.session_state["original_df"][job_col].unique()) if job_col in st.session_state["original_df"].columns else 0)
        with col2:
            st.metric("Cleaned Unique Titles", len(df_export[job_col].unique()) if job_col in df_export.columns else 0)
        with col3:
            try:
                reduction = len(st.session_state["original_df"][job_col].unique()) - len(df_export[job_col].unique())
            except Exception:
                reduction = 0
            st.metric("Titles Consolidated", reduction)
        
        st.markdown("---")

        # Allow user to download the cleaned file
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV (active file)",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
            type="primary"
        )
        st.success("Cleaned data ready for download.")
    else:
        st.info("No cleaned mappings found for the active file. Try running Auto-Cluster first in the Title Cleaning tab.")

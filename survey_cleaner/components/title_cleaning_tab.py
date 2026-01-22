"""Title cleaning tab component."""

import streamlit as st
import pandas as pd
from core.title_clustering import detect_job_title_column, cluster_titles


def render_title_cleaning_tab():
    """Render the title cleaning tab."""
    st.subheader("Job Title Clustering")
    
    df = st.session_state["df"]
    columns = df.columns.tolist()
    
    # Auto-detect job title column
    detected_col = detect_job_title_column(df)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_col = st.selectbox(
            "Job titles column:", 
            columns, 
            index=columns.index(detected_col) if detected_col in columns else 0,
            help="Auto-detected based on column name and content"
        )
    
    with col2:
        st.write("")
        st.write("")
        auto_cluster_btn = st.button("Auto-Cluster", type="primary")
    
    # Reset state if column changes
    if "last_selected_col" not in st.session_state or st.session_state["last_selected_col"] != selected_col:
        st.session_state["last_selected_col"] = selected_col
        st.session_state["mapping"] = {}
        st.session_state["clusters"] = None
        st.session_state["selected_cluster"] = None
    
    titles = df[selected_col].astype(str).str.strip()
    
    # Auto-cluster on button click or first load
    if auto_cluster_btn or st.session_state["clusters"] is None:
        threshold = st.session_state.get("similarity_threshold", 90)
        
        with st.spinner("Auto-clustering titles..."):
            progress_bar = st.progress(0)
            
            clusters, mapping = cluster_titles(titles, threshold)
            
            st.session_state["clusters"] = clusters
            st.session_state["mapping"] = mapping
            st.session_state["selected_cluster"] = None
            
            progress_bar.progress(100)
            progress_bar.empty()
        
        st.success(f"Found {len(clusters)} clusters from {len(titles.unique())} unique titles")
    
    clusters = st.session_state.get("clusters", [])
    mapping = st.session_state.get("mapping", {})
    
    if clusters:
        _render_cluster_summary(clusters, mapping, df, selected_col)
        _render_cluster_details(clusters, mapping, df, selected_col)
        _render_merge_clusters(clusters, mapping, df, selected_col)


def _render_cluster_summary(clusters, mapping, df, selected_col):
    """Render cluster summary table."""
    summary_data = []
    for i, (cluster_orig, cluster_cleaned) in enumerate(clusters):
        canonical = mapping.get(cluster_orig[0], cluster_orig[0])
        summary_data.append({
            "Cluster": i,
            "Canonical Title": canonical,
            "Variations": len(cluster_orig),
            "Total Records": len(df[df[selected_col].isin(cluster_orig)])
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values("Total Records", ascending=False)
    
    # Show summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clusters", len(clusters))
    with col2:
        titles = df[selected_col].astype(str).str.strip()
        st.metric("Unique Titles", len(titles.unique()))
    with col3:
        avg_cluster_size = sum(len(c[0]) for c in clusters) / len(clusters)
        st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
    with col4:
        titles = df[selected_col].astype(str).str.strip()
        reduction_pct = (1 - len(clusters) / len(titles.unique())) * 100
        st.metric("Title Reduction", f"{reduction_pct:.0f}%")
    
    st.markdown("---")
    st.subheader("Clustered Titles (Click to review)")
    
    # Use native Streamlit dataframe with selection
    event = st.dataframe(
        summary_df,
        use_container_width=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True
    )
    
    # Show cluster details when selected
    if event.selection.rows:
        cluster_idx = int(summary_df.iloc[event.selection.rows[0]]["Cluster"])
        st.session_state["selected_cluster"] = cluster_idx


def _render_cluster_details(clusters, mapping, df, selected_col):
    """Render selected cluster details."""
    if st.session_state["selected_cluster"] is None:
        return
    
    cluster_idx = st.session_state["selected_cluster"]
    current_cluster_orig, current_cluster_cleaned = clusters[cluster_idx]
    
    st.markdown("---")
    st.subheader(f"Cluster {cluster_idx} Details")
    
    # Edit canonical title
    col1, col2 = st.columns([2, 1])
    
    with col1:
        canonical = mapping.get(current_cluster_orig[0], current_cluster_orig[0])
        new_canonical = st.text_input(
            "Edit canonical title:",
            value=canonical,
            help="This will be the standardized title for all variations",
            key=f"canonical_input_{cluster_idx}"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Update", key=f"update_btn_{cluster_idx}"):
            for title in current_cluster_orig:
                st.session_state["mapping"][title] = new_canonical
            st.success("Updated!")
            st.rerun()
    
    # Show variations
    st.markdown(f"**{len(current_cluster_orig)} variations in this cluster:**")
    variations_df = pd.DataFrame({
        "Original Title": current_cluster_orig,
        "Count": [len(df[df[selected_col] == t]) for t in current_cluster_orig]
    }).sort_values("Count", ascending=False)
    
    st.dataframe(variations_df, use_container_width=True, height=300, hide_index=True)
    
    # Move variation to another cluster
    _render_move_variation(current_cluster_orig, cluster_idx, clusters, mapping, canonical)
    
    # Split variation to new cluster
    _render_split_variation(current_cluster_orig, cluster_idx)


def _render_move_variation(current_cluster_orig, cluster_idx, clusters, mapping, canonical):
    """Render move variation controls."""
    st.markdown("---")
    st.markdown("**Move a variation to another cluster:**")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        variation_index_input = st.text_input(
            "Enter variation index to move:",
            value="",
            key=f"move_var_{cluster_idx}",
            help=f"Enter index 0-{len(current_cluster_orig)-1}"
        )
    
    with col2:
        target_cluster_input = st.text_input(
            "Move to cluster index (or -1 for new):",
            value="",
            key=f"target_cluster_{cluster_idx}",
            help=f"Enter a cluster index (0-{len(clusters)-1}) or -1 to create a new cluster"
        )
    
    with col3:
        st.write("")
        st.write("")
        if st.button("Move", key=f"move_btn_{cluster_idx}"):
            _handle_move_variation(
                variation_index_input, 
                target_cluster_input, 
                current_cluster_orig, 
                clusters, 
                mapping, 
                canonical
            )


def _handle_move_variation(variation_index_input, target_cluster_input, current_cluster_orig, clusters, mapping, canonical):
    """Handle moving a variation to another cluster."""
    try:
        var_idx = int(variation_index_input)
        target_idx = int(target_cluster_input)
        
        if not (0 <= var_idx < len(current_cluster_orig)):
            st.error(f"Invalid variation index. Enter 0-{len(current_cluster_orig)-1}")
        else:
            variation_to_move = current_cluster_orig[var_idx]
            
            if target_idx == -1:
                # Create a new cluster
                new_canonical = ' '.join(word.capitalize() for word in variation_to_move.split())
                st.session_state["mapping"][variation_to_move] = new_canonical
                _rebuild_clusters()
                st.success(f"Created new cluster with '{new_canonical}'")
                st.rerun()
            elif 0 <= target_idx < len(clusters):
                target_canonical = mapping.get(clusters[target_idx][0][0], clusters[target_idx][0][0])
                
                if target_canonical == canonical:
                    st.warning("Cannot move to the same cluster")
                else:
                    st.session_state["mapping"][variation_to_move] = target_canonical
                    _rebuild_clusters()
                    st.success(f"Moved '{variation_to_move}' to cluster {target_idx}: '{target_canonical}'")
                    st.rerun()
            else:
                st.error(f"Invalid cluster index. Enter 0-{len(clusters)-1} or -1 for new cluster")
    except ValueError:
        st.error("Please enter valid numbers for both variation and cluster index")


def _render_split_variation(current_cluster_orig, cluster_idx):
    """Render split variation controls."""
    st.markdown("**Or split a variation to a new cluster with custom name:**")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        variation_index_for_new = st.text_input(
            "Enter variation index:",
            value="",
            key=f"split_var_idx_{cluster_idx}",
            help=f"Enter index 0-{len(current_cluster_orig)-1}"
        )
    
    with col2:
        new_cluster_name = st.text_input(
            "New cluster name:",
            value="",
            key=f"split_cluster_name_{cluster_idx}",
            help="Enter the canonical title for the new cluster"
        )
    
    with col3:
        st.write("")
        st.write("")
        if st.button("Split Out", key=f"split_btn_{cluster_idx}"):
            _handle_split_variation(
                variation_index_for_new, 
                new_cluster_name, 
                current_cluster_orig
            )


def _handle_split_variation(variation_index_for_new, new_cluster_name, current_cluster_orig):
    """Handle splitting a variation to a new cluster."""
    try:
        var_idx_new = int(variation_index_for_new)
        
        if not (0 <= var_idx_new < len(current_cluster_orig)):
            st.error(f"Invalid variation index. Enter 0-{len(current_cluster_orig)-1}")
        elif not new_cluster_name.strip():
            st.warning("Please enter a cluster name")
        else:
            variation_for_new = current_cluster_orig[var_idx_new]
            st.session_state["mapping"][variation_for_new] = new_cluster_name.strip()
            _rebuild_clusters()
            st.success(f"Created new cluster: '{new_cluster_name.strip()}'")
            st.rerun()
    except ValueError:
        st.error("Please enter a valid number for variation index")


def _render_merge_clusters(clusters, mapping, df, selected_col):
    """Render merge clusters functionality."""
    st.markdown("---")
    st.markdown("### Merge Multiple Clusters")
    st.markdown("Combine multiple clusters into one canonical title")
    
    all_canonicals = sorted(set(st.session_state["mapping"].values()))
    
    if len(all_canonicals) > 1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            clusters_to_merge_input = st.text_input(
                "Enter cluster indices to merge (comma-separated):",
                value="",
                help=f"Example: 0,5,12 (valid indices: 0-{len(clusters)-1})",
                key="merge_clusters_input"
            )
            
            # Parse and preview
            if clusters_to_merge_input.strip():
                _preview_merge(clusters_to_merge_input, clusters, mapping, df, selected_col)
        
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            if clusters_to_merge_input.strip():
                _render_merge_button(clusters_to_merge_input, clusters, mapping)
    else:
        st.info("Need at least 2 clusters to merge. Create more clusters first.")


def _preview_merge(clusters_to_merge_input, clusters, mapping, df, selected_col):
    """Preview what will be merged."""
    try:
        cluster_indices = [int(idx.strip()) for idx in clusters_to_merge_input.split(',')]
        valid_indices = [idx for idx in cluster_indices if 0 <= idx < len(clusters)]
        invalid_indices = [idx for idx in cluster_indices if idx not in valid_indices]
        
        if invalid_indices:
            st.warning(f"Invalid indices removed: {invalid_indices}")
        
        if len(valid_indices) >= 2:
            clusters_to_merge = []
            for idx in valid_indices:
                canonical_temp = mapping.get(clusters[idx][0][0], clusters[idx][0][0])
                clusters_to_merge.append(canonical_temp)
            
            # Remove duplicates
            seen = set()
            clusters_to_merge = [x for x in clusters_to_merge if not (x in seen or seen.add(x))]
            
            st.markdown(f"**Merging {len(clusters_to_merge)} clusters:**")
            
            total_variations = 0
            total_records = 0
            
            for i, canonical_temp in enumerate(clusters_to_merge):
                variations = [title for title, canon in st.session_state["mapping"].items() if canon == canonical_temp]
                records = sum(len(df[df[selected_col] == var]) for var in variations)
                total_variations += len(variations)
                total_records += records
                cluster_idx_display = valid_indices[i]
                st.markdown(f"- **Cluster {cluster_idx_display}: {canonical_temp}**: {len(variations)} variations, {records} records")
            
            st.info(f"Total: {total_variations} variations, {total_records} records")
            
            # New canonical name input
            default_name = clusters_to_merge[0]
            st.text_input(
                "New canonical title for merged cluster:",
                value=default_name,
                key="merged_canonical_name",
                help="All selected clusters will use this title"
            )
        elif len(valid_indices) == 1:
            st.info("Enter at least 2 cluster indices to merge")
    except ValueError:
        st.error("Invalid format. Use comma-separated numbers (e.g., 0,5,12)")


def _render_merge_button(clusters_to_merge_input, clusters, mapping):
    """Render the merge button and handle merging."""
    try:
        cluster_indices = [int(idx.strip()) for idx in clusters_to_merge_input.split(',')]
        valid_indices = [idx for idx in cluster_indices if 0 <= idx < len(clusters)]
        
        if len(valid_indices) >= 2:
            if st.button("Merge Clusters", type="primary", key="merge_btn"):
                new_merged_canonical = st.session_state.get("merged_canonical_name", "").strip()
                
                if new_merged_canonical:
                    clusters_to_merge = []
                    for idx in valid_indices:
                        canonical_temp = mapping.get(clusters[idx][0][0], clusters[idx][0][0])
                        if canonical_temp not in clusters_to_merge:
                            clusters_to_merge.append(canonical_temp)
                    
                    # Update mappings
                    for canonical_temp in clusters_to_merge:
                        for title, canon in st.session_state["mapping"].items():
                            if canon == canonical_temp:
                                st.session_state["mapping"][title] = new_merged_canonical
                    
                    _rebuild_clusters()
                    st.success(f"Merged {len(clusters_to_merge)} clusters into '{new_merged_canonical}'")
                    st.rerun()
                else:
                    st.warning("Please enter a canonical name")
    except ValueError:
        pass


def _rebuild_clusters():
    """Rebuild clusters from updated mapping."""
    new_clusters_dict = {}
    for orig_title, canon_title in st.session_state["mapping"].items():
        if canon_title not in new_clusters_dict:
            new_clusters_dict[canon_title] = []
        new_clusters_dict[canon_title].append(orig_title)
    
    new_clusters = []
    for canon, origs in new_clusters_dict.items():
        new_clusters.append((origs, origs))
    
    st.session_state["clusters"] = new_clusters

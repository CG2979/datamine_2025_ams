import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import re
import numpy as np

# --- App setup ---
st.set_page_config(page_title="Survey Data Cleaning App", layout="wide")

# --- Initialize session state ---
if "history" not in st.session_state:
    st.session_state["history"] = []
if "mapping" not in st.session_state:
    st.session_state["mapping"] = {}
if "clusters" not in st.session_state:
    st.session_state["clusters"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None
if "original_df" not in st.session_state:
    st.session_state["original_df"] = None
if "auto_cleaned" not in st.session_state:
    st.session_state["auto_cleaned"] = False
if "cleaning_report" not in st.session_state:
    st.session_state["cleaning_report"] = {}
if "selected_cluster" not in st.session_state:
    st.session_state["selected_cluster"] = None

# --- Auto-cleaning functions ---
def auto_clean_dataframe(df):
    """Automatically clean the dataframe with smart defaults"""
    report = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "actions": []
    }
    
    # 1. Remove completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        df = df.dropna(how='all')
        report["actions"].append(f"Removed {empty_rows} completely empty rows")
    
    # 3. Clean text columns automatically
    text_cols_cleaned = 0
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            # Strip whitespace and remove extra spaces
            original = df[col].copy()
            cleaned = df[col].astype(str).str.strip()
            cleaned = cleaned.apply(lambda x: re.sub(r'\s+', ' ', x) if x != 'nan' else x)
            
            # Replace 'nan' strings with actual NaN
            cleaned = cleaned.replace('nan', np.nan)
            
            df.loc[:, col] = cleaned
            
            if not df[col].equals(original):
                text_cols_cleaned += 1
    
    if text_cols_cleaned > 0:
        report["actions"].append(f"Auto-cleaned {text_cols_cleaned} text columns (whitespace, extra spaces)")
    
    # 4. Clean Age columns
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    for col in age_cols:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == 'object':
            original_missing = df[col].isnull().sum()
            
            # Convert to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN with 0
            df[col] = df[col].fillna(0)
            
            # Set invalid ages (< 20 or > 100) to 0
            invalid_count = ((df[col] > 100) | (df[col] < 20)).sum()
            df.loc[(df[col] > 100) | (df[col] < 20), col] = 0
            
            if invalid_count > 0 or original_missing > 0:
                report["actions"].append(
                    f"Cleaned '{col}': filled {original_missing} missing, "
                    f"reset {invalid_count} invalid ages (< 20 or > 100) to 0"
                )
    
    # 5. Clean Gender columns
    gender_cols = [col for col in df.columns if 'gender' in col.lower()]
    for col in gender_cols:
        if df[col].dtype == 'object' or pd.api.types.is_numeric_dtype(df[col]):
            original_missing = df[col].isnull().sum()
            
            # Convert to string and fill NaN with empty string
            df[col] = df[col].astype(str).replace('nan', '')
            df[col] = df[col].fillna('')
            
            if original_missing > 0:
                report["actions"].append(f"Cleaned '{col}': filled {original_missing} missing values with empty string")
    
    report["final_rows"] = len(df)
    report["final_cols"] = len(df.columns)
    
    return df, report

def auto_detect_job_title_column(df):
    """Automatically detect which column likely contains job titles"""
    candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Check column name
        if any(keyword in col_lower for keyword in ['title', 'position', 'job', 'role', 'designation', 'Emp_Position_Title']):
            candidates.append((col, 100))
        
        # Check content patterns
        elif df[col].dtype == 'object':
            sample = df[col].dropna().head(100).astype(str).str.lower()
            job_keywords = ['professor', 'lecturer', 'assistant', 'associate', 'director', 
                          'manager', 'specialist', 'coordinator', 'analyst', 'engineer']
            matches = sum(sample.str.contains('|'.join(job_keywords), regex=True))
            if matches > len(sample) * 0.3:  # If 30%+ contain job keywords
                candidates.append((col, matches / len(sample) * 100))
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    return df.columns[0] if len(df.columns) > 0 else None

def auto_cluster_titles(titles, threshold=90):
    """Automatically cluster and clean job titles with smart typo correction"""
    
    # First, separate postdoctoral titles from others
    def is_postdoc_title(title):
        """Check if a title is postdoctoral-related"""
        t = str(title).lower()
        return (
            'postdoctoral' in t or
            'postdoc fellow' in t or
            'post-doc fellow' in t or
            'post doctoral fellow' in t or
            'post doc' in t or
            'postdoctor' in t
        )
    
    postdoc_mask = titles.map(is_postdoc_title)
    postdoc_titles = titles[postdoc_mask]
    non_postdoc_titles = titles[~postdoc_mask]
    
    # Common abbreviations and typos to fix
    REPLACEMENTS = {
        # Common abbreviations
        r'\basst\.?\b': 'assistant',
        r'\bassst\.?\b': 'assistant',  # Triple 's' typo
        r'\basssitant\b': 'assistant',  # Common typo
        r'\bassitant\b': 'assistant',  # Missing 's'
        r'\bassoc\.?\b': 'associate',
        r'\bvist\.?\b': 'visiting',
        r'\bvisit\.?\b': 'visiting',
        r'\bprof\.?\b': 'professor',
        r'\bprofesso\b': 'professor',  # Truncated professor
        r'\bprofesor\b': 'professor',  # Missing 's'
        r'\bdept\.?\b': 'department',
        r'\bdir\.?\b': 'director',
        r'\bmgr\.?\b': 'manager',
        
        # Postdoc variations - normalize to just "postdoctoral" 
        r'\bpos-doc\b': 'postdoctoral',
        r'\bpost-doc\b': 'postdoctoral',
        r'\bpostdoc\b': 'postdoctoral',
        r'\bpstdoctoral\b': 'postdoctoral',  # Missing 'o'
        r'\bpostdoctral\b': 'postdoctoral',  # Missing 'o' in middle
        r'\bposdoctoral\b': 'postdoctoral',
        r'\bpost doctoral\b': 'postdoctoral',
        r'\bpostdoctor\b': 'postdoctoral',
        r'\bpostdoctorate\b': 'postdoctoral',
        r'\bPost Doc\b': 'postdoctoral',
        r'\bpost doc\b': 'postdoctoral',
        
        # Common typos
        r'\breserch\b': 'research',
        r'\bresearch\b': 'research',
        r'\brsearch\b': 'research',
        r'\bscientis\b': 'scientist',
        r'\blecture\b': 'lecturer',
        r'\bintructor\b': 'instructor',
        r'\binstrutor\b': 'instructor',  # Missing 'c'
        r'\binsructor\b': 'instructor',  # Transposed 't' and 'r'
        r'\bfello\b': 'fellow',  # Missing 'w'
    }
    
    def fix_typos_and_abbrev(title: str) -> str:
        """Fix common typos and abbreviations"""
        t = title.lower()
        for pattern, replacement in REPLACEMENTS.items():
            t = re.sub(pattern, replacement, t)
        
        # Clean up duplicate words (like "fellow fellow")
        words = t.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    # Job keywords for normalization (expanded and ordered by specificity)
    JOB_KEYWORDS = [
        # Visiting variations (most specific first)
        "visiting lecturer", "visiting assistant professor", "visiting associate professor", "visiting professor",
        "visiting instructor",
        # Adjunct variations
        "adjunct lecturer", "adjunct assistant professor", "adjunct associate professor", "adjunct professor",
        "adjunct instructor",
        # Acting variations
        "acting assistant professor", "acting associate professor", "acting professor",
        "acting instructor",
        # Temporary variations
        "temporary assistant professor", "temporary associate professor", "temporary professor",
        "temporary instructor",
        # Clinical variations
        "clinical assistant professor", "clinical associate professor", "clinical professor",
        "clinical instructor",
        # Research variations
        "research assistant professor", "research associate professor", "research professor",
        "research instructor",
        # Regular positions
        "lecturer", "assistant professor", "associate professor", "professor",
        "instructor",
        # Postdoctoral positions
        "postdoctoral",
        # Research positions
        "research scientist", "research fellow", "research associate", "research assistant",
        # Director positions
        "director", "associate director", "assistant director",
        # Other positions
        "biostatistician", "statistician",
    ]
    
    def normalize_title(title: str) -> str:
        # First fix typos and abbreviations
        t = fix_typos_and_abbrev(title)
        # Remove punctuation
        t = re.sub(r"[^a-z0-9\s]", "", t)
        # Remove extra spaces
        t = re.sub(r"\s+", " ", t).strip()
        
        # Special handling for postdoctoral - return early
        if re.search(r"\bpostdoctoral\b", t):
            return "postdoctoral"
            
        # Modifiers that should be preserved before job keywords
        PRESERVE_MODIFIERS = ["associate", "assistant"]
        
        # Find longest matching keyword and preserve modifiers + keyword + everything after
        # Sort by length descending to match most specific first (e.g., "adjunct assistant professor" before "assistant professor")
        for keyword in sorted(JOB_KEYWORDS, key=len, reverse=True):
            match = re.search(rf"\b{re.escape(keyword)}\b", t)
            if match:
                start_idx = match.start()
                
                # Check if there's a preserved modifier immediately before the keyword
                prefix = t[:start_idx].strip()
                words_before = prefix.split()
                
                if words_before and words_before[-1] in PRESERVE_MODIFIERS:
                    # Include the modifier
                    modifier_start = t.rfind(words_before[-1], 0, start_idx)
                    return t[modifier_start:].strip()
                else:
                    return t[start_idx:].strip()
        return t

    
    # Apply normalization to NON-POSTDOC titles only
    normalized_titles = non_postdoc_titles.map(normalize_title)
    unique_norm_titles = normalized_titles.unique()
    
    # Also create a cleaned version of titles for canonical selection
    cleaned_titles = non_postdoc_titles.map(fix_typos_and_abbrev)
    
    # Cluster similar NON-POSTDOC titles
    clusters = []
    seen = set()
    
    for norm_title in unique_norm_titles:
        if norm_title in seen:
            continue
        cluster = [norm_title]
        seen.add(norm_title)
        for other in unique_norm_titles:
            if other not in seen and fuzz.ratio(norm_title, other) > threshold:
                cluster.append(other)
                seen.add(other)
        
        # Get both original AND cleaned versions for this cluster
        mask = normalized_titles.isin(cluster)
        full_titles = non_postdoc_titles[mask].unique().tolist()
        full_titles_cleaned = cleaned_titles[mask].unique().tolist()
        
        # Store both for later use
        clusters.append((full_titles, full_titles_cleaned))
    
    # Add one cluster for ALL postdoctoral titles at the end
    if len(postdoc_titles) > 0:
        postdoc_list = postdoc_titles.unique().tolist()
        clusters.append((postdoc_list, postdoc_list))
    
    # Auto-generate canonical titles with proper expansion
    mapping = {}
    
    for cluster_orig, cluster_cleaned in clusters:
        # Use cleaned titles to pick canonical
        expanded_titles = cluster_cleaned
        
        # Filter out empty, nan, or whitespace-only titles
        expanded_titles = [t for t in expanded_titles if t and str(t).strip() and str(t).lower() not in ['nan', 'none', '']]
        
        # Skip if no valid titles remain
        if not expanded_titles:
            continue
        
        # Check if this is the postdoctoral cluster (check original titles)
        is_postdoc_cluster = any(is_postdoc_title(t) for t in cluster_orig)
        
        if is_postdoc_cluster:
            # Map all postdoc titles to "Postdoctoral"
            for title in cluster_orig:
                mapping[title] = "Postdoctoral"
            continue  # Skip to next cluster
        
        # Normal processing for non-postdoc clusters
        # Check if "visiting" appears in majority of titles
        visiting_count = sum(1 for t in expanded_titles if 'visiting' in t.lower())
        has_visiting = visiting_count > len(cluster_orig) / 2
        
        # Filter candidates based on visiting presence
        if has_visiting:
            candidates = [exp for exp in expanded_titles if 'visiting' in exp.lower()]
        else:
            candidates = [exp for exp in expanded_titles if 'visiting' not in exp.lower()]
        
        # If no candidates after filtering, use all titles
        if not candidates:
            candidates = expanded_titles
        
        # Safety check - skip if still no candidates
        if not candidates:
            continue
        
        # Pick the shortest expanded title
        canonical_expanded = min(candidates, key=len)
        
        # Properly title case the canonical name
        canonical = ' '.join(word.capitalize() for word in canonical_expanded.split())
        
        # Map all ORIGINAL titles to this canonical
        for title in cluster_orig:
            mapping[title] = canonical

    # Post-process: rename any canonical title containing "Fellow" to "Postdoctoral"
    for orig_title, canon_title in mapping.items():
        if 'fellow' in canon_title.lower():
            mapping[orig_title] = "Postdoctoral"
            
    # Merge clusters with duplicate canonical titles
    canonical_to_originals = {}
    for orig_title, canon_title in mapping.items():
        if canon_title not in canonical_to_originals:
            canonical_to_originals[canon_title] = []
        canonical_to_originals[canon_title].append(orig_title)
    
    # Rebuild clusters - one cluster per unique canonical title
    merged_clusters = []
    for canon_title, orig_titles in canonical_to_originals.items():
        merged_clusters.append((orig_titles, orig_titles))
    
    return merged_clusters, mapping
    
# --- Sidebar ---
with st.sidebar:
    st.title("Survey Data Cleaner")
    st.markdown("*Automatically combines equivalent job titles and cleans numerical data*")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None and st.session_state["df"] is None:
        with st.spinner("Loading and auto-cleaning..."):
            raw_df = pd.read_csv(uploaded_file, low_memory=False)
            st.session_state["original_df"] = raw_df.copy()
            
            # Auto-clean immediately
            cleaned_df, report = auto_clean_dataframe(raw_df)
            st.session_state["df"] = cleaned_df
            st.session_state["cleaning_report"] = report
            st.session_state["auto_cleaned"] = True
        
        st.success("File loaded & cleaned!")
    
    if st.session_state["df"] is not None:
        st.markdown("---")
        st.subheader("Dataset Stats")
        df = st.session_state["df"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", len(df))
            missing_count = df.isnull().sum().sum()
            st.metric("Missing", missing_count)
        with col2:
            st.metric("Columns", len(df.columns))
            duplicate_count = df.duplicated().sum()
            st.metric("Dupes", duplicate_count)
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.session_state["similarity_threshold"] = st.slider(
                "Clustering threshold:", 80, 100, 90,
                help="Higher = stricter matching"
            )
        
        st.markdown("---")
        
        if st.button("Reset to Original"):
            st.session_state["df"] = st.session_state["original_df"].copy()
            st.session_state["mapping"] = {}
            st.session_state["clusters"] = None
            st.session_state["history"] = []
            st.session_state["auto_cleaned"] = False
            st.session_state["selected_cluster"] = None
            st.rerun()

if st.session_state["df"] is None:
    st.title("Survey Data Cleaner")
    st.info("Upload a CSV file to get started")
    st.stop()

df = st.session_state["df"]
columns = df.columns.tolist()

# --- Main content ---
st.title("Survey Data Cleaner")

# Show cleaning report if auto-cleaned
if st.session_state["auto_cleaned"] and st.session_state["cleaning_report"]:
    report = st.session_state["cleaning_report"]
    
    with st.expander("Auto-Cleaning Report", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", report["original_rows"])
        with col2:
            st.metric("Cleaned Rows", report["final_rows"])
        
        if report["actions"]:
            st.markdown("**Actions Taken:**")
            for action in report["actions"]:
                st.markdown(f"- {action}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Title Cleaning", "Data Overview", "Export"])

# --- TAB 1: Smart Title Cleaning ---
with tab1:
    st.subheader("Job Title Clustering")
    
    # Auto-detect job title column
    detected_col = auto_detect_job_title_column(df)
    
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
            
            clusters, mapping = auto_cluster_titles(titles, threshold)
            
            st.session_state["clusters"] = clusters
            st.session_state["mapping"] = mapping
            st.session_state["selected_cluster"] = None
            
            progress_bar.progress(100)
            progress_bar.empty()
        
        st.success(f"Found {len(clusters)} clusters from {len(titles.unique())} unique titles")
    
    clusters = st.session_state.get("clusters", [])
    mapping = st.session_state.get("mapping", {})
    
    if clusters:
        # Build summary
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
            st.metric("Unique Titles", len(titles.unique()))
        with col3:
            avg_cluster_size = sum(len(c) for c in clusters) / len(clusters)
            st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
        with col4:
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
        
        if st.session_state["selected_cluster"] is not None:
            cluster_idx = st.session_state["selected_cluster"]
            current_cluster_orig, current_cluster_cleaned = clusters[cluster_idx]
            
            st.markdown("---")
            st.subheader(f"Cluster {cluster_idx} Details")
            
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
            
            st.markdown(f"**{len(current_cluster_orig)} variations in this cluster:**")
            variations_df = pd.DataFrame({
                "Original Title": current_cluster_orig,
                "Count": [len(df[df[selected_col] == t]) for t in current_cluster_orig]
            }).sort_values("Count", ascending=False)
            
            st.dataframe(variations_df, use_container_width=True, height=300, hide_index=True)
            
            # Move variation to another cluster
            st.markdown("---")
            st.markdown("**Move a variation to another cluster:**")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                variation_to_move = st.selectbox(
                    "Select variation to move:",
                    current_cluster_orig,
                    key=f"move_var_{cluster_idx}"
                )
            
            with col2:
                # Get all canonical titles except current one
                all_canonicals = sorted(set(st.session_state["mapping"].values()))
                current_canonical = mapping.get(current_cluster_orig[0], current_cluster_orig[0])
                other_canonicals = [c for c in all_canonicals if c != current_canonical]
                
                # Add option to create new cluster
                cluster_options = ["--- Create New Cluster ---"] + (other_canonicals if other_canonicals else [])
                
                target_canonical = st.selectbox(
                    "Move to cluster:",
                    cluster_options,
                    key=f"target_cluster_{cluster_idx}"
                )
            
            with col3:
                st.write("")
                st.write("")
                if st.button("Move", key=f"move_btn_{cluster_idx}"):
                    if target_canonical == "--- Create New Cluster ---":
                        # Create a new cluster with this variation
                        # Use the variation itself as the canonical title (properly capitalized)
                        new_canonical = ' '.join(word.capitalize() for word in variation_to_move.split())
                        st.session_state["mapping"][variation_to_move] = new_canonical
                        
                        # Rebuild clusters from updated mapping
                        new_clusters_dict = {}
                        for orig_title, canon_title in st.session_state["mapping"].items():
                            if canon_title not in new_clusters_dict:
                                new_clusters_dict[canon_title] = []
                            new_clusters_dict[canon_title].append(orig_title)
                        
                        # Convert back to list of tuples format
                        new_clusters = []
                        for canon, origs in new_clusters_dict.items():
                            new_clusters.append((origs, origs))  # simplified
                        
                        st.session_state["clusters"] = new_clusters
                        st.success(f"Created new cluster with '{new_canonical}'")
                        st.rerun()
                    elif other_canonicals:
                        # Update the mapping for this variation
                        st.session_state["mapping"][variation_to_move] = target_canonical
                        
                        # Rebuild clusters from updated mapping
                        new_clusters_dict = {}
                        for orig_title, canon_title in st.session_state["mapping"].items():
                            if canon_title not in new_clusters_dict:
                                new_clusters_dict[canon_title] = []
                            new_clusters_dict[canon_title].append(orig_title)
                        
                        # Convert back to list of tuples format
                        new_clusters = []
                        for canon, origs in new_clusters_dict.items():
                            new_clusters.append((origs, origs))  # simplified
                        
                        st.session_state["clusters"] = new_clusters
                        st.success(f"Moved '{variation_to_move}' to '{target_canonical}'")
                        st.rerun()
                    else:
                        st.warning("No other clusters available")
            
            # Option to create new cluster with custom name
            st.markdown("**Or create a new cluster with a custom name:**")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                variation_for_new = st.selectbox(
                    "Select variation:",
                    current_cluster_orig,
                    key=f"new_cluster_var_{cluster_idx}"
                )
            
            with col2:
                new_cluster_name = st.text_input(
                    "New cluster name:",
                    value=' '.join(word.capitalize() for word in variation_for_new.split()),
                    key=f"new_cluster_name_{cluster_idx}",
                    help="Enter the canonical title for the new cluster"
                )
            
            with col3:
                st.write("")
                st.write("")
                if st.button("Create New", key=f"create_new_btn_{cluster_idx}"):
                    if new_cluster_name.strip():
                        # Create new cluster with custom name
                        st.session_state["mapping"][variation_for_new] = new_cluster_name.strip()
                        
                        # Rebuild clusters from updated mapping
                        new_clusters_dict = {}
                        for orig_title, canon_title in st.session_state["mapping"].items():
                            if canon_title not in new_clusters_dict:
                                new_clusters_dict[canon_title] = []
                            new_clusters_dict[canon_title].append(orig_title)
                        
                        # Convert back to list of tuples format
                        new_clusters = []
                        for canon, origs in new_clusters_dict.items():
                            new_clusters.append((origs, origs))  # simplified
                        
                        st.session_state["clusters"] = new_clusters
                        st.success(f"Created new cluster: '{new_cluster_name.strip()}'")
                        st.rerun()
                    else:
                        st.warning("Please enter a cluster name")
        
        # Merge clusters functionality
        st.markdown("---")
        st.markdown("### Merge Multiple Clusters")
        st.markdown("Combine multiple clusters into one canonical title")
        
        # Get all unique canonical titles
        all_canonicals = sorted(set(st.session_state["mapping"].values()))
        
        if len(all_canonicals) > 1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                clusters_to_merge = st.multiselect(
                    "Select clusters to merge:",
                    all_canonicals,
                    help="Select 2 or more clusters to merge together",
                    key="merge_clusters_select"
                )
                
                if len(clusters_to_merge) >= 2:
                    # Show preview of what will be merged
                    st.markdown(f"**Merging {len(clusters_to_merge)} clusters:**")
                    
                    total_variations = 0
                    total_records = 0
                    
                    for canonical in clusters_to_merge:
                        # Count variations and records for this canonical
                        variations = [title for title, canon in st.session_state["mapping"].items() if canon == canonical]
                        records = sum(len(df[df[selected_col] == var]) for var in variations)
                        total_variations += len(variations)
                        total_records += records
                        st.markdown(f"- **{canonical}**: {len(variations)} variations, {records} records")
                    
                    st.info(f"Total: {total_variations} variations, {total_records} records")
                    
                    # New canonical name input
                    default_name = clusters_to_merge[0]  # Use first selected as default
                    new_merged_canonical = st.text_input(
                        "New canonical title for merged cluster:",
                        value=default_name,
                        key="merged_canonical_name",
                        help="All selected clusters will use this title"
                    )
            
            with col2:
                st.write("")
                st.write("")
                st.write("")
                if len(clusters_to_merge) >= 2:
                    if st.button("Merge Clusters", type="primary", key="merge_btn"):
                        if new_merged_canonical.strip():
                            # Update all mappings for selected clusters
                            for canonical in clusters_to_merge:
                                for title, canon in st.session_state["mapping"].items():
                                    if canon == canonical:
                                        st.session_state["mapping"][title] = new_merged_canonical.strip()
                            
                            # Rebuild clusters from updated mapping
                            new_clusters_dict = {}
                            for orig_title, canon_title in st.session_state["mapping"].items():
                                if canon_title not in new_clusters_dict:
                                    new_clusters_dict[canon_title] = []
                                new_clusters_dict[canon_title].append(orig_title)
                            
                            # Convert back to list of tuples format
                            new_clusters = []
                            for canon, origs in new_clusters_dict.items():
                                new_clusters.append((origs, origs))
                            
                            st.session_state["clusters"] = new_clusters
                            st.success(f"Merged {len(clusters_to_merge)} clusters into '{new_merged_canonical.strip()}'")
                            st.rerun()
                        else:
                            st.warning("Please enter a canonical name")
        else:
            st.info("Need at least 2 clusters to merge. Create more clusters first.")

# --- TAB 2: Data Overview ---
with tab2:
    st.subheader("Dataset Overview")
    
    # Top 100 Job Titles Section
    st.markdown("### Top 100 Job Titles")
    
    # Auto-detect or use selected job title column
    if st.session_state.get("last_selected_col") and st.session_state["last_selected_col"] in df.columns:
        job_col = st.session_state["last_selected_col"]
    else:
        job_col = auto_detect_job_title_column(df)
    
    if job_col and job_col in df.columns:
        # Check if we have mappings (canonical titles)
        if st.session_state.get("mapping"):
            # Use canonical titles
            df_with_canonical = df.copy()
            df_with_canonical[job_col] = df_with_canonical[job_col].replace(st.session_state["mapping"])
            title_counts = df_with_canonical[job_col].value_counts().head(100)
            title_type = "Canonical"
        else:
            # Use original titles
            title_counts = df[job_col].value_counts().head(100)
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
            st.metric("Top 100 Total Records", title_counts.sum())
        with col3:
            coverage = (title_counts.sum() / len(df)) * 100
            st.metric("Coverage", f"{coverage:.1f}%")
    else:
        st.info("No job title column detected. Please run Auto-Cluster in the Title Cleaning tab first.")
    
    st.markdown("---")
    
    # Age and Gender Analysis Section
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    gender_cols = [col for col in df.columns if 'gender' in col.lower()]
    
    if age_cols or gender_cols:
        st.markdown("### Age & Gender Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if age_cols:
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
        
        with col2:
            if gender_cols:
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
        
        st.markdown("---")
        
# --- TAB 3: Export ---
with tab3:
    st.subheader("Export Cleaned Data")

    if st.session_state.get("mapping"):
        df_export = df.copy()
        job_col = st.session_state["last_selected_col"]

        # Apply the canonical mappings to the job titles
        df_export[job_col] = df_export[job_col].replace(st.session_state["mapping"])

        # Show preview
        st.markdown("#### Preview of cleaned data:")
        st.dataframe(df_export.head(20), use_container_width=True, height=300)
        
        st.markdown("---")
        
        # Show mapping summary
        st.markdown("#### Cleaning Summary:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Unique Titles", len(df[job_col].unique()))
        with col2:
            st.metric("Cleaned Unique Titles", len(df_export[job_col].unique()))
        with col3:
            reduction = len(df[job_col].unique()) - len(df_export[job_col].unique())
            st.metric("Titles Consolidated", reduction)
        
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

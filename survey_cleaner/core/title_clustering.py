"""Job title clustering logic."""

import pandas as pd
from rapidfuzz import fuzz
from typing import Tuple, List, Dict
from core.text_normalization import (
    normalize_title, 
    is_postdoc_title, 
    fix_typos_and_abbreviations
)
from config.settings import JOB_KEYWORDS

def detect_job_title_column(df: pd.DataFrame) -> str:
    """Automatically detect which column likely contains job titles."""
    from config.settings import JOB_TITLE_KEYWORDS, JOB_CONTENT_KEYWORDS
    
    candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check column name
        if any(keyword in col_lower for keyword in JOB_TITLE_KEYWORDS):
            candidates.append((col, 100))
        
        # Check content patterns
        elif df[col].dtype == 'object':
            sample = df[col].dropna().head(100).astype(str).str.lower()
            matches = sum(
                sample.str.contains('|'.join(JOB_CONTENT_KEYWORDS), regex=True)
            )
            if matches > len(sample) * 0.3:
                candidates.append((col, matches / len(sample) * 100))
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    return df.columns[0] if len(df.columns) > 0 else None

def cluster_titles(titles: pd.Series, 
                  threshold: int = 90) -> Tuple[List[Tuple], Dict[str, str]]:
    """
    Cluster and clean job titles with smart typo correction.
    
    Returns:
        clusters: List of (original_titles, cleaned_titles) tuples
        mapping: Dict mapping original titles to canonical titles
    """
    # Separate postdoctoral titles
    postdoc_mask = titles.map(is_postdoc_title)
    postdoc_titles = titles[postdoc_mask]
    non_postdoc_titles = titles[~postdoc_mask]
    
    # Normalize non-postdoc titles
    normalized_titles = non_postdoc_titles.map(
        lambda t: normalize_title(t, JOB_KEYWORDS)
    )
    unique_norm_titles = normalized_titles.unique()
    
    # Create cleaned versions
    cleaned_titles = non_postdoc_titles.map(fix_typos_and_abbreviations)
    
    # Cluster similar titles
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
        
        # Get original and cleaned versions
        mask = normalized_titles.isin(cluster)
        full_titles = non_postdoc_titles[mask].unique().tolist()
        full_titles_cleaned = cleaned_titles[mask].unique().tolist()
        
        clusters.append((full_titles, full_titles_cleaned))
    
    # Add postdoctoral cluster
    if len(postdoc_titles) > 0:
        postdoc_list = postdoc_titles.unique().tolist()
        clusters.append((postdoc_list, postdoc_list))
    
    # Generate canonical titles
    mapping = _generate_canonical_mapping(clusters)
    
    return clusters, mapping

def _generate_canonical_mapping(clusters: List[Tuple]) -> Dict[str, str]:
    """Generate mapping from original titles to canonical titles."""
    mapping = {}
    
    for cluster_orig, cluster_cleaned in clusters:
        expanded_titles = cluster_cleaned
        
        # Filter out invalid titles
        expanded_titles = [
            t for t in expanded_titles 
            if t and str(t).strip() and str(t).lower() not in ['nan', 'none', '']
        ]
        
        if not expanded_titles:
            continue
        
        # Check if postdoctoral cluster
        is_postdoc_cluster = any(is_postdoc_title(t) for t in cluster_orig)
        
        if is_postdoc_cluster:
            for title in cluster_orig:
                mapping[title] = "Postdoctoral"
            continue
        
        # Check for visiting majority
        visiting_count = sum(
            1 for t in expanded_titles if 'visiting' in t.lower()
        )
        has_visiting = visiting_count > len(cluster_orig) / 2
        
        # Filter candidates
        if has_visiting:
            candidates = [t for t in expanded_titles if 'visiting' in t.lower()]
        else:
            candidates = [t for t in expanded_titles if 'visiting' not in t.lower()]
        
        if not candidates:
            candidates = expanded_titles
        
        if not candidates:
            continue
        
        # Pick shortest title
        canonical_expanded = min(candidates, key=len)
        canonical = ' '.join(word.capitalize() for word in canonical_expanded.split())
        
        for title in cluster_orig:
            mapping[title] = canonical
    
    # Post-process: rename fellow titles to Postdoctoral
    for orig_title, canon_title in mapping.items():
        if 'fellow' in canon_title.lower():
            mapping[orig_title] = "Postdoctoral"
    
    return mapping

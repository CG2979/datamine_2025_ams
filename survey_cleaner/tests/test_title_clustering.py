"""Tests for title clustering functions."""

import pytest
import pandas as pd
from core.title_clustering import detect_job_title_column, cluster_titles

def test_detect_job_title_column():
    """Test job title column detection."""
    df = pd.DataFrame({
        'name': ['John', 'Jane'],
        'job_title': ['Professor', 'Lecturer'],
        'age': [45, 35]
    })
    
    result = detect_job_title_column(df)
    assert result == 'job_title'

def test_detect_job_title_by_content():
    """Test detection by content analysis."""
    df = pd.DataFrame({
        'column_a': ['data1', 'data2'],
        'column_b': ['Assistant Professor', 'Associate Professor']
    })
    
    result = detect_job_title_column(df)
    assert result == 'column_b'

def test_cluster_titles_basic():
    """Test basic title clustering."""
    titles = pd.Series([
        'Assistant Professor',
        'Asst. Professor',
        'Assistant Prof',
        'Lecturer',
        'Postdoctoral Fellow'
    ])
    
    clusters, mapping = cluster_titles(titles, threshold=85)
    
    # Should cluster the assistant professor variants
    assert len(clusters) < len(titles.unique())
    
    # Postdoctoral should map to "Postdoctoral"
    assert mapping['Postdoctoral Fellow'] == 'Postdoctoral'
    
    # Assistant professor variants should map to same canonical
    canonical_asst = mapping['Assistant Professor']
    assert mapping['Asst. Professor'] == canonical_asst

def test_cluster_titles_postdoc_separation():
    """Test that postdoc titles are separated correctly."""
    titles = pd.Series([
        'Postdoctoral Fellow',
        'Post-doc Researcher',
        'Postdoc',
        'Professor'
    ])
    
    clusters, mapping = cluster_titles(titles, threshold=90)
    
    # All postdoc variants should map to "Postdoctoral"
    assert mapping['Postdoctoral Fellow'] == 'Postdoctoral'
    assert mapping['Post-doc Researcher'] == 'Postdoctoral'
    assert mapping['Postdoc'] == 'Postdoctoral'
    
    # Professor should be separate
    assert mapping['Professor'] != 'Postdoctoral'

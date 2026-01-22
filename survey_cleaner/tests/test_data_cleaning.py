"""Tests for data cleaning functions."""

import pytest
import pandas as pd
import numpy as np
from core.data_cleaning import (
    remove_empty_rows,
    clean_text_columns,
    clean_age_columns,
    clean_gender_columns
)

def test_remove_empty_rows():
    """Test empty row removal."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, np.nan, 6]
    })
    result = remove_empty_rows(df)
    assert len(result) == 2

def test_clean_text_columns():
    """Test text column cleaning."""
    df = pd.DataFrame({
        'title': ['  Assistant Professor  ', 'Lecturer   ', 'nan']
    })
    result = clean_text_columns(df)
    
    assert result['title'].iloc[0] == 'Assistant Professor'
    assert result['title'].iloc[1] == 'Lecturer'
    assert pd.isna(result['title'].iloc[2])

def test_clean_age_columns():
    """Test age column cleaning."""
    df = pd.DataFrame({
        'age': [25, 150, 10, 45, 'invalid', None]
    })
    result = clean_age_columns(df)
    
    assert result['age'].iloc[0] == 25  # Valid
    assert result['age'].iloc[1] == 0   # Too high
    assert result['age'].iloc[2] == 0   # Too low
    assert result['age'].iloc[3] == 45  # Valid
    assert result['age'].iloc[4] == 0   # Invalid converted
    assert result['age'].iloc[5] == 0   # NaN filled

def test_clean_gender_columns():
    """Test gender column cleaning."""
    df = pd.DataFrame({
        'gender': ['Male', 'Female', np.nan, 'nan', 'Other']
    })
    result = clean_gender_columns(df)
    
    assert result['gender'].iloc[0] == 'Male'
    assert result['gender'].iloc[1] == 'Female'
    assert result['gender'].iloc[2] == ''
    assert result['gender'].iloc[3] == ''

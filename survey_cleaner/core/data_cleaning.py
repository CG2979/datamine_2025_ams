"""Data cleaning functions for survey data."""

import pandas as pd
import numpy as np
import re
from config.settings import MIN_VALID_AGE, MAX_VALID_AGE

def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove completely empty rows from dataframe."""
    return df.dropna(how='all')

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text columns by stripping whitespace and normalizing."""
    df = df.copy()
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            # Strip whitespace and remove extra spaces
            cleaned = df[col].astype(str).str.strip()
            cleaned = cleaned.apply(
                lambda x: re.sub(r'\s+', ' ', x) if x != 'nan' else x
            )
            
            # Replace 'nan' strings with actual NaN
            cleaned = cleaned.replace('nan', np.nan)
            df[col] = cleaned
    
    return df

def clean_age_columns(df: pd.DataFrame, 
                     min_age: int = MIN_VALID_AGE, 
                     max_age: int = MAX_VALID_AGE) -> pd.DataFrame:
    """Clean age columns by validating ranges."""
    df = df.copy()
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    
    for col in age_cols:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == 'object':
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN with 0
            df[col] = df[col].fillna(0)
            
            # Set invalid ages to 0
            df.loc[(df[col] > max_age) | (df[col] < min_age), col] = 0
    
    return df

def clean_gender_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean gender columns by normalizing values."""
    df = df.copy()
    gender_cols = [col for col in df.columns if 'gender' in col.lower()]
    
    for col in gender_cols:
        if df[col].dtype == 'object' or pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).replace('nan', '')
            df[col] = df[col].fillna('')
    
    return df

def auto_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically clean the dataframe with smart defaults."""
    df = remove_empty_rows(df)
    df = clean_text_columns(df)
    df = clean_age_columns(df)
    df = clean_gender_columns(df)
    return df


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def safe_to_int(value):
    """
    Safely convert any value to int (0 or 1)
    Handles: strings, bools, ints, floats, None, NaN
    
    Args:
        value: Any type of value
        
    Returns:
        int: 0 or 1
    """
    if value is None or pd.isna(value):
        return 0
    
    if isinstance(value, (int, np.integer)):
        return int(value)
    
    if isinstance(value, bool):
        return int(value)
    
    if isinstance(value, float):
        if pd.isna(value):
            return 0
        return 1 if value > 0.5 else 0
    
    if isinstance(value, str):
        normalized = value.strip().lower()
        true_vals = {'true', 't', '1', '1.0', 'yes', 'y', 'success'}
        false_vals = {'false', 'f', '0', '0.0', 'no', 'n', 'fail', 'failure'}
        
        if normalized in true_vals:
            return 1
        elif normalized in false_vals:
            return 0
    
    return 0  # Default fallback


def convert_column_to_int(df, col_name, safe=True):
    """
    Convert a column safely to int (0 or 1)
    
    Args:
        df: DataFrame
        col_name: Column name to convert
        safe: If True, use safe_to_int; otherwise use astype
        
    Returns:
        DataFrame with converted column
    """
    if col_name not in df.columns:
        return df
    
    if safe:
        df[col_name] = df[col_name].apply(safe_to_int).astype(int)
    else:
        try:
            df[col_name] = df[col_name].astype(int, errors='coerce').fillna(0).astype(int)
        except Exception as e:
            logger.warning(f"Could not convert {col_name}: {e}")
            df[col_name] = df[col_name].apply(safe_to_int).astype(int)
    
    return df


def validate_binary_column(df, col_name):
    """
    Validate that a column contains only binary values (0, 1)
    
    Args:
        df: DataFrame
        col_name: Column name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if col_name not in df.columns:
        return False
    
    unique_vals = df[col_name].dropna().unique()
    return all(v in [0, 1, '0', '1', True, False] for v in unique_vals)


def safe_numeric(df, cols):
    """
    Safely convert columns to numeric
    
    Args:
        df: DataFrame
        cols: List of column names
        
    Returns:
        DataFrame with numeric columns
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

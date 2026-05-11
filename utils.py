import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def safe_to_int(value):
    """
    Safely convert any value to binary int (0 or 1)

    Handles:
    - strings
    - bools
    - ints
    - floats
    - None
    - NaN

    Args:
        value: Any type of value

    Returns:
        int: 0 or 1
    """

    # Handle None / NaN
    if value is None or pd.isna(value):
        return 0

    # Handle booleans first
    if isinstance(value, bool):
        return int(value)

    # Handle integers
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) > 0 else 0

    # Handle floats
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return 0
        return 1 if value > 0.5 else 0

    # Handle strings
    if isinstance(value, str):
        normalized = value.strip().lower()

        true_vals = {
            'true', 't', '1', '1.0',
            'yes', 'y', 'success'
        }

        false_vals = {
            'false', 'f', '0', '0.0',
            'no', 'n', 'fail', 'failure'
        }

        if normalized in true_vals:
            return 1

        if normalized in false_vals:
            return 0

    # Default fallback
    return 0


def convert_column_to_int(df, col_name, safe=True):
    """
    Safely convert a DataFrame column to binary int (0 or 1)

    Args:
        df: pandas DataFrame
        col_name: Column name
        safe: Use safe conversion logic

    Returns:
        DataFrame
    """

    if col_name not in df.columns:
        logger.warning(f"Column '{col_name}' not found.")
        return df

    try:
        if safe:
            df[col_name] = (
                df[col_name]
                .apply(safe_to_int)
                .astype(int)
            )
        else:
            df[col_name] = (
                pd.to_numeric(df[col_name], errors='coerce')
                .fillna(0)
                .astype(int)
            )

    except Exception as e:
        logger.warning(f"Could not safely convert {col_name}: {e}")

        df[col_name] = (
            df[col_name]
            .apply(safe_to_int)
            .astype(int)
        )

    return df


def validate_binary_column(df, col_name):
    """
    Validate that a column contains only binary values

    Args:
        df: pandas DataFrame
        col_name: Column name

    Returns:
        bool
    """

    if col_name not in df.columns:
        logger.warning(f"Column '{col_name}' not found.")
        return False

    unique_vals = df[col_name].dropna().unique()

    valid_values = {0, 1, '0', '1', True, False}

    return all(v in valid_values for v in unique_vals)


def safe_numeric(df, cols):
    """
    Safely convert columns to numeric values

    Args:
        df: pandas DataFrame
        cols: list of column names

    Returns:
        DataFrame
    """

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors='coerce'
            )

    return df

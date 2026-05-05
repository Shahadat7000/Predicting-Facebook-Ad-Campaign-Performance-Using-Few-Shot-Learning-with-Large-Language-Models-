"""
Data Cleaning Script for Facebook Ad Campaign Dataset
Author: Thesis Researcher
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = "data/data.csv"
OUTPUT_CLEAN_FILE = "data/data_clean.csv"
OUTPUT_BAD_FILE = "data/bad_rows_report.csv"

EXPECTED_COLUMNS = [
    "ad_id", "reporting_start", "reporting_end", "campaign_id", "fb_campaign_id",
    "age", "gender", "interest1", "interest2", "interest3",
    "impressions", "clicks", "spent", "total_conversion", "approved_conversion"
]

VALID_AGES = {"30-34", "35-39", "40-44", "45-49"}
VALID_GENDERS = {"M", "F"}


# =========================
# HELPER FUNCTIONS
# =========================
def safe_numeric(df, cols):
    """Safely convert columns to numeric"""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load and validate raw data"""
    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check for missing/extra columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing_cols:
        print(f" Missing columns: {missing_cols}")
    if extra_cols:
        print(f" Extra columns: {extra_cols}")

    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data types and strip spaces"""
    # Strip spaces from object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Convert numeric columns
    numeric_cols = ["ad_id", "interest1", "interest2", "interest3",
                    "impressions", "clicks", "spent",
                    "total_conversion", "approved_conversion"]
    df = safe_numeric(df, numeric_cols)

    return df


def find_corrupted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Identify corrupted rows (column shift)"""
    age_ok = df["age"].astype(str).isin(VALID_AGES)
    gender_ok = df["gender"].astype(str).isin(VALID_GENDERS)

    # Check if ID columns contain age/gender values (indicating shift)
    campaign_bad = df["campaign_id"].astype(str).isin(VALID_AGES | VALID_GENDERS)
    fb_campaign_bad = df["fb_campaign_id"].astype(str).isin(VALID_AGES | VALID_GENDERS)

    corrupted = df[
        (~age_ok) | (~gender_ok) | campaign_bad | fb_campaign_bad
        ].copy()

    if len(corrupted) > 0:
        corrupted["issue_age"] = ~age_ok.loc[corrupted.index]
        corrupted["issue_gender"] = ~gender_ok.loc[corrupted.index]
        corrupted["issue_campaign_shift"] = campaign_bad.loc[corrupted.index]
        corrupted["issue_fb_shift"] = fb_campaign_bad.loc[corrupted.index]

    return corrupted


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics with proper NaN handling"""
    # Fill missing conversions with 0
    df["total_conversion"] = df["total_conversion"].fillna(0)
    df["approved_conversion"] = df["approved_conversion"].fillna(0)

    # CTR: (clicks / impressions) * 100
    df["CTR"] = np.where(
        df["impressions"] > 0,
        (df["clicks"] / df["impressions"]) * 100,
        0.0  # 0 clicks = 0 CTR
    )

    # CPC: spent / clicks
    df["CPC"] = np.where(
        df["clicks"] > 0,
        df["spent"] / df["clicks"],
        np.nan  # CPC undefined when no clicks
    )

    # Conversion Rate: (approved_conversion / clicks) * 100
    df["Conversion_Rate"] = np.where(
        df["clicks"] > 0,
        (df["approved_conversion"] / df["clicks"]) * 100,
        0.0  # No clicks = 0 conversion rate
    )

    return df


def validate_clean_data(df: pd.DataFrame):
    """Print validation summary for clean data"""
    print("\n" + "=" * 60)
    print("CLEAN DATA VALIDATION")
    print("=" * 60)

    print("\n Age distribution:")
    print(df["age"].value_counts().sort_index())

    print("\n Gender distribution:")
    print(df["gender"].value_counts())

    print("\n Missing values:")
    print(df.isna().sum())

    print("\n Metric summary:")
    for col in ["CTR", "CPC", "Conversion_Rate"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Median: {df[col].median():.4f}")
            print(f"  Missing: {df[col].isna().sum()}")


# =========================
# MAIN EXECUTION
# =========================
def main():
    print("\n" + "=" * 60)
    print("FACEBOOK AD DATASET CLEANING")
    print("=" * 60)

    # Check input file
    if not Path(INPUT_FILE).exists():
        print(f" File not found: {INPUT_FILE}")
        return

    # Load data
    df = load_data(INPUT_FILE)
    df = normalize_types(df)

    # Find corrupted rows
    corrupted = find_corrupted_rows(df)

    print("\n" + "=" * 60)
    print("CORRUPTION ANALYSIS")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Corrupted rows: {len(corrupted)}")
    print(f"Clean rows: {len(df) - len(corrupted)}")
    print(f"Corruption rate: {(len(corrupted) / len(df)) * 100:.2f}%")

    # Save corrupted rows report
    if not corrupted.empty:
        print("\n Sample corrupted rows (first 10):")
        cols_to_show = ["ad_id", "campaign_id", "fb_campaign_id", "age", "gender",
                        "issue_age", "issue_gender", "issue_campaign_shift", "issue_fb_shift"]
        cols_to_show = [c for c in cols_to_show if c in corrupted.columns]
        print(corrupted[cols_to_show].head(10).to_string(index=False))

        corrupted.to_csv(OUTPUT_BAD_FILE, index=False)
        print(f"\n Corrupted rows saved to: {OUTPUT_BAD_FILE}")

    # Keep only clean rows
    clean_df = df[
        df["age"].astype(str).isin(VALID_AGES) &
        df["gender"].astype(str).isin(VALID_GENDERS)
        ].copy()

    # Calculate metrics
    clean_df = calculate_metrics(clean_df)

    # Validate
    validate_clean_data(clean_df)

    # Save clean dataset
    clean_df.to_csv(OUTPUT_CLEAN_FILE, index=False)
    print(f"\n Clean dataset saved to: {OUTPUT_CLEAN_FILE}")

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"\nOriginal rows: {len(df)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Removed rows: {len(df) - len(clean_df)}")

    return clean_df


if __name__ == "__main__":
    clean_df = main()
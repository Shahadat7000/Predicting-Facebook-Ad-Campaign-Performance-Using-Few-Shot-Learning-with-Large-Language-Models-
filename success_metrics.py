import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import safe_to_int


def generate_success_metrics_report(
    df: pd.DataFrame,
    output_dir: str = "visualizations"
):
    """
    Generate a report on success metrics and save basic visualizations.
    """

    Path(output_dir).mkdir(exist_ok=True)

    df = df.copy()

    if "is_success" not in df.columns:
        raise ValueError("DataFrame must contain 'is_success' column.")

    df["is_success"] = df["is_success"].apply(safe_to_int)

    print("\n" + "=" * 60)
    print("SUCCESS METRICS COMPREHENSIVE REPORT")
    print("=" * 60)

    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total number of campaigns: {len(df)}")
    print(f"Number of features: {len(df.columns)}")

    preview_cols = ", ".join(df.columns[:10])
    print(f"Features: {preview_cols}...")

    print("\n2. SUCCESS DISTRIBUTION")
    print("-" * 40)

    success_count = int(df["is_success"].sum())
    fail_count = int(len(df) - success_count)
    success_rate = (success_count / len(df)) * 100 if len(df) > 0 else 0.0

    print(f"Successful campaigns: {success_count} ({success_rate:.2f}%)")
    print(f"Unsuccessful campaigns: {fail_count} ({100 - success_rate:.2f}%)")

    print("\n3. DEMOGRAPHIC BREAKDOWN")
    print("-" * 40)

    gender_dist = {}
    age_dist = {}

    if "gender" in df.columns:
        print("\nGender distribution:")
        gender_dist = df["gender"].value_counts().to_dict()

        for gender, count in gender_dist.items():
            pct = (count / len(df)) * 100 if len(df) > 0 else 0.0
            print(f"  {gender}: {count} ({pct:.1f}%)")

    if "age" in df.columns:
        print("\nAge group distribution:")
        age_dist = df["age"].value_counts().sort_index().to_dict()

        for age, count in age_dist.items():
            pct = (count / len(df)) * 100 if len(df) > 0 else 0.0
            print(f"  {age}: {count} ({pct:.1f}%)")

    print("\n4. SUCCESS RATE BY DEMOGRAPHIC")
    print("-" * 40)

    success_by_gender = {}
    success_by_age = {}

    if "gender" in df.columns:
        print("\nBy Gender:")
        success_by_gender = (
            df.groupby("gender")["is_success"].mean() * 100
        ).to_dict()

        for gender, rate in success_by_gender.items():
            print(f"  {gender}: {rate:.2f}%")

    if "age" in df.columns:
        print("\nBy Age Group:")
        success_by_age = (
            df.groupby("age")["is_success"].mean() * 100
        ).to_dict()

        for age, rate in success_by_age.items():
            print(f"  {age}: {rate:.2f}%")

    print("\n5. METRIC STATISTICS")
    print("-" * 40)

    metrics = ["CTR", "CPC", "Conversion_Rate"]
    existing_metrics = [metric for metric in metrics if metric in df.columns]

    for metric in existing_metrics:
        metric_values = pd.to_numeric(df[metric], errors="coerce")

        print(f"\n{metric}:")
        print(f"  Mean: {metric_values.mean():.4f}")
        print(f"  Median: {metric_values.median():.4f}")
        print(f"  Std Dev: {metric_values.std():.4f}")
        print(f"  Min: {metric_values.min():.4f}")
        print(f"  Max: {metric_values.max():.4f}")

    print("\n6. GENERATING VISUALIZATIONS")
    print("-" * 40)

    # Success distribution pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        [success_count, fail_count],
        labels=["Successful", "Unsuccessful"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Campaign Success Distribution", fontsize=14, fontweight="bold")
    plt.savefig(
        f"{output_dir}/success_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {output_dir}/success_distribution.png")

    # Metrics distribution by success status
    if existing_metrics:
        fig, axes = plt.subplots(1, len(existing_metrics), figsize=(5 * len(existing_metrics), 5))

        if len(existing_metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(existing_metrics):
            ax = axes[idx]

            success_data = pd.to_numeric(
                df[df["is_success"] == 1][metric],
                errors="coerce"
            ).dropna()

            fail_data = pd.to_numeric(
                df[df["is_success"] == 0][metric],
                errors="coerce"
            ).dropna()

            ax.hist(success_data, alpha=0.7, bins=20, label="Successful")
            ax.hist(fail_data, alpha=0.7, bins=20, label="Unsuccessful")
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{metric} Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/metrics_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        print(f"  Saved: {output_dir}/metrics_distribution.png")

    report = {
        "total_campaigns": int(len(df)),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "success_rate": float(success_rate),
        "gender_distribution": gender_dist,
        "age_distribution": age_dist,
        "success_by_gender": {
            str(k): float(v) for k, v in success_by_gender.items()
        },
        "success_by_age": {
            str(k): float(v) for k, v in success_by_age.items()
        }
    }

    with open(f"{output_dir}/success_metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"  Saved: {output_dir}/success_metrics_report.json")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)

    return report


if __name__ == "__main__":
    df = pd.read_csv("data/processed_data.csv")
    report = generate_success_metrics_report(df)

"""
Compute H2 and H3 statistics for all models:
DeepSeek v3.2, GPT OSS 120b, and Qwen3.5.
"""

import json
import math
import pandas as pd
from statsmodels.stats.proportion import (
    proportions_ztest,
    confint_proportions_2indep
)

# ------------------------------
# Files
# ------------------------------
MODEL_FILES = {
    "DeepSeek v3.2": "results/experiment_results_deepseek_v3_2.json",
    "GPT OSS 120b": "results/experiment_results_gpt_oss_120b.json",
    "Qwen3.5": "results/experiment_results_qwen3_5.json",
}


SHOT_LEVELS = [1, 5]

# ------------------------------
# Load test data
# ------------------------------
X_test = pd.read_csv("data/X_test.csv")
X_test["ad_id"] = X_test["ad_id"].astype(int)


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def compute_metrics(y_pred, X_subset):
    # H3: CTR condition vs conversion condition
    ctr_success = (X_subset["CTR"] > 0.90).astype(int)
    conv_success = (X_subset["Conversion_Rate"] > 3.0).astype(int)

    ctr_correct = (y_pred == ctr_success).sum()
    conv_correct = (y_pred == conv_success).sum()
    total = len(y_pred)

    ctr_acc = ctr_correct / total
    conv_acc = conv_correct / total
    diff = conv_acc - ctr_acc

    ci_low, ci_high = confint_proportions_2indep(
        conv_correct,
        total,
        ctr_correct,
        total,
        method="agresti-caffo"
    )

    _, p_value = proportions_ztest(
        [conv_correct, ctr_correct],
        [total, total],
        alternative="two-sided"
    )

    h_value = cohens_h(conv_acc, ctr_acc)

    return {
        "ctr_acc": ctr_acc,
        "conv_acc": conv_acc,
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "cohens_h": h_value,
        "ctr_correct": int(ctr_correct),
        "conv_correct": int(conv_correct),
        "total": int(total),
    }


for model_name, file_path in MODEL_FILES.items():
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_df = pd.DataFrame(data["results"])

    results_df["shot_level"] = results_df["shot_level"].astype(int)
    results_df["prediction"] = results_df["prediction"].astype(int)
    results_df["actual"] = results_df["actual"].astype(int)
    results_df["ad_id"] = results_df["ad_id"].astype(int)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    for shot in SHOT_LEVELS:
        shot_df = results_df[results_df["shot_level"] == shot].copy()

        merged = shot_df.merge(
            X_test[["ad_id", "CTR", "Conversion_Rate"]],
            on="ad_id",
            how="left"
        )

        merged = merged.dropna(subset=["CTR", "Conversion_Rate"])

        y_pred = merged["prediction"].values
        X_subset = merged[["CTR", "Conversion_Rate"]].copy()

        metrics = compute_metrics(y_pred, X_subset)

        print(f"\n{shot}-shot H3: CTR vs Conversion Rate")
        print(f"CTR-based accuracy:        {metrics['ctr_acc'] * 100:.2f}% "
              f"({metrics['ctr_correct']}/{metrics['total']})")
        print(f"Conversion-based accuracy: {metrics['conv_acc'] * 100:.2f}% "
              f"({metrics['conv_correct']}/{metrics['total']})")
        print(f"Difference: {metrics['diff'] * 100:.2f} percentage points")
        print(f"95% CI: [{metrics['ci_low'] * 100:.2f}%, {metrics['ci_high'] * 100:.2f}%]")
        print(f"p-value: {metrics['p_value']:.4f}")
        print(f"Cohen's h: {metrics['cohens_h']:.4f}")
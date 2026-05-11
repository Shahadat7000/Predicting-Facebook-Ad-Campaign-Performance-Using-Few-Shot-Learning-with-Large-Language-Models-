import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest

from utils import convert_column_to_int, validate_binary_column, safe_to_int

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical analysis for thesis hypothesis testing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        results_df: Optional[pd.DataFrame] = None,
        alpha: float = 0.05
    ):
        self.df = df.copy()
        self.results_df = results_df.copy() if results_df is not None else None
        self.alpha = alpha
        self.results = {}

        if "is_success" in self.df.columns:
            self.df["is_success"] = self.df["is_success"].apply(safe_to_int)

        for col in ["success_ctr", "success_conversion", "success_cpc"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(safe_to_int)

        if self.results_df is not None:
            for col in ["prediction", "actual", "correct"]:
                if col in self.results_df.columns:
                    self.results_df = convert_column_to_int(
                        self.results_df,
                        col,
                        safe=True
                    )

            logger.info("\n" + "=" * 60)
            logger.info("DATA TYPE CONVERSION")
            logger.info("=" * 60)

            for col in ["prediction", "actual", "correct"]:
                if col in self.results_df.columns:
                    is_valid = validate_binary_column(self.results_df, col)
                    unique_vals = self.results_df[col].unique()
                    logger.info(
                        f"{col}: dtype={self.results_df[col].dtype}, "
                        f"valid={is_valid}, unique={unique_vals}"
                    )

            logger.info("=" * 60 + "\n")

    def test_primary_hypothesis(self) -> Dict:
        """
        Test H1: one-shot learning achieves 60-70% accuracy.
        """
        if self.results_df is None:
            return {"error": "No results data available"}

        accuracy_by_shot = {}

        for shot_level in [0, 1, 3, 5]:
            shot_data = self.results_df[
                self.results_df["shot_level"] == shot_level
            ].copy()

            if shot_data.empty:
                continue

            for col in ["correct", "actual", "prediction"]:
                if col in shot_data.columns:
                    shot_data[col] = shot_data[col].apply(safe_to_int)

            correct = int(shot_data["correct"].sum())
            total = int(len(shot_data))
            accuracy = correct / total if total > 0 else 0.0

            if total > 0:
                try:
                    z_stat, p_value = proportions_ztest(
                        count=correct,
                        nobs=total,
                        value=0.5,
                        alternative="larger"
                    )
                except Exception:
                    z_stat, p_value = 0.0, 1.0
            else:
                z_stat, p_value = 0.0, 1.0

            se = np.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0
            ci_lower = accuracy - 1.96 * se
            ci_upper = accuracy + 1.96 * se

            actuals = shot_data["actual"].values
            preds = shot_data["prediction"].values

            if len(actuals) > 0 and len(preds) > 0 and len(actuals) == len(preds):
                balanced_acc = balanced_accuracy_score(actuals, preds)
                mcc = matthews_corrcoef(actuals, preds)
                cm = confusion_matrix(actuals, preds, labels=[0, 1]).tolist()
            else:
                balanced_acc, mcc, cm = 0.0, 0.0, []

            accuracy_by_shot[int(shot_level)] = {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "mcc": float(mcc),
                "confusion_matrix": cm,
                "correct": int(correct),
                "total": int(total),
                "z_statistic": float(z_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < self.alpha),
                "ci_95": [
                    float(max(0.0, ci_lower)),
                    float(min(1.0, ci_upper))
                ]
            }

            logger.info(
                f"Shot {shot_level}: "
                f"acc={accuracy:.2%}, "
                f"bal_acc={balanced_acc:.2%}, "
                f"mcc={mcc:.4f}"
            )

        if 1 in accuracy_by_shot:
            one_shot_acc = accuracy_by_shot[1]["accuracy"]
            in_range = 0.60 <= one_shot_acc <= 0.70

            accuracy_by_shot["hypothesis_result"] = {
                "one_shot_accuracy": float(one_shot_acc),
                "target_range": [0.60, 0.70],
                "in_target_range": bool(in_range),
                "hypothesis_supported": bool(
                    in_range and accuracy_by_shot[1]["significant"]
                )
            }

        self.results["primary_hypothesis"] = accuracy_by_shot
        return accuracy_by_shot

    def test_sub_hypothesis_1a(self) -> Dict:
        """
        Test H1a: model performs better on high-CTR ads.
        """
        if self.results_df is None:
            return {"error": "No results data available"}

        results_df = self.results_df.copy()

        if "ad_index" in results_df.columns and "ad_id" not in results_df.columns:
            results_df["ad_id"] = results_df["ad_index"]

        if "ad_id" not in results_df.columns or "ad_id" not in self.df.columns:
            logger.warning("ad_id column not available; skipping H1a.")
            return {"error": "ad_id column not available"}

        if "CTR" not in self.df.columns:
            return {"error": "CTR column not found"}

        merged = results_df.merge(
            self.df[["ad_id", "CTR"]],
            on="ad_id",
            how="left"
        )

        ctr_median = self.df["CTR"].median()
        merged["ctr_category"] = np.where(
            merged["CTR"] > ctr_median,
            "high",
            "low"
        )

        results = {}

        for shot_level in [1, 3]:
            shot_data = merged[merged["shot_level"] == shot_level].copy()

            if shot_data.empty:
                continue

            shot_data["correct"] = shot_data["correct"].apply(safe_to_int)

            high_ctr = shot_data[shot_data["ctr_category"] == "high"]
            low_ctr = shot_data[shot_data["ctr_category"] == "low"]

            high_acc = high_ctr["correct"].mean() if len(high_ctr) > 0 else 0.0
            low_acc = low_ctr["correct"].mean() if len(low_ctr) > 0 else 0.0

            if len(high_ctr) > 0 and len(low_ctr) > 0:
                high_correct = int(high_ctr["correct"].sum())
                low_correct = int(low_ctr["correct"].sum())

                try:
                    z_stat, p_value = proportions_ztest(
                        count=[high_correct, low_correct],
                        nobs=[len(high_ctr), len(low_ctr)],
                        alternative="larger"
                    )
                except Exception:
                    z_stat, p_value = 0.0, 1.0

                results[int(shot_level)] = {
                    "high_ctr_accuracy": float(high_acc),
                    "low_ctr_accuracy": float(low_acc),
                    "difference": float(high_acc - low_acc),
                    "z_statistic": float(z_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < self.alpha),
                    "n_high": int(len(high_ctr)),
                    "n_low": int(len(low_ctr))
                }

        self.results["sub_hypothesis_1a"] = results
        return results

    def test_sub_hypothesis_1c(self) -> Dict:
        """
        Test H1c: prediction aligns better with CTR success than conversion success.
        """
        if self.results_df is None:
            return {"error": "No results data available"}

        results_df = self.results_df.copy()

        if "ad_index" in results_df.columns and "ad_id" not in results_df.columns:
            results_df["ad_id"] = results_df["ad_index"]

        required_cols = ["ad_id", "success_ctr", "success_conversion"]

        for col in required_cols:
            if col not in self.df.columns and col != "ad_id":
                return {"error": f"{col} column not found"}

        if "ad_id" not in results_df.columns or "ad_id" not in self.df.columns:
            return {"error": "ad_id column not available"}

        results = {}

        for shot_level in [1, 3]:
            shot_data = results_df[results_df["shot_level"] == shot_level].copy()

            if shot_data.empty:
                continue

            shot_data["prediction"] = shot_data["prediction"].apply(safe_to_int)

            merged_shot = shot_data[["ad_id", "prediction"]].merge(
                self.df[["ad_id", "success_ctr", "success_conversion"]],
                on="ad_id",
                how="left"
            ).dropna()

            if merged_shot.empty:
                continue

            merged_shot["success_ctr"] = merged_shot["success_ctr"].apply(safe_to_int)
            merged_shot["success_conversion"] = merged_shot[
                "success_conversion"
            ].apply(safe_to_int)

            current_preds = merged_shot["prediction"].values
            actual_ctr = merged_shot["success_ctr"].values
            actual_conv = merged_shot["success_conversion"].values

            ctr_acc = np.mean(current_preds == actual_ctr)
            conv_acc = np.mean(current_preds == actual_conv)

            results[int(shot_level)] = {
                "ctr_accuracy": float(ctr_acc),
                "conversion_accuracy": float(conv_acc),
                "difference": float(ctr_acc - conv_acc),
                "n_samples": int(len(merged_shot))
            }

        self.results["sub_hypothesis_1c"] = results
        return results

    def test_metric_differences(self) -> Dict:
        """
        Test metric differences between successful and unsuccessful campaigns.
        """
        results = {}
        metrics = ["CTR", "CPC", "Conversion_Rate", "impressions", "clicks", "spent"]

        if "is_success" not in self.df.columns:
            return {"error": "is_success column not found"}

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            success_data = self.df[self.df["is_success"] == 1][metric].dropna()
            fail_data = self.df[self.df["is_success"] == 0][metric].dropna()

            if len(success_data) > 1 and len(fail_data) > 1:
                try:
                    u_stat, p_value = mannwhitneyu(
                        success_data,
                        fail_data,
                        alternative="two-sided"
                    )

                    pooled_std = np.sqrt(
                        (success_data.std() ** 2 + fail_data.std() ** 2) / 2
                    )

                    cohens_d = (
                        (success_data.mean() - fail_data.mean()) / pooled_std
                        if pooled_std != 0
                        else 0.0
                    )

                    results[metric] = {
                        "test_type": "Mann-Whitney U test",
                        "statistic": float(u_stat),
                        "p_value": float(p_value),
                        "significant": bool(p_value < self.alpha),
                        "cohens_d": float(cohens_d),
                        "success_mean": float(success_data.mean()),
                        "fail_mean": float(fail_data.mean()),
                        "success_n": int(len(success_data)),
                        "fail_n": int(len(fail_data))
                    }

                except Exception as e:
                    logger.warning(f"Error in metric test for {metric}: {e}")

        self.results["metric_differences"] = results
        return results

    def test_demographic_effects(self) -> Dict:
        """
        Test demographic effects on campaign success.
        """
        results = {}

        if "gender" in self.df.columns and "is_success" in self.df.columns:
            male_success = self.df[self.df["gender"] == "M"]["is_success"].dropna()
            female_success = self.df[self.df["gender"] == "F"]["is_success"].dropna()

            if len(male_success) > 1 and len(female_success) > 1:
                try:
                    male_success_count = int(male_success.sum())
                    male_total = int(len(male_success))

                    female_success_count = int(female_success.sum())
                    female_total = int(len(female_success))

                    z_stat, p_value = proportions_ztest(
                        [male_success_count, female_success_count],
                        [male_total, female_total],
                        alternative="two-sided"
                    )

                    results["gender_test"] = {
                        "male_success_rate": float(male_success_count / male_total),
                        "female_success_rate": float(female_success_count / female_total),
                        "difference": float(
                            male_success_count / male_total
                            - female_success_count / female_total
                        ),
                        "z_statistic": float(z_stat),
                        "p_value": float(p_value),
                        "significant": bool(p_value < self.alpha),
                        "male_n": male_total,
                        "female_n": female_total
                    }

                except Exception as e:
                    logger.warning(f"Error in gender test: {e}")

        self.results["demographic_effects"] = results
        return results

    def perform_power_analysis(
        self,
        effect_size: float = 0.3,
        power: float = 0.8
    ) -> Dict:
        """
        Perform statistical power analysis.
        """
        power_analyzer = TTestIndPower()

        try:
            required_n = power_analyzer.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=self.alpha,
                ratio=1.0
            )
        except Exception:
            required_n = 0

        results = {
            "target_power": float(power),
            "alpha": float(self.alpha),
            "effect_size_assumed": float(effect_size),
            "required_sample_size_per_group": (
                int(np.ceil(required_n)) if required_n > 0 else 0
            ),
            "total_required_samples": (
                int(np.ceil(required_n * 2)) if required_n > 0 else 0
            )
        }

        self.results["power_analysis"] = results
        return results

    def generate_comprehensive_report(
        self,
        output_file: str = "results/statistical_report.json"
    ) -> Dict:
        """
        Generate comprehensive statistical report.
        """
        logger.info("Generating comprehensive statistical report...")

        self.test_primary_hypothesis()
        self.test_sub_hypothesis_1a()
        self.test_sub_hypothesis_1c()
        self.test_metric_differences()
        self.test_demographic_effects()
        self.perform_power_analysis()

        Path(output_file).parent.mkdir(exist_ok=True)

        if "is_success" in self.df.columns:
            success_rate = float(self.df["is_success"].mean())
            success_count = int(self.df["is_success"].sum())
            fail_count = int((self.df["is_success"] == 0).sum())
        else:
            success_rate = 0.0
            success_count = 0
            fail_count = 0

        self.results["descriptive_statistics"] = {
            "total_campaigns": int(len(self.df)),
            "success_rate": success_rate,
            "success_count": success_count,
            "fail_count": fail_count,
            "gender_distribution": (
                self.df["gender"].value_counts().to_dict()
                if "gender" in self.df.columns
                else {}
            ),
            "age_distribution": (
                self.df["age"].value_counts().sort_index().to_dict()
                if "age" in self.df.columns
                else {}
            )
        }

        numeric_cols = [
            "CTR", "CPC", "Conversion_Rate",
            "impressions", "clicks", "spent", "is_success"
        ]

        existing_cols = [col for col in numeric_cols if col in self.df.columns]

        if existing_cols:
            corr_df = self.df[existing_cols].dropna()

            if not corr_df.empty:
                self.results["correlations"] = corr_df.corr().to_dict()
            else:
                self.results["correlations"] = {}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, default=str)

        logger.info(f"Statistical report saved to {output_file}")

        return self.results

    def print_summary(self):
        """
        Print summary of statistical findings.
        """
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("=" * 80)

        if "primary_hypothesis" in self.results:
            print("\n1. PRIMARY HYPOTHESIS (H1):")
            print("-" * 40)

            ph = self.results["primary_hypothesis"]

            if 1 in ph:
                print(f"   One-shot accuracy: {ph[1].get('accuracy', 0):.2%}")
                print(
                    f"   One-shot Balanced Accuracy: "
                    f"{ph[1].get('balanced_accuracy', 0):.2%}"
                )
                print(f"   One-shot MCC: {ph[1].get('mcc', 0):.4f}")
                print(f"   Confusion Matrix: {ph[1].get('confusion_matrix', [])}")
                print(
                    f"   Significant vs random? "
                    f"{'YES' if ph[1].get('significant') else 'NO'}"
                )
                print(f"   p-value: {ph[1].get('p_value', 1.0):.4f}")
                print(
                    f"   95% CI: "
                    f"[{ph[1].get('ci_95', [0, 0])[0]:.2%}, "
                    f"{ph[1].get('ci_95', [0, 0])[1]:.2%}]"
                )

            if "hypothesis_result" in ph:
                hr = ph["hypothesis_result"]
                print(
                    f"   In target range 60-70%: "
                    f"{'YES' if hr.get('in_target_range') else 'NO'}"
                )


def main():
    """
    Main statistical analysis execution.
    """
    df = pd.read_csv("data/processed_data.csv")
    results_df = None

    possible_files = list(Path("results").glob("experiment_results_*.json"))

    if possible_files:
        result_file = possible_files[0]

        try:
            with open(result_file, "r", encoding="utf-8") as f:
                results_json = json.load(f)

            if "results" in results_json:
                results_df = pd.DataFrame(results_json["results"])
            elif isinstance(results_json, list):
                results_df = pd.DataFrame(results_json)

            if results_df is not None:
                print("\n" + "=" * 60)
                print("LOADED RESULTS DATAFRAME")
                print("=" * 60)
                print(f"File: {result_file}")
                print(f"Rows: {len(results_df)}")
                print(f"Columns: {results_df.columns.tolist()}")

        except Exception as e:
            print(f"Could not load results: {e}")
            results_df = None
    else:
        print("No experiment_results_*.json file found. Running descriptive analysis only.")

    analyzer = StatisticalAnalyzer(df, results_df)
    report = analyzer.generate_comprehensive_report()
    analyzer.print_summary()

    return report


if __name__ == "__main__":
    main()

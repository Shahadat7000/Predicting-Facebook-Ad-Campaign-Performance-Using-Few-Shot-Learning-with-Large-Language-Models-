

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, f_oneway
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, matthews_corrcoef
import json
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from utils import convert_column_to_int, validate_binary_column, safe_to_int

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for hypothesis testing
    """

    def __init__(self,
                 df: pd.DataFrame,
                 results_df: Optional[pd.DataFrame] = None,
                 alpha: float = 0.05):
        """
        Initialize analyzer with data

        Args:
            df: Original dataframe with campaign data
            results_df: Experiment results dataframe
            alpha: Significance level
        """
        self.df = df
        self.results_df = results_df
        self.alpha = alpha
        self.results = {}

        # ===== IMPROVED: Robust type conversion using utils ====="
        if self.results_df is not None:
            for col in ['prediction', 'actual', 'correct']:
                if col in self.results_df.columns:
                    self.results_df = convert_column_to_int(self.results_df, col, safe=True)
            
            # Validate conversion
            logger.info("\n" + "="*60)
            logger.info("DATA TYPE CONVERSION")
            logger.info("="*60)
            for col in ['prediction', 'actual', 'correct']:
                if col in self.results_df.columns:
                    is_valid = validate_binary_column(self.results_df, col)
                    unique_vals = self.results_df[col].unique()
                    logger.info(f"{col}: dtype={self.results_df[col].dtype}, valid={is_valid}, unique={unique_vals}")
            logger.info("="*60 + "\n")

    def test_primary_hypothesis(self) -> Dict:
        """
        Test H1: One-shot learning achieves 60-70% accuracy

        Returns:
            Dict: Hypothesis test results
        """
        if self.results_df is None:
            return {"error": "No results data available"}

        accuracy_by_shot = {}

        for shot_level in [0, 1, 3, 5]:
            shot_data = self.results_df[self.results_df['shot_level'] == shot_level].copy()

            if len(shot_data) > 0:
                # Ensure correct column is int
                shot_data['correct'] = shot_data['correct'].apply(safe_to_int)
                shot_data['actual'] = shot_data['actual'].apply(safe_to_int)
                shot_data['prediction'] = shot_data['prediction'].apply(safe_to_int)

                correct = int(shot_data['correct'].sum())
                total = int(len(shot_data))
                accuracy = correct / total if total > 0 else 0

                # One-sample proportion test against 50% baseline
                if total > 0 and correct > 0:
                    try:
                        z_stat, p_value = proportions_ztest(
                            count=correct,
                            nobs=total,
                            value=0.5,
                            alternative='larger'
                        )
                    except:
                        z_stat, p_value = 0, 1.0
                else:
                    z_stat, p_value = 0, 1.0

                # Confidence interval
                se = np.sqrt(0.5 * 0.5 / total) if total > 0 else 0
                ci_lower = accuracy - 1.96 * se if total > 0 else 0
                ci_upper = accuracy + 1.96 * se if total > 0 else 0
                
                # New metrics: Balanced Accuracy, MCC, Confusion Matrix
                actuals = shot_data['actual'].dropna().values
                preds = shot_data['prediction'].dropna().values
                
                if len(actuals) > 0 and len(preds) > 0 and len(actuals) == len(preds):
                    balanced_acc = balanced_accuracy_score(actuals, preds)
                    mcc = matthews_corrcoef(actuals, preds)
                    cm = confusion_matrix(actuals, preds, labels=[0, 1]).tolist()
                else:
                    balanced_acc, mcc, cm = 0.0, 0.0, []

                accuracy_by_shot[int(shot_level)] = {
                    'accuracy': float(accuracy),
                    'balanced_accuracy': float(balanced_acc),
                    'mcc': float(mcc),
                    'confusion_matrix': cm,
                    'correct': int(correct),
                    'total': int(total),
                    'z_statistic': float(z_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < self.alpha) if total > 0 else False,
                    'ci_95': [float(max(0, ci_lower)), float(min(1, ci_upper))]
                }

                logger.info(f"Shot {shot_level}: acc={accuracy:.2%}, bal_acc={balanced_acc:.2%}, mcc={mcc:.4f}")

        # Test if one-shot falls in 60-70% range
        if 1 in accuracy_by_shot:
            one_shot_acc = accuracy_by_shot[1]['accuracy']
            in_range = 0.60 <= one_shot_acc <= 0.70

            accuracy_by_shot['hypothesis_result'] = {
                'one_shot_accuracy': float(one_shot_acc),
                'target_range': [0.60, 0.70],
                'in_target_range': bool(in_range),
                'hypothesis_supported': bool(in_range and accuracy_by_shot[1]['significant'])
            }

        self.results['primary_hypothesis'] = accuracy_by_shot
        return accuracy_by_shot

    def test_sub_hypothesis_1a(self) -> Dict:
        """
        Test H1a: Better performance on high-CTR ads

        Returns:
            Dict: Test results
        """
        if self.results_df is None:
            return {"error": "No results data available"}

        if 'ad_index' in self.results_df.columns:
            self.results_df['ad_id'] = self.results_df['ad_index']

        if 'ad_id' not in self.df.columns:
            logger.warning("ad_id column not found in main dataset; skipping H1a")
            return {"error": "ad_id column not found"}

        merged = self.results_df.merge(
            self.df[['ad_id', 'CTR']],
            left_on='ad_id',
            right_on='ad_id',
            how='left'
        )

        ctr_median = self.df['CTR'].median()
        merged['ctr_category'] = np.where(merged['CTR'] > ctr_median, 'high', 'low')

        results = {}

        for shot_level in [1, 3]:
            shot_data = merged[merged['shot_level'] == shot_level]

            if len(shot_data) > 0:
                shot_data['correct'] = shot_data['correct'].apply(safe_to_int)
                
                high_ctr = shot_data[shot_data['ctr_category'] == 'high']
                low_ctr = shot_data[shot_data['ctr_category'] == 'low']

                high_acc = high_ctr['correct'].mean() if len(high_ctr) > 0 else 0
                low_acc = low_ctr['correct'].mean() if len(low_ctr) > 0 else 0

                if len(high_ctr) > 0 and len(low_ctr) > 0:
                    high_correct = int(high_ctr['correct'].sum())
                    low_correct = int(low_ctr['correct'].sum())

                    try:
                        z_stat, p_value = proportions_ztest(
                            count=[high_correct, low_correct],
                            nobs=[len(high_ctr), len(low_ctr)],
                            alternative='larger'
                        )
                    except:
                        z_stat, p_value = 0, 1.0

                    results[int(shot_level)] = {
                        'high_ctr_accuracy': float(high_acc),
                        'low_ctr_accuracy': float(low_acc),
                        'difference': float(high_acc - low_acc),
                        'z_statistic': float(z_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < self.alpha),
                        'n_high': int(len(high_ctr)),
                        'n_low': int(len(low_ctr))
                    }

        self.results['sub_hypothesis_1a'] = results
        return results

    def test_sub_hypothesis_1c(self) -> Dict:
        """
        Test H1c: Better for CTR than Conversion Rate prediction

        Returns:
            Dict: Test results
        """
        if self.results_df is None:
            return {"error": "No results data available"}

        if 'ad_index' in self.results_df.columns:
            self.results_df['ad_id'] = self.results_df['ad_index']

        results = {}

        for shot_level in [1, 3]:
            shot_data = self.results_df[self.results_df['shot_level'] == shot_level]

            if len(shot_data) > 0 and len(shot_data['ad_id'].values) > 0:
                shot_data = shot_data.copy()
                shot_data['prediction'] = shot_data['prediction'].apply(safe_to_int)
                
                merged_shot = shot_data[['ad_id', 'prediction']].merge(
                    self.df[['ad_id', 'success_ctr', 'success_conversion']],
                    on='ad_id',
                    how='left'
                )
                
                actual_ctr = merged_shot['success_ctr'].values
                actual_conv = merged_shot['success_conversion'].values
                current_preds = merged_shot['prediction'].values

                ctr_acc = np.mean(current_preds == actual_ctr)
                conv_acc = np.mean(current_preds == actual_conv)

                results[int(shot_level)] = {
                    'ctr_accuracy': float(ctr_acc),
                    'conversion_accuracy': float(conv_acc),
                    'difference': float(ctr_acc - conv_acc),
                    'n_samples': int(len(shot_data))
                }

        self.results['sub_hypothesis_1c'] = results
        return results

    def test_metric_differences(self) -> Dict:
        """
        Test differences in metrics between successful and unsuccessful campaigns

        Returns:
            Dict: T-test results for each metric
        """
        results = {}
        metrics = ['CTR', 'CPC', 'Conversion_Rate', 'impressions', 'clicks', 'spent']

        for metric in metrics:
            if metric in self.df.columns:
                success_data = self.df[self.df['is_success'] == True][metric].dropna()
                fail_data = self.df[self.df['is_success'] == False][metric].dropna()

                if len(success_data) > 1 and len(fail_data) > 1:
                    try:
                        u_stat, p_value = mannwhitneyu(success_data, fail_data, alternative='two-sided')

                        pooled_std = np.sqrt((success_data.std()**2 + fail_data.std()**2) / 2)
                        cohens_d = (success_data.mean() - fail_data.mean()) / pooled_std if pooled_std != 0 else 0

                        results[metric] = {
                            'test_type': "Mann-Whitney U test",
                            'statistic': float(u_stat),
                            'p_value': float(p_value),
                            'significant': bool(p_value < self.alpha),
                            'cohens_d': float(cohens_d),
                            'success_mean': float(success_data.mean()),
                            'fail_mean': float(fail_data.mean()),
                            'success_n': int(len(success_data)),
                            'fail_n': int(len(fail_data))
                        }
                    except Exception as e:
                        logger.warning(f"Error in metric test for {metric}: {e}")

        self.results['metric_differences'] = results
        return results

    def test_demographic_effects(self) -> Dict:
        """
        Test demographic effects on success

        Returns:
            Dict: ANOVA and t-test results
        """
        results = {}

        if 'gender' in self.df.columns:
            male_success = self.df[self.df['gender'] == 'M']['is_success'].dropna()
            female_success = self.df[self.df['gender'] == 'F']['is_success'].dropna()

            if len(male_success) > 1 and len(female_success) > 1:
                try:
                    male_success_count = int(male_success.sum())
                    male_total = len(male_success)
                    female_success_count = int(female_success.sum())
                    female_total = len(female_success)

                    z_stat, p_value = proportions_ztest(
                        [male_success_count, female_success_count],
                        [male_total, female_total],
                        alternative='two-sided'
                    )

                    results['gender_test'] = {
                        'male_success_rate': float(male_success_count / male_total),
                        'female_success_rate': float(female_success_count / female_total),
                        'difference': float(male_success_count/male_total - female_success_count/female_total),
                        'z_statistic': float(z_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < self.alpha),
                        'male_n': male_total,
                        'female_n': female_total
                    }
                except Exception as e:
                    logger.warning(f"Error in gender test: {e}")

        self.results['demographic_effects'] = results
        return results

    def perform_power_analysis(self, effect_size: float = 0.3, power: float = 0.8) -> Dict:
        """
        Perform statistical power analysis

        Returns:
            Dict: Power analysis results
        """
        power_analyzer = TTestIndPower()

        try:
            required_n = power_analyzer.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=self.alpha,
                ratio=1.0
            )
        except:
            required_n = 0

        results = {
            'target_power': power,
            'alpha': self.alpha,
            'effect_size_assumed': effect_size,
            'required_sample_size_per_group': int(np.ceil(required_n)) if required_n > 0 else 0,
            'total_required_samples': int(np.ceil(required_n * 2)) if required_n > 0 else 0
        }

        self.results['power_analysis'] = results
        return results

    def generate_comprehensive_report(self, output_file: str = 'results/statistical_report.json') -> Dict:
        """
        Generate comprehensive statistical report

        Args:
            output_file: Path to save the report JSON file.

        Returns:
            Dict: Complete statistical analysis report
        """
        logger.info("Generating comprehensive statistical report...")

        self.test_primary_hypothesis()
        self.test_sub_hypothesis_1a()
        self.test_sub_hypothesis_1c()
        self.test_metric_differences()
        self.test_demographic_effects()
        self.perform_power_analysis()

        self.results['descriptive_statistics'] = {
            'total_campaigns': int(len(self.df)),
            'success_rate': float(self.df['is_success'].mean()),
            'success_count': int(self.df['is_success'].sum()),
            'fail_count': int((~self.df['is_success']).sum()),
            'gender_distribution': self.df['gender'].value_counts().to_dict() if 'gender' in self.df.columns else {},
            'age_distribution': self.df['age'].value_counts().sort_index().to_dict() if 'age' in self.df.columns else {}
        }

        numeric_cols = ['CTR', 'CPC', 'Conversion_Rate', 'impressions', 'clicks', 'spent', 'is_success']
        existing_cols = [col for col in numeric_cols if col in self.df.columns]
        if existing_cols:
            corr_df = self.df[existing_cols].dropna()
            self.results['correlations'] = corr_df.corr().to_dict()

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)

        logger.info(f"Statistical report saved to {output_file}")
        return self.results

    def print_summary(self):
        """Print summary of statistical findings"""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*80)

        if 'primary_hypothesis' in self.results:
            print("\n1. PRIMARY HYPOTHESIS (H1):")
            print("-"*40)
            ph = self.results['primary_hypothesis']
            if 1 in ph:
                print(f"   One-shot accuracy: {ph[1].get('accuracy', 0):.2%}")
                print(f"   One-shot Balanced Accuracy: {ph[1].get('balanced_accuracy', 0):.2%}")
                print(f"   One-shot MCC: {ph[1].get('mcc', 0):.4f}")
                print(f"   Confusion Matrix: {ph[1].get('confusion_matrix', [])}")
                print(f"   Significant vs random? {'YES' if ph[1].get('significant') else 'NO'}")
                print(f"   p-value: {ph[1].get('p_value', 1.0):.4f}")
                print(f"   95% CI: [{ph[1].get('ci_95', [0, 0])[0]:.2%}, {ph[1].get('ci_95', [0, 0])[1]:.2%}]")

                if 'hypothesis_result' in ph:
                    print(f"   In target range (60-70%): {'YES' if ph['hypothesis_result'].get('in_target_range') else 'NO'}")


def main():
    """Main statistical analysis execution"""
    df = pd.read_csv("data/processed_data.csv")
    results_df = None

    try:
        import json
        with open("results/experiment_results.json", 'r') as f:
            results_json = json.load(f)

            if 'results' in results_json:
                results_df = pd.DataFrame(results_json['results'])
            elif isinstance(results_json, list):
                results_df = pd.DataFrame(results_json)

            if results_df is not None:
                print(f"\n{'='*60}")
                print("LOADED RESULTS DATAFRAME")
                print(f"{'='*60}")
                print(f"Rows: {len(results_df)}")
                print(f"Columns: {results_df.columns.tolist()}")

    except Exception as e:
        print(f"Could not load results: {e}")
        results_df = None

    analyzer = StatisticalAnalyzer(df, results_df)
    report = analyzer.generate_comprehensive_report()
    analyzer.print_summary()

    return report


if __name__ == "__main__":
    main()

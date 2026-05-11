

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path
from typing import Dict, Optional, List
import json

logger = logging.getLogger(__name__)


# HELPER FUNCTION: Convert string booleans to int

def _convert_to_int(df, col_name):
    """
    Convert string boolean column to integer

    Args:
        df: DataFrame
        col_name: Column name to convert

    Returns:
        DataFrame with converted column
    """
    if col_name in df.columns:
        # Convert to string, strip whitespace, and normalize case
        normalized = df[col_name].astype(str).str.strip().str.lower()

        # Explicitly map common boolean representations to int
        df[col_name] = normalized.map({
            'true': 1, 'false': 0,
            '1': 1, '0': 0,
            '1.0': 1, '0.0': 0,
            'yes': 1, 'no': 0,
            'success': 1, 'fail': 0,
            'failure': 0
        }).fillna(0).astype(int)
    return df


class ThesisVisualizer:
    """
    Comprehensive visualization for thesis results
    """

    def __init__(self,
                 df: pd.DataFrame,
                 results_df: Optional[pd.DataFrame] = None,
                 output_dir: str = "visualizations",
                 model_name: Optional[str] = None):
        """
        Initialize visualizer

        Args:
            df: Original dataframe
            results_df: Experiment results dataframe
            output_dir: Output directory for visualizations
            model_name: Name of the model (used in titles and folder naming).
                        If not given, tries to guess from output_dir.
        """
        self.df = df
        self.results_df = results_df

        # ----- determine model name -----
        if model_name:
            self.model_name = model_name
        else:
            # try to extract from the last folder of output_dir
            p = Path(output_dir)
            if p.name not in ('', 'visualizations', '.'):
                self.model_name = p.name.replace('_', ' ').title()
            else:
                self.model_name = 'Unknown Model'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # ===== FIX: Convert all string booleans to int =====
        if self.results_df is not None:
            for col in ['actual', 'prediction', 'correct']:
                if col in self.results_df.columns:
                    self.results_df = _convert_to_int(self.results_df, col)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Color scheme
        self.colors = {
            'success': '#2ecc71',
            'failure': '#e74c3c',
            'zero_shot': '#3498db',
            'one_shot': '#9b59b6',
            'few_shot': '#e67e22',
            'many_shot': '#1abc9c',
            'baseline': '#95a5a6'
        }
    # Existing plotting methods (with model_name prefix added to titles)
    

    def plot_success_distribution(self, save: bool = True):
        """Plot distribution of successful vs unsuccessful campaigns"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        success_count = self.df['is_success'].sum()
        fail_count = len(self.df) - success_count

        axes[0].pie(
            [success_count, fail_count],
            labels=['Successful', 'Unsuccessful'],
            autopct='%1.1f%%',
            colors=[self.colors['success'], self.colors['failure']],
            startangle=90,
            explode=(0.05, 0)
        )
        axes[0].set_title(f'{self.model_name} - Campaign Success Distribution',
                          fontsize=14, fontweight='bold')

        # Bar chart by demographic (unchanged logic, only title updated)
        success_by_gender = self.df.groupby('gender')['is_success'].mean() * 100

        age_groups = [age for age in self.df['age'].unique() if '-' in str(age)]
        if len(age_groups) > 0:
            age_data = self.df[self.df['age'].isin(age_groups)]
            success_by_age = age_data.groupby('age')['is_success'].mean() * 100

            gender_values = success_by_gender.values
            age_values = success_by_age.values
            age_labels = success_by_age.index

            x = np.arange(len(age_labels))
            width = 0.35

            gender_aligned = []
            for i, _ in enumerate(age_labels):
                if i < len(gender_values):
                    gender_aligned.append(gender_values[i % len(gender_values)])
                else:
                    gender_aligned.append(gender_values[0])

            axes[1].bar(x - width/2, gender_aligned[:len(x)], width,
                       label='Gender', color=['lightblue', 'lightpink'])
            axes[1].bar(x + width/2, age_values, width,
                       label='Age Group', color='lightgreen')

            axes[1].set_xlabel('Age Group')
            axes[1].set_ylabel('Success Rate (%)')
            axes[1].set_title(f'{self.model_name} - Success Rate by Age Group',
                              fontsize=14, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(age_labels, rotation=45)
            axes[1].legend()
            axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].bar(success_by_gender.index, success_by_gender.values,
                       color=['lightblue', 'lightpink'], edgecolor='black')
            axes[1].set_xlabel('Gender')
            axes[1].set_ylabel('Success Rate (%)')
            axes[1].set_title(f'{self.model_name} - Success Rate by Gender',
                              fontsize=14, fontweight='bold')
            axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'success_distribution.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {self.output_dir / 'success_distribution.png'}")

        plt.show()
        return fig

    def plot_metrics_distribution(self, save: bool = True):
        """Plot distribution of key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics = ['CTR', 'CPC', 'Conversion_Rate', 'impressions', 'clicks', 'spent']
        titles = ['CTR Distribution (%)', 'CPC Distribution ($)',
                 'Conversion Rate Distribution (%)', 'Impressions Distribution',
                 'Clicks Distribution', 'Spent Distribution ($)']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            if metric in self.df.columns:
                success_data = self.df[self.df['is_success']][metric].dropna()
                fail_data = self.df[~self.df['is_success']][metric].dropna()

                # Remove extreme outliers for better visualization
                for data in [success_data, fail_data]:
                    if len(data) > 0:
                        q1 = data.quantile(0.25)
                        q3 = data.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 3 * iqr
                        upper = q3 + 3 * iqr
                        # Filtering happens inside but we still pass the filtered version
                        data = data[(data >= lower) & (data <= upper)]

                ax.hist(success_data, alpha=0.7, bins=20, label='Successful',
                       color=self.colors['success'], edgecolor='black')
                ax.hist(fail_data, alpha=0.7, bins=20, label='Unsuccessful',
                       color=self.colors['failure'], edgecolor='black')

                ax.set_xlabel(metric)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{self.model_name} - {title}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {self.output_dir / 'metrics_distribution.png'}")

        plt.show()
        return fig

    def plot_accuracy_by_shot_level(self, save: bool = True):
        """Plot accuracy by shot level"""
        if self.results_df is None:
            logger.warning("No results data available")
            return None

        accuracy_data = []
        for shot_level in sorted(self.results_df['shot_level'].unique()):
            shot_data = self.results_df[self.results_df['shot_level'] == shot_level]
            if len(shot_data) > 0:
                if 'correct' in shot_data.columns:
                    correct_count = shot_data['correct'].sum()
                    accuracy = (correct_count / len(shot_data)) * 100
                else:
                    accuracy = 0

                n_samples = len(shot_data)
                se = np.sqrt(accuracy/100 * (1 - accuracy/100) / n_samples) * 100 if n_samples > 0 else 0

                accuracy_data.append({
                    'shot_level': f"{int(shot_level)}-shot",
                    'accuracy': accuracy,
                    'n': n_samples,
                    'ci_lower': accuracy - 1.96 * se,
                    'ci_upper': accuracy + 1.96 * se
                })

        if not accuracy_data:
            return None

        acc_df = pd.DataFrame(accuracy_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [self.colors['zero_shot'], self.colors['one_shot'],
                 self.colors['few_shot'], self.colors['many_shot']]

        bars = ax.bar(acc_df['shot_level'], acc_df['accuracy'],
                     color=colors[:len(acc_df)], edgecolor='black', linewidth=1.5)

        ax.errorbar(acc_df['shot_level'], acc_df['accuracy'],
                   yerr=[acc_df['accuracy'] - acc_df['ci_lower'],
                         acc_df['ci_upper'] - acc_df['accuracy']],
                   fmt='none', color='black', capsize=5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')

        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=2,
                  label='Random Guessing (50%)')
        ax.axhspan(60, 70, alpha=0.2, color='green', label='Target Range (60-70%)')

        ax.set_xlabel('Shot Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_name} - Prediction Accuracy by Shot Level',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')

        for i, row in acc_df.iterrows():
            ax.text(i, 5, f'n={row["n"]}', ha='center', fontsize=10)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'accuracy_by_shot.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {self.output_dir / 'accuracy_by_shot.png'}")

        plt.show()
        return fig

    def plot_confusion_matrices(self, save: bool = True):
        """Plot confusion matrices for each shot level"""
        if self.results_df is None:
            logger.warning("No results data available")
            return None

        from sklearn.metrics import confusion_matrix

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        shot_levels = sorted(self.results_df['shot_level'].unique())
        titles = [f"{int(shot)}-Shot" for shot in shot_levels]

        for idx, (shot_level, title) in enumerate(zip(shot_levels, titles)):
            if idx >= len(axes):
                break

            ax = axes[idx]
            shot_data = self.results_df[self.results_df['shot_level'] == shot_level]

            if len(shot_data) > 0:
                valid_data = shot_data.dropna(subset=['actual', 'prediction'])
                if len(valid_data) == 0:
                    continue

                actual = valid_data['actual'].astype(int).values
                prediction = valid_data['prediction'].astype(int).values

                cm = confusion_matrix(actual, prediction, labels=[0, 1])
                cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Predicted Fail', 'Predicted Success'],
                           yticklabels=['Actual Fail', 'Actual Success'],
                           ax=ax, cbar=False)

                if 'correct' in shot_data.columns:
                    acc = shot_data['correct'].mean() * 100
                else:
                    acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100

                ax.set_title(f'{self.model_name} - {title}\nAccuracy: {acc:.2f}%',
                            fontsize=12, fontweight='bold')

                for i in range(2):
                    for j in range(2):
                        ax.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)',
                               ha='center', va='center', fontsize=9)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {self.output_dir / 'confusion_matrices.png'}")

        plt.show()
        return fig

    def plot_model_comparison(self, baseline_results: Optional[Dict] = None, save: bool = True):
        """Plot comparison between current model's shot levels and baseline ML"""
        if self.results_df is None:
            logger.warning("No results data available")
            return None

        models = []
        accuracies = []
        colors_list = []

        for shot_level in sorted(self.results_df['shot_level'].unique()):
            shot_data = self.results_df[self.results_df['shot_level'] == shot_level]
            if len(shot_data) > 0:
                models.append(f"{self.model_name} {int(shot_level)}-shot")
                acc = shot_data['correct'].mean() * 100 if 'correct' in shot_data.columns else 0
                accuracies.append(acc)
                if shot_level == 0:
                    colors_list.append(self.colors['zero_shot'])
                elif shot_level == 1:
                    colors_list.append(self.colors['one_shot'])
                elif shot_level == 3:
                    colors_list.append(self.colors['few_shot'])
                else:
                    colors_list.append(self.colors['many_shot'])

        if baseline_results:
            for model_name, results in baseline_results.items():
                if isinstance(results, dict) and 'accuracy' in results:
                    models.append(model_name.replace('_', ' ').title())
                    accuracies.append(results['accuracy'] * 100)
                    colors_list.append(self.colors['baseline'])

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(models, accuracies, color=colors_list, edgecolor='black', linewidth=1.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', rotation=45)

        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=2,
                  label='Random Guessing (50%)')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_name} - Model Performance Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {self.output_dir / 'model_comparison.png'}")

        plt.show()
        return fig

    def plot_cost_analysis(self, save: bool = True):
        """Plot cost analysis"""
        if self.results_df is None:
            logger.warning("No results data available")
            return None

        cost_data = []
        for shot_level in sorted(self.results_df['shot_level'].unique()):
            shot_data = self.results_df[self.results_df['shot_level'] == shot_level]
            if len(shot_data) > 0:
                accuracy = shot_data['correct'].mean() * 100 if 'correct' in shot_data.columns else 0
                total_cost = shot_data['cost_usd'].sum() if 'cost_usd' in shot_data.columns else 0
                n_samples = len(shot_data)
                cost_data.append({
                    'shot_level': f"{int(shot_level)}-shot",
                    'accuracy': accuracy,
                    'total_cost': total_cost,
                    'cost_per_sample': total_cost / n_samples if n_samples > 0 else 0,
                    'n_samples': n_samples
                })

        if not cost_data:
            return None

        cost_df = pd.DataFrame(cost_data)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(cost_df))
        width = 0.35

        bars = ax1.bar(x - width/2, cost_df['cost_per_sample'] * 1000, width,
                      color='lightblue', edgecolor='black', label='Cost per Sample (cents)')
        ax1.set_xlabel('Shot Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost per Sample (cents)', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(cost_df['shot_level'])

        ax2 = ax1.twinx()
        ax2.plot(x + width/2, cost_df['accuracy'], 'ro-', linewidth=2, markersize=8,
                label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 100)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f'{self.model_name} - Cost-Accuracy Tradeoff by Shot Level',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'cost_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {self.output_dir / 'cost_analysis.png'}")

        plt.show()
        return fig

    def plot_error_analysis(self, save: bool = True):
        """Plot error analysis"""
        if self.results_df is None:
            logger.warning("No results data available")
            return None

        if 'correct' not in self.results_df.columns:
            logger.warning("No 'correct' column found")
            return None

        misclassified = self.results_df[self.results_df['correct'] == False]
        if len(misclassified) == 0:
            logger.warning("No misclassified examples")
            return None

        if 'ad_id' in misclassified.columns or 'ad_index' in misclassified.columns:
            id_col = 'ad_id' if 'ad_id' in misclassified.columns else 'ad_index'
            misclassified = misclassified.merge(
                self.df[['ad_id', 'CTR', 'CPC', 'Conversion_Rate', 'impressions', 'clicks']],
                left_on=id_col,
                right_on='ad_id',
                how='left'
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Error distribution by shot level
        ax1 = axes[0, 0]
        error_by_shot = misclassified['shot_level'].value_counts().sort_index()
        ax1.bar([f"{int(s)}-shot" for s in error_by_shot.index], error_by_shot.values,
               color='salmon', edgecolor='black')
        ax1.set_xlabel('Shot Level')
        ax1.set_ylabel('Number of Errors')
        ax1.set_title(f'{self.model_name} - Error Distribution by Shot Level', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Error by CTR range
        ax2 = axes[0, 1]
        if 'CTR' in misclassified.columns:
            ctr_bins = pd.cut(misclassified['CTR'].dropna(), bins=5)
            if len(ctr_bins.dropna()) > 0:
                ctr_counts = ctr_bins.value_counts().sort_index()
                ax2.bar(range(len(ctr_counts)), ctr_counts.values, color='lightcoral', edgecolor='black')
                ax2.set_xlabel('CTR Range')
                ax2.set_ylabel('Number of Errors')
                ax2.set_title(f'{self.model_name} - Errors by CTR Range', fontweight='bold')
                ax2.set_xticks(range(len(ctr_counts)))
                ax2.set_xticklabels([f"{b.left:.3f}-{b.right:.3f}" for b in ctr_counts.index], rotation=45)
                ax2.grid(True, alpha=0.3)

        # 3. Error by impression volume
        ax3 = axes[1, 0]
        if 'impressions' in misclassified.columns:
            impression_bins = pd.cut(misclassified['impressions'].dropna(), bins=5)
            if len(impression_bins.dropna()) > 0:
                imp_counts = impression_bins.value_counts().sort_index()
                ax3.bar(range(len(imp_counts)), imp_counts.values, color='lightblue', edgecolor='black')
                ax3.set_xlabel('Impressions Range')
                ax3.set_ylabel('Number of Errors')
                ax3.set_title(f'{self.model_name} - Errors by Impression Volume', fontweight='bold')
                ax3.set_xticks(range(len(imp_counts)))
                ax3.set_xticklabels([f"{int(b.left):,}-{int(b.right):,}" for b in imp_counts.index], rotation=45)
                ax3.grid(True, alpha=0.3)

        # 4. Confusion matrix for errors
        ax4 = axes[1, 1]
        if 'actual' in misclassified.columns and 'prediction' in misclassified.columns:
            error_types = misclassified.groupby(['actual', 'prediction']).size().reset_index(name='count')
            error_types['type'] = error_types.apply(
                lambda x: f"Actual: {'Success' if x['actual'] else 'Fail'}\nPred: {'Success' if x['prediction'] else 'Fail'}",
                axis=1
            )
            ax4.bar(error_types['type'], error_types['count'], color='lightgreen', edgecolor='black')
            ax4.set_xlabel('Error Type')
            ax4.set_ylabel('Count')
            ax4.set_title(f'{self.model_name} - Error Type Distribution', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {self.output_dir / 'error_analysis.png'}")

        plt.show()
        return fig

    def create_interactive_dashboard(self, baseline_results: Optional[Dict] = None, save: bool = True):
        """Create interactive HTML dashboard using Plotly"""
        if self.results_df is None:
            logger.warning("No results data available")
            return None

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'{self.model_name} - Accuracy by Shot Level',
                f'{self.model_name} - Confusion Matrix (One-Shot)',
                f'{self.model_name} - Metric Distributions',
                f'{self.model_name} - Cost Analysis',
                f'{self.model_name} - Model Comparison',
                f'{self.model_name} - Error Analysis'
            ),
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'box'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'histogram'}]]
        )

        # 1. Accuracy by shot level
        shot_acc = []
        for shot in sorted(self.results_df['shot_level'].unique()):
            data = self.results_df[self.results_df['shot_level'] == shot]
            if len(data) > 0:
                acc = data['correct'].mean() * 100 if 'correct' in data.columns else 0
                shot_acc.append({'shot': f"{int(shot)}-shot", 'accuracy': acc})

        if shot_acc:
            shot_df = pd.DataFrame(shot_acc)
            fig.add_trace(
                go.Bar(x=shot_df['shot'], y=shot_df['accuracy'],
                      name='Accuracy', marker_color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c']),
                row=1, col=1
            )

        # 2. Confusion matrix (one-shot)
        one_shot = self.results_df[self.results_df['shot_level'] == 1]
        if len(one_shot) > 0:
            from sklearn.metrics import confusion_matrix
            actual = one_shot['actual'].values
            pred = one_shot['prediction'].values
            cm = confusion_matrix(actual, pred, labels=[0, 1])
            fig.add_trace(
                go.Heatmap(z=cm, x=['Pred Fail', 'Pred Success'],
                          y=['Actual Fail', 'Actual Success'],
                          colorscale='Blues', showscale=True,
                          text=cm, texttemplate='%{text}'),
                row=1, col=2
            )

        # 3. Metric distributions
        metrics = ['CTR', 'CPC', 'Conversion_Rate']
        for i, metric in enumerate(metrics[:2]):
            if metric in self.df.columns:
                fig.add_trace(
                    go.Box(y=self.df[metric].dropna(), name=metric, boxmean='sd'),
                    row=2, col=1
                )

        # 4. Cost analysis
        cost_data = []
        for shot in sorted(self.results_df['shot_level'].unique()):
            data = self.results_df[self.results_df['shot_level'] == shot]
            if len(data) > 0 and 'cost_usd' in data.columns:
                cost_data.append({'shot': f"{int(shot)}-shot", 'cost': data['cost_usd'].mean() * 1000})
        if cost_data:
            cost_df = pd.DataFrame(cost_data)
            fig.add_trace(
                go.Scatter(x=cost_df['shot'], y=cost_df['cost'],
                          mode='lines+markers', name='Cost per Sample',
                          line=dict(color='blue', width=2)),
                row=2, col=2
            )

        # 5. Model comparison (simplified per model)
        models = []
        accs = []
        for shot in sorted(self.results_df['shot_level'].unique()):
            data = self.results_df[self.results_df['shot_level'] == shot]
            if len(data) > 0:
                models.append(f"{int(shot)}-shot")
                accs.append(data['correct'].mean() * 100 if 'correct' in data.columns else 0)
        fig.add_trace(
            go.Bar(x=models, y=accs, name='GPT Models', marker_color='lightblue'),
            row=3, col=1
        )

        # 6. Error histogram
        if 'correct' in self.results_df.columns:
            errors = self.results_df[self.results_df['correct'] == False]
            if len(errors) > 0:
                fig.add_trace(
                    go.Histogram(x=errors['shot_level'], name='Errors', marker_color='salmon'),
                    row=3, col=2
                )

        fig.update_layout(height=1200, showlegend=True,
                         title_text=f"Facebook Ad Prediction: {self.model_name} - Interactive Dashboard",
                         title_font_size=16)

        if save:
            fig.write_html(self.output_dir / 'interactive_dashboard.html')
            logger.info(f"Saved: {self.output_dir / 'interactive_dashboard.html'}")

        fig.show()
        return fig

    def generate_all_visualizations(self, baseline_results: Optional[Dict] = None):
        """Generate all visualizations"""
        logger.info(f"Generating all visualizations for {self.model_name}...")

        self.plot_success_distribution()
        self.plot_metrics_distribution()
        self.plot_accuracy_by_shot_level()
        self.plot_confusion_matrices()
        self.plot_model_comparison(baseline_results)
        self.plot_cost_analysis()
        self.plot_error_analysis()
        self.create_interactive_dashboard(baseline_results)

        logger.info(f"All visualizations saved to {self.output_dir}")


    # EXISTING STATIC METHOD: Overall model comparison
  
    @staticmethod
    def generate_model_comparison(results_dir="results", output_dir="visualizations"):
        """
        Compare all models from experiment result files and generate:
        - model_comparison_table.csv
        - model_comparison_heatmap.png
        """
        results_path = Path(results_dir)
        model_files = list(results_path.glob("experiment_results_*.json"))
        if not model_files:
            logger.warning("No model files found for comparison.")
            return

        all_models = {}
        for f in model_files:
            model_name = f.stem.replace("experiment_results_", "").replace('_', ' ').title()
            with open(f) as fh:
                data = json.load(fh)
            df_res = pd.DataFrame(data['results'])
            # Ensure correct column types
            for col in ['correct']:
                if col in df_res.columns:
                    df_res[col] = df_res[col].astype(bool).astype(int)

            per_shot = {}
            for shot in [0, 1, 3, 5]:
                sub = df_res[df_res['shot_level'] == shot]
                if len(sub) > 0:
                    acc = sub['correct'].mean() * 100
                    per_shot[shot] = acc
                else:
                    per_shot[shot] = np.nan
            all_models[model_name] = per_shot

        comp_df = pd.DataFrame(all_models).T
        comp_df.columns = [f"{s}-shot" for s in comp_df.columns]
        comp_df.index.name = 'Model'

        # Save CSV
        out_dir = Path(output_dir)
        comp_df.to_csv(out_dir / "model_comparison_table.csv")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(comp_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                    cbar_kws={'label': 'Accuracy (%)'})
        ax.set_title("Model Accuracy Comparison (% by Shot Level)")
        plt.tight_layout()
        plt.savefig(out_dir / "model_comparison_heatmap.png", dpi=300)
        plt.close()
        logger.info(f"Model comparison saved to {output_dir}")


    # NEW STATIC METHOD: Combined accuracy by shot (all models side-by-side)
    
    @staticmethod
    def generate_combined_accuracy_by_shot(results_dir="results", output_path="visualizations/combined_accuracy_by_shot.png"):
        """Grouped bar chart: Accuracy by shot level for all models."""
        import json
        results_path = Path(results_dir)
        model_files = list(results_path.glob("experiment_results_*.json"))
        if not model_files:
            logger.warning("No model files found for combined accuracy chart.")
            return

        data = {}
        for f in model_files:
            model_name = f.stem.replace("experiment_results_", "").replace('_', ' ').title()
            with open(f) as fh:
                content = json.load(fh)
            df = pd.DataFrame(content['results'])
            # ensure correct type
            if 'correct' in df.columns:
                df['correct'] = df['correct'].astype(bool).astype(int)
            for shot in [0, 1, 3, 5]:
                subset = df[df['shot_level'] == shot]
                if len(subset) > 0:
                    acc = subset['correct'].mean() * 100
                    data.setdefault(model_name, {})[shot] = acc
                else:
                    data.setdefault(model_name, {})[shot] = np.nan

        comp = pd.DataFrame(data).T
        comp.columns = [f"{s}-shot" for s in comp.columns]

        fig, ax = plt.subplots(figsize=(10, 6))
        comp.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
        ax.set_title("Model Comparison: Accuracy by Shot Level")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 100)
        ax.legend(title="Shot Level")
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Combined accuracy-by-shot saved to {output_path}")


    # NEW STATIC METHOD: Combined model comparison (1-shot + baselines)
    
    @staticmethod
    def generate_combined_model_comparison(results_dir="results", baseline_path="results/baseline_results.json",
                                           output_path="visualizations/combined_model_comparison.png"):
        """Compare best shot (1-shot) of each model + ML baselines."""
        import json
        results_path = Path(results_dir)
        model_files = list(results_path.glob("experiment_results_*.json"))
        if not model_files:
            logger.warning("No model files found for combined model comparison.")
            return

        models = []
        accuracies = []
        colors = []

        for f in model_files:
            model_name = f.stem.replace("experiment_results_", "").replace('_', ' ').title()
            with open(f) as fh:
                content = json.load(fh)
            df = pd.DataFrame(content['results'])
            if 'correct' in df.columns:
                df['correct'] = df['correct'].astype(bool).astype(int)
            subset = df[df['shot_level'] == 1]   # using 1-shot
            if len(subset) > 0:
                acc = subset['correct'].mean() * 100
            else:
                acc = 0
            models.append(model_name)
            accuracies.append(acc)
            if 'deepseek' in model_name.lower():
                colors.append('#3498db')
            elif 'gpt' in model_name.lower():
                colors.append('#9b59b6')
            else:
                colors.append('#e67e22')

        # Add baselines if exist
        baseline_file = Path(baseline_path)
        if baseline_file.exists():
            with open(baseline_file) as f:
                bl = json.load(f)
            for name, vals in bl.items():
                if isinstance(vals, dict) and 'accuracy' in vals:
                    models.append(name.replace('_', ' ').title())
                    accuracies.append(vals['accuracy'] * 100)
                    colors.append('#95a5a6')

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, accuracies, color=colors, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', fontsize=10)
        ax.set_title("Model Performance Comparison (1-Shot + Baselines)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='red', linestyle='--', label='Random Guessing (50%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Combined model comparison saved to {output_path}")


def main():
    """Main visualization execution (can be used standalone)"""
    df = pd.read_csv("data/processed_data.csv")

    # Try loading a single model file (old style) or per-model files
    results_path = Path("results")
    model_files = list(results_path.glob("experiment_results_*.json"))

    if model_files:
        # For standalone run, process all models as in run_experiment.py
        # But here we can just take the first one, or loop
        for f in model_files:
            model_name = f.stem.replace("experiment_results_", "")
            with open(f) as fh:
                data = json.load(fh)
            results_df = pd.DataFrame(data['results']) if 'results' in data else None
            if results_df is not None:
                for col in ['actual', 'prediction', 'correct']:
                    if col in results_df.columns:
                        results_df = _convert_to_int(results_df, col)

            visualizer = ThesisVisualizer(df, results_df, output_dir=f"visualizations/{model_name}")
            visualizer.generate_all_visualizations()

        # Overall comparison
        ThesisVisualizer.generate_model_comparison("results", "visualizations")
        # New combined charts
        ThesisVisualizer.generate_combined_accuracy_by_shot()
        ThesisVisualizer.generate_combined_model_comparison()
    else:
        print("No experiment result files found.")


if __name__ == "__main__":
    main()

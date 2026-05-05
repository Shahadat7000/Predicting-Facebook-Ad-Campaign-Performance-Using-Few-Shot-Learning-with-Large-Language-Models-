"""
GPT Experiment Module for Few-Shot Learning
Author: Thesis Researcher
Date: 2024
"""

# ============================================
# FIX 1: Windows Encoding Fix - MUST BE AT TOP
# ============================================
import sys
import re

# Fix Windows console encoding for emoji/unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================
# Regular Imports
# ============================================
import openai
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging (base config, will be adapted per model)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Custom adapter to add model name prefix
class ModelLogAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{self.extra.get('model_name', 'unknown')}] {msg}", kwargs

base_logger = logging.getLogger(__name__)


class GPTExperiment:
    """
    Main experiment class for GPT few-shot learning on Facebook ad prediction
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 primary_model: Optional[str] = None,
                 secondary_model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 150):
        """
        Initialize experiment with model configuration

        Args:
            api_key: OpenAI API key (if None, load from env)
            base_url: API base URL (if None, load from env)
            primary_model: Primary GPT model to use
            secondary_model: Secondary model for comparison
            temperature: Temperature for generation (0.0 for deterministic)
            max_tokens: Maximum tokens in response
        """
        # Load from .env if not provided
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://llm.ai.e-infra.cz/v1")
        self.primary_model = primary_model or os.getenv("PRIMARY_MODEL", "gpt-oss-120b")
        self.secondary_model = secondary_model or os.getenv("SECONDARY_MODEL", "llama3.3:latest")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # ----- MODEL-SPECIFIC LOGGING -----
        self.model_name = self.primary_model
        self.logger = ModelLogAdapter(base_logger, {'model_name': self.model_name})

        # Initialize client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        self.results = []
        self.cost_tracker = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_cost_usd': 0.0,
            'api_calls': 0,
            'failed_calls': 0
        }

        # Token pricing (CERIT-SC is free, so costs are 0)
        self.pricing = {
            'gpt-oss-120b': {'prompt': 0.0, 'completion': 0.0},
            'llama3.3:latest': {'prompt': 0.0, 'completion': 0.0},
            'deepseek-v3.2': {'prompt': 0.0, 'completion': 0.0},
            'deepseek-v3.2-thinking': {'prompt': 0.0, 'completion': 0.0},
            'qwen3.5': {'prompt': 0.0, 'completion': 0.0},
            'qwen3-coder': {'prompt': 0.0, 'completion': 0.0}
        }

        self.logger.info(f"Initialized GPTExperiment with model: {self.primary_model}")
        self.logger.info(f"API Base URL: {self.base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError))
    )
    def get_prediction(self,
                       prompt: str,
                       model: Optional[str] = None,
                       temperature: Optional[float] = None) -> Tuple[str, Dict]:
        """
        Get prediction from GPT model with retry logic

        Args:
            prompt: Input prompt
            model: Model to use (defaults to primary_model)
            temperature: Temperature (defaults to instance value)

        Returns:
            Tuple[str, Dict]: (response text, usage info)
        """
        model = model or self.primary_model
        temp = temperature if temperature is not None else self.temperature

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=self.max_tokens
            )

            elapsed = time.time() - start_time

            # Extract response and usage
            response_text = response.choices[0].message.content or ""
            usage = {
                'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                'total_tokens': response.usage.total_tokens if response.usage else 0,
                'model': model,
                'response_time': elapsed
            }

            # Track costs (CERIT-SC is free)
            self.cost_tracker['api_calls'] += 1
            self.cost_tracker['total_prompt_tokens'] += usage['prompt_tokens']
            self.cost_tracker['total_completion_tokens'] += usage['completion_tokens']

            # Calculate cost (0 for CERIT-SC)
            usage['cost_usd'] = 0.0

            self.logger.debug(f"API call successful: {usage['total_tokens']} tokens")

            return response_text, usage

        except Exception as e:
            self.cost_tracker['failed_calls'] += 1
            self.logger.error(f"API call failed: {e}")
            raise

    # ============================================
    # FIX 2: Robust parse_prediction with regex
    # ============================================
    def parse_prediction(self, response: str) -> int:
        """
        Robust parsing of GPT response to binary prediction using regex

        Args:
            response: GPT response text

        Returns:
            int: 1 for success, 0 for failure
        """
        # Handle None or empty response
        if response is None or not isinstance(response, str) or len(response.strip()) == 0:
            self.logger.warning("Empty response, defaulting to 0")
            return 0

        response = response.strip()

        # DIRECT 0/1 MATCH - highest priority
        if response in ('0', '1'):
            return int(response)

        # REGEX: Look for 0/1 anywhere in text (with word boundaries)
        match = re.search(r'\b([01])\b', response)
        if match:
            return int(match.group(1))

        response_lower = response.lower()

        # Check for clear YES/NO at beginning
        lines = response_lower.strip().split('\n')
        first_line = lines[0].strip() if lines else ""

        if 'yes' in first_line and 'no' not in first_line:
            return 1
        elif 'no' in first_line and 'yes' not in first_line:
            return 0

        # Check for final answer markers
        if 'final answer: yes' in response_lower or 'final: yes' in response_lower:
            return 1
        elif 'final answer: no' in response_lower or 'final: no' in response_lower:
            return 0

        # Check for success/failure keywords
        if 'successful' in response_lower and 'not successful' not in response_lower:
            return 1
        elif 'unsuccessful' in response_lower or 'not successful' in response_lower:
            return 0

        # Check for positive/negative indicators (fallback)
        positive_indicators = ['good', 'high', 'strong', 'excellent', 'above']
        negative_indicators = ['poor', 'low', 'bad', 'weak', 'fail', 'below']

        pos_count = sum(1 for word in positive_indicators if word in response_lower)
        neg_count = sum(1 for word in negative_indicators if word in response_lower)

        if pos_count > neg_count:
            return 1
        elif neg_count > pos_count:
            return 0

        # Last resort
        clean_response = response[:100].replace('\n', ' ').replace('\r', '')
        clean_response = ''.join(char for char in clean_response if ord(char) < 128)
        self.logger.warning(f"Unclear response, defaulting to 0: {clean_response}...")
        return 0

    def run_experiment(self,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       prompt_engineer,
                       shot_levels: List[int] = [0, 1, 3, 5],
                       selection_method: str = 'similarity',
                       template_type: str = 'chain_of_thought',
                       model: Optional[str] = None,
                       sample_size: Optional[int] = None,
                       batch_size: int = 5) -> Dict:
        """
        Run complete experiment for specified shot levels

        Args:
            X_test: Test features
            y_test: Test labels
            prompt_engineer: PromptEngineer instance (trained on training data)
            shot_levels: List of shot levels to test
            selection_method: Example selection method
            template_type: Prompt template type
            model: Model to use
            sample_size: Number of test samples to use (None for all)
            batch_size: Batch size for processing

        Returns:
            Dict: Experiment results
        """
        self.logger.info(f"Starting experiment with shot levels: {shot_levels}")

        # Prepare test data
        test_data = X_test.copy()
        if sample_size:
            test_data = test_data.head(sample_size)
            self.logger.info(f"Using sample of {sample_size} test samples")

        results_by_shot = {}

        for n_shots in shot_levels:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running {n_shots}-shot experiment")
            self.logger.info(f"{'='*60}")

            predictions = []
            actuals = []
            usage_info = []
            errors = []

            # Process in batches
            for i in range(0, len(test_data), batch_size):
                batch = test_data.iloc[i:i+batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(test_data)-1)//batch_size + 1}")

                for idx, (_, row) in enumerate(batch.iterrows()):
                    try:
                        # Create prompt
                        if n_shots == 0:
                            prompt = prompt_engineer.create_zero_shot_prompt(row, template_type)
                            examples_used = None
                        else:
                            prompt, examples_used = prompt_engineer.create_few_shot_prompt(
                                row,
                                n_shots=n_shots,
                                selection_method=selection_method,
                                template_type=template_type
                            )

                        # Get prediction
                        response, usage = self.get_prediction(prompt, model=model)
                        prediction = self.parse_prediction(response)

                        # Get actual label
                        actual = y_test.loc[row.name]

                        # Store results
                        predictions.append(prediction)
                        actuals.append(actual)
                        usage_info.append(usage)

                        # FIXED: Explicitly convert boolean to integer
                        is_correct = int(prediction == actual)

                        # Store detailed result
                        examples_used_dict = None
                        if examples_used is not None:
                            try:
                                examples_used_dict = examples_used.to_dict()
                            except Exception as e:
                                self.logger.warning(f"Could not convert examples_used to dict for row {row.name}: {e}")
                                examples_used_dict = str(examples_used)

                        self.results.append({
                            'shot_level': n_shots,
                            'ad_id': row['ad_id'],
                            'prediction': prediction,
                            'actual': actual,
                            'correct': is_correct,
                            'response': response[:200] + "..." if response else "",
                            'prompt_tokens': usage['prompt_tokens'],
                            'completion_tokens': usage['completion_tokens'],
                            'cost_usd': usage.get('cost_usd', 0),
                            'model': usage['model'],
                            'response_time': usage['response_time'],
                            'timestamp': datetime.now().isoformat(),
                            'examples_used': examples_used_dict
                        })

                        # Small delay to avoid rate limits
                        time.sleep(0.5)

                    except Exception as e:
                        self.logger.error(f"Error processing row {row.name}: {e}")
                        errors.append({'index': row.name, 'error': str(e)})
                        predictions.append(None)
                        actual_fallback = y_test.loc[row.name] if row.name in y_test.index else None
                        actuals.append(actual_fallback)

            # Calculate metrics for this shot level
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            valid_predictions = [predictions[i] for i in valid_indices]
            valid_actuals = [actuals[i] for i in valid_indices]

            if valid_predictions:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef

                accuracy = accuracy_score(valid_actuals, valid_predictions)
                balanced_acc = balanced_accuracy_score(valid_actuals, valid_predictions)
                precision = precision_score(valid_actuals, valid_predictions, zero_division=0)
                recall = recall_score(valid_actuals, valid_predictions, zero_division=0)
                f1 = f1_score(valid_actuals, valid_predictions, zero_division=0)
                mcc = matthews_corrcoef(valid_actuals, valid_predictions)
                cm = confusion_matrix(valid_actuals, valid_predictions)

                results_by_shot[n_shots] = {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mcc': mcc,
                    'n_samples': len(valid_predictions),
                    'n_errors': len(errors),
                    'avg_tokens': np.mean([u['total_tokens'] for u in usage_info]) if usage_info else 0,
                    'avg_response_time': np.mean([u['response_time'] for u in usage_info]) if usage_info else 0,
                    'total_cost': 0.0
                }

                self.logger.info(f"\n{n_shots}-shot Results:")
                self.logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                self.logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
                self.logger.info(f"  Precision: {precision:.4f}")
                self.logger.info(f"  Recall: {recall:.4f}")
                self.logger.info(f"  F1-Score: {f1:.4f}")
                self.logger.info(f"  MCC: {mcc:.4f}")
                self.logger.info(f"  Samples: {len(valid_predictions)}")
                self.logger.info(f"  Errors: {len(errors)}")
                self.logger.info(f"  Confusion Matrix:\n{cm}")
            else:
                results_by_shot[n_shots] = {
                    'error': 'No valid predictions',
                    'n_errors': len(errors)
                }

        return results_by_shot

    def safe_encode_categorical(self, X_train, X_test):
        """
        Safely encode categorical variables handling unseen labels
        """
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = ['age', 'gender']

        X_train = X_train.copy()
        X_test = X_test.copy()

        for col in categorical_cols:
            if col in X_train.columns:
                all_unique = list(X_train[col].dropna().unique())

                X_train[col] = X_train[col].fillna('unknown').astype(str)
                X_test[col] = X_test[col].apply(
                    lambda x: str(x) if x in all_unique else 'unknown'
                ).fillna('unknown')

                le = LabelEncoder()
                le.fit(all_unique + ['unknown'])

                X_train[col] = le.transform(X_train[col])
                X_test[col] = le.transform(X_test[col])

        return X_train, X_test

    def run_baseline_comparison(self,
                                 X_train: pd.DataFrame,
                                 X_test: pd.DataFrame,
                                 y_train: pd.Series,
                                 y_test: pd.Series) -> Dict:
        """
        Run traditional ML baseline for comparison (Acting as a Mathematical Oracle)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef

        self.logger.info("Running baseline ML models (Mathematical Oracles) for comparison")

        results = {}

        try:
            X_train_encoded, X_test_encoded = self.safe_encode_categorical(X_train, X_test)

            numeric_cols = ['interest1', 'interest2', 'interest3', 'impressions',
                           'clicks', 'spent']

            oracle_features = ['CTR', 'CPC', 'Conversion_Rate', 'ad_id']
            for col in oracle_features:
                if col in X_train_encoded.columns:
                    X_train_encoded = X_train_encoded.drop(columns=[col])
                if col in X_test_encoded.columns:
                    X_test_encoded = X_test_encoded.drop(columns=[col])

            for col in numeric_cols:
                if col in X_train_encoded.columns:
                    median_val = X_train_encoded[col].median()
                    X_train_encoded[col] = X_train_encoded[col].fillna(median_val)
                    X_test_encoded[col] = X_test_encoded[col].fillna(median_val)

            y_train_encoded = y_train.astype(str).str.lower().map({
                'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'yes': 1, 'no': 0
            }).fillna(y_train).astype(int)

            y_test_encoded = y_test.astype(str).str.lower().map({
                 'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'yes': 1, 'no': 0
            }).fillna(y_test).astype(int)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_encoded)
            X_test_scaled = scaler.transform(X_test_encoded)

            # Logistic Regression
            self.logger.info("Training Logistic Regression (Realistic Baseline)...")
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train_scaled, y_train_encoded)
            lr_pred = lr.predict(X_test_scaled)

            results['logistic_regression'] = {
                'accuracy': accuracy_score(y_test_encoded, lr_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test_encoded, lr_pred),
                'precision': precision_score(y_test_encoded, lr_pred, zero_division=0),
                'recall': recall_score(y_test_encoded, lr_pred, zero_division=0),
                'f1_score': f1_score(y_test_encoded, lr_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_test_encoded, lr_pred)
            }

            # Random Forest
            self.logger.info("Training Random Forest (Realistic Baseline)...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train_encoded)
            rf_pred = rf.predict(X_test_scaled)

            results['random_forest'] = {
                'accuracy': accuracy_score(y_test_encoded, rf_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test_encoded, rf_pred),
                'precision': precision_score(y_test_encoded, rf_pred, zero_division=0),
                'recall': recall_score(y_test_encoded, rf_pred, zero_division=0),
                'f1_score': f1_score(y_test_encoded, rf_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_test_encoded, rf_pred)
            }

            self.logger.info(f"\nRealistic ML Baseline Results (No Oracle Features):")
            self.logger.info(f"  Logistic Regression - Accuracy: {results['logistic_regression']['accuracy']:.4f}, Balanced Acc: {results['logistic_regression']['balanced_accuracy']:.4f}")
            self.logger.info(f"  Random Forest - Accuracy: {results['random_forest']['accuracy']:.4f}, Balanced Acc: {results['random_forest']['balanced_accuracy']:.4f}")

            self.logger.info("\nNOTE: These ML models act as realistic baselines. The oracle features (CTR, CPC, Conversion Rate)")
            self.logger.info("   have been removed to prevent data leakage and provide a fair comparison with GPT.")

        except Exception as e:
            self.logger.error(f"Error in baseline comparison: {e}")
            results = {
                'logistic_regression': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'random_forest': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'error': str(e)
            }

        return results

    def save_results(self, filepath: str = "results/experiment_results.json"):
        """Save all results to JSON file"""
        output = {
            'results': self.results,
            'cost_tracker': self.cost_tracker,
            'metadata': {
                'primary_model': self.primary_model,
                'secondary_model': self.secondary_model,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'timestamp': datetime.now().isoformat()
            }
        }

        Path(filepath).parent.mkdir(exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, default=str)

        self.logger.info(f"Results saved to {filepath}")

    def generate_cost_report(self) -> Dict:
        """Generate cost analysis report (all 0 for CERIT-SC)"""
        report = {
            'total_api_calls': self.cost_tracker['api_calls'],
            'failed_calls': self.cost_tracker['failed_calls'],
            'success_rate': ((self.cost_tracker['api_calls'] - self.cost_tracker['failed_calls']) /
                           max(self.cost_tracker['api_calls'], 1) * 100),
            'total_prompt_tokens': self.cost_tracker['total_prompt_tokens'],
            'total_completion_tokens': self.cost_tracker['total_completion_tokens'],
            'total_tokens': self.cost_tracker['total_prompt_tokens'] + self.cost_tracker['total_completion_tokens'],
            'total_cost_usd': 0.0,
            'avg_cost_per_call': 0.0,
            'avg_tokens_per_call': (self.cost_tracker['total_prompt_tokens'] + self.cost_tracker['total_completion_tokens']) /
                                   max(self.cost_tracker['api_calls'], 1)
        }

        with open('results/cost_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)

        return report


def main():
    """Main experiment execution (standalone)"""
    base_logger.info("Starting main experiment...")

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    full_df = pd.read_csv("data/processed_data.csv")

    from prompt_engineering import PromptEngineer
    train_df_with_labels = X_train.copy()
    train_df_with_labels['is_success'] = y_train.values
    prompt_engineer = PromptEngineer(train_df_with_labels)

    experiment = GPTExperiment(
        primary_model=os.getenv('PRIMARY_MODEL', 'gpt-oss-120b'),
        secondary_model=os.getenv('SECONDARY_MODEL', 'llama3.3:latest'),
        temperature=0.0
    )

    total_test = len(X_test)
    print(f"\nTotal test samples available: {total_test}")
    sample_size = input("Enter number of test samples to use (recommended >= 100, or 'all' for all): ")
    sample_size = total_test if sample_size.lower() == 'all' else int(sample_size)

    results = experiment.run_experiment(
        X_test=X_test,
        y_test=y_test,
        prompt_engineer=prompt_engineer,
        shot_levels=[0, 1, 3, 5],
        selection_method='similarity',
        template_type='chain_of_thought',
        sample_size=sample_size,
        batch_size=5
    )

    baseline_results = experiment.run_baseline_comparison(
        X_train, X_test, y_train, y_test
    )

    experiment.save_results()
    cost_report = experiment.generate_cost_report()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("="*80)
    print("\nFew-Shot Learning Results (Reasoning-Based):")
    for shot_level, shot_results in results.items():
        if 'accuracy' in shot_results:
            print(f"  {shot_level}-shot: Accuracy={shot_results['accuracy']:.4f}, Balanced Acc={shot_results['balanced_accuracy']:.4f}, F1={shot_results['f1_score']:.4f}, MCC={shot_results['mcc']:.4f}")

    print("\nRealistic ML Baseline Results (No Oracle Features):")
    for model_name, model_results in baseline_results.items():
        if isinstance(model_results, dict) and 'accuracy' in model_results:
            print(f"  {model_name}: Accuracy={model_results['accuracy']:.4f}, Balanced Acc={model_results.get('balanced_accuracy', 0.0):.4f}")

    print("\nCost Summary (CERIT-SC is free!):")
    print(f"  Total API Calls: {cost_report['total_api_calls']}")
    print(f"  Total Cost: $0.00")

    print("\nResults saved to results/experiment_results.json")

    return results, baseline_results, cost_report


if __name__ == "__main__":
    main()
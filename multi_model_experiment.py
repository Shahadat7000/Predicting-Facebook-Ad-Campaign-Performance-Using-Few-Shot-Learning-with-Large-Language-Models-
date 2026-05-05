"""
Multi-Model Experiment Module for Few-Shot Learning
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiModelExperiment:
    """
    Main experiment class for multiple LLM few-shot learning on Facebook ad prediction
    """

    def __init__(self,
                 model_type: str = "gpt",  # 'gpt', 'deepseek', 'qwen'
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 150):
        """
        Initialize experiment with model configuration

        Args:
            model_type: Type of model ('gpt', 'deepseek', 'qwen')
            model_name: Specific model name
            api_key: API key for the model
            base_url: API base URL
            temperature: Temperature for generation (0.0 for deterministic)
            max_tokens: Maximum tokens in response
        """
        # Load from .env if not provided
        load_dotenv()

        self.model_type = model_type.lower()

        # Configure based on model type
        if self.model_type == "gpt":
            # CERIT-SC API (Free for Masaryk University students) [1]
            self.base_url = base_url or os.getenv("GPT_BASE_URL", "https://llm.ai.e-infra.cz/v1")
            self.model_name = model_name or os.getenv("GPT_MODEL", "gpt-oss-120b")
            self.api_key = api_key or os.getenv("GPT_API_KEY")
            if not self.api_key:
                # CERIT-SC API is free for students [1], may not need key
                self.api_key = "dummy_key_for_ceritsc"

        elif self.model_type == "deepseek":
            # DeepSeek API
            self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "sk-5f50c3d729194caca142d97eed2ac512")

        elif self.model_type == "qwen":
            # Qwen API (Alibaba Cloud)
            self.base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.model_name = model_name or os.getenv("QWEN_MODEL", "qwen-max")
            self.api_key = api_key or os.getenv("QWEN_API_KEY")

        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'gpt', 'deepseek', or 'qwen'")

        if not self.api_key:
            raise ValueError(
                f"API key not found for {self.model_type}. Set {self.model_type.upper()}_API_KEY in .env file")

        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        # Results storage
        self.results = []
        self.cost_tracker = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_cost_usd': 0.0,
            'api_calls': 0,
            'failed_calls': 0
        }

        # Pricing information (CERIT-SC is free) [1]
        self.pricing = {
            'gpt-oss-120b': {'prompt': 0.0, 'completion': 0.0},
            'deepseek-chat': {'prompt': 0.03, 'completion': 0.06},  # Example pricing
            'qwen-max': {'prompt': 0.02, 'completion': 0.08},  # Example pricing
        }

        logger.info(f"Initialized {self.model_type.upper()} Experiment with model: {self.model_name}")
        logger.info(f"API Base URL: {self.base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError))
    )
    def get_prediction(self,
                       prompt: str,
                       temperature: Optional[float] = None) -> Tuple[str, Dict]:
        """
        Get prediction from model with retry logic

        Args:
            prompt: Input prompt
            temperature: Temperature (defaults to instance value)

        Returns:
            Tuple[str, Dict]: (response text, usage info)
        """
        temp = temperature if temperature is not None else self.temperature
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
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
                'model': self.model_name,
                'model_type': self.model_type,
                'response_time': elapsed
            }

            # Track costs
            self.cost_tracker['api_calls'] += 1
            self.cost_tracker['total_prompt_tokens'] += usage['prompt_tokens']
            self.cost_tracker['total_completion_tokens'] += usage['completion_tokens']

            # Calculate cost
            if self.model_name in self.pricing:
                prompt_cost = (usage['prompt_tokens'] / 1000) * self.pricing[self.model_name]['prompt']
                completion_cost = (usage['completion_tokens'] / 1000) * self.pricing[self.model_name]['completion']
                usage['cost_usd'] = prompt_cost + completion_cost
                self.cost_tracker['total_cost_usd'] += usage['cost_usd']
            else:
                usage['cost_usd'] = 0.0  # CERIT-SC is free [1]

            logger.debug(f"API call successful: {usage['total_tokens']} tokens")
            return response_text, usage

        except Exception as e:
            self.cost_tracker['failed_calls'] += 1
            logger.error(f"API call failed for {self.model_type}: {e}")
            raise

    def parse_prediction(self, response: str) -> int:
        """
        Robust parsing of response to binary prediction using regex
        """
        # Same as your existing parse_prediction method
        # ... [Keep your existing parse_prediction code exactly as is]
        # Handle None or empty response
        if response is None or not isinstance(response, str) or len(response.strip()) == 0:
            logger.warning("Empty response, defaulting to 0")
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
        logger.warning(f"Unclear response, defaulting to 0: {clean_response}...")
        return 0

    def run_experiment(self,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       prompt_engineer,
                       shot_levels: List[int] = [0, 1, 3, 5],
                       selection_method: str = 'similarity',
                       template_type: str = 'chain_of_thought',
                       sample_size: Optional[int] = None,
                       batch_size: int = 5) -> Dict:
        """
        Run complete experiment for specified shot levels
        """
        logger.info(f"Starting {self.model_type.upper()} experiment with shot levels: {shot_levels}")

        # Prepare test data
        test_data = X_test.copy()
        if sample_size:
            test_data = test_data.head(sample_size)
            logger.info(f"Using sample of {sample_size} test samples")

        results_by_shot = {}

        for n_shots in shot_levels:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running {n_shots}-shot experiment for {self.model_type.upper()}")
            logger.info(f"{'=' * 60}")

            predictions = []
            actuals = []
            usage_info = []
            errors = []

            # Process in batches
            for i in range(0, len(test_data), batch_size):
                batch = test_data.iloc[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(test_data) - 1) // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")

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
                        response, usage = self.get_prediction(prompt)
                        prediction = self.parse_prediction(response)

                        # Get actual label
                        actual = y_test.loc[row.name]

                        # Store results
                        predictions.append(prediction)
                        actuals.append(actual)
                        usage_info.append(usage)

                        # Convert boolean to integer
                        is_correct = int(prediction == actual)

                        # Store detailed result
                        examples_used_dict = None
                        if examples_used is not None:
                            try:
                                examples_used_dict = examples_used.to_dict()
                            except Exception as e:
                                logger.warning(f"Could not convert examples_used to dict for row {row.name}: {e}")
                                examples_used_dict = str(examples_used)

                        self.results.append({
                            'model_type': self.model_type,
                            'model_name': self.model_name,
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
                        logger.error(f"Error processing row {row.name}: {e}")
                        errors.append({'index': row.name, 'error': str(e)})
                        predictions.append(None)
                        actual_fallback = y_test.loc[row.name] if row.name in y_test.index else None
                        actuals.append(actual_fallback)

            # Calculate metrics for this shot level
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            valid_predictions = [predictions[i] for i in valid_indices]
            valid_actuals = [actuals[i] for i in valid_indices]

            if valid_predictions:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
                    balanced_accuracy_score, matthews_corrcoef

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
                    'total_cost': self.cost_tracker['total_cost_usd']
                }

                logger.info(f"\n{self.model_type.upper()} {n_shots}-shot Results:")
                logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                logger.info(f"  MCC: {mcc:.4f}")
                logger.info(f"  Samples: {len(valid_predictions)}")
                logger.info(f"  Errors: {len(errors)}")
                logger.info(f"  Confusion Matrix:\n{cm}")
            else:
                results_by_shot[n_shots] = {
                    'error': 'No valid predictions',
                    'n_errors': len(errors)
                }

        return results_by_shot

    # Keep your existing safe_encode_categorical and run_baseline_comparison methods
    # ... [Copy your existing methods exactly as they are]

    def save_results(self, filepath: Optional[str] = None):
        """Save all results to JSON file with model-specific naming"""
        if filepath is None:
            # Auto-generate filename based on model type and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/{self.model_type}_{self.model_name}_{timestamp}_results.json"

        output = {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'results': self.results,
            'cost_tracker': self.cost_tracker,
            'metadata': {
                'base_url': self.base_url,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def generate_cost_report(self) -> Dict:
        """Generate cost analysis report"""
        report = {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'total_api_calls': self.cost_tracker['api_calls'],
            'failed_calls': self.cost_tracker['failed_calls'],
            'success_rate': ((self.cost_tracker['api_calls'] - self.cost_tracker['failed_calls']) /
                             max(self.cost_tracker['api_calls'], 1) * 100),
            'total_prompt_tokens': self.cost_tracker['total_prompt_tokens'],
            'total_completion_tokens': self.cost_tracker['total_completion_tokens'],
            'total_tokens': self.cost_tracker['total_prompt_tokens'] + self.cost_tracker['total_completion_tokens'],
            'total_cost_usd': self.cost_tracker['total_cost_usd'],
            'avg_cost_per_call': self.cost_tracker['total_cost_usd'] / max(self.cost_tracker['api_calls'], 1),
            'avg_tokens_per_call': (self.cost_tracker['total_prompt_tokens'] + self.cost_tracker[
                'total_completion_tokens']) /
                                   max(self.cost_tracker['api_calls'], 1)
        }

        # Save report
        cost_filepath = f"results/{self.model_type}_{self.model_name}_cost_report.json"
        with open(cost_filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)

        return report


# Main function to run experiments for all models
def main():
    """Run experiments for GPT, DeepSeek, and Qwen models"""
    logger.info("Starting multi-model experiment...")

    # Load data
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    # Load full processed data for threshold info
    full_df = pd.read_csv("data/processed_data.csv")

    # Initialize prompt engineer with ONLY training data
    from prompt_engineering import PromptEngineer
    train_df_with_labels = X_train.copy()
    train_df_with_labels['is_success'] = y_train.values
    prompt_engineer = PromptEngineer(train_df_with_labels)

    # Models to test (with their configurations)
    models_to_test = [
        {
            "name": "GPT OSS 120B",
            "type": "gpt",
            "model_name": "gpt-oss-120b",
            "api_key": os.getenv("GPT_API_KEY", "dummy_key_for_ceritsc"),
            "base_url": os.getenv("GPT_BASE_URL", "https://llm.ai.e-infra.cz/v1")
        },
        {
            "name": "DeepSeek v3.2",
            "type": "deepseek",
            "model_name": "deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-5f50c3d729194caca142d97eed2ac512"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        },
        {
            "name": "Qwen 3.5",
            "type": "qwen",
            "model_name": "qwen-max",
            "api_key": os.getenv("QWEN_API_KEY"),
            "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        }
    ]

    all_results = {}

    # Ask for sample size
    total_test = len(X_test)
    print(f"\nTotal test samples available: {total_test}")
    sample_size = input("Enter number of test samples to use (recommended >= 100, or 'all' for all): ")
    if sample_size.lower() == 'all':
        sample_size = total_test
    else:
        sample_size = int(sample_size)

    for model_config in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing {model_config['name']}")
        print(f"{'=' * 80}")

        # Skip if API key is not available
        if not model_config.get('api_key') and model_config['type'] != 'gpt':
            print(f"Skipping {model_config['name']}: API key not found")
            continue

        try:
            # Initialize experiment for this model
            experiment = MultiModelExperiment(
                model_type=model_config["type"],
                model_name=model_config["model_name"],
                api_key=model_config["api_key"],
                base_url=model_config.get("base_url"),
                temperature=0.0
            )

            # Run experiment
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

            # Store results
            all_results[model_config['name']] = results

            # Save results to model-specific file
            results_file = experiment.save_results()

            # Generate cost report
            cost_report = experiment.generate_cost_report()

            # Run baseline comparison (once is enough)
            if model_config['type'] == 'gpt':  # Run baseline only once
                baseline_results = experiment.run_baseline_comparison(
                    X_train, X_test, y_train, y_test
                )

        except Exception as e:
            print(f"Error testing {model_config['name']}: {e}")
            continue

    # Generate comparison table
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL MODELS")
    print("=" * 80)

    headers = ["Model", "0-shot", "1-shot", "3-shot", "5-shot", "Best Accuracy", "Total Cost"]
    print(
        f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<15} {headers[6]:<10}")
    print("-" * 100)

    for model_name, results in all_results.items():
        accuracies = []
        total_cost = 0

        for shot in [0, 1, 3, 5]:
            if shot in results and 'accuracy' in results[shot]:
                accuracies.append(results[shot]['accuracy'] * 100)
                total_cost += results[shot].get('total_cost', 0)
            else:
                accuracies.append(0.0)

        best_acc = max(accuracies) if accuracies else 0.0
        print(
            f"{model_name:<20} {accuracies[0]:<10.2f} {accuracies[1]:<10.2f} {accuracies[2]:<10.2f} {accuracies[3]:<10.2f} {best_acc:<15.2f} ${total_cost:<10.2f}")

    # Print thesis results for comparison [1]
    print("\n" + "-" * 100)
    print("THESIS RESULTS (GPT-OSS-120B) [1]:")
    print("0-shot: 86.90% | 1-shot: 85.59% | 3-shot: 88.65% | 5-shot: 90.39% | Cost: $0.00")

    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    print("   - ML models act as realistic baselines without access to formula components")
    print("   - This enables a fair comparison with GPT models")
    print("   - GPT achieving comparable or higher accuracy demonstrates strong reasoning capabilities")
    print("-" * 60)

    print("\n All experiments completed!")
    print("Results saved to separate files in 'results/' directory:")
    for model_config in models_to_test:
        print(f"   - {model_config['name']}: {model_config['type']}_{model_config['model_name']}_*.json")

    return all_results, baseline_results if 'baseline_results' in locals() else None


if __name__ == "__main__":
    main()
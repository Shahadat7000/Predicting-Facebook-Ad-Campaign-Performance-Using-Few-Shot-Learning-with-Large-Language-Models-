"""
Multi-Model Experiment Module for Few-Shot Learning
Final thesis-ready version.

This module evaluates multiple OpenAI-compatible LLM endpoints on
Facebook ad campaign success prediction using zero-shot and few-shot prompts.
"""

# ============================================
# Windows Encoding Fix - MUST BE NEAR TOP
# ============================================
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================
# Imports
# ============================================
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================
# Setup
# ============================================
load_dotenv()
Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/experiment.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class MultiModelExperiment:
    """Experiment class for multiple LLM few-shot learning runs."""

    def __init__(
        self,
        model_type: str = "gpt",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ):
        load_dotenv()

        self.model_type = model_type.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        # All thesis models can run through CERIT / OpenAI-compatible endpoint.
        # Keep model_type mostly for reporting.
        if self.model_type in {"gpt", "deepseek", "qwen", "cerit"}:
            self.base_url = base_url or os.getenv("GPT_BASE_URL") or os.getenv(
                "OPENAI_BASE_URL", "https://llm.ai.e-infra.cz/v1"
            )
            self.model_name = model_name or os.getenv("GPT_MODEL", "gpt-oss-120b")
            self.api_key = api_key or os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Use 'gpt', 'deepseek', 'qwen', or 'cerit'."
            )

        if not self.api_key:
            # Some CERIT-compatible setups still require an auth header.
            # This avoids putting secrets in code. Prefer setting GPT_API_KEY in .env.
            self.api_key = "dummy_key_for_ceritsc"
            logger.warning("No API key found. Using dummy key. Set GPT_API_KEY/OPENAI_API_KEY if your endpoint requires it.")

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        self.results = []
        self.cost_tracker = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
            "api_calls": 0,
            "failed_calls": 0,
        }

        # CERIT-SC academic endpoint treated as cost-free in the thesis.
        self.pricing = {
            "gpt-oss-120b": {"prompt": 0.0, "completion": 0.0},
            "deepseek-v3.2": {"prompt": 0.0, "completion": 0.0},
            "qwen3.5": {"prompt": 0.0, "completion": 0.0},
        }

        logger.info(f"Initialized MultiModelExperiment with model_type={self.model_type}, model={self.model_name}")
        logger.info(f"API Base URL: {self.base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError)),
    )
    def get_prediction(self, prompt: str, temperature: Optional[float] = None) -> Tuple[str, Dict]:
        """Call the LLM and return response text plus usage metadata."""
        temp = self.temperature if temperature is None else temperature
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=self.max_tokens,
            )

            elapsed = time.time() - start_time
            response_text = response.choices[0].message.content or ""

            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "model": self.model_name,
                "model_type": self.model_type,
                "response_time": elapsed,
            }

            self.cost_tracker["api_calls"] += 1
            self.cost_tracker["total_prompt_tokens"] += usage["prompt_tokens"]
            self.cost_tracker["total_completion_tokens"] += usage["completion_tokens"]

            prices = self.pricing.get(self.model_name, {"prompt": 0.0, "completion": 0.0})
            usage["cost_usd"] = (usage["prompt_tokens"] / 1000) * prices["prompt"] + (
                usage["completion_tokens"] / 1000
            ) * prices["completion"]
            self.cost_tracker["total_cost_usd"] += usage["cost_usd"]

            return response_text, usage

        except Exception as exc:
            self.cost_tracker["failed_calls"] += 1
            logger.error(f"API call failed for model={self.model_name}: {exc}")
            raise

    def parse_prediction(self, response: str) -> int:
        """Parse an LLM response into a binary prediction: 0 or 1."""
        if response is None or not isinstance(response, str) or not response.strip():
            logger.warning("Empty response, defaulting to 0")
            return 0

        text = response.strip()

        if text in {"0", "1"}:
            return int(text)

        # Prefer explicit final answer patterns.
        final_match = re.search(r"(?:final\s*answer|answer|prediction)\s*[:\-]?\s*([01])\b", text, re.IGNORECASE)
        if final_match:
            return int(final_match.group(1))

        # Then any isolated 0/1.
        isolated_match = re.search(r"\b([01])\b", text)
        if isolated_match:
            return int(isolated_match.group(1))

        lower = text.lower()
        first_line = lower.splitlines()[0].strip() if lower.splitlines() else ""

        if "yes" in first_line and "no" not in first_line:
            return 1
        if "no" in first_line and "yes" not in first_line:
            return 0

        if "final answer: yes" in lower or "final: yes" in lower:
            return 1
        if "final answer: no" in lower or "final: no" in lower:
            return 0

        # Important: check unsuccessful before successful because unsuccessful contains successful.
        if "unsuccessful" in lower or "not successful" in lower or "failed" in lower:
            return 0
        if "successful" in lower:
            return 1

        logger.warning(f"Unclear response, defaulting to 0: {text[:100].replace(chr(10), ' ')}...")
        return 0

    def run_experiment(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        prompt_engineer,
        shot_levels: List[int] = [0, 1, 3, 5],
        selection_method: str = "similarity",
        template_type: str = "chain_of_thought",
        sample_size: Optional[int] = None,
        batch_size: int = 5,
    ) -> Dict:
        """Run experiment for all requested shot levels."""
        logger.info(f"Starting experiment for {self.model_name} with shot levels: {shot_levels}")

        test_data = X_test.copy()
        if sample_size is not None:
            test_data = test_data.head(sample_size)
            logger.info(f"Using sample of {sample_size} test samples")

        results_by_shot = {}

        for n_shots in shot_levels:
            logger.info("\n" + "=" * 60)
            logger.info(f"Running {n_shots}-shot experiment for {self.model_name}")
            logger.info("=" * 60)

            predictions = []
            actuals = []
            usage_info = []
            errors = []

            for i in range(0, len(test_data), batch_size):
                batch = test_data.iloc[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(test_data) - 1) // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")

                for _, row in batch.iterrows():
                    try:
                        if n_shots == 0:
                            prompt = prompt_engineer.create_zero_shot_prompt(row, template_type)
                            examples_used = None
                        else:
                            prompt, examples_used = prompt_engineer.create_few_shot_prompt(
                                row,
                                n_shots=n_shots,
                                selection_method=selection_method,
                                template_type=template_type,
                            )

                        response, usage = self.get_prediction(prompt)
                        prediction = self.parse_prediction(response)
                        actual = int(y_test.loc[row.name])

                        predictions.append(prediction)
                        actuals.append(actual)
                        usage_info.append(usage)

                        examples_used_dict = None
                        if examples_used is not None:
                            try:
                                examples_used_dict = examples_used.to_dict()
                            except Exception:
                                examples_used_dict = str(examples_used)

                        self.results.append(
                            {
                                "model_type": self.model_type,
                                "model_name": self.model_name,
                                "shot_level": n_shots,
                                "ad_id": row.get("ad_id", row.name),
                                "prediction": prediction,
                                "actual": actual,
                                "correct": int(prediction == actual),
                                "response": (response[:200] + "...") if response and len(response) > 200 else response,
                                "prompt_tokens": usage["prompt_tokens"],
                                "completion_tokens": usage["completion_tokens"],
                                "cost_usd": usage.get("cost_usd", 0.0),
                                "model": usage["model"],
                                "response_time": usage["response_time"],
                                "timestamp": datetime.now().isoformat(),
                                "examples_used": examples_used_dict,
                            }
                        )

                        time.sleep(0.5)

                    except Exception as exc:
                        logger.error(f"Error processing row {row.name}: {exc}")
                        errors.append({"index": row.name, "error": str(exc)})
                        predictions.append(None)
                        actuals.append(int(y_test.loc[row.name]) if row.name in y_test.index else None)

            valid_indices = [idx for idx, pred in enumerate(predictions) if pred is not None and actuals[idx] is not None]
            valid_predictions = [predictions[idx] for idx in valid_indices]
            valid_actuals = [actuals[idx] for idx in valid_indices]

            if valid_predictions:
                accuracy = accuracy_score(valid_actuals, valid_predictions)
                balanced_acc = balanced_accuracy_score(valid_actuals, valid_predictions)
                precision = precision_score(valid_actuals, valid_predictions, zero_division=0)
                recall = recall_score(valid_actuals, valid_predictions, zero_division=0)
                f1 = f1_score(valid_actuals, valid_predictions, zero_division=0)
                mcc = matthews_corrcoef(valid_actuals, valid_predictions)
                cm = confusion_matrix(valid_actuals, valid_predictions)

                results_by_shot[n_shots] = {
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "mcc": mcc,
                    "confusion_matrix": cm.tolist(),
                    "n_samples": len(valid_predictions),
                    "n_errors": len(errors),
                    "avg_tokens": float(np.mean([u["total_tokens"] for u in usage_info])) if usage_info else 0.0,
                    "avg_response_time": float(np.mean([u["response_time"] for u in usage_info])) if usage_info else 0.0,
                    "total_cost": self.cost_tracker["total_cost_usd"],
                }

                logger.info(f"\n{self.model_name} {n_shots}-shot Results:")
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
                results_by_shot[n_shots] = {"error": "No valid predictions", "n_errors": len(errors)}

        return results_by_shot

    def safe_encode_categorical(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Safely encode categorical columns for baseline models.
        Kept for compatibility, but run_baseline_comparison uses a sklearn Pipeline.
        """
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()

        categorical_cols = [col for col in train_encoded.columns if train_encoded[col].dtype == "object"]
        for col in categorical_cols:
            combined_values = pd.concat([train_encoded[col], test_encoded[col]], axis=0).astype(str)
            categories = {value: idx for idx, value in enumerate(sorted(combined_values.unique()))}
            train_encoded[col] = train_encoded[col].astype(str).map(categories)
            test_encoded[col] = test_encoded[col].astype(str).map(categories)

        return train_encoded, test_encoded

    def run_baseline_comparison(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict:
        """
        Run Logistic Regression and Random Forest baselines.
        Oracle metrics are removed so baselines do not directly see the success formula.
        """
        logger.info("Running traditional ML baseline comparison...")

        oracle_cols = ["CTR", "CPC", "Conversion_Rate", "is_success"]
        drop_cols = [col for col in oracle_cols if col in X_train.columns]

        X_train_base = X_train.drop(columns=drop_cols, errors="ignore").copy()
        X_test_base = X_test.drop(columns=drop_cols, errors="ignore").copy()

        # Remove identifier/date columns that are not meaningful predictive features.
        non_feature_cols = [
            "ad_id",
            "reporting_start",
            "reporting_end",
            "campaign_id",
            "fb_campaign_id",
            "xyz_campaign_id",
        ]
        X_train_base = X_train_base.drop(columns=[c for c in non_feature_cols if c in X_train_base.columns], errors="ignore")
        X_test_base = X_test_base.drop(columns=[c for c in non_feature_cols if c in X_test_base.columns], errors="ignore")

        numeric_cols = X_train_base.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in X_train_base.columns if col not in numeric_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ],
            remainder="drop",
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=None, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None),
        }

        baseline_results = {}
        for name, model in models.items():
            pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            pipeline.fit(X_train_base, y_train)
            preds = pipeline.predict(X_test_base)

            baseline_results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1_score": f1_score(y_test, preds, zero_division=0),
                "mcc": matthews_corrcoef(y_test, preds),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
                "features_used": X_train_base.columns.tolist(),
                "removed_oracle_cols": drop_cols,
            }

            logger.info(f"{name} baseline: accuracy={baseline_results[name]['accuracy']:.4f}, "
                        f"balanced_accuracy={baseline_results[name]['balanced_accuracy']:.4f}")

        filepath = "results/baseline_comparison.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(baseline_results, f, indent=4, default=str)
        logger.info(f"Baseline comparison saved to {filepath}")

        return baseline_results

    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save detailed LLM results to JSON."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
            filepath = f"results/{self.model_type}_{safe_model_name}_{timestamp}_results.json"

        output = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "results": self.results,
            "cost_tracker": self.cost_tracker,
            "metadata": {
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timestamp": datetime.now().isoformat(),
            },
        }

        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def generate_cost_report(self) -> Dict:
        """Generate cost/token report."""
        total_calls = max(self.cost_tracker["api_calls"], 1)
        total_tokens = self.cost_tracker["total_prompt_tokens"] + self.cost_tracker["total_completion_tokens"]

        report = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "total_api_calls": self.cost_tracker["api_calls"],
            "failed_calls": self.cost_tracker["failed_calls"],
            "success_rate": ((self.cost_tracker["api_calls"] - self.cost_tracker["failed_calls"]) / total_calls) * 100,
            "total_prompt_tokens": self.cost_tracker["total_prompt_tokens"],
            "total_completion_tokens": self.cost_tracker["total_completion_tokens"],
            "total_tokens": total_tokens,
            "total_cost_usd": self.cost_tracker["total_cost_usd"],
            "avg_cost_per_call": self.cost_tracker["total_cost_usd"] / total_calls,
            "avg_tokens_per_call": total_tokens / total_calls,
        }

        safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        cost_filepath = f"results/{self.model_type}_{safe_model_name}_cost_report.json"
        with open(cost_filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Cost report saved to {cost_filepath}")
        return report


def main():
    """Run thesis experiments for GPT OSS 120B, DeepSeek v3.2, and Qwen3.5."""
    logger.info("Starting multi-model experiment...")

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    from prompt_engineering import PromptEngineer

    train_df_with_labels = X_train.copy()
    train_df_with_labels["is_success"] = y_train.values
    prompt_engineer = PromptEngineer(train_df_with_labels)

    # Use the same OpenAI-compatible CERIT endpoint for all three thesis models.
    shared_api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    shared_base_url = os.getenv("GPT_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://llm.ai.e-infra.cz/v1")

    models_to_test = [
        {"display_name": "DeepSeek v3.2", "type": "deepseek", "model_name": "deepseek-v3.2"},
        {"display_name": "GPT OSS 120B", "type": "gpt", "model_name": "gpt-oss-120b"},
        {"display_name": "Qwen3.5", "type": "qwen", "model_name": "qwen3.5"},
    ]

    total_test = len(X_test)
    print(f"\nTotal test samples available: {total_test}")
    sample_size_input = input("Enter number of test samples to use (recommended >= 100, or 'all' for all): ").strip()
    if sample_size_input.lower() == "all":
        sample_size = total_test
    else:
        sample_size = int(sample_size_input)

    all_results = {}
    baseline_results = None

    for model_config in models_to_test:
        print("\n" + "=" * 80)
        print(f"Testing {model_config['display_name']}")
        print("=" * 80)

        try:
            experiment = MultiModelExperiment(
                model_type=model_config["type"],
                model_name=model_config["model_name"],
                api_key=shared_api_key,
                base_url=shared_base_url,
                temperature=0.0,
                max_tokens=150,
            )

            results = experiment.run_experiment(
                X_test=X_test,
                y_test=y_test,
                prompt_engineer=prompt_engineer,
                shot_levels=[0, 1, 3, 5],
                selection_method="similarity",
                template_type="chain_of_thought",
                sample_size=sample_size,
                batch_size=5,
            )

            all_results[model_config["display_name"]] = results
            experiment.save_results()
            experiment.generate_cost_report()

            if baseline_results is None:
                baseline_results = experiment.run_baseline_comparison(X_train, X_test, y_train, y_test)

        except Exception as exc:
            logger.error(f"Error testing {model_config['display_name']}: {exc}")
            print(f"Error testing {model_config['display_name']}: {exc}")
            continue

    print("\n" + "=" * 80)
    print("COMPARISON OF ALL MODELS")
    print("=" * 80)

    print(f"{'Model':<20} {'0-shot':<15} {'1-shot':<15} {'3-shot':<15} {'5-shot':<15} {'Best Acc':<12}")
    print("-" * 95)

    for model_name, results in all_results.items():
        acc_strings = []
        acc_values = []
        for shot in [0, 1, 3, 5]:
            if shot in results and "accuracy" in results[shot]:
                acc = results[shot]["accuracy"] * 100
                bal = results[shot]["balanced_accuracy"] * 100
                acc_values.append(acc)
                acc_strings.append(f"{acc:.2f}/{bal:.2f}")
            else:
                acc_strings.append("N/A")
        best_acc = max(acc_values) if acc_values else 0.0
        print(f"{model_name:<20} {acc_strings[0]:<15} {acc_strings[1]:<15} {acc_strings[2]:<15} {acc_strings[3]:<15} {best_acc:<12.2f}")

    summary_path = "results/all_model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"llm_results": all_results, "baseline_results": baseline_results}, f, indent=4, default=str)
    logger.info(f"All-model summary saved to {summary_path}")

    print("\nAll experiments completed.")
    print("Accuracy cells are shown as Accuracy/Balanced Accuracy (%).")
    print("Results saved in results/ directory.")

    return all_results, baseline_results


if __name__ == "__main__":
    main()

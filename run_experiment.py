#!/usr/bin/env python3
"""
Master Experiment Runner for Facebook Ad Prediction Thesis
Run this script to execute the complete experiment pipeline.

Final pipeline:
1. Data preprocessing
2. Train-test split
3. Multi-model LLM experiments using MultiModelExperiment
4. Statistical analysis
5. Visualization generation
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# ------------------------------------------------------------
# IMPORTANT: create logs folder before FileHandler is created
# ------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)

# Add src to path if you keep modules inside src/
sys.path.append("src")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/master_experiment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories."""
    dirs = ["data", "results", "visualizations", "logs", "checkpoints", "src"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    logger.info("Directories created/verified")


def step1_data_preprocessing():
    """Step 1: Load and preprocess data using clean dataset."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)

    from data_preprocessing import FacebookAdDataPreprocessor
    from success_metrics import generate_success_metrics_report

    preprocessor = FacebookAdDataPreprocessor("data/data_clean.csv")
    preprocessor.load_data()
    preprocessor.calculate_performance_metrics()
    preprocessor.split_data(test_size=0.2, random_state=42)
    preprocessor.define_success_labels(method="absolute")

    validation = preprocessor.validate_data_quality()
    if validation["has_issues"]:
        logger.warning(f"Data quality issues found: {validation['issues']}")
        response = input("Continue despite issues? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)

    preprocessor.save_processed_data("data/")
    df = preprocessor.preprocessed_df
    generate_success_metrics_report(df)

    logger.info("Data preprocessing completed")
    return df


def step2_train_test_split(df):
    """Step 2: Create train-test split."""
    logger.info("=" * 60)
    logger.info("STEP 2: TRAIN-TEST SPLIT")
    logger.info("=" * 60)

    from train_test_split import prepare_train_test_split, validate_split

    X_train, X_test, y_train, y_test, split_info = prepare_train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=True
    )

    validation = validate_split(X_train, X_test, y_train, y_test)
    if validation["warnings"]:
        logger.warning("Split validation warnings:")
        for warning in validation["warnings"]:
            logger.warning(f"  - {warning}")

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.astype(int).to_csv("data/y_train.csv", index=False)
    y_test.astype(int).to_csv("data/y_test.csv", index=False)

    logger.info(f"Split complete: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def step3_run_experiment(X_train, X_test, y_train, y_test):
    """
    Step 3: Run multi-model LLM experiments.

    This function uses multi_model_experiment.py, not model_experiment.py.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: LLM EXPERIMENTS USING MultiModelExperiment")
    logger.info("=" * 60)

    from dotenv import load_dotenv
    load_dotenv()

    from prompt_engineering import PromptEngineer
    from multi_model_experiment import MultiModelExperiment

    train_df = X_train.copy()
    train_df["is_success"] = y_train.values
    prompt_engineer = PromptEngineer(train_df)

    # Thesis model names
    models_to_test = [
        "deepseek-v3.2",
        "gpt-oss-120b",
        "qwen3.5"
    ]

    total_test = len(X_test)
    print(f"\nTotal test samples available: {total_test}")
    sample_size_input = input("Enter number of test samples to use (recommended >= 100, or 'all' for all): ").strip()

    if sample_size_input.lower() == "all":
        sample_size = total_test
    else:
        try:
            sample_size = int(sample_size_input)
            if sample_size <= 0 or sample_size > total_test:
                print(f"Invalid sample size. Using all {total_test} samples.")
                sample_size = total_test
        except ValueError:
            print(f"Invalid input. Using all {total_test} samples.")
            sample_size = total_test

    api_key = (
        os.getenv("GPT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "dummy_key_for_ceritsc"
    )
    base_url = (
        os.getenv("GPT_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://llm.ai.e-infra.cz/v1"
    )

    all_results = {}
    baseline_results = None

    for model_name in models_to_test:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"TESTING MODEL: {model_name}")
        logger.info("=" * 60)

        experiment = MultiModelExperiment(
            model_type="gpt",
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=150
        )

        results = experiment.run_experiment(
            X_test=X_test,
            y_test=y_test,
            prompt_engineer=prompt_engineer,
            shot_levels=[0, 1, 3, 5],
            selection_method="similarity",
            template_type="chain_of_thought",
            sample_size=sample_size,
            batch_size=5
        )

        safe_name = model_name.replace(".", "_").replace("-", "_")
        experiment.save_results(f"results/experiment_results_{safe_name}.json")
        experiment.generate_cost_report()

        all_results[model_name] = results

        # Baseline is model-independent, so run only once.
        if baseline_results is None:
            if hasattr(experiment, "run_baseline_comparison"):
                baseline_results = experiment.run_baseline_comparison(
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )
            else:
                logger.warning(
                    "run_baseline_comparison() not found in MultiModelExperiment. "
                    "Skipping baseline."
                )
                baseline_results = {}

    with open("results/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=4, default=str)

    print("\n" + "=" * 80)
    print("MULTI-MODEL EXPERIMENT SUMMARY")
    print("=" * 80)

    for model, res in all_results.items():
        print(f"\n{model}:")
        for shot in [0, 1, 3, 5]:
            metrics = res.get(shot, {})
            if isinstance(metrics, dict) and "accuracy" in metrics:
                print(
                    f"  {shot}-shot: "
                    f"Accuracy={metrics['accuracy']:.4f}, "
                    f"Balanced Acc={metrics['balanced_accuracy']:.4f}, "
                    f"MCC={metrics.get('mcc', 0):.4f}"
                )
            else:
                print(f"  {shot}-shot: No valid result")

    return all_results, baseline_results, None


def step4_statistical_analysis(df):
    """Step 4: Perform statistical analysis for all model result files."""
    logger.info("=" * 60)
    logger.info("STEP 4: STATISTICAL ANALYSIS (Per Model)")
    logger.info("=" * 60)

    from statistical_analysis import StatisticalAnalyzer
    import pandas as pd

    results_path = Path("results")
    model_files = list(results_path.glob("experiment_results_*.json"))

    if not model_files:
        logger.warning("No experiment results found. Running descriptive analysis only.")
        analyzer = StatisticalAnalyzer(df, None)
        report = analyzer.generate_comprehensive_report()
        return report

    for model_file in model_files:
        model_name = model_file.stem.replace("experiment_results_", "")
        safe_model = model_name.replace(".", "_").replace("-", "_")
        logger.info(f"Analyzing {model_name}...")

        with open(model_file, "r", encoding="utf-8") as f:
            results_json = json.load(f)

        if "results" in results_json:
            results_df = pd.DataFrame(results_json["results"])
            try:
                from visualization import _convert_to_int
                for col in ["actual", "prediction", "correct"]:
                    if col in results_df.columns:
                        results_df = _convert_to_int(results_df, col)
            except Exception as e:
                logger.warning(f"Could not convert result columns to int: {e}")
        else:
            results_df = None
            logger.warning(f"No 'results' key in {model_file}")

        analyzer = StatisticalAnalyzer(df, results_df)
        analyzer.generate_comprehensive_report(
            f"results/statistical_report_{safe_model}.json"
        )

        print(f"\n--- Statistical Summary for {model_name} ---")
        analyzer.print_summary()

    logger.info("Statistical reports saved for each model in results/")
    return None


def step5_generate_visualizations(df):
    """Step 5: Generate visualizations for each model and overall comparison."""
    logger.info("=" * 60)
    logger.info("STEP 5: VISUALIZATION (Per Model + Comparison)")
    logger.info("=" * 60)

    from visualization import ThesisVisualizer, _convert_to_int
    import pandas as pd

    baseline_path = Path("results/baseline_results.json")
    baseline_results = None
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)

    results_path = Path("results")
    model_files = list(results_path.glob("experiment_results_*.json"))

    if not model_files:
        logger.warning("No model result files found. Skipping visualizations.")
        return

    # Per-model visualizations
    for model_file in model_files:
        model_name = model_file.stem.replace("experiment_results_", "")
        logger.info(f"Generating visualizations for: {model_name}")

        with open(model_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "results" in data:
            results_df = pd.DataFrame(data["results"])
            for col in ["actual", "prediction", "correct"]:
                if col in results_df.columns:
                    results_df = _convert_to_int(results_df, col)
        else:
            results_df = None
            logger.warning(f"No 'results' key in {model_file}")

        model_output_dir = f"visualizations/{model_name}"
        visualizer = ThesisVisualizer(df, results_df, output_dir=model_output_dir)
        visualizer.generate_all_visualizations(baseline_results)

    # Overall comparison
    ThesisVisualizer.generate_model_comparison("results", "visualizations")

    # Combined comparison charts
    if hasattr(ThesisVisualizer, "generate_combined_accuracy_by_shot"):
        ThesisVisualizer.generate_combined_accuracy_by_shot()
    if hasattr(ThesisVisualizer, "generate_combined_model_comparison"):
        ThesisVisualizer.generate_combined_model_comparison()

    logger.info("Visualizations and comparison saved to visualizations/")


def main():
    start_time = time.time()

    print("\n" + "=" * 80)
    print("FACEBOOK AD PREDICTION THESIS - MASTER EXPERIMENT RUNNER")
    print("=" * 80)
    print("\nPipeline steps:")
    print("  1. Data preprocessing using clean dataset")
    print("  2. Train-test split")
    print("  3. Multi-model LLM experiments using multi_model_experiment.py")
    print("  4. Statistical analysis per model")
    print("  5. Visualization generation per model + comparison")

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("GPT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://llm.ai.e-infra.cz/v1"

    if not api_key:
        print("\nWARNING: GPT_API_KEY / OPENAI_API_KEY not found in .env file.")
        print("If CERIT-SC endpoint works without a real key, the script will use a dummy key.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)
    else:
        print("\nAPI key found")
        print(f"Base URL: {base_url}")

    setup_directories()

    try:
        df = step1_data_preprocessing()
        X_train, X_test, y_train, y_test = step2_train_test_split(df)

        response = input("\nRun LLM experiments? This may call the API (y/n): ")
        if response.lower() == "y":
            results, baseline_results, experiment = step3_run_experiment(
                X_train,
                X_test,
                y_train,
                y_test
            )
            print("\nCost details are saved in each model's result/cost report file.")

        step4_statistical_analysis(df)
        step5_generate_visualizations(df)

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"EXPERIMENT COMPLETE! Time elapsed: {elapsed / 60:.2f} minutes")
        print("=" * 80)
        print("\nResults saved to:")
        print("  - results/experiment_results_<model>.json")
        print("  - results/statistical_report_<model>.json")
        print("  - results/baseline_results.json")
        print("  - visualizations/<model>/")
        print("  - visualizations/model_comparison_heatmap.png")
        print("  - visualizations/model_comparison_table.csv")
        print("  - visualizations/combined_accuracy_by_shot.png")
        print("  - visualizations/combined_model_comparison.png")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\nExperiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

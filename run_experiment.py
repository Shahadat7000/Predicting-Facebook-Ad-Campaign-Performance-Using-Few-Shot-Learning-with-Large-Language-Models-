#!/usr/bin/env python3
"""
Master Experiment Runner for Facebook Ad Prediction Thesis
Run this script to execute the complete experiment pipeline
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'results', 'visualizations', 'logs', 'checkpoints', 'src']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    logger.info("Directories created/verified")


def step1_data_preprocessing():
    """Step 1: Load and preprocess data using CLEAN dataset"""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)

    from data_preprocessing import FacebookAdDataPreprocessor
    from success_metrics import generate_success_metrics_report

    preprocessor = FacebookAdDataPreprocessor("data/data_clean.csv")
    preprocessor.load_data()
    preprocessor.calculate_performance_metrics()
    preprocessor.split_data(test_size=0.2, random_state=42)
    preprocessor.define_success_labels(method='absolute')

    validation = preprocessor.validate_data_quality()
    if validation['has_issues']:
        logger.warning(f"Data quality issues found: {validation['issues']}")
        response = input("Continue despite issues? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    preprocessor.save_processed_data("data/")
    df = preprocessor.preprocessed_df
    generate_success_metrics_report(df)
    logger.info("Data preprocessing completed")
    return df


def step2_train_test_split(df):
    """Step 2: Create train-test split"""
    logger.info("=" * 60)
    logger.info("STEP 2: TRAIN-TEST SPLIT")
    logger.info("=" * 60)

    from train_test_split import prepare_train_test_split, validate_split

    X_train, X_test, y_train, y_test, split_info = prepare_train_test_split(
        df, test_size=0.2, random_state=42, stratify=True
    )
    validation = validate_split(X_train, X_test, y_train, y_test)
    if validation['warnings']:
        logger.warning("Split validation warnings:")
        for w in validation['warnings']:
            logger.warning(f"  - {w}")

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.astype(int).to_csv("data/y_train.csv", index=False)
    y_test.astype(int).to_csv("data/y_test.csv", index=False)

    logger.info(f"Split complete: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def step3_run_experiment(X_train, X_test, y_train, y_test):
    """Step 3: Run GPT experiments on multiple models"""
    logger.info("=" * 60)
    logger.info("STEP 3: GPT EXPERIMENTS (Multi-Model Loop)")
    logger.info("=" * 60)

    from dotenv import load_dotenv
    load_dotenv()

    from prompt_engineering import PromptEngineer
    from model_experiment import GPTExperiment

    train_df = X_train.copy()
    train_df['is_success'] = y_train.values
    prompt_engineer = PromptEngineer(train_df)

    models_to_test = ['deepseek-v3.2', 'gpt-oss-120b', 'qwen3.5']

    total_test = len(X_test)
    print(f"\nTotal test samples available: {total_test}")
    sample_size_input = input("Enter number of test samples to use (recommended >= 100, or 'all' for all): ")
    sample_size = total_test if sample_size_input.lower() == 'all' else int(sample_size_input)

    all_results = {}
    baseline_results = None

    for model_name in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING MODEL: {model_name}")
        logger.info(f"{'='*60}")

        experiment = GPTExperiment(
            primary_model=model_name,
            secondary_model=None,
            temperature=0.0
        )

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

        safe_name = model_name.replace('.', '_').replace('-', '_')
        experiment.save_results(f"results/experiment_results_{safe_name}.json")
        _ = experiment.generate_cost_report()

        all_results[model_name] = results

        if baseline_results is None:
            baseline_results = experiment.run_baseline_comparison(
                X_train, X_test, y_train, y_test
            )

    import json
    with open("results/baseline_results.json", 'w') as f:
        json.dump(baseline_results, f, indent=4)

    print("\n" + "="*80)
    print("MULTI-MODEL EXPERIMENT SUMMARY")
    print("="*80)
    for model, res in all_results.items():
        print(f"\n{model}:")
        for shot, metrics in res.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                print(f"  {shot}-shot: Accuracy={metrics['accuracy']:.4f}, Balanced Acc={metrics['balanced_accuracy']:.4f}")

    return all_results, baseline_results, None


def step4_statistical_analysis(df):
    """Step 4: Perform statistical analysis for ALL models separately"""
    logger.info("=" * 60)
    logger.info("STEP 4: STATISTICAL ANALYSIS (Per Model)")
    logger.info("=" * 60)

    from statistical_analysis import StatisticalAnalyzer
    import json
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
        safe_model = model_name.replace('.', '_').replace('-', '_')
        logger.info(f"Analyzing {model_name}...")

        with open(model_file, 'r') as f:
            results_json = json.load(f)
        if 'results' in results_json:
            results_df = pd.DataFrame(results_json['results'])
            from visualization import _convert_to_int
            for col in ['actual', 'prediction', 'correct']:
                if col in results_df.columns:
                    results_df = _convert_to_int(results_df, col)
        else:
            results_df = None
            logger.warning(f"No 'results' key in {model_file}")

        analyzer = StatisticalAnalyzer(df, results_df)


        model_report = analyzer.generate_comprehensive_report(
            f"results/statistical_report_{safe_model}.json"
        )

        print(f"\n--- Statistical Summary for {model_name} ---")
        analyzer.print_summary()

    logger.info("Statistical reports saved for each model in results/")
    return None


def step5_generate_visualizations(df):
    """Step 5: Generate visualizations for each model separately and overall comparison"""
    logger.info("=" * 60)
    logger.info("STEP 5: VISUALIZATION (Per Model + Comparison)")
    logger.info("=" * 60)

    from visualization import ThesisVisualizer, _convert_to_int

    baseline_path = Path("results/baseline_results.json")
    baseline_results = None
    if baseline_path.exists():
        import json
        with open(baseline_path, 'r') as f:
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

        import json
        import pandas as pd
        with open(model_file, 'r') as f:
            data = json.load(f)
        if 'results' in data:
            results_df = pd.DataFrame(data['results'])
            for col in ['actual', 'prediction', 'correct']:
                if col in results_df.columns:
                    results_df = _convert_to_int(results_df, col)
        else:
            results_df = None
            logger.warning(f"No 'results' key in {model_file}")

        model_output_dir = f"visualizations/{model_name}"
        visualizer = ThesisVisualizer(df, results_df, output_dir=model_output_dir)
        visualizer.generate_all_visualizations(baseline_results)

    # Overall comparison (heatmap + CSV)
    ThesisVisualizer.generate_model_comparison("results", "visualizations")

    # NEW: Combined comparison charts (all models in one image)
    ThesisVisualizer.generate_combined_accuracy_by_shot()
    ThesisVisualizer.generate_combined_model_comparison()

    logger.info("Visualizations and comparison saved to visualizations/")


def main():
    start_time = time.time()

    print("\n" + "=" * 80)
    print("FACEBOOK AD PREDICTION THESIS - MASTER EXPERIMENT RUNNER")
    print("=" * 80)
    print("\nPipeline steps:")
    print("  1. Data preprocessing (using CLEAN dataset)")
    print("  2. Train-test split")
    print("  3. GPT experiments (requires OpenAI API key)")
    print("  4. Statistical analysis (per model)")
    print("  5. Visualization generation (per model + comparison)")

    from dotenv import load_dotenv
    load_dotenv()

    # Warn if API key not set - but not needed if only running stats/viz
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or api_key == "your_openai_api_key_here":
        print("\n  WARNING: OpenAI API key not found in .env file")
        print("   Experiments requiring API calls will fail.")
        print("   Please edit the .env file and add your API key.")

        response = input("\nContinue without API key? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"\n API Key found")
        print(f" Base URL: {base_url}")

    setup_directories()

    try:
        df = step1_data_preprocessing()
        X_train, X_test, y_train, y_test = step2_train_test_split(df)

        response = input("\nRun GPT experiments? This will incur API costs (y/n): ")
        if response.lower() == 'y':
            results, baseline_results, experiment = step3_run_experiment(
                X_train, X_test, y_train, y_test
            )
            if experiment is not None:
                cost_report = experiment.generate_cost_report()
                print("\n" + "-" * 40)
                print("COST SUMMARY:")
                print(f"  Total API Calls: {cost_report['total_api_calls']}")
                print(f"  Total Cost: ${cost_report['total_cost_usd']:.4f}")
                print("-" * 40)
            else:
                print("\nCost details are saved in each model's result file.")

        report = step4_statistical_analysis(df)
        step5_generate_visualizations(df)

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f" EXPERIMENT COMPLETE! Time elapsed: {elapsed / 60:.2f} minutes")
        print("=" * 80)
        print("\nResults saved to:")
        print("  - results/experiment_results_<model>.json")
        print("  - results/statistical_report_<model>.json (for each model)")
        print("  - results/baseline_results.json")
        print("  - visualizations/<model>/ for each model")
        print("  - visualizations/model_comparison_heatmap.png")
        print("  - visualizations/model_comparison_table.csv")
        print("  - visualizations/combined_accuracy_by_shot.png")
        print("  - visualizations/combined_model_comparison.png")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\n Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
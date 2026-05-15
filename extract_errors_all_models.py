import json
import pandas as pd

# Load test data to get metrics
X_test = pd.read_csv("data/X_test.csv")
X_test['ad_id'] = X_test['ad_id'].astype(int)

# Define models and their result files
models = {
    "DeepSeek v3.2": "results/experiment_results_deepseek_v3_2.json",
    "GPT OSS 120b": "results/experiment_results_gpt_oss_120b.json",
    "Qwen3.5": "results/experiment_results_qwen3_5.json"
}

# Choose shot levels to analyse (e.g., [0, 1, 3, 5] or [5] for best)
shot_levels = [5]  # change to [0,1,3,5] for all

for model_name, filepath in models.items():
    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print('=' * 60)

    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["results"])
    df['ad_id'] = df['ad_id'].astype(int)
    df['shot_level'] = df['shot_level'].astype(int)

    for shot in shot_levels:
        df_shot = df[df["shot_level"] == shot].copy()
        if df_shot.empty:
            print(f"\nNo data for {shot}-shot")
            continue

        df_shot["error"] = df_shot["prediction"] != df_shot["actual"]
        misclassified = df_shot[df_shot["error"] == True]

        print(f"\n{shot}-shot: Total misclassified = {len(misclassified)}")
        if len(misclassified) == 0:
            continue

        # Merge with X_test to get metrics
        merged = misclassified.merge(X_test[['ad_id', 'CTR', 'CPC', 'Conversion_Rate']], on='ad_id', how='left')

        # Select relevant columns for display
        cols = ['ad_id', 'CTR', 'CPC', 'Conversion_Rate', 'actual', 'prediction']
        print(merged[cols].head(10).to_string(index=False))
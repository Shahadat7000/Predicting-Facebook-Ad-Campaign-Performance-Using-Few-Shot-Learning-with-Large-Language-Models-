"""
Retrieval-Only Baseline: Majority vote on top-k similar training examples.
No LLM is used – only cosine similarity and label copying.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Load data
# ------------------------------
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze().astype(int)
y_test = pd.read_csv("data/y_test.csv").squeeze().astype(int)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train success rate: {y_train.mean() * 100:.2f}%")
print(f"Test success rate: {y_test.mean() * 100:.2f}%")

# ------------------------------
# Feature columns for similarity (same as in prompt_engineering.py)
# ------------------------------
similarity_feature_cols = ["CTR", "CPC", "Conversion_Rate", "impressions", "clicks", "spent"]

# Ensure all columns exist
missing_cols = [c for c in similarity_feature_cols if c not in X_train.columns]
if missing_cols:
    raise ValueError(f"Missing columns in train data: {missing_cols}")


# ------------------------------
# Prepare feature matrices (fill NaN with 0, scale)
# ------------------------------
def prepare_features(df, cols):
    """Extract, fill NaN, and return numpy array."""
    X = df[cols].copy()
    X = X.fillna(0)  # same as in prompt_engineering
    return X.values


X_train_raw = prepare_features(X_train, similarity_feature_cols)
X_test_raw = prepare_features(X_test, similarity_feature_cols)

# Scale features (fit on train, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# ------------------------------
# Majority vote prediction for a given k
# ------------------------------
def majority_vote_predict(X_train_scaled, y_train, X_test_scaled, k=5):
    """
    For each test sample, find k most similar training samples (cosine similarity),
    take majority vote of their labels. Ties broken by random (or by smaller index).
    """
    y_pred = []
    # Compute similarity matrix (test x train) – may be large, but okay for 153 test, 608 train
    sim = cosine_similarity(X_test_scaled, X_train_scaled)  # shape (n_test, n_train)

    for i in range(sim.shape[0]):
        # Get indices of top-k most similar training examples
        top_k_idx = np.argsort(sim[i])[-k:][::-1]
        top_labels = y_train.iloc[top_k_idx].values  # use .iloc because y_train is Series with original index
        # Majority vote
        vote = np.bincount(top_labels).argmax()
        y_pred.append(vote)
    return np.array(y_pred)


# ------------------------------
# Evaluate for k = 1, 3, 5
# ------------------------------
results = {}
for k in [1, 3, 5]:
    y_pred = majority_vote_predict(X_train_scaled, y_train, X_test_scaled, k=k)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    results[k] = {"accuracy": acc, "balanced_accuracy": bal_acc}
    print(f"\n{k}-shot majority vote:")
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc * 100:.2f}%")

# ------------------------------
# Also compute for comparison: random baseline (always predict majority class)
# ------------------------------
majority_class = y_train.mode()[0]
y_random = np.full_like(y_test, majority_class)
random_acc = accuracy_score(y_test, y_random)
random_bal_acc = balanced_accuracy_score(y_test, y_random)
print(f"\nBaseline: Always predict majority class ({majority_class})")
print(f"  Accuracy: {random_acc * 100:.2f}%")
print(f"  Balanced Accuracy: {random_bal_acc * 100:.2f}%")

# ------------------------------
# Save results to JSON for easy inclusion in thesis
# ------------------------------
import json

output = {
    "retrieval_baseline": results,
    "majority_class_baseline": {
        "accuracy": random_acc,
        "balanced_accuracy": random_bal_acc
    }
}
with open("results/retrieval_baseline_results.json", "w") as f:
    json.dump(output, f, indent=4)
print("\nResults saved to results/retrieval_baseline_results.json")
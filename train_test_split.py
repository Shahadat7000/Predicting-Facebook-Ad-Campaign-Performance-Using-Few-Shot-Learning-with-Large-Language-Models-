import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import safe_to_int

logger = logging.getLogger(__name__)


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Prepare train-test split with optional stratification.
    """

    logger.info(
        f"Preparing train-test split "
        f"(test_size={test_size}, random_state={random_state})"
    )

    Path("results").mkdir(exist_ok=True)

    df = df.copy()

    feature_cols = [
        "ad_id", "age", "gender",
        "interest1", "interest2", "interest3",
        "impressions", "clicks", "spent",
        "CTR", "CPC", "Conversion_Rate"
    ]

    available_feature_cols = [
        col for col in feature_cols
        if col in df.columns
    ]

    missing_feature_cols = [
        col for col in feature_cols
        if col not in df.columns
    ]

    if missing_feature_cols:
        logger.warning(f"Missing feature columns skipped: {missing_feature_cols}")

    X = df[available_feature_cols].copy()

    has_labels = "is_success" in df.columns

    if has_labels:
        y = df["is_success"].apply(safe_to_int).astype(int)
    else:
        y = pd.Series([0] * len(df), name="is_success")
        stratify = False
        logger.warning("is_success column missing. Stratification disabled.")

    X = X.fillna(0)

    stratify_target = None

    if stratify and has_labels:
        class_counts = y.value_counts()

        if len(class_counts) < 2 or class_counts.min() < 2:
            logger.warning(
                "Not enough samples per class for stratification. "
                "Using non-stratified split."
            )
            stratify_target = None
            stratify = False
        else:
            stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    logger.info("\n" + "=" * 50)
    logger.info("TRAIN-TEST SPLIT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Training set size: {len(X_train)} samples ({len(X_train) / len(df) * 100:.1f}%)")
    logger.info(f"Test set size: {len(X_test)} samples ({len(X_test) / len(df) * 100:.1f}%)")

    split_info = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "stratify": bool(stratify),
        "train_size": int(len(X_train)),
        "test_size_samples": int(len(X_test)),
        "feature_columns": available_feature_cols,
        "missing_feature_columns": missing_feature_cols
    }

    if has_labels:
        train_success_rate = y_train.mean() * 100
        test_success_rate = y_test.mean() * 100
        full_success_rate = y.mean() * 100

        logger.info("\nSuccess Rate Distribution:")
        logger.info(f"  Full dataset: {full_success_rate:.2f}%")
        logger.info(f"  Training set: {train_success_rate:.2f}%")
        logger.info(f"  Test set: {test_success_rate:.2f}%")
        logger.info(f"  Difference: {abs(train_success_rate - test_success_rate):.2f}%")

        split_info.update(
            {
                "train_success_rate": float(train_success_rate),
                "test_success_rate": float(test_success_rate),
                "full_success_rate": float(full_success_rate)
            }
        )

    with open("results/split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)

    return X_train, X_test, y_train, y_test, split_info


def validate_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict:
    """
    Validate train-test split and check for possible leakage.
    """

    y_train = y_train.apply(safe_to_int).astype(int)
    y_test = y_test.apply(safe_to_int).astype(int)

    success_rate_diff = abs(y_train.mean() - y_test.mean())

    validation_results = {
        "train_test_overlap": False,
        "index_overlap": False,
        "success_rate_diff": float(success_rate_diff),
        "warnings": []
    }

    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)

    if overlap:
        validation_results["index_overlap"] = True
        validation_results["warnings"].append(
            f"Index overlap found: {len(overlap)} samples"
        )

    common_cols = [
        col for col in X_train.columns
        if col in X_test.columns
    ]

    if common_cols:
        train_tuples = set(
            tuple(row)
            for row in X_train[common_cols].astype(str).values
        )

        test_tuples = set(
            tuple(row)
            for row in X_test[common_cols].astype(str).values
        )

        data_overlap = train_tuples.intersection(test_tuples)

        if data_overlap:
            validation_results["train_test_overlap"] = True
            validation_results["warnings"].append(
                f"Duplicate row overlap found: {len(data_overlap)} rows"
            )

    if success_rate_diff > 0.10:
        validation_results["warnings"].append(
            f"Large success rate difference: {success_rate_diff:.2%}"
        )

    logger.info("\n" + "=" * 50)
    logger.info("SPLIT VALIDATION RESULTS")
    logger.info("=" * 50)

    if validation_results["warnings"]:
        logger.warning(f"Found {len(validation_results['warnings'])} warnings:")

        for warning in validation_results["warnings"]:
            logger.warning(f"  - {warning}")
    else:
        logger.info("Split validation passed - no issues detected.")

    return validation_results


if __name__ == "__main__":
    df = pd.read_csv("data/processed_data.csv")

    X_train, X_test, y_train, y_test, split_info = prepare_train_test_split(df)

    validation = validate_split(X_train, X_test, y_train, y_test)

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    logger.info("Training and test sets saved to data/ directory.")

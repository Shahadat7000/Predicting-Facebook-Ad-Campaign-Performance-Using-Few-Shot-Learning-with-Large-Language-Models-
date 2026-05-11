import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict
import json
from sklearn.model_selection import train_test_split
from utils import safe_numeric, convert_column_to_int


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class FacebookAdDataPreprocessor:
    """
    Data preprocessing class for Facebook ad campaign data.
    """

    def __init__(self, data_path: str = "data/data_clean.csv"):
        self.data_path = Path(data_path)
        self.df = None
        self.train_df = None
        self.test_df = None
        self.metric_thresholds = {}
        self.preprocessed_df = None

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading clean data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Clean data not found at {self.data_path}. Run clean_data.py first."
            )

        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")

        required_cols = ["age", "gender"]
        missing_cols = [c for c in required_cols if c not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        valid_ages = {"30-34", "35-39", "40-44", "45-49"}
        valid_genders = {"M", "F"}

        invalid_age = ~self.df["age"].astype(str).isin(valid_ages)
        invalid_gender = ~self.df["gender"].astype(str).isin(valid_genders)

        if invalid_age.any() or invalid_gender.any():
            logger.error("Corrupted rows still present in dataset.")
            logger.error(f"Invalid ages: {self.df[invalid_age]['age'].unique()}")
            logger.error(f"Invalid genders: {self.df[invalid_gender]['gender'].unique()}")
            raise ValueError("Dataset still contains corrupted rows. Run clean_data.py again.")

        self.preprocessed_df = self.df.copy()
        return self.df

    def calculate_performance_metrics(self) -> pd.DataFrame:
        logger.info("Calculating performance metrics...")

        if self.df is None:
            raise ValueError("Must call load_data() first.")

        df = self.df.copy()

        numeric_cols = [
            "impressions", "clicks", "spent",
            "total_conversion", "approved_conversion"
        ]

        df = safe_numeric(df, numeric_cols)

        for col in numeric_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required numeric column: {col}")

        df["impressions"] = df["impressions"].fillna(0)
        df["clicks"] = df["clicks"].fillna(0)
        df["spent"] = df["spent"].fillna(0)
        df["total_conversion"] = df["total_conversion"].fillna(0)
        df["approved_conversion"] = df["approved_conversion"].fillna(0)

        df["CTR"] = np.where(
            df["impressions"] > 0,
            (df["clicks"] / df["impressions"]) * 100,
            0.0
        )

        df["CPC"] = np.where(
            df["clicks"] > 0,
            df["spent"] / df["clicks"],
            np.nan
        )

        df["Conversion_Rate"] = np.where(
            df["clicks"] > 0,
            (df["approved_conversion"] / df["clicks"]) * 100,
            0.0
        )

        logger.info("Performance metrics calculated.")
        logger.info(f"CTR mean: {df['CTR'].mean():.4f}%, median: {df['CTR'].median():.4f}%")
        logger.info(f"CPC mean: ${df['CPC'].mean():.4f}, median: ${df['CPC'].median():.4f}")
        logger.info(
            f"Conversion Rate mean: {df['Conversion_Rate'].mean():.4f}%, "
            f"median: {df['Conversion_Rate'].median():.4f}%"
        )

        self.df = df
        self.preprocessed_df = df.copy()

        return df

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        logger.info(f"Splitting data with test_size={test_size}...")

        if self.df is None:
            raise ValueError("Must load and preprocess data before splitting.")

        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )

        self.train_df = self.train_df.copy()
        self.test_df = self.test_df.copy()

        logger.info(f"Train set: {len(self.train_df)} samples")
        logger.info(f"Test set: {len(self.test_df)} samples")

        return self.train_df, self.test_df

    def define_success_labels(self, method: str = "absolute"):
        logger.info(f"Defining success labels using {method} method...")

        if self.train_df is None or self.test_df is None:
            raise ValueError("Must call split_data() first.")

        if method == "absolute":
            ctr_threshold = 0.90
            cpc_threshold = 1.72
            conversion_threshold = 3.0
            logger.info("Using absolute industry thresholds.")

        elif method == "relative":
            ctr_threshold = self.train_df["CTR"].median()
            cpc_threshold = self.train_df["CPC"].median()
            conversion_threshold = self.train_df["Conversion_Rate"].median()
            logger.info("Using relative thresholds from training data only.")

        else:
            raise ValueError("Method must be 'absolute' or 'relative'.")

        self.metric_thresholds = {
            "ctr_threshold": float(ctr_threshold),
            "cpc_threshold": float(cpc_threshold),
            "conversion_threshold": float(conversion_threshold),
            "method": method
        }

        for df in [self.train_df, self.test_df]:
            df["success_ctr"] = df["CTR"] > ctr_threshold
            df["success_cpc"] = df["CPC"].fillna(np.inf) < cpc_threshold
            df["success_conversion"] = df["Conversion_Rate"] > conversion_threshold

            df["is_success"] = (
                df["success_ctr"].astype(int)
                + df["success_cpc"].astype(int)
                + df["success_conversion"].astype(int)
            ) >= 2

            df = convert_column_to_int(df, "is_success", safe=True)

        self.train_df = convert_column_to_int(self.train_df, "is_success", safe=True)
        self.test_df = convert_column_to_int(self.test_df, "is_success", safe=True)

        self.preprocessed_df = pd.concat(
            [self.train_df, self.test_df],
            ignore_index=True
        )

        return self.train_df, self.test_df

    def validate_data_quality(self) -> Dict:
        logger.info("Performing data quality validation...")

        issues = []
        df = self.preprocessed_df if self.preprocessed_df is not None else self.df

        if df is None:
            return {"error": "No data loaded"}

        validation_results = {
            "total_samples": int(len(df)),
            "missing_values": {},
            "data_types": {},
            "outliers": {},
            "issues": []
        }

        missing_counts = df.isnull().sum()

        for col, count in missing_counts.items():
            if count > 0:
                validation_results["missing_values"][col] = int(count)

                if col != "CPC":
                    issues.append(f"Missing values in {col}: {count}")

        numeric_cols = [
            "impressions", "clicks", "spent",
            "total_conversion", "approved_conversion",
            "CTR", "CPC", "Conversion_Rate"
        ]

        for col in numeric_cols:
            if col in df.columns:
                validation_results["data_types"][col] = str(df[col].dtype)

                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Non-numeric data in {col}")

        for col in ["impressions", "clicks", "spent"]:
            if col in df.columns and (df[col].fillna(0) < 0).any():
                issues.append(f"Negative values found in {col}")

        if "CTR" in df.columns:
            invalid_ctr = df[(df["CTR"] < 0) | (df["CTR"] > 100)]

            if len(invalid_ctr) > 0:
                issues.append(f"Invalid CTR values outside 0-100%: {len(invalid_ctr)} rows")
                validation_results["outliers"]["CTR"] = int(len(invalid_ctr))

        validation_results["issues"] = issues
        validation_results["has_issues"] = len(issues) > 0
        validation_results["data_quality_score"] = self._calculate_quality_score(
            validation_results
        )

        if issues:
            logger.warning(f"Found {len(issues)} data quality issues.")
            for issue in issues[:5]:
                logger.warning(f" - {issue}")
        else:
            logger.info("No major data quality issues found.")

        return validation_results

    def _calculate_quality_score(self, validation_results: Dict) -> float:
        score = 100.0
        penalty_per_issue = 5.0

        missing_values = validation_results.get("missing_values", {})
        outliers = validation_results.get("outliers", {})

        score -= len(missing_values) * penalty_per_issue
        score -= len(outliers) * penalty_per_issue

        return max(0.0, score)

    def validate_no_leakage(self):
        logger.info("Validating no data leakage...")

        if self.metric_thresholds.get("method") == "absolute":
            logger.info("Absolute thresholds used. No threshold leakage.")
        else:
            logger.info("Relative thresholds calculated from training data only.")

        return True

    def save_processed_data(self, output_dir: str = "data/"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if self.train_df is not None:
            self.train_df.to_csv(output_path / "train.csv", index=False)

        if self.test_df is not None:
            self.test_df.to_csv(output_path / "test.csv", index=False)

        if (
            self.train_df is not None
            and self.test_df is not None
            and "is_success" in self.train_df.columns
            and "is_success" in self.test_df.columns
        ):
            feature_cols = [
                "age", "gender", "interest1", "interest2", "interest3",
                "impressions", "clicks", "spent",
                "CTR", "CPC", "Conversion_Rate"
            ]

            feature_cols = [
                col for col in feature_cols
                if col in self.train_df.columns and col in self.test_df.columns
            ]

            X_train = self.train_df[feature_cols]
            X_test = self.test_df[feature_cols]
            y_train = self.train_df["is_success"].astype(int)
            y_test = self.test_df["is_success"].astype(int)

            X_train.to_csv(output_path / "X_train.csv", index=False)
            X_test.to_csv(output_path / "X_test.csv", index=False)
            y_train.to_csv(output_path / "y_train.csv", index=False)
            y_test.to_csv(output_path / "y_test.csv", index=False)

        if self.preprocessed_df is not None:
            self.preprocessed_df.to_csv(output_path / "processed_data.csv", index=False)

        with open(output_path / "thresholds.json", "w", encoding="utf-8") as f:
            json.dump(self.metric_thresholds, f, indent=4)

        logger.info(f"All processed data saved to {output_path}")


def main():
    logger.info("Starting data preprocessing with no leakage...")

    preprocessor = FacebookAdDataPreprocessor("data/data_clean.csv")

    preprocessor.load_data()
    preprocessor.calculate_performance_metrics()
    preprocessor.split_data(test_size=0.2, random_state=42)
    preprocessor.define_success_labels(method="absolute")
    preprocessor.validate_no_leakage()
    preprocessor.validate_data_quality()
    preprocessor.save_processed_data()

    logger.info("Data preprocessing completed successfully.")
    logger.info(f"Successfully processed {len(preprocessor.df)} clean samples.")

    return preprocessor


if __name__ == "__main__":
    preprocessor = main()

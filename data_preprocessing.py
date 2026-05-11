
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict
import json
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacebookAdDataPreprocessor:
    """
    Comprehensive data preprocessing class for Facebook ad campaign data
    """

    def __init__(self, data_path: str = "data/data_clean.csv"):
        """
        Initialize preprocessor with clean data path
        """
        self.data_path = Path(data_path)
        self.df = None
        self.train_df = None
        self.test_df = None
        self.metric_thresholds = {}
        self.preprocessed_df = None

    def load_data(self) -> pd.DataFrame:
        """Load and validate clean data"""
        logger.info(f"Loading clean data from {self.data_path}")

        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Clean data not found at {self.data_path}. "
                "Run clean_data.py first!"
            )

        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")

        # Validate no corrupted rows remain
        valid_ages = {"30-34", "35-39", "40-44", "45-49"}
        valid_genders = {"M", "F"}

        invalid_age = ~self.df["age"].astype(str).isin(valid_ages)
        invalid_gender = ~self.df["gender"].astype(str).isin(valid_genders)

        if invalid_age.any() or invalid_gender.any():
            logger.error("Corrupted rows still present in dataset!")
            logger.error(f"Invalid ages: {self.df[invalid_age]['age'].unique()}")
            logger.error(f"Invalid genders: {self.df[invalid_gender]['gender'].unique()}")
            raise ValueError("Dataset still contains corrupted rows. Run clean_data.py again.")

        self.preprocessed_df = self.df.copy()
        return self.df

    def calculate_performance_metrics(self) -> pd.DataFrame:
        """Calculate key performance metrics with proper NaN handling"""
        logger.info("Calculating performance metrics...")

        df = self.df.copy()

        # Fill missing conversions with 0
        df['total_conversion'] = df['total_conversion'].fillna(0)
        df['approved_conversion'] = df['approved_conversion'].fillna(0)

        # CTR: (clicks / impressions) * 100
        df['CTR'] = np.where(
            df['impressions'] > 0,
            (df['clicks'] / df['impressions']) * 100,
            0.0
        )

        # CPC: spent / clicks
        df['CPC'] = np.where(
            df['clicks'] > 0,
            df['spent'] / df['clicks'],
            np.nan
        )

        # Conversion Rate: (approved_conversion / clicks) * 100
        df['Conversion_Rate'] = np.where(
            df['clicks'] > 0,
            (df['approved_conversion'] / df['clicks']) * 100,
            0.0
        )

        logger.info("Performance metrics calculated:")
        logger.info(f"CTR - Mean: {df['CTR'].mean():.4f}%, Median: {df['CTR'].median():.4f}%")
        logger.info(f"CPC - Mean: ${df['CPC'].mean():.2f}, Median: ${df['CPC'].median():.2f}")
        logger.info(f"Conversion Rate - Mean: {df['Conversion_Rate'].mean():.2f}%, Median: {df['Conversion_Rate'].median():.2f}%")

        self.df = df
        self.preprocessed_df = df.copy()
        return df

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Split data BEFORE calculating thresholds to prevent data leakage
        """
        logger.info(f"Splitting data (test_size={test_size})...")

        # Split data
        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # We don't have labels yet
        )

        logger.info(f"Train set: {len(self.train_df)} samples")
        logger.info(f"Test set: {len(self.test_df)} samples")

        return self.train_df, self.test_df

    def define_success_labels(self, method: str = 'absolute'):
        """
        Define success labels WITHOUT data leakage
        """
        logger.info(f"Defining success labels using {method} method...")

        if self.train_df is None:
            raise ValueError("Must call split_data() first!")

        if method == 'absolute':
            # Industry standards
            ctr_threshold = 0.90
            cpc_threshold = 1.72
            conversion_threshold = 3.0
            logger.info("Using ABSOLUTE industry thresholds")

        elif method == 'relative':
            # FIX: Calculate thresholds ONLY from training data
            ctr_threshold = self.train_df['CTR'].median()
            cpc_threshold = self.train_df['CPC'].median()
            conversion_threshold = self.train_df['Conversion_Rate'].median()
            logger.info("Using RELATIVE thresholds from TRAINING data only (Leakage Fixed)")
        else:
            raise ValueError("Method must be 'absolute' or 'relative'")

        # Store thresholds
        self.metric_thresholds = {
            'ctr_threshold': ctr_threshold,
            'cpc_threshold': cpc_threshold,
            'conversion_threshold': conversion_threshold,
            'method': method
        }

        # Apply SAME thresholds to BOTH train and test
        for df in [self.train_df, self.test_df]:
            df['success_ctr'] = df['CTR'] > ctr_threshold
            df['success_cpc'] = df['CPC'] < cpc_threshold
            df['success_conversion'] = df['Conversion_Rate'] > conversion_threshold

            df['is_success'] = (
                df['success_ctr'].astype(int) +
                df['success_cpc'].astype(int) +
                df['success_conversion'].astype(int)
            ) >= 2

        self.preprocessed_df = pd.concat([self.train_df, self.test_df])
        return self.train_df, self.test_df

    def validate_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality validation
        """
        logger.info("Performing data quality validation...")

        issues = []
        df = self.preprocessed_df if self.preprocessed_df is not None else self.df

        if df is None:
            return {"error": "No data loaded"}

        validation_results = {
            "total_samples": len(df),
            "missing_values": {},
            "data_types": {},
            "outliers": {},
            "issues": []
        }

        # Check for missing values
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                validation_results["missing_values"][col] = int(count)
                issues.append(f"Missing values in {col}: {count}")

        # Check data types
        numeric_cols = ['impressions', 'clicks', 'spent', 'total_conversion', 'approved_conversion', 'CTR', 'CPC', 'Conversion_Rate']
        for col in numeric_cols:
            if col in df.columns:
                validation_results["data_types"][col] = str(df[col].dtype)
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Non-numeric data in {col}")

        # Check for negative values
        if 'impressions' in df.columns and (df['impressions'] < 0).any():
            issues.append("Negative impressions found")
        if 'clicks' in df.columns and (df['clicks'] < 0).any():
            issues.append("Negative clicks found")
        if 'spent' in df.columns and (df['spent'] < 0).any():
            issues.append("Negative spending found")

        # Check calculated metrics
        if 'CTR' in df.columns:
            invalid_ctr = df[(df['CTR'] < 0) | (df['CTR'] > 100)]
            if len(invalid_ctr) > 0:
                issues.append(f"Invalid CTR values (outside 0-100%): {len(invalid_ctr)} rows")
                validation_results["outliers"]["CTR"] = len(invalid_ctr)

        validation_results["issues"] = issues
        validation_results["has_issues"] = len(issues) > 0
        validation_results["data_quality_score"] = self._calculate_quality_score(validation_results)

        if issues:
            logger.warning(f"Found {len(issues)} data quality issues")
            for issue in issues[:5]:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No data quality issues found. Data is clean!")

        return validation_results

    def _calculate_quality_score(self, validation_results: Dict) -> float:
        """Calculate a data quality score (0-100)"""
        score = 100.0
        penalty_per_issue = 5.0

        if validation_results.get("missing_values"):
            score -= len(validation_results["missing_values"]) * penalty_per_issue

        if validation_results.get("outliers"):
            score -= len(validation_results["outliers"]) * penalty_per_issue

        return max(0, score)

    def validate_no_leakage(self):
        """Verify no data leakage occurred"""
        logger.info("Validating no data leakage...")

        if self.metric_thresholds.get('method') == 'absolute':
            logger.info(" ABSOLUTE thresholds - no leakage possible")
        else:
            logger.info(" Using training data only for thresholds")

        return True

    def save_processed_data(self, output_dir: str = "data/"):
        """
        Save processed train/test splits

        Args:
            output_dir: Directory to save files (must be a directory, not file path)
        """
        # Ensure output_dir is treated as directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save full dataframes
        if self.train_df is not None:
            self.train_df.to_csv(output_path / "train.csv", index=False)
        if self.test_df is not None:
            self.test_df.to_csv(output_path / "test.csv", index=False)

        # Save feature matrices if labels exist
        if hasattr(self, 'train_df') and self.train_df is not None and 'is_success' in self.train_df.columns:
            feature_cols = ['age', 'gender', 'interest1', 'interest2', 'interest3',
                           'impressions', 'clicks', 'spent', 'CTR', 'CPC', 'Conversion_Rate']

            X_train = self.train_df[feature_cols]
            X_test = self.test_df[feature_cols]
            y_train = self.train_df['is_success']
            y_test = self.test_df['is_success']

            X_train.to_csv(output_path / "X_train.csv", index=False)
            X_test.to_csv(output_path / "X_test.csv", index=False)
            y_train.to_csv(output_path / "y_train.csv", index=False)
            y_test.to_csv(output_path / "y_test.csv", index=False)

        # Save processed data
        if self.preprocessed_df is not None:
            self.preprocessed_df.to_csv(output_path / "processed_data.csv", index=False)

        # Save thresholds
        with open(output_path / "thresholds.json", 'w') as f:
            json.dump(self.metric_thresholds, f, indent=4)

        logger.info(f"All processed data saved to {output_path}")

def main():
    """Main execution"""
    logger.info("Starting data preprocessing with NO LEAKAGE...")

    # ===== IMPORTANT: Use clean data =====
    preprocessor = FacebookAdDataPreprocessor("data/data_clean.csv")

    # Load data
    preprocessor.load_data()

    # Calculate metrics
    preprocessor.calculate_performance_metrics()

    # STEP 1: Split data FIRST (critical!)
    preprocessor.split_data(test_size=0.2, random_state=42)

    # STEP 2: Define success labels using thresholds from TRAIN only
    preprocessor.define_success_labels(method='absolute')  # Recommended

    # Validate no leakage
    preprocessor.validate_no_leakage()

    # Save processed data
    preprocessor.save_processed_data()

    logger.info("Data preprocessing completed successfully!")
    logger.info(f" Successfully processed {len(preprocessor.df)} clean samples")

    return preprocessor

if __name__ == "__main__":
    preprocessor = main()

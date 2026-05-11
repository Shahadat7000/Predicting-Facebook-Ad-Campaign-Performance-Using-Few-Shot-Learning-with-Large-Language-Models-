

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict
import json

logger = logging.getLogger(__name__)

def prepare_train_test_split(df: pd.DataFrame, 
                            test_size: float = 0.2, 
                            random_state: int = 42,
                            stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Prepare train-test split with stratification
    
    Args:
        df: Full dataframe with features and target
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by success rate
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test, split_info
    """
    logger.info(f"Preparing train-test split (test_size={test_size}, random_state={random_state})")
    
    # Define features and target
    feature_cols = ['ad_id', 'age', 'gender', 'interest1', 'interest2', 'interest3', 
                    'impressions', 'clicks', 'spent', 'CTR', 'CPC', 'Conversion_Rate']
    
    # We'll split the data safely. If 'is_success' is missing, stratify must be False.
    X = df[feature_cols].copy()
    
    has_labels = 'is_success' in df.columns
    if has_labels:
        y = df['is_success'].copy()
    else:
        y = pd.Series([0] * len(df), name='is_success')
        stratify = False
    
    # Handle NaN values in features (fill with 0 for split, actual handling in model)
    X = X.fillna(0)
    
    # Perform split
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        logger.info("Split with stratification by success rate")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        logger.info("Split without stratification")
    
    # Log split info
    logger.info(f"\n{'='*50}")
    logger.info("TRAIN-TEST SPLIT SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Training set size: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"Test set size: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    # Check success rate distribution (only if labels exist)
    if has_labels:
        train_success_rate = y_train.mean() * 100
        test_success_rate = y_test.mean() * 100
        full_success_rate = y.mean() * 100

        logger.info(f"\nSuccess Rate Distribution:")
        logger.info(f"  Full dataset: {full_success_rate:.2f}%")
        logger.info(f"  Training set: {train_success_rate:.2f}%")
        logger.info(f"  Test set: {test_success_rate:.2f}%")
        logger.info(f"  Difference: {abs(train_success_rate - test_success_rate):.2f}%")
    
    # Store split info
    split_info = {
        'test_size': test_size,
        'random_state': random_state,
        'stratify': stratify,
        'train_size': len(X_train),
        'test_size_samples': len(X_test)
    }
    
    if has_labels:
        split_info.update({
            'train_success_rate': float(train_success_rate),
            'test_success_rate': float(test_success_rate),
            'full_success_rate': float(full_success_rate)
        })
    
    # Save split info
    with open('results/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=4)
    
    return X_train, X_test, y_train, y_test, split_info

def validate_split(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Validate that split is appropriate and no data leakage
    
    Args:
        X_train, X_test, y_train, y_test: Split data
        
    Returns:
        Dict: Validation results
    """
    validation_results = {
        'train_test_overlap': False,
        'index_overlap': False,
        'success_rate_diff': abs(y_train.mean() - y_test.mean()),
        'warnings': []
    }
    
    # Check for index overlap (should be none)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)
    
    if overlap:
        validation_results['index_overlap'] = True
        validation_results['warnings'].append(f"Index overlap found: {len(overlap)} samples")
    
    # Check for data overlap (duplicate rows)
    # Convert to tuples for comparison
    train_tuples = set(tuple(x) for x in X_train.values)
    test_tuples = set(tuple(x) for x in X_test.values)
    data_overlap = train_tuples.intersection(test_tuples)
    
    if data_overlap:
        validation_results['train_test_overlap'] = True
        validation_results['warnings'].append(f"Data overlap found: {len(data_overlap)} rows")
    
    # Check success rate difference
    if validation_results['success_rate_diff'] > 0.1:  # More than 10% difference
        validation_results['warnings'].append(
            f"Large success rate difference: {validation_results['success_rate_diff']:.2%}"
        )
    
    # Log results
    logger.info(f"\n{'='*50}")
    logger.info("SPLIT VALIDATION RESULTS")
    logger.info(f"{'='*50}")
    
    if validation_results['warnings']:
        logger.warning(f"Found {len(validation_results['warnings'])} warnings:")
        for warning in validation_results['warnings']:
            logger.warning(f"  - {warning}")
    else:
        logger.info("✓ Split validation passed - no issues detected")
    
    return validation_results

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("data/processed_data.csv")
    
    # Prepare split
    X_train, X_test, y_train, y_test, split_info = prepare_train_test_split(df)
    
    # Validate split
    validation = validate_split(X_train, X_test, y_train, y_test)
    
    # Save splits
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    
    logger.info("Training and test sets saved to data/ directory")

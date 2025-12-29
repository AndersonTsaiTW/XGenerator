"""
Prediction service
"""
import pandas as pd
import joblib
from typing import List, Dict, Any, Tuple
from pathlib import Path


def validate_prediction_input(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any]
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate prediction input against model schema.
    
    Enhancement: Allow missing features (will be imputed by pipeline)
    - Missing features generate warnings but don't prevent prediction
    - Unknown features generate warnings and will be ignored
    
    Args:
        rows: List of prediction input dictionaries
        schema: Model schema containing required features
        
    Returns:
        Tuple of (is_valid, list_of_errors, list_of_warnings)
    """
    errors = []
    warnings = []
    
    # Extract required features from schema
    numeric_features = schema.get('numeric_features', [])
    categorical_features = schema.get('categorical_features', [])
    required_features = set(numeric_features + categorical_features)
    
    for idx, row in enumerate(rows):
        row_features = set(row.keys())
        
        # Check for missing required features - WARNING instead of ERROR
        missing = required_features - row_features
        if missing:
            warnings.append(
                f"Row {idx}: Missing features {sorted(list(missing))} - "
                f"will be imputed using training data statistics"
            )
        
        # Check for unknown features - WARNING
        unknown = row_features - required_features
        if unknown:
            warnings.append(
                f"Row {idx}: Unknown features {sorted(list(unknown))} - "
                f"will be ignored during prediction"
            )
    
    # Always return True - allow prediction to proceed
    return (True, errors, warnings)


def make_predictions(
    pipeline: Any,
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    task_type: str
) -> List[float]:
    """
    Make predictions using the trained pipeline.
    
    Handles missing features by creating a complete DataFrame with NaN values,
    which will be imputed by the pipeline's SimpleImputer.
    
    Args:
        pipeline: Trained sklearn pipeline
        rows: List of input dictionaries
        schema: Model schema with required features
        task_type: 'classification' or 'regression'
        
    Returns:
        List of predictions
    """
    # Get all required features from schema
    numeric_features = schema.get('numeric_features', [])
    categorical_features = schema.get('categorical_features', [])
    all_features = numeric_features + categorical_features
    
    # Convert rows to DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure all required features exist in DataFrame (add as NaN if missing)
    import numpy as np
    for feature in all_features:
        if feature not in df.columns:
            df[feature] = np.nan  # Will be imputed by pipeline
    
    # Reorder columns to match schema (and drop unknown columns)
    df = df[all_features]
    
    # Make predictions
    if task_type == 'classification':
        # For binary classification, return probability of positive class
        predictions = pipeline.predict_proba(df)[:, 1].tolist()
    else:  # regression
        predictions = pipeline.predict(df).tolist()
    
    return predictions

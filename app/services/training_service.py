"""
Training pipeline builder service
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    task_type: str,
    xgb_params: Dict[str, Any],
    df: pd.DataFrame = None
) -> Pipeline:
    """
    Build a sklearn Pipeline with ColumnTransformer and XGBoost model.
    
    Step 2 enhancement:
    - Uses ColumnTransformer for separate numeric/categorical handling
    - OneHotEncoder with handle_unknown='ignore' for robustness
    - Proper imputation strategies for each feature type
    - Date columns are converted to timestamps if detected
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        task_type: 'classification' or 'regression'
        xgb_params: XGBoost parameters
        df: Optional DataFrame for date detection
        
    Returns:
        Configured sklearn Pipeline
    """
    from app.utils.transformers import DateToTimestampTransformer
    
    # Build preprocessor using ColumnTransformer
    transformers = []
    
    if numeric_features:
        # Detect date columns in numeric features
        date_features = []
        pure_numeric = []
        
        if df is not None:
            for feat in numeric_features:
                if feat in df.columns and df[feat].dtype == 'object':
                    # Try to parse as date
                    try:
                        pd.to_datetime(df[feat].dropna().head(100))
                        date_features.append(feat)
                    except:
                        pure_numeric.append(feat)
                else:
                    pure_numeric.append(feat)
        else:
            pure_numeric = numeric_features
        
        # Build transformers for date and pure numeric separately
        if date_features:
            date_transformer = Pipeline(steps=[
                ('date_to_timestamp', DateToTimestampTransformer())
            ])
            transformers.append(('date', date_transformer, date_features))
        
        if pure_numeric:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ])
            transformers.append(('num', numeric_transformer, pure_numeric))
    
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not in numeric or categorical
    )
    
    # Select appropriate model
    if task_type == 'classification':
        model = XGBClassifier(
            random_state=42,
            **xgb_params
        )
    else:  # regression
        model = XGBRegressor(
            random_state=42,
            **xgb_params
        )
    
    # Build full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline


def validate_features_and_target(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    numeric_features: List[str],
    categorical_features: List[str]
) -> None:
    """
    Validate that features and target are valid for training.
    
    Step 2 enhancement: comprehensive validation
    
    Raises:
        ValueError: If validation fails
    """
    all_columns = set(df.columns)
    
    # Target must exist
    if target not in all_columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    
    # Features cannot include target
    if target in features:
        raise ValueError(f"Target column '{target}' cannot be in features list")
    
    # All features must exist
    missing_features = set(features) - all_columns
    if missing_features:
        raise ValueError(f"Features not found in dataset: {list(missing_features)}")
    
    # All schema features must exist
    all_schema_features = set(numeric_features + categorical_features)
    missing_schema = all_schema_features - all_columns
    if missing_schema:
        raise ValueError(f"Schema features not found in dataset: {list(missing_schema)}")
    
    # No duplicates in schema
    overlap = set(numeric_features) & set(categorical_features)
    if overlap:
        raise ValueError(f"Features cannot be both numeric and categorical: {list(overlap)}")
    
    # Features must be covered by schema
    features_not_in_schema = set(features) - all_schema_features
    if features_not_in_schema:
        raise ValueError(f"Features not defined in schema: {list(features_not_in_schema)}")


def train_model_with_validation(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    task_type: str,
    xgb_params: Dict[str, Any],
    validation_split: float = 0.2,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train model with automatic validation split and metrics calculation.
    
    Step 2 enhancement:
    - Automatic train/validation split
    - Metrics calculation
    - Stratified split for classification
    
    Args:
        df: Full dataset
        target: Target column name
        features: List of feature column names (subset of numeric + categorical)
        numeric_features: List of numeric features from schema
        categorical_features: List of categorical features from schema
        task_type: 'classification' or 'regression'
        xgb_params: XGBoost parameters
        validation_split: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_pipeline, metrics_dict)
    """
    # Filter schema features to only those in the features list
    active_numeric = [f for f in numeric_features if f in features]
    active_categorical = [f for f in categorical_features if f in features]
    
    # Prepare X and y
    X = df[features].copy()
    y = df[target].copy()
    
    # Split data
    if task_type == 'classification':
        # Stratified split for classification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=random_state,
            stratify=y
        )
    else:
        # Regular split for regression
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=random_state
        )
    
    # Build and train pipeline (on train split only)
    pipeline = build_pipeline(active_numeric, active_categorical, task_type, xgb_params, df=X_train)
    pipeline.fit(X_train, y_train)
    
    # Calculate validation metrics
    y_pred = pipeline.predict(X_val)
    
    metrics = {}
    if task_type == 'classification':
        metrics['accuracy'] = float(accuracy_score(y_val, y_pred))
        
        # ROC AUC only for binary classification
        unique_classes = y.nunique()
        if unique_classes == 2:
            y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_val, y_pred_proba))
    else:  # regression
        metrics['mae'] = float(mean_absolute_error(y_val, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    
    # Add evaluation metadata
    evaluation_info = {
        'metrics': metrics,
        'evaluation_method': 'holdout',
        'validation_split': validation_split,
        'random_state': random_state,
        'train_samples': len(X_train),
        'validation_samples': len(X_val)
    }
    
    return pipeline, evaluation_info

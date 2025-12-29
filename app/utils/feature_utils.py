"""
Feature filtering utilities
"""
from typing import List, Tuple
import re


def detect_id_columns(columns: List[str]) -> List[str]:
    """
    Detect columns that are likely ID fields.
    
    Args:
        columns: List of column names
        
    Returns:
        List of column names that appear to be IDs
    """
    id_columns = []
    
    # Pattern to detect ID columns (case insensitive)
    # Matches: *_id, id_*, *id (but not inside words)
    id_pattern = re.compile(r'(^|_)id($|_)', re.IGNORECASE)
    
    for col in columns:
        if id_pattern.search(col):
            id_columns.append(col)
    
    return id_columns


def filter_features(
    all_columns: List[str],
    target: str,
    features: List[str] = None,
    exclude_features: List[str] = None,
    auto_exclude_ids: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    Filter features based on inclusion/exclusion rules.
    
    Args:
        all_columns: All columns in the dataset
        target: Target column name
        features: Explicit list of features to include (None = all except target)
        exclude_features: Features to explicitly exclude
        auto_exclude_ids: Whether to automatically exclude ID columns
        
    Returns:
        Tuple of (final_features, excluded_features, warnings)
    """
    warnings = []
    excluded = set()
    
    # Start with specified features or all columns
    if features is not None:
        selected = set(features)
    else:
        selected = set(all_columns) - {target}
    
    # Remove target if somehow included
    if target in selected:
        selected.discard(target)
        excluded.add(target)
        warnings.append(f"Target column '{target}' cannot be used as a feature")
    
    # Auto-exclude ID columns
    if auto_exclude_ids:
        id_cols = detect_id_columns(list(selected))
        if id_cols:
            for col in id_cols:
                selected.discard(col)
                excluded.add(col)
            warnings.append(
                f"Auto-excluded ID columns: {id_cols}. "
                "ID columns typically don't generalize well and can cause overfitting."
            )
    
    # Apply explicit exclusions
    if exclude_features:
        for col in exclude_features:
            if col in selected:
                selected.discard(col)
                excluded.add(col)
            else:
                warnings.append(f"Column '{col}' in exclude_features was not in selected features")
    
    final_features = sorted(list(selected))
    excluded_features = sorted(list(excluded))
    
    return final_features, excluded_features, warnings

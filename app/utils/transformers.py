"""
Custom transformers for feature preprocessing
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DateToTimestampTransformer(BaseEstimator, TransformerMixin):
    """
    Convert date strings to Unix timestamps (seconds since epoch).
    
    Handles:
    - Date strings in various formats
    - Missing values (filled with median timestamp)
    - Invalid dates (filled with median timestamp)
    """
    
    def __init__(self):
        self.median_timestamp_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer by computing median timestamp.
        
        Args:
            X: DataFrame or array-like of date strings
            y: Ignored
            
        Returns:
            self
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Convert to datetime and then to timestamp
        timestamps = []
        for col in X.columns:
            try:
                dt = pd.to_datetime(X[col], errors='coerce')
                ts = dt.astype('int64') / 10**9  # Convert to seconds
                timestamps.extend(ts.dropna().values)
            except:
                pass
        
        # Store median for imputation
        if timestamps:
            self.median_timestamp_ = np.median(timestamps)
        else:
            # Fallback: current time
            self.median_timestamp_ = pd.Timestamp.now().timestamp()
        
        return self
    
    def transform(self, X):
        """
        Transform date strings to timestamps.
        
        Args:
            X: DataFrame or array-like of date strings
            
        Returns:
            Array of timestamps
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        result = []
        for col in X.columns:
            try:
                dt = pd.to_datetime(X[col], errors='coerce')
                ts = dt.astype('int64') / 10**9  # Convert to seconds
                # Fill NaN with median
                ts = ts.fillna(self.median_timestamp_)
                result.append(ts.values)
            except:
                # If conversion fails, use median for all values
                result.append(np.full(len(X), self.median_timestamp_))
        
        # Return as 2D array
        return np.column_stack(result) if len(result) > 0 else np.array([]).reshape(len(X), 0)

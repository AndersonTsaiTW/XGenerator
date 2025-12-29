"""
Schema inference service using OpenAI API
"""
import pandas as pd
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
import json


def infer_schema_with_openai(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Use OpenAI API to intelligently infer which columns are numeric vs categorical.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    if not OPENAI_API_KEY:
        # Fallback to basic pandas dtype inference
        return _basic_schema_inference(df)
    
    try:
        # Prepare column information for OpenAI
        column_info = []
        for col in df.columns:
            info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": df[col].dropna().head(10).tolist()
            }
            column_info.append(info)
        
        # Create OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare prompt
        prompt = f"""You are a data science expert. Classify each column as "numeric" or "categorical" based on its SEMANTIC MEANING.

Key Principle: What does this feature represent?

**Numeric (Quantitative)**: Measurable quantities where math operations make sense
- Examples: age, price, weight, voltage, wattage, length, count, ratings, temperature
- Use for: continuous measurements, counts, physical properties

**Categorical (Qualitative)**: Labels, categories, or groups where order doesn't matter
- Examples: gender, color, material, status, type, ID
- Use for: discrete groups, labels, binary yes/no, unique identifiers

**Special cases**:
- Dates/timestamps → NUMERIC (will be converted to Unix timestamps)
- IDs (customer_id, product_id) → CATEGORICAL (will be auto-excluded)
- Target/label → Keep as-is based on its nature

Consider: column name meaning + unique_count + sample values

Column data:
{json.dumps(column_info, indent=2)}

Return only valid JSON (no markdown):
{{
  "numeric_features": ["col1", "col2"],
  "categorical_features": ["col3", "col4"]
}}"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a data science expert specializing in feature type classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        numeric_features = result.get("numeric_features", [])
        categorical_features = result.get("categorical_features", [])
        
        # Validate that all columns are classified
        all_classified = set(numeric_features + categorical_features)
        all_columns = set(df.columns)
        
        if all_classified != all_columns:
            # Missing some columns, fallback to basic inference
            return _basic_schema_inference(df)
        
        return numeric_features, categorical_features
        
    except Exception as e:
        print(f"OpenAI schema inference failed: {e}. Falling back to basic inference.")
        return _basic_schema_inference(df)


def _basic_schema_inference(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Fallback schema inference using pandas dtypes only.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    numeric_features = []
    categorical_features = []
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if dtype in ['int64', 'float64']:
            # Simple heuristic: if unique values < 20, treat as categorical
            if df[col].nunique() < 20:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        elif dtype == 'bool':
            categorical_features.append(col)
        elif dtype == 'object':
            # Try to detect if it's a date column
            try:
                pd.to_datetime(df[col].dropna().head(100))
                # Successfully parsed as datetime - treat as numeric
                numeric_features.append(col)
            except:
                # Not a date - treat as categorical string
                categorical_features.append(col)
        else:
            # Default to categorical for unknown types
            categorical_features.append(col)
    
    return numeric_features, categorical_features


def calculate_feature_statistics(df: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
    """
    Calculate statistics for a single feature.
    
    Args:
        df: The pandas DataFrame
        feature_name: The column name
        
    Returns:
        Dictionary with missing_rate and unique_count
    """
    total_rows = len(df)
    missing_count = df[feature_name].isnull().sum()
    missing_rate = float(missing_count / total_rows) if total_rows > 0 else 0.0
    unique_count = int(df[feature_name].nunique())
    
    return {
        "missing_rate": round(missing_rate, 4),
        "unique_count": unique_count
    }

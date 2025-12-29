import json
import sys
sys.path.insert(0, 'c:/Users/ander/Documents/GitHub/XGenerator')

# Load schema
with open('c:/Users/ander/Documents/GitHub/XGenerator/data/metadata/datasets/cd8b8c70cccc40368a047f531eda147b_schema.json') as f:
    schema = json.load(f)
    
print("Numeric features:", schema['numeric_features'])
print("\nCategorical features:", schema['categorical_features'])

# Simulate filter_features
from app.utils.feature_utils import filter_features
import pandas as pd

df = pd.read_csv('c:/Users/ander/Documents/GitHub/XGenerator/data/datasets/cd8b8c70cccc40368a047f531eda147b.csv')

features, excluded, warnings = filter_features(
    all_columns=list(df.columns),
    target='label',
    features=None,
    exclude_features=None,
    auto_exclude_ids=True
)

print("\n\nFiltered features:", features)
print("Excluded features:", excluded)
print("Warnings:", warnings)

# Check if features are in schema
all_schema = set(schema['numeric_features'] + schema['categorical_features'])
not_in_schema = set(features) - all_schema

print("\n\nFeatures not in schema:", not_in_schema)

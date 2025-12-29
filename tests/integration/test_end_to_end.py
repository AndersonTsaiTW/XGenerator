"""
End-to-End API Test Script
Tests the complete workflow: Dataset Upload → Training → Prediction → Model Management
"""

import requests
import json
import time
from pathlib import Path

# API Base URL
BASE_URL = "http://localhost:8000"

# Test data file
CSV_FILE = "train_full.csv"

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_step(step_num, description):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Step {step_num}: {description}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


def print_success(message):
    print(f"{GREEN}[OK] {message}{RESET}")


def print_error(message):
    print(f"{RED}[ERROR] {message}{RESET}")


def print_info(message):
    print(f"{YELLOW}[INFO] {message}{RESET}")


def test_health_check():
    """Test 0: Health check"""
    print_step(0, "Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"API is healthy!")
        print(f"Status: {data.get('status')}")
        print(f"Directories OK: {data.get('directories')}")
        print(f"OpenAI configured: {data.get('openai_configured')}")
        return True
    else:
        print_error(f"Health check failed: {response.status_code}")
        return False


def test_dataset_upload():
    """Test 1: Upload dataset"""
    print_step(1, "Upload Dataset with OpenAI Schema Inference")
    
    if not Path(CSV_FILE).exists():
        print_error(f"Test file {CSV_FILE} not found!")
        return None
    
    with open(CSV_FILE, 'rb') as f:
        files = {'file': (CSV_FILE, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/datasets", files=files)
    
    if response.status_code == 200:
        data = response.json()
        dataset_id = data['dataset_id']
        print_success(f"Dataset uploaded successfully!")
        print(f"Dataset ID: {dataset_id}")
        print(f"Filename: {data['filename']}")
        print(f"Rows: {data['row_count']}, Columns: {data['column_count']}")
        print(f"\nSchema inferred:")
        print(f"  Numeric features ({len(data['schema']['numeric_features'])}): {data['schema']['numeric_features'][:5]}...")
        print(f"  Categorical features ({len(data['schema']['categorical_features'])}): {data['schema']['categorical_features'][:5]}...")
        
        if data['schema'].get('features'):
            print(f"\nSample feature details:")
            for feat in data['schema']['features'][:3]:
                print(f"  - {feat['name']} ({feat['type']}): missing_rate={feat.get('missing_rate', 'N/A')}, unique_count={feat.get('unique_count', 'N/A')}")
        
        return dataset_id
    else:
        print_error(f"Dataset upload failed: {response.status_code}")
        print(response.text)
        return None


def test_train_classification(dataset_id):
    """Test 2: Train a classification model"""
    print_step(2, "Train Classification Model")
    
    payload = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "label",
        "features": None,  # Use all features except target
        "xgb_params": {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 5
        }
    }
    
    print_info("Training with parameters:")
    print(json.dumps(payload, indent=2))
    print_info("Training started (this may take a while)...")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/train", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        model_id = data['model_id']
        print_success(f"Model trained successfully in {elapsed:.2f}s!")
        print(f"Model ID: {model_id}")
        print(f"Task: {data['task_type']}")
        print(f"Target: {data['target']}")
        print(f"Features used: {data['feature_count']}")
        print(f"Training samples: {data['row_count']}")
        print(f"Training duration: {data['training_duration']:.2f}s")
        
        if data.get('metrics'):
            print(f"\nValidation Metrics:")
            for metric, value in data['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        return model_id
    else:
        print_error(f"Training failed: {response.status_code}")
        print(response.text)
        return None


def test_get_model_schema(model_id):
    """Test 3: Get model schema"""
    print_step(3, "Get Model Input Schema")
    
    response = requests.get(f"{BASE_URL}/models/{model_id}/schema")
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"Model schema retrieved!")
        print(f"Numeric features: {len(data['numeric_features'])}")
        print(f"Categorical features: {len(data['categorical_features'])}")
        return data
    else:
        print_error(f"Failed to get schema: {response.status_code}")
        return None


def test_prediction(model_id, schema):
    """Test 4: Make predictions"""
    print_step(4, "Make Batch Predictions")
    
    # Create sample prediction data
    sample_rows = [
        {
            "age": 45,
            "lifetime_value": 1200.50,
            "total_orders": 15,
            "is_subscriber": 1,
            "gender": 0,
            "income_bracket": 3,
            "customer_segment": 2,
            "preferred_channel": 1,
            "wattage": 60,
            "lumens": 800,
            "voltage": 120,
            "length_mm": 150,
            "width_mm": 50,
            "height_mm": 100,
            "weight_kg": 0.5,
            "price_usd": 29.99,
            "warranty_months": 24,
            "rating_avg": 4.5,
            "num_reviews": 120,
            "inventory_qty": 500,
            "lead_time_days": 7,
            "category": 1,
            "subcategory": 3,
            "color": 2,
            "finish": 1,
            "material": 0,
            "purchase_date": "2024-07-18"
        },
        {
            "age": 32,
            "lifetime_value": 550.00,
            "total_orders": 5,
            "is_subscriber": 0,
            "gender": 1,
            "income_bracket": 2,
            "customer_segment": 1,
            "preferred_channel": 0,
            "wattage": 40,
            "lumens": 600,
            "voltage": 120,
            "length_mm": 100,
            "width_mm": 40,
            "height_mm": 80,
            "weight_kg": 0.3,
            "price_usd": 19.99,
            "warranty_months": 12,
            "rating_avg": 4.0,
            "num_reviews": 50,
            "inventory_qty": 300,
            "lead_time_days": 5,
            "category": 1,
            "subcategory": 2,
            "color": 0,
            "finish": 0,
            "material": 1,
            "purchase_date": "2024-10-23"
        }
    ]
    
    payload = {
        "model_id": model_id,
        "rows": sample_rows
    }
    
    print_info(f"Making predictions for {len(sample_rows)} rows...")
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"Predictions completed!")
        print(f"\nResults:")
        for pred in data['predictions']:
            print(f"  Row {pred['row_index']}: {pred['prediction']:.4f}")
        
        if data.get('warnings'):
            print(f"\nWarnings:")
            for warning in data['warnings']:
                print(f"  ⚠ {warning}")
        
        return True
    else:
        print_error(f"Prediction failed: {response.status_code}")
        print(response.text)
        return False


def test_list_models():
    """Test 5: List all models"""
    print_step(5, "List All Models")
    
    response = requests.get(f"{BASE_URL}/models")
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"Found {data['total']} models!")
        
        if data['models']:
            print(f"\nModels:")
            for model in data['models'][:5]:  # Show first 5
                print(f"  - {model['model_id']}: {model['task_type']} on {model['target']}")
        
        return True
    else:
        print_error(f"Failed to list models: {response.status_code}")
        return False


def test_get_model_details(model_id):
    """Test 6: Get model details"""
    print_step(6, "Get Model Metadata")
    
    response = requests.get(f"{BASE_URL}/models/{model_id}")
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"Model details retrieved!")
        print(f"\nMetadata:")
        print(f"  Model ID: {data['model_id']}")
        print(f"  Task Type: {data['task_type']}")
        print(f"  Target: {data['target']}")
        print(f"  Features: {data['feature_count']} features")
        print(f"  Created: {data['created_at']}")
        print(f"  Training Duration: {data.get('training_duration', 'N/A')}s")
        
        if data.get('metrics'):
            print(f"\n  Metrics:")
            for metric, value in data['metrics'].items():
                print(f"    {metric}: {value}")
        
        if data.get('evaluation_method'):
            print(f"\n  Evaluation: {data['evaluation_method']}")
        
        return True
    else:
        print_error(f"Failed to get model details: {response.status_code}")
        return False


def main():
    """Run all tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}XGBoost Training Service - End-to-End Test{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    try:
        # Test 0: Health check
        if not test_health_check():
            print_error("Server is not healthy. Please start the server first:")
            print_info("Run: .\.venv\Scripts\Activate.ps1; python -m app.main")
            return
        
        # Test 1: Upload dataset
        dataset_id = test_dataset_upload()
        if not dataset_id:
            return
        
        # Test 2: Train model
        model_id = test_train_classification(dataset_id)
        if not model_id:
            return
        
        # Test 3: Get model schema
        schema = test_get_model_schema(model_id)
        if not schema:
            return
        
        # Test 4: Make predictions
        if not test_prediction(model_id, schema):
            return
        
        # Test 5: List models
        if not test_list_models():
            return
        
        # Test 6: Get model details
        if not test_get_model_details(model_id):
            return
        
        # Success!
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}✓ All tests passed successfully!{RESET}")
        print(f"{GREEN}{'='*60}{RESET}\n")
        
        print(f"Summary:")
        print(f"  Dataset ID: {dataset_id}")
        print(f"  Model ID: {model_id}")
        print(f"\nYou can now:")
        print(f"  - View Swagger UI: http://localhost:8000/docs")
        print(f"  - View ReDoc: http://localhost:8000/redoc")
        print(f"  - Make more predictions using the model ID above")
        
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API server!")
        print_info("Please start the server first:")
        print_info("Run: .\.venv\Scripts\Activate.ps1; python -m app.main")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Unit tests for prediction endpoint
"""
import pytest
from io import BytesIO


@pytest.mark.unit
def test_predict_with_complete_features(authenticated_client, sample_csv_data):
    """Test prediction with all required features"""
    # Upload and train first
    files = {"file": ("train.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Prediction Test Dataset"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    # Train model
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Prediction Test Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "label"
    }
    train_response = authenticated_client.post("/train", json=train_request)
    model_id = train_response.json()["model_id"]
    
    # Make prediction
    predict_request = {
        "model_id": model_id,
        "rows": [
            {
                "age": 30,
                "gender": 1,
                "product_id": 105,
                "customer_id": 11,
                "purchase_date": "2024-01-11"
            }
        ]
    }
    response = authenticated_client.post("/predict", json=predict_request)
    
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 1


@pytest.mark.unit
def test_predict_with_missing_features(authenticated_client, sample_csv_data):
    """Test prediction with missing features (should return warnings)"""
    # Upload and train
    files = {"file": ("train.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Missing Features Test"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Missing Features Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "label"
    }
    train_response = authenticated_client.post("/train", json=train_request)
    model_id = train_response.json()["model_id"]
    
    # Predict with missing features
    predict_request = {
        "model_id": model_id,
        "rows": [
            {
                "age": 30,
                "gender": 1
                # Missing other features - should still work
            }
        ]
    }
    response = authenticated_client.post("/predict", json=predict_request)
    
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result


@pytest.mark.unit
def test_predict_with_invalid_model(authenticated_client):
    """Test prediction with non-existent model"""
    predict_request = {
        "model_id": "nonexistent",
        "rows": [{"age": 30}]
    }
    response = authenticated_client.post("/predict", json=predict_request)
    assert response.status_code == 404

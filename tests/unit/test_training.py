"""
Unit tests for training endpoint
"""
import pytest
from io import BytesIO


@pytest.mark.unit
def test_train_model_success(authenticated_client, sample_csv_data):
    """Test successful model training"""
    # Upload dataset first
    files = {"file": ("train.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Training Test Dataset"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    # Train model
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Test Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "label"
    }
    response = authenticated_client.post("/train", json=train_request)
    
    assert response.status_code == 200
    result = response.json()
    assert "model_id" in result
    assert result["task_type"] == "classification"


@pytest.mark.unit
def test_train_with_invalid_dataset(authenticated_client):
    """Test training with non-existent dataset"""
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Invalid Test",
        "dataset_id": "nonexistent",
        "task_type": "classification",
        "target": "label"
    }
    response = authenticated_client.post("/train", json=train_request)
    assert response.status_code == 404


@pytest.mark.unit
def test_train_with_invalid_target(authenticated_client, sample_csv_data):
    """Test training with invalid target column"""
    files = {"file": ("train.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Invalid Target Test"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Invalid Target Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "nonexistent_column"
    }
    response = authenticated_client.post("/train", json=train_request)
    assert response.status_code == 400


@pytest.mark.unit
def test_train_with_valid_xgb_params(authenticated_client, sample_csv_data):
    """Test training with custom XGBoost parameters"""
    files = {"file": ("train.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "XGB Params Test"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Custom Params Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "label",
        "xgb_params": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.05
        }
    }
    response = authenticated_client.post("/train", json=train_request)
    assert response.status_code == 200

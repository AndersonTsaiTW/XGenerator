"""
Unit tests for model management endpoints
"""
import pytest
from io import BytesIO


@pytest.mark.unit
def test_list_models_empty(client):
    """Test listing models when none exist"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


@pytest.mark.unit
def test_get_model_details(authenticated_client, sample_csv_data):
    """Test getting model details"""
    # Create a model first
    files = {"file": ("train.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Model Details Test"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Details Test Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "label"
    }
    train_response = authenticated_client.post("/train", json=train_request)
    model_id = train_response.json()["model_id"]
    
    # Get model details (no auth required for GET)
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    response = client.get(f"/models/{model_id}")
    
    assert response.status_code == 200
    model = response.json()
    assert model["model_id"] == model_id
    assert model["task_type"] == "classification"


@pytest.mark.unit
def test_get_model_schema(authenticated_client, sample_csv_data):
    """Test getting model input schema"""
    # Create a model
    files = {"file": ("train.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Schema Test Dataset"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    train_request = {
        "user_id": authenticated_client.test_user["user_id"],
        "model_name": "Schema Test Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "label"
    }
    train_response = authenticated_client.post("/train", json=train_request)
    model_id = train_response.json()["model_id"]
    
    # Get schema (no auth required)
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    response = client.get(f"/models/{model_id}/schema")
    
    assert response.status_code == 200
    schema = response.json()
    assert "numeric_features" in schema
    assert "categorical_features" in schema


@pytest.mark.unit
def test_get_nonexistent_model(client):
    """Test getting a model that doesn't exist"""
    response = client.get("/models/nonexistent")
    assert response.status_code == 404

"""
Unit tests for dataset endpoints
"""
import pytest
from io import BytesIO


@pytest.mark.unit
def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "directories" in data
    assert "openai_configured" in data


@pytest.mark.unit
def test_upload_csv_success(authenticated_client, sample_csv_data):
    """Test successful CSV upload with authentication"""
    files = {"file": ("test.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Test Dataset"
    }
    response = authenticated_client.post("/datasets", files=files, data=data)
    
    assert response.status_code == 200
    result = response.json()
    assert "dataset_id" in result
    assert result["filename"] == "test.csv"
    assert result["row_count"] == 10  # Updated to match new fixture
    assert "schema" in result
    assert result["user_id"] == authenticated_client.test_user["user_id"]
    assert result["dataset_name"] == "Test Dataset"


@pytest.mark.unit
def test_upload_non_csv_fails(authenticated_client):
    """Test that non-CSV files are rejected"""
    files = {"file": ("test.txt", BytesIO(b"not a csv"), "text/plain")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Test"
    }
    response = authenticated_client.post("/datasets", files=files, data=data)
    
    assert response.status_code == 400
    assert "detail" in response.json()
    assert response.json()["detail"]["error"] == "invalid_file_type"


@pytest.mark.unit
def test_upload_empty_csv_fails(authenticated_client):
    """Test that empty CSV files are rejected"""
    files = {"file": ("empty.csv", BytesIO(b""), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Empty Test"
    }
    response = authenticated_client.post("/datasets", files=files, data=data)
    
    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.unit
def test_update_schema(authenticated_client, sample_csv_data):
    """Test schema update endpoint"""
    # First upload a dataset
    files = {"file": ("test.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Schema Test"
    }
    upload_response = authenticated_client.post("/datasets", files=files, data=data)
    dataset_id = upload_response.json()["dataset_id"]
    
    # Update schema
    schema_update = {
        "numeric_features": ["age"],
        "categorical_features": ["gender", "product_id", "customer_id"]
    }
    response = authenticated_client.patch(f"/datasets/{dataset_id}/schema", json=schema_update)
    
    assert response.status_code == 200
    result = response.json()
    # Response structure contains numeric/categorical at top level after update
    assert "age" in result["numeric_features"] or "age" in result.get("schema", {}).get("numeric_features", [])


@pytest.mark.unit
def test_update_schema_invalid_dataset(client):
    """Test schema update with non-existent dataset"""
    response = client.patch(
        "/datasets/nonexistent_id/schema",
        json={"numeric_features": ["age"]}
    )
    assert response.status_code == 404

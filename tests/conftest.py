"""
Test configuration and fixtures
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path
import shutil
import os
import uuid


@pytest.fixture
def client():
    """Basic test client without authentication"""
    return TestClient(app)


@pytest.fixture
def test_user(client):
    """Create a test user and return user data with API key"""
    username = f"test_user_{uuid.uuid4().hex[:8]}"
    
    response = client.post(
        "/users",
        json={"username": username, "email": "test@example.com"}
    )
    
    assert response.status_code == 201
    user_data = response.json()
    assert "api_key" in user_data
    assert "user_id" in user_data
    
    return user_data


@pytest.fixture
def authenticated_client(test_user):
    """Test client with API key authentication header"""
    client = TestClient(app)
    client.headers.update({"X-API-Key": test_user["api_key"]})
    # Store user info in client for easy access in tests
    client.test_user = test_user
    return client


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return """customer_id,product_id,age,gender,purchase_date,label
1,101,25,0,2024-01-01,1
2,102,35,1,2024-01-02,0
3,103,45,0,2024-01-03,1
4,104,28,1,2024-01-04,0
5,105,52,0,2024-01-05,1
6,106,42,1,2024-01-06,0
7,107,33,0,2024-01-07,1
8,108,29,1,2024-01-08,0
9,109,38,0,2024-01-09,1
10,110,31,1,2024-01-10,0"""


@pytest.fixture
def sample_prediction_row():
    """Sample prediction input"""
    return {
        "age": 30,
        "gender": 1,
        "category": 1,
        "price_usd": 29.99
    }

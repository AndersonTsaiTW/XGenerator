"""
Unit tests for user management endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_create_user():
    """Test user creation with API key generation"""
    import uuid
    response = client.post("/users", json={
        "username": f"test_{uuid.uuid4().hex[:12]}",
        "email": "test@example.com"
    })
    
    assert response.status_code == 201
    data = response.json()
    
    assert "user_id" in data
    assert "api_key" in data
    assert data["api_key"].startswith("sk_live_")
    assert "warning" in data


def test_create_duplicate_username():
    """Test that duplicate usernames are rejected"""
    import uuid
    username = f"dup_{uuid.uuid4().hex[:12]}"
    
    # Create first user
    response1 = client.post("/users", json={"username": username})
    assert response1.status_code == 201
    
    # Try to create duplicate
    response2 = client.post("/users", json={"username": username})
    assert response2.status_code == 400
    assert "username_exists" in str(response2.json())


def test_list_users():
    """Test listing users with masked API keys"""
    response = client.get("/users")
    
    assert response.status_code == 200
    users = response.json()
    
    assert isinstance(users, list)
    if len(users) > 0:
        user = users[0]
        assert "api_key_preview" in user
        assert "..." in user["api_key_preview"]
        assert "api_key" not in user  # Full key should not be present


def test_get_user_by_id():
    """Test getting user details by ID"""
    import uuid
    # Create a user first
    unique_username = f"get_{uuid.uuid4().hex[:12]}"
    create_response = client.post("/users", json={"username": unique_username})
    assert create_response.status_code == 201
    user_id = create_response.json()["user_id"]
    
    # Get user details
    response = client.get(f"/users/{user_id}")
    
    assert response.status_code == 200
    user = response.json()
    
    assert user["user_id"] == user_id
    assert "api_key_preview" in user
    assert "api_key" not in user  # Full key should not be in GET response


def test_get_nonexistent_user():
    """Test getting a user that doesn't exist"""
    response = client.get("/users/nonexistent_user_123")
    
    assert response.status_code == 404

# Testing Guide for XGenerator

## Running Tests

### Prerequisites
```bash
# Activate virtual environment  
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install test dependencies (already in requirements.txt)
pip install pytest pytest-cov pytest-asyncio
```

### Run All Tests
```bash
# Run all unit tests with TESTING mode enabled
$env:TESTING="true"; pytest tests/unit -m unit -v

# With coverage report
$env:TESTING="true"; pytest tests/unit -m unit --cov=app --cov-report=html

# Stop on first failure (useful for debugging)
$env:TESTING="true"; pytest tests/unit -m unit -x
```

### Run Specific Test Files
```bash
# Dataset tests only
$env:TESTING="true"; pytest tests/unit/test_datasets.py -v

# Training tests only
$env:TESTING="true"; pytest tests/unit/test_training.py -v  

# Models tests only
$env:TESTING="true"; pytest tests/unit/test_models.py -v

# Prediction tests only
$env:TESTING="true"; pytest tests/unit/test_prediction.py -v
```

## Test Environment

### TESTING Environment Variable
**IMPORTANT**: Always set `TESTING=true` when running tests to disable rate limiting.

```bash
# Windows PowerShell
$env:TESTING="true"

# Linux/Mac
export TESTING=true
```

This prevents 429 (Too Many Requests) errors during test execution.

## Test Structure

### Fixtures (conftest.py)
- `client`: Basic TestClient without authentication
- `test_user`: Creates a unique test user with API key
- `authenticated_client`: TestClient with X-API-Key header pre-configured
- `sample_csv_data`: Sample CSV data for dataset tests

### Test Files
1. **test_datasets.py** - Dataset upload and schema management
2. **test_training.py** - Model training endpoints
3. **test_prediction.py** - Prediction endpoints  
4. **test_models.py** - Model management endpoints
5. **test_users.py** - User management endpoints

## Authentication in Tests

All endpoints requiring authentication now use the `authenticated_client` fixture:

```python
def test_upload_dataset(authenticated_client, sample_csv_data):
    # Client already has X-API-Key header
    # Access user info via authenticated_client.test_user
    files = {"file": ("test.csv", BytesIO(sample_csv_data.encode()), "text/csv")}
    data = {
        "user_id": authenticated_client.test_user["user_id"],
        "dataset_name": "Test Dataset"
    }
    response = authenticated_client.post("/datasets", files=files, data=data)
    assert response.status_code == 200
```

## Common Issues

### 1. Rate Limit Errors (429)
**Problem**: Tests failing with "Too Many Requests"
**Solution**: Ensure `TESTING=true` environment variable is set

### 2. Authentication Errors (422/401)
**Problem**: Missing or invalid X-API-Key
**Solution**: Use `authenticated_client` fixture instead of `client`

### 3. KeyError: 'dataset_id'  
**Problem**: Upload endpoint hit rate limit or authentication failed
**Solution**: 
- Check `TESTING=true` is set
- Ensure using `authenticated_client`
- Add debug print: `print(upload_response.json())`

## Test Coverage

Current coverage: ~50%

To improve coverage:
1. Add tests for error scenarios
2. Test edge cases (empty inputs, invalid formats)
3. Test authentication/authorization edge cases
4. Test  rate limiting behavior

## Continuous Integration

For CI/CD pipelines:
```yaml
# GitHub Actions example
- name: Run tests
  env:
    TESTING: "true"
  run: |
    pytest tests/unit -m unit --cov=app --cov-report=xml
```

## Debugging Failed Tests

### View full traceback
```bash
pytest tests/unit/test_file.py::test_function -vv
```

### Use pytest debugger
```bash
pytest tests/unit/test_file.py::test_function --pdb
```

### Print request/response
```python
def test_something(authenticated_client):
    response = authenticated_client.post("/endpoint", json=data)
    print(f"Status:\n{response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
```

---

**Last Updated**: 2025-12-28  
**Test Framework**: pytest 9.0.2  
**Python Version**: 3.13.1

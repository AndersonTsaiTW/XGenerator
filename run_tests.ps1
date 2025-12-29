# Quick Test Script
# Save as: run_tests.ps1

# Set testing environment variable
$env:TESTING = "true"

# Activate venv
.\.venv\Scripts\Activate.ps1

# Run tests
pytest tests/unit -m unit -v --tb=short

# Or run specific test file:
# pytest tests/unit/test_datasets.py -v

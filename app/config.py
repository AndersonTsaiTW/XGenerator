"""
Configuration settings for the XGBoost Training Service
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"  # NEW
DATASETS_DIR = DATA_DIR / "datasets"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
METADATA_DIR = DATA_DIR / "metadata"
DATASET_SCHEMAS_DIR = METADATA_DIR / "datasets"
MODEL_METADATA_DIR = METADATA_DIR / "models"
MODEL_SCHEMAS_DIR = METADATA_DIR / "schemas"
JOBS_DIR = DATA_DIR / "jobs"  # NEW: Job metadata storage

# Ensure directories exist
for dir_path in [USERS_DIR, DATASETS_DIR, ARTIFACTS_DIR, DATASET_SCHEMAS_DIR, MODEL_METADATA_DIR, MODEL_SCHEMAS_DIR, JOBS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Redis configuration for Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# XGBoost parameter whitelist and ranges
ALLOWED_XGB_PARAMS = {
    "n_estimators": (1, 5000, int),
    "learning_rate": (0.0001, 1.0, float),
    "max_depth": (1, 16, int),
    "subsample": (0.0, 1.0, float),
    "colsample_bytree": (0.0, 1.0, float),
}

# Model evaluation settings
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

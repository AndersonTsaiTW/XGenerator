"""Main FastAPI application for XGBoost Training Service"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from pathlib import Path
import os

from app.routers import datasets, training, prediction, models, users, jobs
from app.config import DATASETS_DIR, ARTIFACTS_DIR, METADATA_DIR, DATA_DIR
from app.utils.rate_limit import limiter, _rate_limit_exceeded_handler, TESTING
from slowapi.errors import RateLimitExceeded

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="XGBoost Training Service API",
    description="""
## Advanced XGBoost Model Training & Prediction Service (v2.0)

### Features
- 📤 CSV dataset upload with intelligent schema inference
- 🚀 **NEW**: Asynchronous background training with Celery + Redis
- 🤖 Automatic XGBoost model training with validation
- 🔮 Batch predictions with missing value handling
- 📊 Model performance metrics & management
- 👥 Multi-user support with API key authentication
- 🎯 Tier-based features (Premium: OpenAI, Free: Pandas)
- ⚡ Rate limiting for abuse prevention

### Quick Start
1. Create a user account to get your API key
2. Upload your CSV dataset (auto schema inference)
3. Submit a training job (returns job_id)
4. Poll job status until complete
5. Make predictions with trained model!

### Rate Limits
- Dataset upload: 5/min (API Key)
- Training: 3/min (API Key)
- Predictions: 120/min (API Key)
- Public queries: 120/min (IP)

### Contact
- **Repository**: https://github.com/yourusername/xgenerator
- **Issues**: https://github.com/yourusername/xgenerator/issues

### License
MIT License
    """,
    version="2.0.0",  # Breaking change: async training with Celery + Redis
    contact={
        "name": "XGenerator Team",
        "url": "https://github.com/yourusername/xgenerator",
    },
    license_info={
        "name": "MIT"
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure rate limiting (disabled in test environment)
if not TESTING:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    # In test mode, create a mock limiter that does nothing
    from unittest.mock import MagicMock
    app.state.limiter = MagicMock()

# Include routers
app.include_router(users.router)  # NEW - Users first
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(models.router)
app.include_router(jobs.router)  # NEW - Job management


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request, exc):
    """Handle Pydantic validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Data validation failed",
            "details": exc.errors()
        }
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "XGBoost Training Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    from app.config import (
        DATASETS_DIR, ARTIFACTS_DIR, METADATA_DIR,
        OPENAI_API_KEY
    )
    
    return {
        "status": "healthy",
        "directories": {
            "datasets": str(DATASETS_DIR.exists()),
            "artifacts": str(ARTIFACTS_DIR.exists()),
            "metadata": str(METADATA_DIR.exists())
        },
        "openai_configured": bool(OPENAI_API_KEY)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

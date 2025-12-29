"""
Model management endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Request
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel

from app.config import MODEL_METADATA_DIR, MODEL_SCHEMAS_DIR, ARTIFACTS_DIR
from app.models.schemas import (
    ModelListResponse, ModelSummary, ModelMetadata,
    DatasetSchema, ErrorResponse
)
from app.utils.auth import verify_api_key, verify_ownership
from app.utils.file_utils import atomic_write_json
from app.utils.rate_limit import limiter

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
@limiter.limit("120/minute")
async def list_models(request: Request, user_id: str = None):
    """
    List all trained models, optionally filtered by user.
    
    **Rate Limit**: 120 per minute (IP-based)
    
    **Query Parameters:**
    - **user_id** (optional): Filter models by owner user ID
    
    Returns a summary of each model including:
    - model_id, model_name
    - task_type, target
    - dataset_id
    - created_at
    """
    # Scan metadata directory
    metadata_files = list(MODEL_METADATA_DIR.glob("model_*.json"))
    
    models = []
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            # Filter by user_id if provided
            if user_id and metadata.get('user_id') != user_id:
                continue
            
            models.append(ModelSummary(
                model_id=metadata['model_id'],
                user_id=metadata.get('user_id', ''),
                username=metadata.get('username', ''),
                model_name=metadata.get('model_name', ''),
                task_type=metadata['task_type'],
                target=metadata['target'],
                dataset_id=metadata['dataset_id'],
                created_at=metadata['created_at']
            ))
        except Exception:
            # Skip corrupted metadata files
            continue
    
    # Sort by created_at (newest first)
    models.sort(key=lambda m: m.created_at, reverse=True)
    
    return ModelListResponse(
        models=models,
        total=len(models)
    )


@router.get("/{model_id}", response_model=ModelMetadata)
@limiter.limit("120/minute")
async def get_model(request: Request, model_id: str):
    """
    Get detailed information about a specific model.
    
    **Rate Limit**: 120 per minute (IP-based)
    
    Returns all stored information including:
    - Training configuration
    - Features used
    - XGBoost parameters
    - Training metrics
    - Evaluation details
    """
    metadata_path = MODEL_METADATA_DIR / f"model_{model_id}.json"
    
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model {model_id} not found",
                "details": None
            }
        )
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return ModelMetadata(**metadata)


@router.get("/{model_id}/schema", response_model=DatasetSchema)
@limiter.limit("120/minute")
async def get_model_schema(request: Request, model_id: str):
    """
    Get the input schema (features) required by a model.
    
    **Rate Limit**: 120 per minute (IP-based)
    
    Returns:
    - List of numeric features
    - List of categorical features
    - Feature statistics (missing_rate, unique_count)
    
    Use this endpoint to understand what features are required for prediction.
    """
    schema_path = MODEL_SCHEMAS_DIR / f"model_{model_id}_schema.json"
    
    if not schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "schema_not_found",
                "message": f"Schema for model {model_id} not found",
                "details": {
                    "hint": "This model may be corrupted."
                }
            }
        )
    
    with open(schema_path, "r") as f:
        schema_data = json.load(f)
    
    # Return as DatasetSchema format (reusing the response model)
    return DatasetSchema(
        dataset_id=model_id,  # Use model_id as identifier
        user_id=schema_data.get('user_id', ''),  # NEW
        dataset_name=schema_data.get('dataset_name', ''),  # NEW
        numeric_features=schema_data.get('numeric_features', []),
        categorical_features=schema_data.get('categorical_features', []),
        features=None  # Could add detailed feature info if needed
    )


# NEW endpoints

class ModelUpdateRequest(BaseModel):
    """Request to update model name"""
    model_name: str


@router.patch("/{model_id}", response_model=ModelMetadata)
@limiter.limit("30/minute")
async def update_model(
    request: Request,
    model_id: str,
    update_request: ModelUpdateRequest,
    current_user: dict = Depends(verify_api_key)
):
    """
    Update model name.
    
    **Requires Authentication:** X-API-Key header
    
    Only the model owner can update the model.
    """
    metadata_path = MODEL_METADATA_DIR / f"model_{model_id}.json"
    
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model {model_id} not found",
                "details": None
            }
        )
    
    # Load current metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Verify ownership
    await verify_ownership(metadata.get('user_id', ''), current_user)
    
    # Update model name
    metadata['model_name'] = update_request.model_name
    
    # Save updated metadata
    atomic_write_json(metadata_path, metadata)
    
    return ModelMetadata(**metadata)


@router.delete("/{model_id}")
@limiter.limit("30/minute")
async def delete_model(
    request: Request,
    model_id: str,
    current_user: dict = Depends(verify_api_key)
):
    """
    Delete a model.
    
    **Requires Authentication:** X-API-Key header
    
    Only the model owner can delete the model.
    Deletes both the model file and metadata.
    """
    metadata_path = MODEL_METADATA_DIR / f"model_{model_id}.json"
    model_path = ARTIFACTS_DIR / f"{model_id}.joblib"
    schema_path = MODEL_SCHEMAS_DIR / f"model_{model_id}_schema.json"
    
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model {model_id} not found",
                "details": None
            }
        )
    
    # Load metadata to check ownership
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Verify ownership
    await verify_ownership(metadata.get('user_id', ''), current_user)
    
    # Delete files
    if metadata_path.exists():
        metadata_path.unlink()
    if model_path.exists():
        model_path.unlink()
    if schema_path.exists():
        schema_path.unlink()
    
    return {"message": f"Model {model_id} deleted successfully"}

"""
Training endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Request
import pandas as pd
import joblib
import json
import time
from datetime import datetime
from pathlib import Path

from app.config import (
    DATASETS_DIR, DATASET_SCHEMAS_DIR, ARTIFACTS_DIR,
    MODEL_METADATA_DIR, MODEL_SCHEMAS_DIR,
    VALIDATION_SPLIT, RANDOM_STATE
)
from app.models.schemas import (
    TrainRequest, TrainResponse, JobCreateResponse,  # NEW
    RetrainRequest,
    ErrorResponse
)
from app.utils.file_utils import generate_id, atomic_write_json
from app.services.training_service import (
    validate_features_and_target,
    train_model_with_validation
)
from app.services.job_service import create_job  # NEW
from app.tasks.training_tasks import train_model_task  # NEW
from app.utils.auth import verify_api_key
from app.utils.rate_limit import limiter

router = APIRouter(prefix="/train", tags=["training"])


@router.post("", response_model=JobCreateResponse)
@limiter.limit("3/minute")
async def train_model(
    request: Request,
    train_request: TrainRequest,
    current_user: dict = Depends(verify_api_key)
):
    """
    Submit a model training job (asynchronous).
    
    **Breaking Change (v2.0)**: This endpoint now returns a job_id instead of model_id.
    Use GET /jobs/{job_id} to check training status and retrieve the model_id when complete.
    
    **Rate Limit**: 3 per minute (API Key-based)
    
    - Creates a training job
    - Queues the job to Celery worker
    - Returns immediately with job_id
    
    Use GET /jobs/{job_id} to poll for completion.
    """
    # Validate dataset exists
    dataset_path = DATASETS_DIR / f"{train_request.dataset_id}.csv"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "dataset_not_found",
                "message": f"Dataset {train_request.dataset_id} not found",
                "details": None
            }
        )
    
    # Load dataset schema
    schema_path = DATASET_SCHEMAS_DIR / f"{train_request.dataset_id}_schema.json"
    if not schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "schema_not_found",
                "message": f"Schema for dataset {train_request.dataset_id} not found",
                "details": None
            }
        )
    
    # Prepare training configuration
    train_config = {
        "dataset_id": train_request.dataset_id,
        "model_name": train_request.model_name,
        "task_type": train_request.task_type,
        "target": train_request.target,
        "features": train_request.features,
        "exclude_features": train_request.exclude_features,
        "xgb_params": train_request.xgb_params or {},
        "username": current_user.get("username", ""),
        "dataset_name": ""  # Will be populated from schema if needed
    }
    
    # Create job record first with a placeholder task_id
    job_data = create_job(
        user_id=train_request.user_id,
        train_config=train_config,
        celery_task_id="pending"  # Will be updated after task is queued
    )
    
    # Now queue the actual training task with the real job_id
    task = train_model_task.apply_async(args=[job_data['job_id']])
    
    # Update the job with the actual Celery task ID
    # Note: We do this synchronously before returning to avoid race conditions
    job_data['celery_task_id'] = task.id
    from app.config import JOBS_DIR
    from app.utils.file_utils import atomic_write_json
    job_path = JOBS_DIR / f"job_{job_data['job_id']}.json"
    atomic_write_json(job_path, job_data)
    
    return JobCreateResponse(
        job_id=job_data['job_id'],
        status="queued",
        created_at=job_data['created_at']
    )


@router.post("/{model_id}/retrain", response_model=TrainResponse, tags=["training"])
async def retrain_model(model_id: str, request: RetrainRequest):
    """
    Retrain an existing model with new data.
    
    - Uses the same task_type, target, and features as the original model
    - Validates that the new dataset has all required features
    - Creates a new model_id (does not overwrite the original)
    - Tracks lineage via parent_model_id
    """
    # Load old model metadata
    old_metadata_path = MODEL_METADATA_DIR / f"model_{model_id}.json"
    
    if not old_metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model {model_id} not found",
                "details": None
            }
        )
    
    with open(old_metadata_path, "r") as f:
        old_metadata = json.load(f)
    
    # Load old model schema
    old_schema_path = MODEL_SCHEMAS_DIR / f"model_{model_id}_schema.json"
    with open(old_schema_path, "r") as f:
        old_schema = json.load(f)
    
    # Extract configuration from old model
    task_type = old_metadata['task_type']
    target = old_metadata['target']
    features = old_metadata['features']
    
    # Use new xgb_params if provided, otherwise use old ones
    if train_request.xgb_params is not None:
        xgb_params = train_request.xgb_params
    else:
        xgb_params = old_metadata['xgb_params']
    
    # Load new dataset
    new_csv_path = DATASETS_DIR / f"{train_request.dataset_id}.csv"
    new_schema_path = DATASET_SCHEMAS_DIR / f"{train_request.dataset_id}_schema.json"
    
    if not new_csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "dataset_not_found",
                "message": f"Dataset {train_request.dataset_id} not found",
                "details": None
            }
        )
    
    if not new_schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "schema_not_found",
                "message": f"Schema for dataset {train_request.dataset_id} not found",
                "details": None
            }
        )
    
    # Load new dataset and schema
    df = pd.read_csv(new_csv_path)
    
    with open(new_schema_path, "r") as f:
        new_schema_data = json.load(f)
    
    # Validate that new dataset has all required features
    required_columns = set(features + [target])
    available_columns = set(df.columns)
    
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_features",
                "message": f"New dataset is missing required features: {list(missing_columns)}",
                "details": {
                    "required": list(required_columns),
                    "available": list(available_columns)
                }
            }
        )
    
    # Warn about extra columns (they will be ignored)
    extra_columns = available_columns - required_columns
    warnings = []
    if extra_columns:
        warnings.append(f"Extra columns in new dataset will be ignored: {list(extra_columns)}")
    
    # Use old model's schema for feature types
    numeric_features = old_schema['numeric_features']
    categorical_features = old_schema['categorical_features']
    
    # Train new model
    start_time = time.time()
    
    try:
        pipeline, evaluation_info = train_model_with_validation(
            df=df,
            target=target,
            features=features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            task_type=task_type,
            xgb_params=xgb_params,
            validation_split=VALIDATION_SPLIT,
            random_state=RANDOM_STATE
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "training_failed",
                "message": f"Model retraining failed: {str(e)}",
                "details": None
            }
        )
    
    training_duration = time.time() - start_time
    
    # Generate new model ID
    new_model_id = generate_id()
    
    # Save model artifact
    artifact_path = ARTIFACTS_DIR / f"model_{new_model_id}.joblib"
    joblib.dump(pipeline, artifact_path)
    
    # Get actual XGBoost parameters
    actual_xgb_params = pipeline.named_steps['model'].get_params()
    from app.config import ALLOWED_XGB_PARAMS
    final_xgb_params = {
        k: actual_xgb_params[k]
        for k in ALLOWED_XGB_PARAMS.keys()
        if k in actual_xgb_params
    }
    
    # Save model schema (same as old model)
    schema_artifact_path = MODEL_SCHEMAS_DIR / f"model_{new_model_id}_schema.json"
    atomic_write_json(schema_artifact_path, old_schema)
    
    # Create new model metadata with parent reference
    created_at = datetime.utcnow().isoformat() + "Z"
    
    metadata = {
        "model_id": new_model_id,
        "task_type": task_type,
        "target": target,
        "features": features,
        "xgb_params": final_xgb_params,
        "dataset_id": train_request.dataset_id,
        "created_at": created_at,
        "parent_model_id": model_id,  # Track lineage
        "training_duration": training_duration,
        "row_count": len(df),
        "feature_count": len(features),
        "metrics": evaluation_info['metrics'],
        "evaluation_method": evaluation_info['evaluation_method'],
        "validation_split": evaluation_info['validation_split'],
        "random_state": evaluation_info['random_state'],
        "train_samples": evaluation_info['train_samples'],
        "validation_samples": evaluation_info['validation_samples']
    }
    
    metadata_path = MODEL_METADATA_DIR / f"model_{new_model_id}.json"
    atomic_write_json(metadata_path, metadata)
    
    # Return response
    return TrainResponse(
        model_id=new_model_id,
        task_type=task_type,
        target=target,
        features=features,
        xgb_params=final_xgb_params,
        dataset_id=train_request.dataset_id,
        training_duration=training_duration,
        row_count=len(df),
        feature_count=len(features),
        metrics=evaluation_info['metrics'],
        created_at=created_at
    )

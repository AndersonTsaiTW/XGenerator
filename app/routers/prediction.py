"""
Prediction endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Request
import joblib
import json
from pathlib import Path

from app.config import ARTIFACTS_DIR, MODEL_METADATA_DIR, MODEL_SCHEMAS_DIR
from app.models.schemas import (
    PredictRequest, PredictResponse, PredictionResult,
    ErrorResponse
)
from app.services.prediction_service import (
    validate_prediction_input,
    make_predictions
)
from app.utils.auth import verify_api_key
from app.utils.rate_limit import limiter

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictResponse)
@limiter.limit("120/minute")
async def predict(
    request: Request,
    predict_request: PredictRequest,
    current_user: dict = Depends(verify_api_key)
):
    """
    Make predictions using a trained model.
    
    - Validates input against model schema
    - Returns predictions for all rows
    - For classification: returns probability of positive class
    - For regression: returns predicted value
    """
    # Load model metadata
    metadata_path = MODEL_METADATA_DIR / f"model_{predict_request.model_id}.json"
    
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model {predict_request.model_id} not found",
                "details": None
            }
        )
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Load model schema
    schema_path = MODEL_SCHEMAS_DIR / f"model_{predict_request.model_id}_schema.json"
    
    if not schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "schema_not_found",
                "message": f"Schema for model {predict_request.model_id} not found",
                "details": {
                    "hint": "This model may be corrupted. Please retrain."
                }
            }
        )
    
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    
    # Validate prediction input (now returns warnings instead of hard errors for missing features)
    is_valid, errors, warnings = validate_prediction_input(predict_request.rows, schema)
    
    if not is_valid:
        # Only fail if there are actual errors (shouldn't happen with current implementation)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "schema_validation_failed",
                "message": "Input data validation failed",
                "details": {
                    "errors": errors,
                    "hint": f"Refer to GET /models/{predict_request.model_id}/schema"
                }
            }
        )
    
    # Load model artifact
    artifact_path = ARTIFACTS_DIR / f"model_{predict_request.model_id}.joblib"
    
    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": f"Model artifact for {predict_request.model_id} not found",
                "details": {
                    "hint": "This model may be corrupted. Please retrain."
                }
            }
        )
    
    try:
        pipeline = joblib.load(artifact_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "load_failed",
                "message": f"Failed to load model: {str(e)}",
                "details": None
            }
        )
    
    # Make predictions (now passes schema for proper feature handling)
    try:
        predictions = make_predictions(
            pipeline,
            predict_request.rows,
            schema,
            metadata['task_type']
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "message": f"Prediction failed: {str(e)}",
                "details": None
            }
        )
    
    # Format response
    prediction_results = [
        PredictionResult(row_index=idx, prediction=pred)
        for idx, pred in enumerate(predictions)
    ]
    
    return PredictResponse(
        model_id=predict_request.model_id,
        predictions=prediction_results,
        warnings=warnings if warnings else None
    )

"""
Dataset management endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse
import pandas as pd
import json
from pathlib import Path

from app.config import DATASETS_DIR, DATASET_SCHEMAS_DIR, USERS_DIR
from app.utils.auth import verify_api_key, verify_ownership
from app.utils.rate_limit import limiter
from app.models.schemas import (
    DatasetUploadResponse,
    DatasetSchema,
    DatasetSchemaUpdateRequest,
    FeatureSchema,
    ErrorResponse
)
from app.utils.file_utils import generate_id, atomic_write_json
from app.services.schema_service import (
    infer_schema_with_openai,
    calculate_feature_statistics
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetUploadResponse)
@limiter.limit("5/minute")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(..., description="User ID of the dataset owner"),
    dataset_name: str = Form(..., description="Name for this dataset"),
    current_user: dict = Depends(verify_api_key)
):
    """
    Upload a CSV dataset and infer its schema.
    
    **Rate Limit**: 5 per minute (API Key-based)
    **Authentication**: Required (X-API-Key header)
    
    Schema inference:
    - Premium tier: OpenAI-powered (intelligent)
    - Free tier: Pandas-based (heuristic)
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file_type",
                "message": "Only CSV files are supported",
                "details": {"filename": file.filename}
            }
        )
    
    try:
        # Generate dataset ID
        dataset_id = generate_id()
        
        # Save CSV file
        csv_path = DATASETS_DIR / f"{dataset_id}.csv"
        content = await file.read()
        
        with open(csv_path, "wb") as f:
            f.write(content)
        
        # Load into pandas
        df = pd.read_csv(csv_path)
        
        if df.empty:
            csv_path.unlink()  # Clean up
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "empty_dataset",
                    "message": "The uploaded CSV file is empty",
                    "details": None
                }
            )
        
        # Infer schema based on user tier
        user_tier = current_user.get('tier', 'premium')  # Default to premium
        
        if user_tier == 'premium':
            # Premium users get OpenAI-powered inference
            numeric_features, categorical_features = infer_schema_with_openai(df)
        else:
            # Free users get basic pandas inference
            from app.services.schema_service import _basic_schema_inference
            numeric_features, categorical_features = _basic_schema_inference(df)
        
        # Calculate detailed feature statistics (Step 2)
        feature_details = []
        for col in df.columns:
            feature_type = "numeric" if col in numeric_features else "categorical"
            stats = calculate_feature_statistics(df, col)
            
            feature_details.append(FeatureSchema(
                name=col,
                type=feature_type,
                missing_rate=stats["missing_rate"],
                unique_count=stats["unique_count"]
            ))
        
        # Create schema object
        schema = DatasetSchema(
            dataset_id=dataset_id,
            dataset_name=dataset_name,  # NEW
            user_id=user_id,            # NEW
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            features=feature_details
        )
        
        # Save schema to metadata
        schema_path = DATASET_SCHEMAS_DIR / f"{dataset_id}_schema.json"
        atomic_write_json(schema_path, schema.model_dump())
        
        # Return response
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            dataset_name=dataset_name,  # NEW
            user_id=user_id,            # NEW
            filename=file.filename,
            row_count=len(df),
            column_count=len(df.columns),
            schema=schema
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_csv",
                "message": "The CSV file could not be parsed",
                "details": None
            }
        )
    except Exception as e:
        # Clean up on error
        if csv_path and csv_path.exists():
            csv_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upload_failed",
                "message": f"Dataset upload failed: {str(e)}",
                "details": None
            }
        )


@router.patch("/{dataset_id}/schema", response_model=DatasetSchema)
async def update_dataset_schema(
    dataset_id: str,
    request: DatasetSchemaUpdateRequest
):
    """
    Update the schema for a dataset.
    
    Allows users to override the automatically inferred schema by specifying
    which features should be treated as numeric or categorical.
    """
    # Check if dataset exists
    csv_path = DATASETS_DIR / f"{dataset_id}.csv"
    schema_path = DATASET_SCHEMAS_DIR / f"{dataset_id}_schema.json"
    
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "dataset_not_found",
                "message": f"Dataset {dataset_id} not found",
                "details": None
            }
        )
    
    if not schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "schema_not_found",
                "message": f"Schema for dataset {dataset_id} not found",
                "details": None
            }
        )
    
    # Load current schema
    with open(schema_path, "r") as f:
        import json
        current_schema_data = json.load(f)
    
    current_schema = DatasetSchema(**current_schema_data)
    
    # Load dataset to validate columns
    df = pd.read_csv(csv_path)
    all_columns = set(df.columns)
    
    # Prepare updated features
    updated_numeric = list(current_schema.numeric_features)
    updated_categorical = list(current_schema.categorical_features)
    
    # Apply user overrides
    if request.numeric_features is not None:
        # Validate all specified features exist
        invalid_features = set(request.numeric_features) - all_columns
        if invalid_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_features",
                    "message": f"Features not found in dataset: {list(invalid_features)}",
                    "details": {"available_columns": list(all_columns)}
                }
            )
        
        # Remove from categorical if present
        for feat in request.numeric_features:
            if feat in updated_categorical:
                updated_categorical.remove(feat)
            if feat not in updated_numeric:
                updated_numeric.append(feat)
    
    if request.categorical_features is not None:
        # Validate all specified features exist
        invalid_features = set(request.categorical_features) - all_columns
        if invalid_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_features",
                    "message": f"Features not found in dataset: {list(invalid_features)}",
                    "details": {"available_columns": list(all_columns)}
                }
            )
        
        # Remove from numeric if present
        for feat in request.categorical_features:
            if feat in updated_numeric:
                updated_numeric.remove(feat)
            if feat not in updated_categorical:
                updated_categorical.append(feat)
    
    # Check for duplicates
    overlap = set(updated_numeric) & set(updated_categorical)
    if overlap:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "duplicate_features",
                "message": f"Features cannot be both numeric and categorical: {list(overlap)}",
                "details": None
            }
        )
    
    # Recalculate feature details with updated types
    feature_details = []
    for col in df.columns:
        feature_type = "numeric" if col in updated_numeric else "categorical"
        stats = calculate_feature_statistics(df, col)
        
        feature_details.append(FeatureSchema(
            name=col,
            type=feature_type,
            missing_rate=stats["missing_rate"],
            unique_count=stats["unique_count"]
        ))
    
    # Create updated schema
    updated_schema = DatasetSchema(
        dataset_id=dataset_id,
        dataset_name=current_schema.dataset_name,  # Preserve
        user_id=current_schema.user_id,  # Preserve
        numeric_features=updated_numeric,
        categorical_features=updated_categorical,
        features=feature_details
    )
    
    # Save updated schema
    atomic_write_json(schema_path, updated_schema.model_dump())
    
    return updated_schema

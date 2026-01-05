"""
Pydantic models for request/response validation
"""
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator


# ===== Error Response Models =====
class ErrorResponse(BaseModel):
    """Unified error response format"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ===== User Management Models =====
class UserCreate(BaseModel):
    """Request to create a new user"""
    username: str = Field(
        ..., 
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Username (3-50 chars, alphanumeric, underscore, hyphen only)",
        json_schema_extra={"example": "john_doe"}
    )
    email: Optional[str] = Field(
        None,
        description="Optional email address",
        json_schema_extra={"example": "john@example.com"}
    )


class UserCreateResponse(BaseModel):
    """Response after creating a user (includes full API key)"""
    user_id: str
    username: str
    email: Optional[str]
    tier: str = "free"  # User tier: "free" or "premium"
    api_key: str  # Full API key - shown ONLY once
    created_at: str
    warning: str = "⚠️ Save this API key! It won't be shown again."


class UserResponse(BaseModel):
    """User response (with masked API key)"""
    user_id: str
    username: str
    email: Optional[str]
    tier: str = "premium"  # User tier: "free" or "premium"
    api_key_preview: str  # Masked API key
    created_at: str


# ===== Schema Models =====
class FeatureSchema(BaseModel):
    """Schema for a single feature"""
    name: str
    type: Literal["numeric", "categorical"]
    missing_rate: Optional[float] = None
    unique_count: Optional[int] = None


class DatasetSchema(BaseModel):
    """Dataset schema information"""
    dataset_id: str
    dataset_name: Optional[str] = Field(default="", description="Name of the dataset")  # Optional for old data
    user_id: Optional[str] = Field(default="", description="Owner user ID")  # Optional for old data
    numeric_features: List[str]
    categorical_features: List[str]
    features: Optional[List[FeatureSchema]] = None  # Step 2: detailed feature info


# ===== Dataset Upload Models =====
class DatasetUploadResponse(BaseModel):
    """Response after uploading a dataset"""
    dataset_id: str
    dataset_name: str = Field(..., description="Name of the dataset")  # NEW
    user_id: str = Field(..., description="Owner user ID")  # NEW
    filename: str
    row_count: int
    column_count: int
    schema: DatasetSchema


class DatasetSchemaUpdateRequest(BaseModel):
    """Request to update dataset schema"""
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None


# ===== Training Models =====
class TrainRequest(BaseModel):
    """Request to train a new model"""
    user_id: str = Field(
        ...,
        description="User ID of the model owner",
        json_schema_extra={"example": "user_abc123"}
    )  # NEW
    model_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name for this model",
        json_schema_extra={"example": "Churn Predictor v1"}
    )  # NEW
    dataset_id: str = Field(
        ..., 
        description="ID of the uploaded dataset",
        json_schema_extra={"example": "abc123def456"}
    )
    task_type: Literal["classification", "regression"] = Field(
        ..., 
        description="Type of ML task",
        json_schema_extra={"example": "classification"}
    )
    target: str = Field(
        ..., 
        description="Name of the target column to predict",
        json_schema_extra={"example": "label"}
    )
    features: Optional[List[str]] = Field(
        None, 
        description="Specific features to use (None = all columns except target)",
        json_schema_extra={"example": ["age", "price", "category"]}
    )
    exclude_features: Optional[List[str]] = Field(
        None, 
        description="Features to explicitly exclude from training (e.g.,['customer_id', 'product_id'])",
        json_schema_extra={"example": ["customer_id", "order_id"]}
    )
    xgb_params: Optional[Dict[str, Union[int, float]]] = Field(
        None, 
        description=(
            "XGBoost hyperparameters. Allowed: n_estimators (1-5000), "
            "learning_rate (0.0001-1.0), max_depth (1-16), "
            "subsample (0.0-1.0), colsample_bytree (0.0-1.0)"
        ),
        json_schema_extra={"example": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}}
    )
    
    @field_validator("xgb_params")
    @classmethod
    def validate_xgb_params(cls, v):
        """Validate XGBoost parameters against whitelist and ranges"""
        if v is None:
            return {}
        
        from app.config import ALLOWED_XGB_PARAMS
        
        # Check for unknown parameters
        unknown_params = set(v.keys()) - set(ALLOWED_XGB_PARAMS.keys())
        if unknown_params:
            allowed_list = ', '.join(ALLOWED_XGB_PARAMS.keys())
            raise ValueError(
                f"Unknown XGBoost parameters: {', '.join(unknown_params)}. "
                f"Allowed parameters: {allowed_list}"
            )
        
        # Validate parameter ranges
        for param_name, param_value in v.items():
            min_val, max_val, param_type = ALLOWED_XGB_PARAMS[param_name]
            
            # Type check
            if not isinstance(param_value, param_type):
                raise ValueError(
                    f"Parameter '{param_name}' must be of type {param_type.__name__}, "
                    f"got {type(param_value).__name__}"
                )
            
            # Range check
            if not (min_val <= param_value <= max_val):
                raise ValueError(
                    f"Parameter '{param_name}' value {param_value} is out of range. "
                    f"Must be between {min_val} and {max_val}"
                )
        
        return v


class TrainResponse(BaseModel):
    """Response from training"""
    model_id: str
    model_name: str  # NEW
    user_id: str  # NEW
    task_type: str
    target: str
    features: List[str]
    excluded_features: Optional[List[str]] = None  # Features that were excluded
    xgb_params: Dict[str, Any]
    dataset_id: str
    training_duration: float
    row_count: int
    feature_count: int
    metrics: Optional[Dict[str, float]] = None  # Step 2: evaluation metrics
    created_at: str
    warnings: Optional[List[str]] = None  # Warnings about excluded features


class RetrainRequest(BaseModel):
    """Request to retrain an existing model"""
    dataset_id: str
    xgb_params: Optional[Dict[str, Any]] = None
    
    @field_validator("xgb_params")
    @classmethod
    def validate_xgb_params(cls, v):
        if v is None:
            return None
        # Reuse the validation from TrainRequest
        return TrainRequest.validate_xgb_params(v)


# ===== Job Management Models (NEW for v2.0) =====
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobCreateResponse(BaseModel):
    """Response when creating a training job"""
    job_id: str
    status: JobStatus
    created_at: str
    message: str = "Training job created. Use GET /jobs/{job_id} to check status."


class JobResult(BaseModel):
    """Job result when training succeeds"""
    model_id: str
    metrics: Optional[Dict[str, float]] = None
    training_duration: float


class JobError(BaseModel):
    """Job error details when training fails"""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None


class JobDetail(BaseModel):
    """Detailed job information"""
    job_id: str
    job_type: Literal["train", "retrain"] = "train"
    status: JobStatus
    user_id: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Training configuration
    train_config: Dict[str, Any]
    
    # Results (only when succeeded)
    result: Optional[JobResult] = None
    
    # Error info (only when failed)
    error: Optional[JobError] = None
    
    # Celery task ID (for debugging)
    celery_task_id: Optional[str] = None


class JobSummary(BaseModel):
    """Summary of a job for listing"""
    job_id: str
    job_type: Literal["train", "retrain"] = "train"
    status: JobStatus
    user_id: str
    model_name: str
    created_at: str
    completed_at: Optional[str] = None
    model_id: Optional[str] = None  # Only when succeeded


class JobListResponse(BaseModel):
    """Response for listing jobs"""
    jobs: List[JobSummary]
    total: int
    limit: int
    offset: int


# ===== Prediction Models =====
class PredictRequest(BaseModel):
    """Request for batch prediction"""
    model_id: str
    rows: List[Dict[str, Any]] = Field(min_length=1)


class PredictionResult(BaseModel):
    """Single prediction result"""
    row_index: int
    prediction: float


class PredictResponse(BaseModel):
    """Response from prediction"""
    model_id: str
    predictions: List[PredictionResult]
    warnings: Optional[List[str]] = None  # Step 2: unknown category warnings


# ===== Model Management Models =====
class ModelSummary(BaseModel):
    """Summary of a model for listing"""
    model_id: str
    user_id: str
    username: Optional[str] = ""  # Backward compatibility
    model_name: Optional[str] = ""  # Backward compatibility
    task_type: str
    target: str
    dataset_id: str
    created_at: str


class ModelListResponse(BaseModel):
    """Response for listing models"""
    models: List[ModelSummary]
    total: int


class ModelMetadata(BaseModel):
    """Complete model metadata"""
    model_id: str
    user_id: str
    username: Optional[str] = None  # Backward compatibility
    model_name: Optional[str] = None  # Backward compatibility
    task_type: str
    target: str
    features: List[str]
    xgb_params: Dict[str, Any]
    dataset_id: str
    created_at: str
    parent_model_id: Optional[str] = None
    training_duration: Optional[float] = None
    row_count: Optional[int] = None
    feature_count: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None  # Step 2
    evaluation_method: Optional[str] = None  # Step 2

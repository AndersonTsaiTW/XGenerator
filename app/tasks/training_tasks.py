"""Celery tasks for model training"""
import pandas as pd
import joblib
import json
import time
from datetime import datetime
from pathlib import Path

from app.celery_app import celery_app
from app.services.job_service import update_job_status, get_job
from app.services.training_service import (
    validate_features_and_target,
    train_model_with_validation
)
from app.config import (
    DATASETS_DIR, DATASET_SCHEMAS_DIR, ARTIFACTS_DIR,
    MODEL_METADATA_DIR, MODEL_SCHEMAS_DIR
)
from app.utils.file_utils import generate_id, atomic_write_json


@celery_app.task(bind=True, max_retries=3)
def train_model_task(self, job_id: str):
    """
    Background task to train a model.
    
    Args:
        job_id: The job ID to track progress
    
    Returns:
        model_id: The ID of the trained model
    """
    try:
        # Update status to running
        update_job_status(job_id, "running")
        
        # Load job config
        job_data = get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")
        
        config = job_data["train_config"]
        user_id = job_data["user_id"]
        
        start_time = time.time()
        
        # 1. Load dataset
        dataset_path = DATASETS_DIR / f"{config['dataset_id']}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {config['dataset_id']} not found")
        
        df = pd.read_csv(dataset_path)
        
        # 2. Load dataset schema
        schema_path = DATASET_SCHEMAS_DIR / f"{config['dataset_id']}_schema.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)
        
        numeric_features = schema['numeric_features']
        categorical_features = schema['categorical_features']
        
        # 3. Smart feature filtering with auto-exclusion of ID columns
        from app.utils.feature_utils import filter_features
        
        features, excluded_features, feature_warnings = filter_features(
            all_columns=list(df.columns),
            target=config["target"],
            features=config.get("features"),  # Can be None
            exclude_features=config.get("exclude_features"),
            auto_exclude_ids=True
        )
        
        # 4. Validate features and target
        from app.services.training_service import validate_features_and_target
        validate_features_and_target(
            df=df,
            target=config["target"],
            features=features,
            numeric_features=numeric_features,
            categorical_features=categorical_features
        )
        
        # Filter schema features to only active features
        active_numeric = [f for f in numeric_features if f in features]
        active_categorical = [f for f in categorical_features if f in features]
        
        # 5. Train model
        from app.services.training_service import train_model_with_validation
        pipeline, evaluation_info = train_model_with_validation(
            df=df,
            target=config["target"],
            features=features,
            numeric_features=active_numeric,
            categorical_features=active_categorical,
            task_type=config["task_type"],
            xgb_params=config.get("xgb_params", {}),
            validation_split=0.2,
            random_state=42
        )
        
        # 5. Generate model ID
        model_id = generate_id()
        
        # 6. Save model artifact
        artifact_path = ARTIFACTS_DIR / f"model_{model_id}.joblib"
        joblib.dump(pipeline, artifact_path)
        
        # 7. Save model metadata
        metadata = {
            "model_id": model_id,
            "user_id": user_id,
            "username": config.get("username", ""),
            "model_name": config["model_name"],
            "task_type": config["task_type"],
            "target": config["target"],
            "features": features,
            "xgb_params": config.get("xgb_params", {}),
            "dataset_id": config["dataset_id"],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "training_duration": time.time() - start_time,
            "row_count": len(df),
            "feature_count": len(features),
            "metrics": evaluation_info['metrics'],
            "evaluation_method": evaluation_info['evaluation_method'],
            "validation_split": evaluation_info['validation_split'],
            "random_state": evaluation_info['random_state'],
            "train_samples": evaluation_info['train_samples'],
            "validation_samples": evaluation_info['validation_samples']
        }
        
        metadata_path = MODEL_METADATA_DIR / f"model_{model_id}.json"
        atomic_write_json(metadata_path, metadata)
        
        # 8. Save model schema
        model_schema = {
            "model_id": model_id,
            "user_id": user_id,
            "dataset_name": config.get("dataset_name", ""),
            "numeric_features": schema["numeric_features"],
            "categorical_features": schema["categorical_features"]
        }
        
        schema_path = MODEL_SCHEMAS_DIR / f"model_{model_id}_schema.json"
        atomic_write_json(schema_path, model_schema)
        
        # 9. Update job status to succeeded
        training_duration = time.time() - start_time
        update_job_status(
            job_id,
            "succeeded",
            result={
                "model_id": model_id,
                "metrics": evaluation_info['metrics'],
                "training_duration": training_duration
            }
        )
        
        return model_id
        
    except Exception as e:
        # Update job status to failed
        error_info = {
            "error_type": type(e).__name__,
            "message": str(e),
            "details": None
        }
        
        update_job_status(
            job_id,
            "failed",
            error=error_info
        )
        
        # Re-raise for Celery retry mechanism
        raise

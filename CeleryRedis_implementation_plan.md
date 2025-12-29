# Celery + Redis 背景訓練系統 - 實作計畫

## 目標
將同步的訓練端點 (`POST /train`) 改為非同步執行，使用 Celery + Redis 實現背景訓練，發布為 **v2.0.0**（Breaking Change）。

---

## 📋 關鍵決策總結

| 項目 | 決策 |
|------|------|
| Worker 數量 | 2 個 (可同時訓練 2 個模型) |
| 並發限制 | Worker `--concurrency=3` |
| Job 保留時間 | succeeded: 90天, failed: 30天 |
| Celery Flower | 暫不安裝 |
| Job Storage | JSON Files（與現有架構一致） |
| 向後相容性 | ❌ 完全非同步（Breaking Change） |
| Retrain 端點 | 不修改（保持原狀或考慮廢除） |

---

## 🔄 核心改變

### API 行為變更
**Before (v1.x - 同步):**
```python
POST /train → 等待訓練完成 (30-60秒) → 回傳 model_id
```

**After (v2.0 - 非同步):**
```python
POST /train → 立即回傳 job_id
GET /jobs/{job_id} → 查詢狀態 → succeeded 時取得 model_id
```

### 新增 API 端點
- `GET /jobs` - 列出訓練任務
- `GET /jobs/{job_id}` - 查詢任務狀態
- `DELETE /jobs/{job_id}` - 取消/刪除任務

---

## 🗂️ 實作變更詳細說明

### Phase 1: 基礎設施設定

#### [NEW] [docker-compose.yml](file:///c:/Users/ander/Documents/GitHub/XGenerator/docker-compose.yml)
創建 Docker Compose 配置，包含三個服務：
- **api**: 現有的 FastAPI 應用（port 8000）
- **redis**: Redis 作為 message broker（port 6379）
- **worker**: Celery worker 進程（背景訓練）

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    depends_on:
      - redis
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  worker:
    build: .
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    depends_on:
      - redis
    command: celery -A app.celery_app worker --loglevel=info --concurrency=2
```

> **注意**: 兩個 worker 進程 (`--concurrency=2`)，共享 `./data` 目錄以存取資料集和模型。

---

#### [MODIFY] [requirements.txt](file:///c:/Users/ander/Documents/GitHub/XGenerator/requirements.txt)
新增 Celery 和 Redis 依賴：

```diff
 fastapi==0.109.0
 uvicorn[standard]==0.27.0
 pydantic==2.5.3
 python-multipart==0.0.6
 
 xgboost==2.0.3
 scikit-learn==1.4.0
 pandas==2.2.0
 numpy==1.26.3
 joblib==1.3.2
 
 openai==1.6.1
 python-dotenv==1.0.0
 slowapi==0.1.9
+
+# Celery + Redis for background tasks
+celery==5.3.4
+redis==5.0.1
```

---

#### [MODIFY] [app/config.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/config.py)
新增 Redis URL 和 Jobs 目錄配置：

```python
# 新增
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 新增 Jobs 目錄
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
```

---

#### [NEW] [app/celery_app.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/celery_app.py)
創建 Celery 應用配置：

```python
"""Celery application configuration"""
from celery import Celery
from app.config import REDIS_URL

# Create Celery app
celery_app = Celery(
    "xgenerator_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['app.tasks'])
```

> **關鍵配置**:
> - `task_time_limit`: 單個訓練任務最多 1 小時
> - `task_track_started`: 追蹤任務開始時間
> - Auto-discover tasks from `app.tasks` module

---

### Phase 2: Job 管理系統

#### [MODIFY] [app/models/schemas.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/models/schemas.py)
新增 Job 相關的 Pydantic 模型：

```python
from typing import Literal

# Job Status Enum
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

# Job Create Response (when POST /train)
class JobCreateResponse(BaseModel):
    """Response when creating a training job"""
    job_id: str
    status: JobStatus
    created_at: str
    message: str = "Training job created. Use GET /jobs/{job_id} to check status."

# Job Result (when succeeded)
class JobResult(BaseModel):
    """Job result when training succeeds"""
    model_id: str
    metrics: Optional[Dict[str, float]] = None
    training_duration: float

# Job Error (when failed)
class JobError(BaseModel):
    """Job error details when training fails"""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Job Detail (GET /jobs/{job_id})
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

# Job Summary (for listing)
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

# Job List Response
class JobListResponse(BaseModel):
    """Response for listing jobs"""
    jobs: List[JobSummary]
    total: int
    limit: int
    offset: int
```

---

#### [NEW] [app/services/job_service.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/services/job_service.py)
實作 Job CRUD 函數：

```python
"""Job management service"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.config import JOBS_DIR
from app.utils.file_utils import generate_id, atomic_write_json

def create_job(
    user_id: str,
    train_config: Dict[str, Any],
    celery_task_id: str
) -> Dict[str, Any]:
    """Create a new training job"""
    job_id = generate_id()
    job_data = {
        "job_id": job_id,
        "job_type": "train",
        "status": "queued",
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
        "train_config": train_config,
        "result": None,
        "error": None,
        "celery_task_id": celery_task_id
    }
    
    job_path = JOBS_DIR / f"job_{job_id}.json"
    atomic_write_json(job_path, job_data)
    return job_data

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job by ID"""
    job_path = JOBS_DIR / f"job_{job_id}.json"
    if not job_path.exists():
        return None
    
    with open(job_path, "r") as f:
        return json.load(f)

def update_job_status(
    job_id: str,
    status: str,
    **kwargs
) -> None:
    """Update job status and other fields"""
    job_data = get_job(job_id)
    if not job_data:
        raise ValueError(f"Job {job_id} not found")
    
    job_data["status"] = status
    
    # Update timestamps
    if status == "running" and not job_data.get("started_at"):
        job_data["started_at"] = datetime.utcnow().isoformat() + "Z"
    
    if status in ["succeeded", "failed", "cancelled"]:
        job_data["completed_at"] = datetime.utcnow().isoformat() + "Z"
    
    # Update other fields
    for key, value in kwargs.items():
        job_data[key] = value
    
    job_path = JOBS_DIR / f"job_{job_id}.json"
    atomic_write_json(job_path, job_data)

def list_jobs(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> tuple[List[Dict[str, Any]], int]:
    """List jobs with filters"""
    all_jobs = []
    
    for job_file in sorted(JOBS_DIR.glob("job_*.json"), reverse=True):
        try:
            with open(job_file, "r") as f:
                job_data = json.load(f)
            
            # Apply filters
            if user_id and job_data.get("user_id") != user_id:
                continue
            if status and job_data.get("status") != status:
                continue
            
            all_jobs.append(job_data)
        except Exception:
            continue
    
    total = len(all_jobs)
    paginated_jobs = all_jobs[offset:offset + limit]
    
    return paginated_jobs, total

def delete_job(job_id: str) -> bool:
    """Delete a job"""
    job_path = JOBS_DIR / f"job_{job_id}.json"
    if not job_path.exists():
        return False
    
    job_path.unlink()
    return True
```

---

### Phase 3: Celery Worker 實作

#### [NEW] [app/tasks/training_tasks.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/tasks/training_tasks.py)
實作訓練任務（核心邏輯）：

```python
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
    DATASETS_DIR, DATASET_METADATA_DIR, ARTIFACTS_DIR,
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
        schema_path = DATASET_METADATA_DIR / f"schema_{config['dataset_id']}.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)
        
        # 3. Validate features and target
        features = validate_features_and_target(
            df=df,
            schema=schema,
            target=config["target"],
            features=config.get("features"),
            exclude_features=config.get("exclude_features")
        )
        
        # 4. Train model
        pipeline, metrics = train_model_with_validation(
            df=df,
            features=features,
            target=config["target"],
            task_type=config["task_type"],
            xgb_params=config.get("xgb_params", {})
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
            "metrics": metrics,
            "evaluation_method": "train_test_split"
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
                "metrics": metrics,
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
```

> **重點**:
> - 使用 `@celery_app.task(bind=True, max_retries=3)` 支援重試
> - 更新 job 狀態：queued → running → succeeded/failed
> - 複用現有的訓練邏輯 (`train_model_with_validation`)
> - 錯誤處理：捕獲異常並記錄到 job

---

#### [NEW] [app/tasks/__init__.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/tasks/__init__.py)
```python
"""Celery tasks module"""
from app.tasks.training_tasks import train_model_task

__all__ = ["train_model_task"]
```

---

### Phase 4: API 端點修改

#### [MODIFY] [app/routers/training.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/routers/training.py)
將 `POST /train` 改為非同步：

**修改範圍**: 整個 `train_model` 函數 (line 33-224)

**主要改變**:
1. 回傳類型：`TrainResponse` → `JobCreateResponse`
2. 不再執行訓練，改為創建 job 並送入 Celery queue
3. 立即回傳 `job_id`

```python
from app.models.schemas import JobCreateResponse
from app.services.job_service import create_job
from app.tasks.training_tasks import train_model_task

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
    
    - Creates a training job
    - Queues the job to Celery worker
    - Returns immediately with job_id
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
    schema_path = DATASET_METADATA_DIR / f"schema_{train_request.dataset_id}.json"
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
        "username": current_user.get("username", "")
    }
    
    # Queue training task to Celery
    task = train_model_task.delay(job_id=None)  # Will be updated after job creation
    
    # Create job record
    job_data = create_job(
        user_id=train_request.user_id,
        train_config=train_config,
        celery_task_id=task.id
    )
    
    # Update task with job_id (hack: store in task args)
    task.update_state(meta={'job_id': job_data['job_id']})
    
    # Actually queue the task with job_id
    task = train_model_task.apply_async(args=[job_data['job_id']])
    
    # Update job with correct task_id
    from app.services.job_service import update_job_status
    update_job_status(job_data['job_id'], 'queued', celery_task_id=task.id)
    
    return JobCreateResponse(
        job_id=job_data['job_id'],
        status="queued",
        created_at=job_data['created_at']
    )
```

> **注意**: 移除了原本的訓練邏輯，改為創建 job 並送入 queue。

---

#### [NEW] [app/routers/jobs.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/routers/jobs.py)
新增 Jobs 管理端點：

```python
"""Job management endpoints"""
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from typing import Optional

from app.models.schemas import (
    JobDetail, JobListResponse, JobSummary,
    ErrorResponse
)
from app.services.job_service import (
    get_job, list_jobs, delete_job, update_job_status
)
from app.utils.auth import verify_api_key
from app.utils.rate_limit import limiter
from app.celery_app import celery_app

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=JobListResponse)
@limiter.limit("120/minute")
async def list_training_jobs(
    request: Request,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    List training jobs with optional filters.
    
    **Rate Limit**: 120 per minute (IP-based)
    
    **Query Parameters:**
    - **user_id** (optional): Filter by user ID
    - **status** (optional): Filter by status (queued/running/succeeded/failed)
    - **limit** (default: 50, max: 100): Number of results
    - **offset** (default: 0): Pagination offset
    """
    jobs_data, total = list_jobs(
        user_id=user_id,
        status=status,
        limit=limit,
        offset=offset
    )
    
    # Convert to JobSummary
    job_summaries = []
    for job in jobs_data:
        summary = JobSummary(
            job_id=job['job_id'],
            job_type=job.get('job_type', 'train'),
            status=job['status'],
            user_id=job['user_id'],
            model_name=job['train_config'].get('model_name', ''),
            created_at=job['created_at'],
            completed_at=job.get('completed_at'),
            model_id=job.get('result', {}).get('model_id') if job.get('result') else None
        )
        job_summaries.append(summary)
    
    return JobListResponse(
        jobs=job_summaries,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/{job_id}", response_model=JobDetail)
@limiter.limit("120/minute")
async def get_job_status(request: Request, job_id: str):
    """
    Get detailed status of a training job.
    
    **Rate Limit**: 120 per minute (IP-based)
    
    Returns complete job information including:
    - Current status (queued/running/succeeded/failed)
    - Training configuration
    - Result (model_id, metrics) when succeeded
    - Error details when failed
    """
    job_data = get_job(job_id)
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"Job {job_id} not found",
                "details": None
            }
        )
    
    return JobDetail(**job_data)


@router.delete("/{job_id}")
@limiter.limit("30/minute")
async def cancel_job(
    request: Request,
    job_id: str,
    current_user: dict = Depends(verify_api_key)
):
    """
    Cancel or delete a training job.
    
    **Requires Authentication**: X-API-Key header
    **Rate Limit**: 30 per minute (API Key-based)
    
    Behavior:
    - **queued**: Remove from queue and mark as cancelled
    - **running**: Attempt to cancel (may not stop immediately)
    - **succeeded/failed**: Delete job record (model is preserved)
    """
    job_data = get_job(job_id)
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"Job {job_id} not found",
                "details": None
            }
        )
    
    # Verify ownership
    if job_data['user_id'] != current_user['user_id']:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "You don't have permission to cancel this job",
                "details": None
            }
        )
    
    previous_status = job_data['status']
    
    # Cancel Celery task if queued or running
    if previous_status in ['queued', 'running']:
        celery_task_id = job_data.get('celery_task_id')
        if celery_task_id:
            celery_app.control.revoke(celery_task_id, terminate=True)
        
        update_job_status(job_id, 'cancelled')
        message = "Job cancelled successfully"
    else:
        # Delete completed/failed jobs
        delete_job(job_id)
        message = "Job deleted successfully"
    
    return {
        "message": message,
        "job_id": job_id,
        "previous_status": previous_status
    }
```

---

#### [MODIFY] [app/main.py](file:///c:/Users/ander/Documents/GitHub/XGenerator/app/main.py)
引入 jobs router：

```python
from app.routers import datasets, training, prediction, models, users, jobs  # NEW

# Include routers
app.include_router(users.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(models.router)
app.include_router(jobs.router)  # NEW
```

並更新版本號：

```python
app = FastAPI(
    title="XGBoost Training Service API",
    version="2.0.0",  # Breaking change: async training
    # ...
)
```

---

#### [NEW] [Dockerfile](file:///c:/Users/ander/Documents/GitHub/XGenerator/Dockerfile)
創建 Dockerfile（如果不存在）：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Default command (overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### Phase 5: 測試與驗證

## 驗證計畫

### 1. 自動化測試

#### 單元測試: Job Service
**新增測試**: `tests/unit/test_job_service.py`

```python
"""Unit tests for job service"""
import pytest
import json
from pathlib import Path
from app.services.job_service import (
    create_job, get_job, update_job_status, list_jobs, delete_job
)
from app.config import JOBS_DIR

def test_create_job():
    """Test job creation"""
    train_config = {
        "dataset_id": "test123",
        "model_name": "Test Model",
        "task_type": "classification",
        "target": "label"
    }
    
    job_data = create_job(
        user_id="user_123",
        train_config=train_config,
        celery_task_id="task_abc"
    )
    
    assert job_data['status'] == 'queued'
    assert job_data['user_id'] == 'user_123'
    assert job_data['train_config'] == train_config
    
    # Verify file exists
    job_path = JOBS_DIR / f"job_{job_data['job_id']}.json"
    assert job_path.exists()

def test_update_job_status():
    """Test job status update"""
    # Create job first
    job_data = create_job(
        user_id="user_123",
        train_config={},
        celery_task_id="task_abc"
    )
    
    # Update to running
    update_job_status(job_data['job_id'], 'running')
    updated = get_job(job_data['job_id'])
    assert updated['status'] == 'running'
    assert updated['started_at'] is not None
    
    # Update to succeeded
    update_job_status(
        job_data['job_id'],
        'succeeded',
        result={'model_id': 'model_xyz'}
    )
    final = get_job(job_data['job_id'])
    assert final['status'] == 'succeeded'
    assert final['completed_at'] is not None
    assert final['result']['model_id'] == 'model_xyz'
```

**執行命令**:
```bash
pytest tests/unit/test_job_service.py -v
```

---

#### 整合測試: 非同步訓練流程
**修改**: `tests/integration/test_end_to_end.py`

新增測試函數 `test_async_training_workflow`:

```python
def test_async_training_workflow():
    """Test complete async training workflow"""
    print_step(7, "Test Async Training Workflow (v2.0)")
    
    # 1. Create user and upload dataset (reuse existing)
    # ...
    
    # 2. Submit training job
    train_payload = {
        "user_id": user_id,
        "model_name": "Async Test Model",
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target": "Survived",
        "features": None,
        "xgb_params": {"n_estimators": 50, "max_depth": 3}
    }
    
    response = requests.post(
        f"{BASE_URL}/train",
        json=train_payload,
        headers={"X-API-Key": api_key}
    )
    
    assert response.status_code == 200, "Training job creation failed"
    job_data = response.json()
    assert "job_id" in job_data
    assert job_data["status"] == "queued"
    
    job_id = job_data["job_id"]
    print_success(f"Job created: {job_id}")
    
    # 3. Poll job status until complete
    max_wait = 120  # 2 minutes
    poll_interval = 5
    elapsed = 0
    
    while elapsed < max_wait:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        assert response.status_code == 200
        
        job_status = response.json()
        print_info(f"Job status: {job_status['status']}")
        
        if job_status['status'] == 'succeeded':
            assert 'result' in job_status
            assert 'model_id' in job_status['result']
            model_id = job_status['result']['model_id']
            print_success(f"Training succeeded! Model ID: {model_id}")
            break
        elif job_status['status'] == 'failed':
            pytest.fail(f"Training failed: {job_status.get('error')}")
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        pytest.fail("Training timeout")
    
    # 4. Verify model exists
    response = requests.get(f"{BASE_URL}/models/{model_id}")
    assert response.status_code == 200
    print_success("Model verified successfully")
```

**執行命令**:
```bash
# 需要先啟動所有服務
docker-compose up -d

# 等待服務就緒
sleep 10

# 執行整合測試
pytest tests/integration/test_end_to_end.py::test_async_training_workflow -v
```

---

### 2. 手動測試

#### 測試步驟

**前置條件**: 啟動所有服務
```bash
docker-compose up --build
```

**測試流程**:

1. **創建用戶**
   ```bash
   curl -X POST http://localhost:8000/users \
     -H "Content-Type: application/json" \
     -d '{"username": "testuser", "email": "test@example.com"}'
   ```
   記下 `api_key`

2. **上傳資料集**
   ```bash
   curl -X POST http://localhost:8000/datasets \
     -H "X-API-Key: YOUR_API_KEY" \
     -F "file=@train_full.csv" \
     -F "user_id=YOUR_USER_ID" \
     -F "dataset_name=Titanic Dataset"
   ```
   記下 `dataset_id`

3. **提交訓練任務**
   ```bash
   curl -X POST http://localhost:8000/train \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "YOUR_USER_ID",
       "model_name": "Titanic Classifier",
       "dataset_id": "YOUR_DATASET_ID",
       "task_type": "classification",
       "target": "Survived",
       "features": null,
       "xgb_params": {"n_estimators": 100, "max_depth": 5}
     }'
   ```
   **預期結果**: 立即回傳 job_id，狀態為 `queued`

4. **查詢任務狀態**
   ```bash
   curl http://localhost:8000/jobs/YOUR_JOB_ID
   ```
   **預期狀態轉換**: queued → running → succeeded

5. **列出所有任務**
   ```bash
   curl http://localhost:8000/jobs
   ```

6. **驗證模型**
   ```bash
   curl http://localhost:8000/models/YOUR_MODEL_ID
   ```

7. **取消任務（可選）**
   ```bash
   curl -X DELETE http://localhost:8000/jobs/YOUR_JOB_ID \
     -H "X-API-Key: YOUR_API_KEY"
   ```

---

### 3. 監控驗證

#### 檢查 Worker Logs
```bash
docker-compose logs -f worker
```

**預期輸出**:
```
worker_1  | [2024-01-01 00:00:00,000: INFO/MainProcess] Connected to redis://redis:6379/0
worker_1  | [2024-01-01 00:00:05,000: INFO/MainProcess] Task train_model_task[abc123] received
worker_1  | [2024-01-01 00:00:45,000: INFO/MainProcess] Task train_model_task[abc123] succeeded
```

#### 檢查 Redis
```bash
docker-compose exec redis redis-cli
> KEYS *
> GET celery-task-meta-XXX
```

---

### 4. 並發訓練測試

**測試目標**: 驗證可以同時訓練多個模型

**步驟**:
1. 快速連續提交 3 個訓練任務
2. 觀察 worker logs，確認有 2 個任務同時執行，1 個排隊
3. 等待全部完成，驗證 3 個模型都成功訓練

**驗證命令**:
```bash
# 提交第1個任務
curl -X POST http://localhost:8000/train ... > job1.json

# 提交第2個任務
curl -X POST http://localhost:8000/train ... > job2.json

# 提交第3個任務
curl -X POST http://localhost:8000/train ... > job3.json

# 同時查詢
curl http://localhost:8000/jobs | jq '.jobs[] | {job_id, status}'
```

---

### 5. 失敗場景測試

#### 測試 1: Invalid Dataset
提交訓練任務時使用不存在的 dataset_id

**預期**: 立即回傳 404 error（不創建 job）

#### 測試 2: Invalid Target Column
使用不存在的 target column

**預期**: Job 狀態變為 `failed`，error 記錄錯誤訊息

#### 測試 3: Worker Crash
手動停止 worker，然後重啟

**預期**: 正在執行的任務會重試或失敗，queued 任務會被新 worker 接手

---

## 📋 實作檢查清單

完成後確認：

- [ ] Docker Compose 可正常啟動（3個服務都 running）
- [ ] Redis 連線正常
- [ ] Worker 可從 queue 取出任務
- [ ] POST /train 立即回傳 job_id
- [ ] GET /jobs/{job_id} 可查詢狀態
- [ ] 訓練完成後 job 狀態變為 succeeded
- [ ] 模型檔案和 metadata 正確儲存
- [ ] 可同時訓練多個模型（並發）
- [ ] 錯誤處理正確（failed 狀態 + error 訊息）
- [ ] 單元測試通過
- [ ] 整合測試通過

---

## 🚧 已知限制與未來改進

### 目前不實作的功能
1. **自動清理舊 Job** - 等需要時再加（定期刪除 90天前的記錄）
2. **Celery Flower** - 監控介面，需要時 5 分鐘可加入
3. **進度更新** - 任務執行百分比（需要修改訓練邏輯）
4. **WebSocket 通知** - 即時推送狀態變更（目前靠輪詢）

### 注意事項
- 這是 **Breaking Change**，需要更新所有客戶端代碼
- Worker 和 API 共享 `./data` 目錄（Docker volume）
- Redis 不持久化（重啟後 queue 清空）

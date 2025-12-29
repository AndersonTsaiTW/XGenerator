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

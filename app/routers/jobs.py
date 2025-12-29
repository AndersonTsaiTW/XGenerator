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

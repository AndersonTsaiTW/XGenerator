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

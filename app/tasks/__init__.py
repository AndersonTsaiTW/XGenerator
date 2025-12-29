"""Celery tasks module"""
from app.tasks.training_tasks import train_model_task

__all__ = ["train_model_task"]

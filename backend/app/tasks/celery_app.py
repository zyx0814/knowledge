from celery import Celery
from config.config import settings
celery_app = Celery(
    "knowledge",
    broker=f"redis://{getattr(settings, 'REDIS_HOST', 'localhost')}:{getattr(settings, 'REDIS_PORT', 6379)}/0",
    backend=f"redis://{getattr(settings, 'REDIS_HOST', 'localhost')}:{getattr(settings, 'REDIS_PORT', 6379)}/1",
    include=["app.tasks.file_tasks"]
)
celery_app.conf.update(
    task_serializer="json",
    result_expires=3600,
    worker_concurrency=8,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

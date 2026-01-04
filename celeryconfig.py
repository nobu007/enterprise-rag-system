"""
Celery Configuration for Enterprise RAG System

This module contains all Celery-related configuration settings.
"""

from celery import Celery
from app.core.config import settings


# Create Celery app instance
app = Celery("rag_tasks")

# Configure Celery using settings
app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task tracking
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes hard limit
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    task_acks_late=True,  # Acknowledge task after execution
    worker_prefetch_multiplier=1,  # Disable prefetching for better error isolation

    # Worker settings
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_concurrency=4,  # Number of worker processes

    # Task result backend
    result_backend=settings.celery_result_backend,
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Keep results until explicitly removed

    # Task routing (can be extended for specific queues)
    task_routes={
        'tasks.process_document_batch': {'queue': 'batch_processing'},
        'tasks.process_document_batch_retry': {'queue': 'batch_processing'},
    },

    # Task execution settings
    task_always_eager=False,  # Don't execute tasks synchronously
    task_eager_propagates=True,  # Propagate exceptions in eager mode

    # Rate limiting (optional)
    # task_annotations={
    #     'tasks.process_document_batch': {'rate_limit': '10/m'}
    # },

    # Retry settings
    task_default_retry_delay=60,  # Retry after 60 seconds
    task_max_retries=3,  # Maximum retry attempts

    # Logging
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',

    # Security
    worker_send_task_events=True,  # Send monitoring events
    task_send_sent_event=True,  # Send task-sent events
)

# Auto-discover tasks (if using multiple task modules)
# app.autodiscover_tasks(['app.tasks'], force=True)


# Optional: Configure monitoring with Flower
# flower_basic_auth = ['admin:password']
# flower_port = 5555

"""
Celery tasks for asynchronous batch processing
"""

from app.tasks.batch_tasks import process_document_batch

__all__ = ['process_document_batch']

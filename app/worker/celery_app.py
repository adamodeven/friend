from __future__ import annotations

import logging

from celery import Celery
from celery.signals import worker_ready

from app.core.config import get_settings
from app.llm.warmup import warmup_ollama_text_model

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "friend_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.beat_schedule = {
    "schedule-active-task-reminders": {
        "task": "app.worker.tasks.schedule_reminders_task",
        "schedule": 300.0,
    },
    "dispatch-due-reminders": {
        "task": "app.worker.tasks.send_due_reminders_task",
        "schedule": 60.0,
    },
    "daily-summary-snapshot": {
        "task": "app.worker.tasks.daily_summary_snapshot_task",
        "schedule": 3600.0,
    },
}
celery_app.conf.timezone = settings.timezone


@worker_ready.connect
def _warmup_llm_on_worker_start(**_kwargs) -> None:
    if settings.llm_provider.lower().strip() != "ollama":
        return
    warmup_ollama_text_model(logger=logger)

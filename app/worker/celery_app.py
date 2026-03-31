from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

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


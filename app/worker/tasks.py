from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from celery import shared_task
from sqlalchemy import select

from app.db.models import DailySummarySnapshot, MessageDirection, ReminderStatus, Task, TaskStatus
from app.db.repositories.message_repo import create_message
from app.db.repositories.task_repo import list_active_tasks
from app.db.repositories.user_repo import get_or_create_primary_user
from app.db.session import SessionLocal
from app.domain.reminder_engine import ReminderEngine
from app.transport.twilio_adapter import TwilioTransport

logger = logging.getLogger(__name__)


@shared_task(name="app.worker.tasks.schedule_reminders_task")
def schedule_reminders_task() -> dict:
    session = SessionLocal()
    try:
        user = get_or_create_primary_user(session)
        engine = ReminderEngine()
        tasks = list_active_tasks(session, user.id)
        created = 0
        now = datetime.now(tz=ZoneInfo(user.timezone))
        for task in tasks:
            reminder = engine.schedule_for_task(session, task, now=now)
            if reminder:
                created += 1
        session.commit()
        return {"created": created, "active_tasks": len(tasks)}
    except Exception as exc:  # pragma: no cover
        session.rollback()
        logger.exception("schedule reminders failed: %s", exc)
        raise
    finally:
        session.close()


@shared_task(name="app.worker.tasks.send_due_reminders_task")
def send_due_reminders_task() -> dict:
    session = SessionLocal()
    transport = TwilioTransport()
    try:
        user = get_or_create_primary_user(session)
        engine = ReminderEngine()
        now = datetime.now(tz=ZoneInfo(user.timezone))
        reminders = engine.due_reminders(session, user.id, now)
        sent = 0
        skipped = 0

        for reminder in reminders:
            task = None
            if reminder.task_id:
                task = session.execute(select(Task).where(Task.id == reminder.task_id)).scalars().first()
            if task and task.status == TaskStatus.completed:
                reminder.status = ReminderStatus.skipped
                skipped += 1
                continue

            body = _compose_reminder_text(task_title=task.title if task else "that thing", escalation=reminder.escalation_level)
            sid = transport.send_sms(to_number=user.phone_number, body=body)
            create_message(
                session,
                user_id=user.id,
                direction=MessageDirection.outbound,
                body=body,
                external_id=sid,
                metadata_json={"source": "reminder", "reminder_id": str(reminder.id)},
            )
            reminder.status = ReminderStatus.sent
            reminder.sent_at = now
            sent += 1

        session.commit()
        return {"sent": sent, "skipped": skipped, "due": len(reminders)}
    except Exception as exc:  # pragma: no cover
        session.rollback()
        logger.exception("send due reminders failed: %s", exc)
        raise
    finally:
        session.close()


@shared_task(name="app.worker.tasks.daily_summary_snapshot_task")
def daily_summary_snapshot_task() -> dict:
    session = SessionLocal()
    try:
        user = get_or_create_primary_user(session)
        now = datetime.now(tz=ZoneInfo(user.timezone))
        week_out = now + timedelta(days=7)
        open_tasks = list_active_tasks(session, user.id)
        due_soon = [t for t in open_tasks if t.deadline_at and t.deadline_at <= week_out]
        summary = DailySummarySnapshot(
            user_id=user.id,
            snapshot_date=now.date(),
            summary_text=f"{len(open_tasks)} open tasks, {len(due_soon)} due within 7d",
            open_task_count=len(open_tasks),
            due_soon_count=len(due_soon),
            payload={"top_tasks": [t.title for t in open_tasks[:5]]},
        )
        session.add(summary)
        session.commit()
        return {"snapshot_date": str(now.date()), "open_tasks": len(open_tasks), "due_soon": len(due_soon)}
    except Exception as exc:  # pragma: no cover
        session.rollback()
        logger.exception("daily summary task failed: %s", exc)
        raise
    finally:
        session.close()


def _compose_reminder_text(*, task_title: str, escalation: int) -> str:
    if escalation <= 0:
        return f"quick check: where are you at on '{task_title}'?"
    if escalation == 1:
        return f"still waiting on '{task_title}'. what's the blocker?"
    return f"we're slipping on '{task_title}'. give me the next concrete move right now."


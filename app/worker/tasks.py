from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from billiard.exceptions import SoftTimeLimitExceeded
from celery import shared_task
from sqlalchemy import select

from app.db.models import DailySummarySnapshot, MessageDirection, ReminderStatus, Task, TaskStatus
from app.db.repositories.message_repo import create_message, inbound_message_exists
from app.db.repositories.task_repo import list_active_tasks
from app.db.repositories.user_repo import get_or_create_primary_user, get_user_by_phone
from app.db.session import SessionLocal
from app.domain.conversation_manager import ConversationManager
from app.domain.reminder_engine import ReminderEngine
from app.schemas.transport import InboundSmsPayload
from app.transport.twilio_adapter import TwilioTransport

logger = logging.getLogger(__name__)


@shared_task(name="app.worker.tasks.process_inbound_sms_task", soft_time_limit=45)
def process_inbound_sms_task(payload_data: dict[str, Any]) -> dict[str, Any]:
    session = SessionLocal()
    transport = TwilioTransport()
    manager = ConversationManager()
    payload = InboundSmsPayload.model_validate(payload_data)
    try:
        result = manager.process_inbound(session, payload)
        if result.skipped_duplicate:
            session.commit()
            return {"skipped_duplicate": True, "message_sid": payload.message_sid}

        sent = 0
        for chunk in result.outgoing_messages:
            transport.send_sms(to_number=payload.from_number, body=chunk)
            sent += 1

        session.commit()
        return {
            "skipped_duplicate": False,
            "message_sid": payload.message_sid,
            "outgoing_count": sent,
        }
    except SoftTimeLimitExceeded as exc:  # pragma: no cover
        session.rollback()
        logger.exception("inbound sms task timed out for sid=%s: %s", payload.message_sid, exc)
        fallback_body = "quick heads up: response took too long on my side. i still logged this, resend and i got you."
        try:
            user = get_user_by_phone(session, payload.from_number) or get_or_create_primary_user(session)
            if not inbound_message_exists(session, payload.message_sid):
                create_message(
                    session,
                    user_id=user.id,
                    direction=MessageDirection.inbound,
                    body=payload.body,
                    external_id=payload.message_sid,
                    metadata_json={
                        "from": payload.from_number,
                        "to": payload.to_number,
                        "num_media": payload.num_media,
                        "processing_timed_out": True,
                    },
                )
            sid = transport.send_sms(to_number=payload.from_number, body=fallback_body)
            create_message(
                session,
                user_id=user.id,
                direction=MessageDirection.outbound,
                body=fallback_body,
                external_id=sid,
                metadata_json={"source": "fallback_after_timeout"},
            )
            session.commit()
        except Exception as send_exc:
            session.rollback()
            logger.exception("timeout fallback sms send failed for sid=%s: %s", payload.message_sid, send_exc)
        return {"skipped_duplicate": False, "message_sid": payload.message_sid, "error": "timeout"}
    except Exception as exc:  # pragma: no cover
        session.rollback()
        logger.exception("inbound sms task failed for sid=%s: %s", payload.message_sid, exc)
        fallback_body = "my bad, i hit a processing hiccup. resend that and i got you."
        try:
            user = get_user_by_phone(session, payload.from_number) or get_or_create_primary_user(session)
            if not inbound_message_exists(session, payload.message_sid):
                create_message(
                    session,
                    user_id=user.id,
                    direction=MessageDirection.inbound,
                    body=payload.body,
                    external_id=payload.message_sid,
                    metadata_json={
                        "from": payload.from_number,
                        "to": payload.to_number,
                        "num_media": payload.num_media,
                        "processing_failed": True,
                    },
                )
            sid = transport.send_sms(to_number=payload.from_number, body=fallback_body)
            create_message(
                session,
                user_id=user.id,
                direction=MessageDirection.outbound,
                body=fallback_body,
                external_id=sid,
                metadata_json={"source": "fallback_after_processing_error"},
            )
            session.commit()
        except Exception as send_exc:  # pragma: no cover
            session.rollback()
            logger.exception("fallback sms send failed for sid=%s: %s", payload.message_sid, send_exc)
        return {"skipped_duplicate": False, "message_sid": payload.message_sid, "error": str(exc)}
    finally:
        session.close()


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

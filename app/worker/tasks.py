from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from billiard.exceptions import SoftTimeLimitExceeded
from celery import shared_task
from sqlalchemy import select

from app.db.models import (
    ConversationMessage,
    DailySummarySnapshot,
    MessageDirection,
    Reminder,
    ReminderStatus,
    Task,
    TaskStatus,
)
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

        # Persist inbound/state/outbound draft rows before transport send attempts.
        # This keeps admin/debug history durable even if Twilio sending fails.
        session.commit()

        sent = 0
        send_failures = 0
        for chunk in result.outgoing_messages:
            try:
                transport.send_sms(to_number=payload.from_number, body=chunk)
                sent += 1
            except Exception as send_exc:  # pragma: no cover
                send_failures += 1
                logger.exception(
                    "outbound sms send failed sid=%s to=%s error=%s",
                    payload.message_sid,
                    payload.from_number,
                    send_exc,
                )
        if send_failures:
            logger.warning(
                "inbound processed with partial outbound failure sid=%s sent=%s failed=%s",
                payload.message_sid,
                sent,
                send_failures,
            )
        return {
            "skipped_duplicate": False,
            "message_sid": payload.message_sid,
            "outgoing_count": sent,
            "send_failures": send_failures,
        }
    except SoftTimeLimitExceeded as exc:  # pragma: no cover
        session.rollback()
        logger.exception("inbound sms task timed out for sid=%s: %s", payload.message_sid, exc)
        fallback_body = "i got your text and logged it. tiny lag on my side, but keep going and i'll sync right after."
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
        fallback_body = "i got this and logged it. quick processing miss on my side, keep texting and i'll catch up."
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
        max_send_per_run = 2
        sent_normal = 0
        conversation_deferral_until = _conversation_deferral_until(session=session, user_id=user.id, now=now)
        items = []
        for reminder in reminders:
            task = session.execute(select(Task).where(Task.id == reminder.task_id)).scalars().first() if reminder.task_id else None
            items.append((reminder, task))
        items.sort(key=lambda item: _dispatch_sort_key(reminder=item[0], task=item[1], now=now))

        for reminder, task in items:
            if task and task.status == TaskStatus.completed:
                reminder.status = ReminderStatus.skipped
                skipped += 1
                continue
            if task and task.status not in (TaskStatus.active, TaskStatus.blocked):
                reminder.status = ReminderStatus.skipped
                skipped += 1
                continue

            if conversation_deferral_until is not None and conversation_deferral_until > now:
                _defer_reminder(
                    engine=engine,
                    session=session,
                    reminder=reminder,
                    user=user,
                    task=task,
                    candidate=conversation_deferral_until,
                )
                skipped += 1
                continue

            available_at = engine.next_available_time(
                session=session,
                user=user,
                candidate=now,
                now=now,
                task=task,
            )
            if available_at > now:
                _defer_reminder(
                    engine=engine,
                    session=session,
                    reminder=reminder,
                    user=user,
                    task=task,
                    candidate=available_at,
                )
                skipped += 1
                continue

            if sent >= max_send_per_run:
                _defer_reminder(
                    engine=engine,
                    session=session,
                    reminder=reminder,
                    user=user,
                    task=task,
                    candidate=now + timedelta(minutes=35 + (skipped * 5)),
                )
                skipped += 1
                continue

            urgent = _is_urgent_dispatch(task=task, reminder=reminder, now=now)
            if sent_normal >= 1 and not urgent:
                _defer_reminder(
                    engine=engine,
                    session=session,
                    reminder=reminder,
                    user=user,
                    task=task,
                    candidate=now + timedelta(minutes=25 + (skipped * 5)),
                )
                skipped += 1
                continue

            body = _compose_reminder_text(task=task, escalation=reminder.escalation_level)
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
            reminder.attempt_count += 1
            if task:
                task.last_reminder_at = now
                task.reminder_escalation_level = reminder.escalation_level
                task.reminder_pause_until = now + _post_send_pause(escalation=reminder.escalation_level)
            if not urgent:
                sent_normal += 1
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


def _compose_reminder_text(*, task: Task | None, escalation: int) -> str:
    title = _compact_task_title(task.title if task else "that thing")
    if escalation <= 0:
        return f"quick check on '{title}'?"
    if escalation == 1:
        if task and (task.status == TaskStatus.blocked or task.blocked_reason):
            return f"checking on '{title}'. what's the blocker?"
        if task and (task.slip_count > 0 or task.last_slip_reason):
            return f"checking on '{title}'. what happened?"
        return f"still on '{title}'?"
    if task and (task.status == TaskStatus.blocked or task.blocked_reason):
        return f"'{title}' is slipping. what's the blocker?"
    return f"'{title}' is slipping. what's the next concrete move?"


def _conversation_deferral_until(*, session, user_id, now: datetime) -> datetime | None:
    latest_inbound = (
        session.execute(
            select(ConversationMessage.created_at)
            .where(
                ConversationMessage.user_id == user_id,
                ConversationMessage.direction == MessageDirection.inbound,
            )
            .order_by(ConversationMessage.created_at.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    latest_outbound = (
        session.execute(
            select(ConversationMessage.created_at)
            .where(
                ConversationMessage.user_id == user_id,
                ConversationMessage.direction == MessageDirection.outbound,
            )
            .order_by(ConversationMessage.created_at.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if latest_inbound is not None:
        inbound_at = _ensure_datetime_tz(latest_inbound, now.tzinfo)
        if inbound_at >= now - timedelta(minutes=12):
            return inbound_at + timedelta(minutes=15)
    if latest_outbound is not None:
        outbound_at = _ensure_datetime_tz(latest_outbound, now.tzinfo)
        if outbound_at >= now - timedelta(minutes=4):
            return outbound_at + timedelta(minutes=8)
    return None


def _dispatch_sort_key(*, reminder: Reminder, task: Task | None, now: datetime) -> tuple:
    deadline_delta_seconds = float("inf")
    if task and task.deadline_at is not None:
        deadline = _ensure_datetime_tz(task.deadline_at, now.tzinfo)
        deadline_delta_seconds = (deadline - now).total_seconds()
    priority = task.priority if task else 0
    blocked = 0 if task and task.status == TaskStatus.blocked else 1
    return (-reminder.escalation_level, blocked, deadline_delta_seconds, -priority, reminder.scheduled_for)


def _is_urgent_dispatch(*, task: Task | None, reminder: Reminder, now: datetime) -> bool:
    if reminder.escalation_level >= 2:
        return True
    if task and task.deadline_at is not None:
        deadline = _ensure_datetime_tz(task.deadline_at, now.tzinfo)
        return deadline <= now + timedelta(minutes=90)
    return False


def _defer_reminder(*, engine: ReminderEngine, session, reminder: Reminder, user, task: Task | None, candidate: datetime) -> None:
    deferred_until = engine.next_available_time(
        session=session,
        user=user,
        candidate=candidate,
        task=task,
    )
    current_scheduled_for = _ensure_datetime_tz(reminder.scheduled_for, deferred_until.tzinfo)
    reminder.scheduled_for = max(current_scheduled_for, deferred_until)
    reminder.cooldown_until = reminder.scheduled_for
    if task:
        task.reminder_pause_until = reminder.scheduled_for


def _post_send_pause(*, escalation: int) -> timedelta:
    if escalation >= 2:
        return timedelta(minutes=100)
    if escalation == 1:
        return timedelta(minutes=75)
    return timedelta(minutes=55)


def _compact_task_title(task_title: str, max_chars: int = 72) -> str:
    title = " ".join(task_title.split()).strip()
    if len(title) <= max_chars:
        return title
    return f"{title[: max_chars - 3].rstrip()}..."


def _ensure_datetime_tz(value: datetime, tzinfo) -> datetime:
    if value.tzinfo is None and tzinfo is not None:
        return value.replace(tzinfo=tzinfo)
    return value

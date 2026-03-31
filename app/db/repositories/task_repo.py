from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.models import Reminder, ReminderStatus, Task, TaskStatus


def create_task(
    session: Session,
    *,
    user_id: uuid.UUID,
    title: str,
    description: str | None = None,
    project_id: uuid.UUID | None = None,
    parent_task_id: uuid.UUID | None = None,
    deadline_at: datetime | None = None,
    soft_deadline_at: datetime | None = None,
    priority: int = 2,
    extraction_confidence: float = 0.6,
    metadata_json: dict | None = None,
) -> Task:
    task = Task(
        user_id=user_id,
        title=title,
        description=description,
        project_id=project_id,
        parent_task_id=parent_task_id,
        deadline_at=deadline_at,
        soft_deadline_at=soft_deadline_at,
        priority=priority,
        extraction_confidence=extraction_confidence,
        metadata_json=metadata_json or {},
    )
    session.add(task)
    session.flush()
    return task


def find_active_task_by_title(session: Session, user_id: uuid.UUID, title_fragment: str) -> Task | None:
    fragment = f"%{title_fragment.lower()}%"
    stmt = (
        select(Task)
        .where(
            Task.user_id == user_id,
            Task.status.in_([TaskStatus.active, TaskStatus.blocked]),
            Task.title.ilike(fragment),
        )
        .order_by(Task.updated_at.desc())
    )
    return session.execute(stmt).scalars().first()


def list_active_tasks(session: Session, user_id: uuid.UUID) -> list[Task]:
    stmt = (
        select(Task)
        .where(Task.user_id == user_id, Task.status.in_([TaskStatus.active, TaskStatus.blocked]))
        .order_by(Task.deadline_at.asc().nulls_last(), Task.priority.desc(), Task.created_at.asc())
    )
    return list(session.execute(stmt).scalars().all())


def list_upcoming_deadlines(session: Session, user_id: uuid.UUID, within_days: int = 7) -> list[Task]:
    now = datetime.now(tz=timezone.utc)
    horizon = now + timedelta(days=within_days)
    stmt = (
        select(Task)
        .where(
            Task.user_id == user_id,
            Task.status.in_([TaskStatus.active, TaskStatus.blocked]),
            Task.deadline_at.is_not(None),
            Task.deadline_at <= horizon,
        )
        .order_by(Task.deadline_at.asc())
    )
    return list(session.execute(stmt).scalars().all())


def mark_task_complete(task: Task) -> None:
    task.status = TaskStatus.completed
    task.completed_at = datetime.now(tz=timezone.utc)


def create_reminder(
    session: Session,
    *,
    user_id: uuid.UUID,
    task_id: uuid.UUID | None,
    scheduled_for: datetime,
    kind: str = "checkin",
    escalation_level: int = 0,
    reason: str | None = None,
    payload: dict | None = None,
) -> Reminder:
    reminder = Reminder(
        user_id=user_id,
        task_id=task_id,
        kind=kind,
        scheduled_for=scheduled_for,
        escalation_level=escalation_level,
        reason=reason,
        payload=payload or {},
    )
    session.add(reminder)
    session.flush()
    return reminder


def list_due_pending_reminders(
    session: Session,
    *,
    user_id: uuid.UUID,
    now: datetime,
) -> list[Reminder]:
    stmt = (
        select(Reminder)
        .where(
            and_(
                Reminder.user_id == user_id,
                Reminder.status == ReminderStatus.pending,
                Reminder.scheduled_for <= now,
            )
        )
        .order_by(Reminder.scheduled_for.asc())
    )
    return list(session.execute(stmt).scalars().all())


def has_pending_reminder_within(
    session: Session,
    *,
    user_id: uuid.UUID,
    task_id: uuid.UUID | None,
    earliest: datetime,
    latest: datetime,
) -> bool:
    stmt = select(Reminder.id).where(
        Reminder.user_id == user_id,
        Reminder.task_id == task_id,
        Reminder.status == ReminderStatus.pending,
        Reminder.scheduled_for >= earliest,
        Reminder.scheduled_for <= latest,
    )
    return session.execute(stmt).first() is not None

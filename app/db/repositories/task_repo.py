from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.models import Reminder, ReminderStatus, Task, TaskDependency, TaskStatus


def create_task(
    session: Session,
    *,
    user_id: uuid.UUID,
    title: str,
    description: str | None = None,
    project_id: uuid.UUID | None = None,
    source_message_id: uuid.UUID | None = None,
    parent_task_id: uuid.UUID | None = None,
    next_step: str | None = None,
    deadline_at: datetime | None = None,
    soft_deadline_at: datetime | None = None,
    start_after: datetime | None = None,
    deadline_source_phrase: str | None = None,
    deadline_confidence: float = 0.0,
    deadline_is_ambiguous: bool = False,
    deadline_granularity: str = "unknown",
    deadline_timezone: str | None = None,
    priority: int = 2,
    started_at: datetime | None = None,
    last_progress_at: datetime | None = None,
    blocked_reason: str | None = None,
    blocked_at: datetime | None = None,
    blocker_details_json: dict | None = None,
    slip_count: int = 0,
    last_slipped_at: datetime | None = None,
    last_slip_reason: str | None = None,
    reminder_escalation_level: int = 0,
    last_reminder_at: datetime | None = None,
    reminder_pause_until: datetime | None = None,
    extraction_confidence: float = 0.6,
    metadata_json: dict | None = None,
) -> Task:
    task = Task(
        user_id=user_id,
        title=title,
        description=description,
        project_id=project_id,
        source_message_id=source_message_id,
        parent_task_id=parent_task_id,
        next_step=next_step,
        deadline_at=deadline_at,
        soft_deadline_at=soft_deadline_at,
        start_after=start_after,
        deadline_source_phrase=deadline_source_phrase,
        deadline_confidence=deadline_confidence,
        deadline_is_ambiguous=deadline_is_ambiguous,
        deadline_granularity=deadline_granularity,
        deadline_timezone=deadline_timezone,
        priority=priority,
        started_at=started_at,
        last_progress_at=last_progress_at,
        blocked_reason=blocked_reason,
        blocked_at=blocked_at,
        blocker_details_json=blocker_details_json or {},
        slip_count=slip_count,
        last_slipped_at=last_slipped_at,
        last_slip_reason=last_slip_reason,
        reminder_escalation_level=reminder_escalation_level,
        last_reminder_at=last_reminder_at,
        reminder_pause_until=reminder_pause_until,
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
    now = datetime.now(tz=timezone.utc)
    task.status = TaskStatus.completed
    task.completed_at = now
    task.last_progress_at = now
    task.blocked_at = None
    task.blocked_reason = None


def record_task_progress(
    task: Task,
    *,
    at: datetime | None = None,
    next_step: str | None = None,
) -> None:
    now = at or datetime.now(tz=timezone.utc)
    if task.started_at is None:
        task.started_at = now
    task.last_progress_at = now
    if next_step is not None:
        task.next_step = next_step


def set_task_blocked(
    task: Task,
    *,
    reason: str,
    blocker_details_json: dict | None = None,
    at: datetime | None = None,
) -> None:
    task.status = TaskStatus.blocked
    task.blocked_reason = reason
    task.blocked_at = at or datetime.now(tz=timezone.utc)
    if blocker_details_json is not None:
        task.blocker_details_json = blocker_details_json


def record_task_slip(
    task: Task,
    *,
    reason: str | None = None,
    at: datetime | None = None,
    next_step: str | None = None,
    escalation_level: int | None = None,
) -> None:
    now = at or datetime.now(tz=timezone.utc)
    task.slip_count += 1
    task.last_slipped_at = now
    task.last_slip_reason = reason
    if next_step is not None:
        task.next_step = next_step
    if escalation_level is not None:
        task.reminder_escalation_level = escalation_level


def update_task_reminder_state(
    task: Task,
    *,
    escalation_level: int | None = None,
    last_reminder_at: datetime | None = None,
    reminder_pause_until: datetime | None = None,
) -> None:
    if escalation_level is not None:
        task.reminder_escalation_level = escalation_level
    if last_reminder_at is not None:
        task.last_reminder_at = last_reminder_at
    if reminder_pause_until is not None:
        task.reminder_pause_until = reminder_pause_until


def create_task_dependency(
    session: Session,
    *,
    user_id: uuid.UUID,
    predecessor_task_id: uuid.UUID,
    successor_task_id: uuid.UUID,
    dependency_type: str = "finish_to_start",
    metadata_json: dict | None = None,
) -> TaskDependency:
    dependency = TaskDependency(
        user_id=user_id,
        predecessor_task_id=predecessor_task_id,
        successor_task_id=successor_task_id,
        dependency_type=dependency_type,
        metadata_json=metadata_json or {},
    )
    session.add(dependency)
    session.flush()
    return dependency


def create_reminder(
    session: Session,
    *,
    user_id: uuid.UUID,
    task_id: uuid.UUID | None,
    scheduled_for: datetime,
    kind: str = "checkin",
    escalation_level: int = 0,
    attempt_count: int = 0,
    cooldown_until: datetime | None = None,
    reason: str | None = None,
    payload: dict | None = None,
) -> Reminder:
    reminder = Reminder(
        user_id=user_id,
        task_id=task_id,
        kind=kind,
        scheduled_for=scheduled_for,
        escalation_level=escalation_level,
        attempt_count=attempt_count,
        cooldown_until=cooldown_until,
        reason=reason,
        payload=payload or {},
    )
    session.add(reminder)
    session.flush()
    if task_id is not None:
        task = session.get(Task, task_id)
        if task is not None:
            task.reminder_escalation_level = max(task.reminder_escalation_level, escalation_level)
            if cooldown_until is not None:
                task.reminder_pause_until = cooldown_until
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

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import PlanningNote, Reminder, ReminderStatus, ScheduleBlock, Task, TaskStatus
from app.db.repositories.task_repo import create_reminder, has_pending_reminder_within


class ReminderEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    def schedule_for_task(self, session: Session, task: Task, *, now: datetime | None = None) -> Reminder | None:
        now = now or datetime.now(tz=ZoneInfo(task.user.timezone if task.user else self.settings.timezone))
        if task.status not in (TaskStatus.active, TaskStatus.blocked):
            return None

        next_time = self._next_checkin_time(session=session, task=task, now=now)
        active_block_end = self._active_block_end(session, task.user_id, next_time)
        if active_block_end:
            next_time = max(next_time, active_block_end + timedelta(minutes=10))

        spacing_start = next_time - timedelta(minutes=self.settings.reminder_min_spacing_minutes)
        spacing_end = next_time + timedelta(minutes=self.settings.reminder_min_spacing_minutes)
        if has_pending_reminder_within(
            session,
            user_id=task.user_id,
            task_id=task.id,
            earliest=spacing_start,
            latest=spacing_end,
        ):
            return None

        reminder = create_reminder(
            session,
            user_id=task.user_id,
            task_id=task.id,
            scheduled_for=next_time,
            kind="checkin",
            reason="auto-schedule",
            escalation_level=0,
        )
        return reminder

    def _next_checkin_time(self, *, session: Session, task: Task, now: datetime) -> datetime:
        spacing = timedelta(minutes=self.settings.checkin_default_minutes)
        if task.deadline_at:
            deadline = self._ensure_tz(task.deadline_at, now.tzinfo or timezone.utc)
            delta = deadline - now
            if delta <= timedelta(hours=3):
                spacing = timedelta(minutes=25)
            elif delta <= timedelta(hours=12):
                spacing = timedelta(minutes=45)
            elif delta <= timedelta(days=1):
                spacing = timedelta(minutes=60)
        if task.priority >= 4:
            spacing = min(spacing, timedelta(minutes=40))
        if task.status == TaskStatus.blocked:
            spacing = min(spacing, timedelta(minutes=35))

        # Adapt cadence to observed friction while still avoiding spam.
        multiplier = self._behavior_multiplier(session=session, user_id=task.user_id, now=now)
        spacing = timedelta(seconds=max(20 * 60, int(spacing.total_seconds() * multiplier)))

        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = now.replace(hour=23, minute=59, second=59, microsecond=0)
        if self.daily_reminder_count(session, task.user_id, day_start, day_end) >= self.settings.reminder_max_per_day:
            spacing += timedelta(minutes=20)

        scheduled = now + spacing

        local_tz = ZoneInfo(task.user.timezone if task.user else self.settings.timezone)
        hour = scheduled.astimezone(local_tz).hour
        if self.settings.sleepy_hours_start <= hour < self.settings.sleepy_hours_end:
            scheduled = scheduled + timedelta(hours=(self.settings.sleepy_hours_end - hour))
        return scheduled

    @staticmethod
    def _behavior_multiplier(*, session: Session, user_id, now: datetime) -> float:
        lookback_start = now - timedelta(days=14)
        stmt = (
            select(PlanningNote)
            .where(
                PlanningNote.user_id == user_id,
                PlanningNote.created_at >= lookback_start,
                PlanningNote.note_type.in_(("slip_reason", "behavior_pattern")),
            )
            .order_by(PlanningNote.created_at.desc())
            .limit(20)
        )
        notes = list(session.execute(stmt).scalars().all())
        if not notes:
            return 1.0
        friction_tokens = ("underestimated", "distracted", "behind", "stuck", "context switching", "conflict")
        friction_hits = sum(1 for note in notes if any(token in note.content.lower() for token in friction_tokens))
        if friction_hits >= 6:
            return 0.72
        if friction_hits >= 3:
            return 0.85
        return 1.0

    @staticmethod
    def _ensure_tz(value: datetime, tz) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value

    @staticmethod
    def _active_block_end(session: Session, user_id, candidate_time: datetime) -> datetime | None:
        stmt = select(ScheduleBlock).where(
            ScheduleBlock.user_id == user_id,
            ScheduleBlock.starts_at <= candidate_time,
            ScheduleBlock.ends_at >= candidate_time,
        )
        block = session.execute(stmt).scalars().first()
        if not block:
            return None
        ends_at = block.ends_at
        if ends_at.tzinfo is None and candidate_time.tzinfo is not None:
            return ends_at.replace(tzinfo=candidate_time.tzinfo)
        return ends_at

    def due_reminders(self, session: Session, user_id, now: datetime) -> list[Reminder]:
        stmt = (
            select(Reminder)
            .where(
                Reminder.user_id == user_id,
                Reminder.status == ReminderStatus.pending,
                Reminder.scheduled_for <= now,
            )
            .order_by(Reminder.scheduled_for.asc())
        )
        return list(session.execute(stmt).scalars().all())

    def daily_reminder_count(self, session: Session, user_id, day_start: datetime, day_end: datetime) -> int:
        stmt = select(func.count(Reminder.id)).where(
            and_(
                Reminder.user_id == user_id,
                Reminder.created_at >= day_start,
                Reminder.created_at <= day_end,
            )
        )
        return int(session.execute(stmt).scalar_one())

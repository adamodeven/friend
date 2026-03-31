from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Reminder, ReminderStatus, ScheduleBlock, Task, TaskStatus
from app.db.repositories.task_repo import create_reminder, has_pending_reminder_within


class ReminderEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    def schedule_for_task(self, session: Session, task: Task, *, now: datetime | None = None) -> Reminder | None:
        now = now or datetime.now(tz=ZoneInfo(task.user.timezone if task.user else self.settings.timezone))
        if task.status not in (TaskStatus.active, TaskStatus.blocked):
            return None

        next_time = self._next_checkin_time(task=task, now=now)
        if self._is_in_block(session, task.user_id, next_time):
            next_time += timedelta(minutes=45)

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

    def _next_checkin_time(self, task: Task, now: datetime) -> datetime:
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
        scheduled = now + spacing

        hour = scheduled.astimezone(ZoneInfo(self.settings.timezone)).hour
        if self.settings.sleepy_hours_start <= hour < self.settings.sleepy_hours_end:
            scheduled = scheduled + timedelta(hours=(self.settings.sleepy_hours_end - hour))
        return scheduled

    @staticmethod
    def _ensure_tz(value: datetime, tz) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value

    @staticmethod
    def _is_in_block(session: Session, user_id, candidate_time: datetime) -> bool:
        stmt = select(ScheduleBlock.id).where(
            ScheduleBlock.user_id == user_id,
            ScheduleBlock.starts_at <= candidate_time,
            ScheduleBlock.ends_at >= candidate_time,
        )
        return session.execute(stmt).first() is not None

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

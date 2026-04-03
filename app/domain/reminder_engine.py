from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import PlanningNote, Reminder, ReminderStatus, ScheduleBlock, Task, TaskStatus
from app.db.repositories.task_repo import create_reminder, has_pending_reminder_within


class ReminderEngine:
    _MAX_ESCALATION_LEVEL = 2
    _QUEUE_SPACING_MINUTES = 18

    def __init__(self) -> None:
        self.settings = get_settings()

    def schedule_for_task(self, session: Session, task: Task, *, now: datetime | None = None) -> Reminder | None:
        now = now or datetime.now(tz=ZoneInfo(task.user.timezone if task.user else self.settings.timezone))
        if task.status not in (TaskStatus.active, TaskStatus.blocked):
            return None
        if self._has_pending_reminder(session=session, task_id=task.id, now=now):
            return None

        next_time = self._next_checkin_time(session=session, task=task, now=now)
        next_time = self.next_available_time(session=session, user=task.user, candidate=next_time, now=now, task=task)
        next_time = self._spread_user_queue(session=session, user_id=task.user_id, candidate=next_time)

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
        escalation_level = self._escalation_level(task=task, now=now)
        task.reminder_escalation_level = escalation_level

        reminder = create_reminder(
            session,
            user_id=task.user_id,
            task_id=task.id,
            scheduled_for=next_time,
            kind="checkin",
            reason=self._schedule_reason(task=task, escalation_level=escalation_level),
            escalation_level=escalation_level,
        )
        return reminder

    def _next_checkin_time(self, *, session: Session, task: Task, now: datetime) -> datetime:
        spacing = timedelta(minutes=self.settings.checkin_default_minutes)
        if task.deadline_at:
            deadline = self._ensure_tz(task.deadline_at, now.tzinfo or timezone.utc)
            delta = deadline - now
            if delta <= timedelta():
                spacing = timedelta(minutes=18)
            elif delta <= timedelta(hours=1):
                spacing = timedelta(minutes=20)
            elif delta <= timedelta(hours=3):
                spacing = timedelta(minutes=25)
            elif delta <= timedelta(hours=8):
                spacing = timedelta(minutes=35)
            elif delta <= timedelta(hours=12):
                spacing = timedelta(minutes=45)
            elif delta <= timedelta(days=1):
                spacing = timedelta(minutes=60)
        if task.deadline_is_ambiguous or task.deadline_confidence < 0.55:
            spacing += timedelta(minutes=15)
        if task.priority >= 4:
            spacing = min(spacing, timedelta(minutes=35))
        if task.status == TaskStatus.blocked:
            spacing = min(spacing, timedelta(minutes=35))
        if task.slip_count > 0 or task.last_slip_reason:
            spacing = min(spacing, timedelta(minutes=50))

        recent_progress_at = self._recent_progress_anchor(task=task, now=now)
        if recent_progress_at is not None:
            progress_gap = now - recent_progress_at
            if progress_gap <= timedelta(minutes=25):
                spacing = max(spacing, timedelta(minutes=75))
            elif progress_gap <= timedelta(minutes=90):
                spacing = max(spacing, timedelta(minutes=55))

        if task.last_reminder_at:
            last_reminder_at = self._ensure_tz(task.last_reminder_at, now.tzinfo or timezone.utc)
            if recent_progress_at is None or recent_progress_at <= last_reminder_at:
                unanswered_gap = now - last_reminder_at
                if unanswered_gap >= timedelta(hours=4):
                    spacing = min(spacing, timedelta(minutes=30))
                elif unanswered_gap >= timedelta(minutes=90):
                    spacing = min(spacing, timedelta(minutes=45))

        # Adapt cadence to observed friction while still avoiding spam.
        multiplier = self._behavior_multiplier(session=session, task=task, now=now)
        spacing = timedelta(seconds=max(20 * 60, int(spacing.total_seconds() * multiplier)))

        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = now.replace(hour=23, minute=59, second=59, microsecond=0)
        if self.daily_reminder_count(session, task.user_id, day_start, day_end) >= self.settings.reminder_max_per_day:
            spacing = max(spacing, timedelta(hours=3))

        scheduled = now + spacing
        return scheduled

    @staticmethod
    def _behavior_multiplier(*, session: Session, task: Task, now: datetime) -> float:
        lookback_start = now - timedelta(days=14)
        stmt = (
            select(PlanningNote)
            .where(
                PlanningNote.user_id == task.user_id,
                PlanningNote.created_at >= lookback_start,
                PlanningNote.note_type.in_(("slip_reason", "behavior_pattern")),
            )
            .order_by(PlanningNote.created_at.desc())
            .limit(20)
        )
        notes = list(session.execute(stmt).scalars().all())
        friction_tokens = ("underestimated", "distracted", "behind", "stuck", "context switching", "conflict")
        availability_tokens = ("class", "sleep", "driving", "meeting", "social", "family", "commute")
        multiplier = 1.0
        if notes:
            friction_hits = sum(1 for note in notes if any(token in note.content.lower() for token in friction_tokens))
            availability_hits = sum(1 for note in notes if any(token in note.content.lower() for token in availability_tokens))
            if friction_hits >= 6:
                multiplier *= 0.72
            elif friction_hits >= 3:
                multiplier *= 0.85
            if availability_hits >= 4:
                multiplier *= 1.18
            elif availability_hits >= 2:
                multiplier *= 1.08

        slip_text = (task.last_slip_reason or "").lower()
        if any(token in slip_text for token in ("distracted", "avoid", "forgot", "procrast")):
            multiplier *= 0.88
        if any(token in slip_text for token in ("class", "driving", "sleep", "meeting", "family", "social")):
            multiplier *= 1.15
        return multiplier

    def next_available_time(
        self,
        *,
        session: Session,
        user,
        candidate: datetime,
        now: datetime | None = None,
        task: Task | None = None,
    ) -> datetime:
        local_tz = ZoneInfo(user.timezone if user else self.settings.timezone)
        aligned = self._ensure_tz(candidate, local_tz)

        if task and task.start_after:
            start_after = self._ensure_tz(task.start_after, aligned.tzinfo or timezone.utc)
            aligned = max(aligned, start_after + timedelta(minutes=10))
        if task and task.reminder_pause_until:
            pause_until = self._ensure_tz(task.reminder_pause_until, aligned.tzinfo or timezone.utc)
            aligned = max(aligned, pause_until)

        while True:
            active_block = self._active_block(session, user.id, aligned)
            if active_block:
                block_end = self._ensure_tz(active_block.ends_at, aligned.tzinfo or timezone.utc)
                aligned = block_end + timedelta(
                    minutes=self._context_buffer_minutes(active_block.block_type, confidence=active_block.confidence)
                )
                continue
            if self._is_sleep_window(user=user, candidate=aligned) and not self._has_active_context(
                session,
                user_id=user.id,
                candidate=aligned,
                block_type="all_nighter",
            ):
                aligned = self._next_wake_time(user=user, candidate=aligned) + timedelta(minutes=15)
                continue
            break

        return aligned

    @staticmethod
    def _ensure_tz(value: datetime, tz) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value

    @staticmethod
    def _active_block(session: Session, user_id, candidate_time: datetime) -> ScheduleBlock | None:
        stmt = (
            select(ScheduleBlock)
            .where(
                ScheduleBlock.user_id == user_id,
                ScheduleBlock.starts_at <= candidate_time,
                ScheduleBlock.ends_at >= candidate_time,
            )
            .order_by(ScheduleBlock.confidence.desc(), ScheduleBlock.ends_at.asc())
        )
        return session.execute(stmt).scalars().first()

    def _has_active_context(self, session: Session, *, user_id, candidate: datetime, block_type: str) -> bool:
        stmt = select(ScheduleBlock.id).where(
            ScheduleBlock.user_id == user_id,
            ScheduleBlock.block_type == block_type,
            ScheduleBlock.starts_at <= candidate,
            ScheduleBlock.ends_at >= candidate,
        )
        return session.execute(stmt).first() is not None

    def _is_sleep_window(self, *, user, candidate: datetime) -> bool:
        profile = getattr(user, "profile", None)
        bedtime = profile.bedtime if profile and profile.bedtime else time(self.settings.sleepy_hours_start, 0)
        wake_time = profile.wake_time if profile and profile.wake_time else time(self.settings.sleepy_hours_end, 0)

        local_time = candidate.timetz().replace(tzinfo=None)
        if bedtime == wake_time:
            return False
        if bedtime < wake_time:
            return bedtime <= local_time < wake_time
        return local_time >= bedtime or local_time < wake_time

    def _next_wake_time(self, *, user, candidate: datetime) -> datetime:
        profile = getattr(user, "profile", None)
        bedtime = profile.bedtime if profile and profile.bedtime else time(self.settings.sleepy_hours_start, 0)
        wake_time = profile.wake_time if profile and profile.wake_time else time(self.settings.sleepy_hours_end, 0)

        local_time = candidate.timetz().replace(tzinfo=None)
        wake_today = candidate.replace(
            hour=wake_time.hour,
            minute=wake_time.minute,
            second=0,
            microsecond=0,
        )
        if bedtime < wake_time:
            if local_time < wake_time:
                return wake_today
            return wake_today + timedelta(days=1)
        if local_time >= bedtime:
            return wake_today + timedelta(days=1)
        return wake_today

    @staticmethod
    def _context_buffer_minutes(block_type: str, *, confidence: float) -> int:
        base = {
            "sleeping": 20,
            "in_class": 12,
            "driving": 18,
            "social_event": 30,
            "all_nighter": 30,
            "focused_sprint": 25,
            "busy": 15,
        }.get(block_type, 15)
        if confidence >= 0.9:
            return base + 5
        if confidence < 0.6 and block_type not in {"sleeping", "in_class", "driving"}:
            return max(10, base - 5)
        return base

    def _spread_user_queue(self, *, session: Session, user_id, candidate: datetime) -> datetime:
        spacing = timedelta(minutes=self._QUEUE_SPACING_MINUTES)
        aligned = candidate
        while self._has_user_reminder_near(session=session, user_id=user_id, candidate=aligned, spacing=spacing):
            aligned += spacing
        return aligned

    @staticmethod
    def _has_user_reminder_near(*, session: Session, user_id, candidate: datetime, spacing: timedelta) -> bool:
        stmt = select(Reminder.id).where(
            Reminder.user_id == user_id,
            Reminder.status == ReminderStatus.pending,
            Reminder.scheduled_for >= candidate - spacing,
            Reminder.scheduled_for <= candidate + spacing,
        )
        return session.execute(stmt).first() is not None

    @staticmethod
    def _has_pending_reminder(*, session: Session, task_id, now: datetime) -> bool:
        stmt = select(Reminder.id).where(
            Reminder.task_id == task_id,
            Reminder.status == ReminderStatus.pending,
            Reminder.scheduled_for >= now - timedelta(days=2),
        )
        return session.execute(stmt).first() is not None

    def _escalation_level(self, *, task: Task, now: datetime) -> int:
        if task.last_progress_at and task.last_reminder_at:
            last_progress_at = self._ensure_tz(task.last_progress_at, now.tzinfo or timezone.utc)
            last_reminder_at = self._ensure_tz(task.last_reminder_at, now.tzinfo or timezone.utc)
            if last_progress_at > last_reminder_at:
                return 0

        recent_progress_at = self._recent_progress_anchor(task=task, now=now)
        if recent_progress_at is not None and now - recent_progress_at <= timedelta(minutes=45):
            return 0

        escalation = 0
        if task.status == TaskStatus.blocked or task.blocked_reason:
            escalation = max(escalation, 1)
        if task.slip_count >= 1 or task.last_slip_reason:
            escalation = max(escalation, 1)
        if task.slip_count >= 2:
            escalation = max(escalation, 2)

        if task.last_reminder_at:
            last_reminder_at = self._ensure_tz(task.last_reminder_at, now.tzinfo or timezone.utc)
            if recent_progress_at is None or recent_progress_at <= last_reminder_at:
                unanswered_gap = now - last_reminder_at
                if unanswered_gap >= timedelta(hours=4):
                    escalation = max(escalation, 2)
                elif unanswered_gap >= timedelta(minutes=90):
                    escalation = max(escalation, 1)

        if task.deadline_at:
            deadline = self._ensure_tz(task.deadline_at, now.tzinfo or timezone.utc)
            if deadline <= now - timedelta(minutes=30):
                escalation = max(escalation, 2)
            elif deadline <= now + timedelta(hours=2) and recent_progress_at is None:
                escalation = max(escalation, 1)

        if task.reminder_escalation_level and recent_progress_at is None:
            escalation = max(escalation, min(task.reminder_escalation_level, self._MAX_ESCALATION_LEVEL))
        return min(escalation, self._MAX_ESCALATION_LEVEL)

    @staticmethod
    def _recent_progress_anchor(*, task: Task, now: datetime) -> datetime | None:
        candidates = []
        for value in (task.last_progress_at, task.started_at):
            if value is not None:
                candidates.append(value if value.tzinfo is not None else value.replace(tzinfo=now.tzinfo))
        if not candidates:
            return None
        return max(candidates)

    @staticmethod
    def _schedule_reason(*, task: Task, escalation_level: int) -> str:
        if escalation_level >= 2:
            return "auto-schedule-slip-warning"
        if escalation_level == 1:
            return "auto-schedule-blocker-check"
        if task.deadline_at is not None:
            return "auto-schedule-deadline-check"
        return "auto-schedule-checkin"

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

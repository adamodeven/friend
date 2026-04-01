from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Task, TaskStatus


class TimelineService:
    def build_today_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        end = now.replace(hour=23, minute=59, second=59, microsecond=0)
        tasks = self._tasks_due_between(session, user_id, now, end)
        if tasks:
            lines = [f"today ({len(tasks)} due):"]
            for t in tasks[:8]:
                due = t.deadline_at.astimezone(ZoneInfo(timezone)).strftime("%-I:%M%p").lower() if t.deadline_at else "no time"
                lines.append(f"- {t.title} ({due})")
            return "\n".join(lines)

        active = self._top_active_tasks(session, user_id, limit=4)
        if not active:
            return "today is clear right now. we can pick one high-impact move."
        lines = ["today has no hard due times in the system. priority stack:"]
        for t in active:
            lines.append(f"- {t.title}")
        return "\n".join(lines)

    def build_week_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        end = now + timedelta(days=7)
        tasks = self._tasks_due_between(session, user_id, now, end)
        if not tasks:
            return "week is light in the system rn. ping me anything new and i'll slot it."
        lines = [f"this week ({len(tasks)}):"]
        for t in tasks[:12]:
            due = t.deadline_at.astimezone(ZoneInfo(timezone)).strftime("%a %-m/%-d %-I:%M%p").lower() if t.deadline_at else "flex"
            lines.append(f"- {t.title} ({due})")
        return "\n".join(lines)

    def build_tonight_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        tonight_end = now.replace(hour=23, minute=59, second=59, microsecond=0)
        overdue = self._tasks_due_between(session, user_id, now - timedelta(days=7), now)
        due_tonight = self._tasks_due_between(session, user_id, now, tonight_end)
        combined = [*overdue[:2], *due_tonight]
        if not combined:
            active = self._top_active_tasks(session, user_id, limit=3)
            if not active:
                return "tonight is clear right now."
            lines = ["tonight focus:"]
            for task in active:
                lines.append(f"- {task.title}")
            return "\n".join(lines)

        lines = [f"tonight ({len(combined)} priority items):"]
        for task in combined[:6]:
            due = task.deadline_at.astimezone(ZoneInfo(timezone)).strftime("%-I:%M%p").lower() if task.deadline_at else "no time"
            lines.append(f"- {task.title} ({due})")
        return "\n".join(lines)

    def build_tomorrow_morning_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        tomorrow = now + timedelta(days=1)
        start = tomorrow.replace(hour=6, minute=0, second=0, microsecond=0)
        end = tomorrow.replace(hour=12, minute=0, second=0, microsecond=0)
        tasks = self._tasks_due_between(session, user_id, start, end)
        if not tasks:
            return "tomorrow morning is open in the system right now."
        lines = [f"tomorrow morning ({len(tasks)}):"]
        for task in tasks[:8]:
            due = task.deadline_at.astimezone(ZoneInfo(timezone)).strftime("%-I:%M%p").lower() if task.deadline_at else "no time"
            lines.append(f"- {task.title} ({due})")
        return "\n".join(lines)

    def build_weekend_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        weekday = now.weekday()
        days_until_sat = (5 - weekday) % 7
        saturday = (now + timedelta(days=days_until_sat)).replace(hour=0, minute=0, second=0, microsecond=0)
        sunday_end = (saturday + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
        tasks = self._tasks_due_between(session, user_id, saturday, sunday_end)
        if not tasks:
            return "weekend is clear in the system right now."
        lines = [f"this weekend ({len(tasks)}):"]
        for task in tasks[:10]:
            due = task.deadline_at.astimezone(ZoneInfo(timezone)).strftime("%a %-I:%M%p").lower() if task.deadline_at else "flex"
            lines.append(f"- {task.title} ({due})")
        return "\n".join(lines)

    def next_hour_recommendation(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        stmt = (
            select(Task)
            .where(
                Task.user_id == user_id,
                Task.status.in_([TaskStatus.active, TaskStatus.blocked]),
            )
            .order_by(Task.priority.desc(), Task.deadline_at.asc().nulls_last(), Task.created_at.asc())
        )
        tasks = list(session.execute(stmt).scalars().all())
        if not tasks:
            return "you're clear rn. drop the next task + deadline and i'll lock in the plan."
        top = tasks[0]
        if top.status == TaskStatus.blocked:
            return f"next hour move: unblock '{top.title}' first so downstream stuff can start."
        if top.deadline_at and top.deadline_at - now < timedelta(hours=8):
            return f"next hour move: push hard on '{top.title}' because it's due soon."
        return f"next hour move: ship a concrete chunk of '{top.title}' and text me when it's done."

    @staticmethod
    def _tasks_due_between(session: Session, user_id, start: datetime, end: datetime) -> list[Task]:
        stmt = (
            select(Task)
            .where(
                Task.user_id == user_id,
                Task.status.in_([TaskStatus.active, TaskStatus.blocked]),
                Task.deadline_at.is_not(None),
                Task.deadline_at >= start,
                Task.deadline_at <= end,
            )
            .order_by(Task.deadline_at.asc())
        )
        return list(session.execute(stmt).scalars().all())

    @staticmethod
    def _top_active_tasks(session: Session, user_id, limit: int) -> list[Task]:
        stmt = (
            select(Task)
            .where(
                Task.user_id == user_id,
                Task.status.in_([TaskStatus.active, TaskStatus.blocked]),
            )
            .order_by(Task.priority.desc(), Task.deadline_at.asc().nulls_last(), Task.created_at.asc())
            .limit(limit)
        )
        return list(session.execute(stmt).scalars().all())

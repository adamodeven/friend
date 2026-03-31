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
        if not tasks:
            return "today is clear right now. we can pick one high-impact move."
        lines = [f"today ({len(tasks)}):"]
        for t in tasks[:8]:
            due = t.deadline_at.astimezone(ZoneInfo(timezone)).strftime("%-I:%M%p").lower() if t.deadline_at else "no time"
            lines.append(f"- {t.title} ({due})")
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
            return "clean slate. let's add the next task you care about."
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


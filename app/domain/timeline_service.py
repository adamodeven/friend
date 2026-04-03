from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Task, TaskDependency, TaskStatus


@dataclass(slots=True)
class RankedTask:
    task: Task
    score: int
    actionable: bool
    blocked_by: tuple[str, ...]
    unlocks: tuple[str, ...]
    due_at: datetime | None
    due_label: str | None


class TimelineService:
    def build_today_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        end = now.replace(hour=23, minute=59, second=59, microsecond=0)
        ranked = self._ranked_tasks(session, user_id, timezone, horizon=end)
        if not ranked:
            return "today is clear right now. we can pick one high-impact move."
        return self._render_plan("today plan", ranked[:5], timezone)

    def build_week_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        end = now + timedelta(days=7)
        ranked = self._ranked_tasks(session, user_id, timezone, horizon=end)
        if not ranked:
            return "week is light in the system rn. ping me anything new and i'll slot it."
        return self._render_plan("this week plan", ranked[:6], timezone)

    def build_tonight_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        tonight_end = now.replace(hour=23, minute=59, second=59, microsecond=0)
        ranked = self._ranked_tasks(session, user_id, timezone, horizon=tonight_end)
        if not ranked:
            return "tonight is clear right now."
        return self._render_plan("tonight plan", ranked[:5], timezone)

    def build_tomorrow_morning_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        tomorrow = now + timedelta(days=1)
        start = tomorrow.replace(hour=6, minute=0, second=0, microsecond=0)
        end = tomorrow.replace(hour=12, minute=0, second=0, microsecond=0)
        ranked = self._ranked_tasks(session, user_id, timezone, horizon=end)
        filtered = [rank for rank in ranked if self._is_due_between(rank.task, start, end) or rank.unlocks]
        if not filtered:
            return "tomorrow morning is open in the system right now."
        return self._render_plan("tomorrow morning plan", filtered[:5], timezone)

    def build_weekend_view(self, session: Session, user_id, timezone: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone))
        weekday = now.weekday()
        days_until_sat = (5 - weekday) % 7
        saturday = (now + timedelta(days=days_until_sat)).replace(hour=0, minute=0, second=0, microsecond=0)
        sunday_end = (saturday + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
        ranked = self._ranked_tasks(session, user_id, timezone, horizon=sunday_end)
        filtered = [rank for rank in ranked if self._is_due_between(rank.task, saturday, sunday_end) or rank.unlocks]
        if not filtered:
            return "weekend is clear in the system right now."
        return self._render_plan("weekend plan", filtered[:5], timezone)

    def next_hour_recommendation(self, session: Session, user_id, timezone: str) -> str:
        ranked = self._ranked_tasks(session, user_id, timezone, horizon=datetime.now(tz=ZoneInfo(timezone)) + timedelta(hours=8))
        if not ranked:
            return "no active tasks are tracked right now. send the next thing you want handled and i'll slot it."
        top = ranked[0]
        move = top.task.next_step or self._default_move_text(top.task.title)
        if top.unlocks:
            return f"next hour move: {move}. that unlocks {top.unlocks[0]}."
        if top.blocked_by:
            return f"next hour move: clear {top.blocked_by[0]} first so {top.task.title} can move."
        if top.due_label:
            return f"next hour move: {move}. keep it tight because it's due {top.due_label}."
        return f"next hour move: {move}."

    def recommend_next_task(self, session: Session, user_id, timezone: str) -> Task | None:
        ranked = self._ranked_tasks(session, user_id, timezone, horizon=datetime.now(tz=ZoneInfo(timezone)) + timedelta(days=7))
        return ranked[0].task if ranked else None

    def _ranked_tasks(self, session: Session, user_id, timezone: str, *, horizon: datetime) -> list[RankedTask]:
        now = datetime.now(tz=ZoneInfo(timezone))
        tasks = self._active_tasks(session, user_id)
        if not tasks:
            return []

        task_by_id = {task.id: task for task in tasks}
        dependency_stmt = select(TaskDependency).where(
            TaskDependency.user_id == user_id,
            TaskDependency.predecessor_task_id.in_(tuple(task_by_id)),
            TaskDependency.successor_task_id.in_(tuple(task_by_id)),
        )
        dependencies = list(session.execute(dependency_stmt).scalars().all())

        predecessors: dict = {task_id: [] for task_id in task_by_id}
        successors: dict = {task_id: [] for task_id in task_by_id}
        for edge in dependencies:
            predecessors[edge.successor_task_id].append(task_by_id[edge.predecessor_task_id])
            successors[edge.predecessor_task_id].append(task_by_id[edge.successor_task_id])

        active_subtask_count: dict = {task_id: 0 for task_id in task_by_id}
        for task in tasks:
            if task.parent_task_id in active_subtask_count and task.status in {TaskStatus.active, TaskStatus.blocked}:
                active_subtask_count[task.parent_task_id] += 1

        ranked: list[RankedTask] = []
        for task in tasks:
            unresolved = [pred for pred in predecessors[task.id] if pred.status != TaskStatus.completed]
            unlocks = [succ for succ in successors[task.id] if succ.status in {TaskStatus.active, TaskStatus.blocked}]
            actionable = task.status == TaskStatus.active and not unresolved

            score = task.priority * 18
            due_at = self._ensure_tz(task.deadline_at, now.tzinfo) if task.deadline_at else None
            due_label: str | None = None
            if due_at is not None:
                delta = due_at - now
                if delta <= timedelta(0):
                    score += 420
                    due_label = "now"
                elif delta <= timedelta(hours=2):
                    score += 360
                    due_label = due_at.astimezone(ZoneInfo(timezone)).strftime("%-I:%M%p").lower()
                elif delta <= timedelta(hours=8):
                    score += 310
                    due_label = due_at.astimezone(ZoneInfo(timezone)).strftime("%-I:%M%p").lower()
                elif delta <= timedelta(days=1):
                    score += 250
                    due_label = due_at.astimezone(ZoneInfo(timezone)).strftime("%-I:%M%p").lower()
                elif delta <= timedelta(days=3):
                    score += 180
                    due_label = due_at.astimezone(ZoneInfo(timezone)).strftime("%a %-m/%-d").lower()
                elif delta <= timedelta(days=7):
                    score += 120
                    due_label = due_at.astimezone(ZoneInfo(timezone)).strftime("%a %-m/%-d").lower()
                elif due_at <= horizon:
                    score += 80
                    due_label = due_at.astimezone(ZoneInfo(timezone)).strftime("%a %-m/%-d").lower()
            else:
                score += 45

            if actionable:
                score += 35
            if task.status == TaskStatus.blocked:
                score -= 105
            if unresolved:
                score -= 140 + (len(unresolved) * 15)

            unlock_bonus = 0
            unlock_titles: list[str] = []
            for successor in unlocks:
                unlock_titles.append(successor.title)
                unlock_bonus += 120
                successor_due = self._ensure_tz(successor.deadline_at, now.tzinfo) if successor.deadline_at else None
                if successor_due is not None and successor_due <= horizon:
                    unlock_bonus += 140
            score += unlock_bonus

            if active_subtask_count[task.id] > 0:
                score -= 50
            if task.parent_task_id is not None:
                score += 15
            if task.slip_count > 0:
                score += min(task.slip_count, 3) * 8

            ranked.append(
                RankedTask(
                    task=task,
                    score=score,
                    actionable=actionable,
                    blocked_by=tuple(pred.title for pred in unresolved[:2]),
                    unlocks=tuple(unlock_titles[:2]),
                    due_at=due_at,
                    due_label=due_label,
                )
            )

        ranked.sort(
            key=lambda rank: (
                -rank.score,
                0 if rank.actionable else 1,
                rank.due_at or datetime.max.replace(tzinfo=now.tzinfo),
                self._ensure_tz(rank.task.created_at, now.tzinfo),
            )
        )
        return ranked

    @staticmethod
    def _render_plan(label: str, ranked: list[RankedTask], timezone: str) -> str:
        lines = [f"{label}:"]
        for index, rank in enumerate(ranked, start=1):
            bits: list[str] = []
            if rank.due_label:
                prefix = "overdue" if rank.due_label == "now" and rank.due_at and rank.due_at <= datetime.now(tz=ZoneInfo(timezone)) else "due"
                bits.append(f"{prefix} {rank.due_label}")
            if rank.unlocks:
                bits.append(f"unlocks {rank.unlocks[0]}")
            if not rank.actionable and rank.blocked_by:
                bits.append(f"blocked by {rank.blocked_by[0]}")
            elif rank.task.status == TaskStatus.blocked and rank.task.blocked_reason:
                bits.append(f"blocked: {rank.task.blocked_reason[:48]}")
            line = f"{index}. {rank.task.title}"
            if bits:
                line = f"{line} - {'; '.join(bits)}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _active_tasks(session: Session, user_id) -> list[Task]:
        stmt = (
            select(Task)
            .where(Task.user_id == user_id, Task.status.in_([TaskStatus.active, TaskStatus.blocked]))
            .order_by(Task.created_at.asc())
        )
        return list(session.execute(stmt).scalars().all())

    @staticmethod
    def _ensure_tz(value: datetime | None, tzinfo) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=tzinfo)
        return value

    @staticmethod
    def _is_due_between(task: Task, start: datetime, end: datetime) -> bool:
        if task.deadline_at is None:
            return False
        due_at = task.deadline_at if task.deadline_at.tzinfo else task.deadline_at.replace(tzinfo=start.tzinfo)
        return start <= due_at <= end

    @staticmethod
    def _default_move_text(task_title: str) -> str:
        return f"make a concrete dent in {task_title}"

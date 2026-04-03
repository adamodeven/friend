from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import PlanningNote, Task, TaskStatus, UserProfile


class ProfileMemoryService:
    def get_profile_snapshot(self, session: Session, user_id) -> dict:
        profile = session.execute(select(UserProfile).where(UserProfile.user_id == user_id)).scalars().first()
        notes = (
            session.execute(
                select(PlanningNote)
                .where(PlanningNote.user_id == user_id)
                .order_by(PlanningNote.created_at.desc())
                .limit(16)
            )
            .scalars()
            .all()
        )
        blocked_tasks = (
            session.execute(
                select(Task)
                .where(Task.user_id == user_id, Task.status == TaskStatus.blocked)
                .order_by(Task.updated_at.desc())
                .limit(6)
            )
            .scalars()
            .all()
        )
        active_tasks = (
            session.execute(
                select(Task)
                .where(Task.user_id == user_id, Task.status.in_((TaskStatus.active, TaskStatus.blocked)))
                .order_by(Task.deadline_at.asc().nulls_last(), Task.priority.desc(), Task.created_at.asc())
                .limit(8)
            )
            .scalars()
            .all()
        )

        behavior_patterns = [note.content for note in notes if note.note_type == "behavior_pattern"]
        slip_reasons = [note.content for note in notes if note.note_type == "slip_reason"]

        return {
            "style": profile.style.value if profile else "casual_cool",
            "planning_preferences": profile.planning_preferences if profile else {},
            "baseline_profile_text": self._baseline_profile_text(),
            "baseline_highlights": self._baseline_highlights(),
            "recent_patterns": [note.content for note in notes if note.note_type in {"behavior_pattern", "slip_reason"}],
            "recent_behavior_patterns": behavior_patterns[:8],
            "recent_slip_reasons": slip_reasons[:8],
            "active_blockers": [
                {
                    "task_title": task.title,
                    "blocked_reason": task.blocked_reason,
                    "dependency_titles": (task.blocker_details_json or {}).get("dependency_titles", []),
                }
                for task in blocked_tasks
            ],
            "active_commitments": [
                {
                    "title": task.title,
                    "next_step": task.next_step,
                    "deadline_at": task.deadline_at.isoformat() if task.deadline_at else None,
                    "status": task.status.value,
                }
                for task in active_tasks
            ],
        }

    @staticmethod
    def _baseline_profile_path() -> Path:
        return Path(__file__).resolve().parents[2] / "USER_PROFILE.md"

    def _baseline_profile_text(self) -> str:
        path = self._baseline_profile_path()
        if not path.exists():
            return ""
        return path.read_text().strip()

    def _baseline_highlights(self) -> list[str]:
        text = self._baseline_profile_text()
        if not text:
            return []
        highlights: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.lower().startswith("file:"):
                continue
            if line.startswith("-"):
                highlights.append(line.lstrip("-").strip())
            elif line.endswith(".") and len(highlights) < 3:
                highlights.append(line)
            if len(highlights) >= 8:
                break
        return highlights

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import PlanningNote, UserProfile


class ProfileMemoryService:
    def get_profile_snapshot(self, session: Session, user_id) -> dict:
        profile = session.execute(select(UserProfile).where(UserProfile.user_id == user_id)).scalars().first()
        notes = (
            session.execute(
                select(PlanningNote)
                .where(PlanningNote.user_id == user_id)
                .order_by(PlanningNote.created_at.desc())
                .limit(10)
            )
            .scalars()
            .all()
        )
        return {
            "style": profile.style.value if profile else "casual_cool",
            "planning_preferences": profile.planning_preferences if profile else {},
            "recent_patterns": [n.content for n in notes if n.note_type in {"behavior_pattern", "slip_reason"}],
        }


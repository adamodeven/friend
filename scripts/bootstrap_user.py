from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import UserProfile
from app.db.repositories.user_repo import get_or_create_primary_user
from app.db.session import SessionLocal


def main() -> None:
    settings = get_settings()
    session = SessionLocal()
    try:
        user = get_or_create_primary_user(session)
        profile = session.execute(select(UserProfile).where(UserProfile.user_id == user.id)).scalars().first()
        profile_path = Path("USER_PROFILE.md")
        bio = profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
        if profile:
            profile.bio = bio
            prefs = dict(profile.planning_preferences or {})
            prefs.update(
                {
                    "communication": "concise, sharp, low fluff",
                    "pressure_style": "real urgency, no fake hype",
                    "class_context_aware": True,
                }
            )
            profile.planning_preferences = prefs
        session.commit()
        print(f"bootstrapped user: {user.phone_number}")
    finally:
        session.close()


if __name__ == "__main__":
    main()


from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ProfileStyle, User, UserProfile


def get_or_create_primary_user(session: Session) -> User:
    settings = get_settings()
    phone = settings.user_phone_number or settings.twilio_to_number
    if not phone:
        phone = "+10000000000"

    user = session.execute(select(User).where(User.phone_number == phone)).scalar_one_or_none()
    if user:
        return user

    user = User(phone_number=phone, name=settings.user_name, timezone=settings.timezone)
    session.add(user)
    session.flush()
    try:
        style = ProfileStyle(settings.default_style)
    except ValueError:
        style = ProfileStyle.casual_cool
    profile = UserProfile(
        user_id=user.id,
        style=style,
        planning_preferences={"source": "bootstrap"},
    )
    session.add(profile)
    session.flush()
    return user


def get_user_by_phone(session: Session, phone_number: str) -> User | None:
    return session.execute(select(User).where(User.phone_number == phone_number)).scalar_one_or_none()

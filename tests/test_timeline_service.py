from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import User
from app.db.repositories.task_repo import create_task
from app.domain.timeline_service import TimelineService


def test_today_view_falls_back_to_priority_stack_when_no_due_tasks(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Website polish", priority=5)
    create_task(db_session, user_id=user.id, title="Portfolio cleanup", priority=4)
    db_session.commit()

    service = TimelineService()
    text = service.build_today_view(db_session, user.id, user.timezone)
    lowered = text.lower()
    assert "priority stack" in lowered
    assert "website polish" in lowered


def test_tomorrow_morning_view_lists_due_items_in_window(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    tomorrow_morning = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
    create_task(db_session, user_id=user.id, title="Submit scout app", deadline_at=tomorrow_morning, priority=5)
    db_session.commit()

    service = TimelineService()
    text = service.build_tomorrow_morning_view(db_session, user.id, user.timezone)
    lowered = text.lower()
    assert "tomorrow morning" in lowered
    assert "submit scout app" in lowered


def test_weekend_view_handles_empty_case(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    service = TimelineService()
    text = service.build_weekend_view(db_session, user.id, user.timezone)
    assert "weekend is clear" in text.lower()

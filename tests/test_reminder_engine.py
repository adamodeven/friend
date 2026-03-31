from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import Reminder, Task, User
from app.db.repositories.task_repo import create_task
from app.domain.reminder_engine import ReminderEngine


def test_schedule_for_task_creates_pending_reminder(db_session):
    user = db_session.execute(select(User)).scalars().first()
    task = create_task(db_session, user_id=user.id, title="Do assignment", deadline_at=datetime.now(tz=ZoneInfo(user.timezone)) + timedelta(hours=5))
    db_session.commit()

    engine = ReminderEngine()
    reminder = engine.schedule_for_task(db_session, task, now=datetime.now(tz=ZoneInfo(user.timezone)))
    db_session.commit()

    assert reminder is not None
    reminders = db_session.execute(select(Reminder)).scalars().all()
    assert len(reminders) == 1


def test_schedule_does_not_duplicate_within_spacing(db_session):
    user = db_session.execute(select(User)).scalars().first()
    task = create_task(db_session, user_id=user.id, title="Website fix")
    db_session.commit()

    engine = ReminderEngine()
    now = datetime.now(tz=ZoneInfo(user.timezone))
    first = engine.schedule_for_task(db_session, task, now=now)
    second = engine.schedule_for_task(db_session, task, now=now)
    db_session.commit()

    assert first is not None
    assert second is None


from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from app.db.models import Reminder, ScheduleBlock, User
from app.db.repositories.task_repo import create_task
from app.domain.reminder_engine import ReminderEngine


def _fixed_now(user: User, *, hour: int = 13, minute: int = 0) -> datetime:
    return datetime(2026, 4, 3, hour, minute, tzinfo=ZoneInfo(user.timezone))


def test_schedule_for_task_creates_pending_reminder(db_session):
    user = db_session.execute(select(User)).scalars().first()
    now = _fixed_now(user)
    task = create_task(db_session, user_id=user.id, title="Do assignment", deadline_at=now + timedelta(hours=5))
    db_session.commit()

    engine = ReminderEngine()
    reminder = engine.schedule_for_task(db_session, task, now=now)
    db_session.commit()

    assert reminder is not None
    reminders = db_session.execute(select(Reminder)).scalars().all()
    assert len(reminders) == 1


def test_schedule_does_not_duplicate_within_spacing(db_session):
    user = db_session.execute(select(User)).scalars().first()
    task = create_task(db_session, user_id=user.id, title="Website fix")
    db_session.commit()

    engine = ReminderEngine()
    now = _fixed_now(user)
    first = engine.schedule_for_task(db_session, task, now=now)
    second = engine.schedule_for_task(db_session, task, now=now)
    db_session.commit()

    assert first is not None
    assert second is None


@pytest.mark.parametrize(
    ("block_type", "expected_buffer_minutes"),
    [
        ("in_class", 12),
        ("driving", 18),
        ("social_event", 30),
        ("focused_sprint", 25),
        ("all_nighter", 30),
    ],
)
def test_schedule_respects_active_schedule_block_end(db_session, block_type: str, expected_buffer_minutes: int):
    user = db_session.execute(select(User)).scalars().first()
    now = _fixed_now(user)
    task = create_task(db_session, user_id=user.id, title="Write outline", deadline_at=now + timedelta(hours=2))
    db_session.flush()
    block = ScheduleBlock(
        user_id=user.id,
        block_type=block_type,
        starts_at=now,
        ends_at=now + timedelta(minutes=50),
        confidence=0.7,
        notes=f"{block_type} rn",
    )
    db_session.add(block)
    db_session.commit()

    engine = ReminderEngine()
    reminder = engine.schedule_for_task(db_session, task, now=now)
    assert reminder is not None
    block_end = block.ends_at
    if block_end.tzinfo is None:
        block_end = block_end.replace(tzinfo=now.tzinfo)
    assert reminder.scheduled_for >= block_end + timedelta(minutes=expected_buffer_minutes)


def test_schedule_uses_profile_sleep_window(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user.profile is not None
    user.profile.bedtime = time(23, 0)
    user.profile.wake_time = time(7, 0)
    now = _fixed_now(user, hour=23, minute=30)
    task = create_task(db_session, user_id=user.id, title="Prep notes for studio")
    db_session.commit()

    engine = ReminderEngine()
    reminder = engine.schedule_for_task(db_session, task, now=now)

    assert reminder is not None
    assert reminder.scheduled_for == datetime(2026, 4, 4, 7, 15, tzinfo=ZoneInfo(user.timezone))


def test_schedule_advances_escalation_after_unanswered_checkin(db_session):
    user = db_session.execute(select(User)).scalars().first()
    now = _fixed_now(user)
    task = create_task(
        db_session,
        user_id=user.id,
        title="Finish portfolio deck",
        deadline_at=now + timedelta(hours=4),
        last_progress_at=now - timedelta(hours=6),
        last_reminder_at=now - timedelta(hours=5),
        reminder_escalation_level=1,
    )
    db_session.commit()

    engine = ReminderEngine()
    reminder = engine.schedule_for_task(db_session, task, now=now)

    assert reminder is not None
    assert reminder.escalation_level == 2
    assert task.reminder_escalation_level == 2


def test_schedule_staggers_multiple_active_tasks(db_session):
    user = db_session.execute(select(User)).scalars().first()
    now = _fixed_now(user)
    tasks = [
        create_task(db_session, user_id=user.id, title="Task one", deadline_at=now + timedelta(hours=5)),
        create_task(db_session, user_id=user.id, title="Task two", deadline_at=now + timedelta(hours=5)),
        create_task(db_session, user_id=user.id, title="Task three", deadline_at=now + timedelta(hours=5)),
    ]
    db_session.commit()

    engine = ReminderEngine()
    for task in tasks:
        assert engine.schedule_for_task(db_session, task, now=now) is not None

    reminders = db_session.execute(select(Reminder).order_by(Reminder.scheduled_for.asc())).scalars().all()
    assert len(reminders) == 3
    assert reminders[1].scheduled_for - reminders[0].scheduled_for >= timedelta(minutes=18)
    assert reminders[2].scheduled_for - reminders[1].scheduled_for >= timedelta(minutes=18)

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import Task, TaskStatus, User
from app.db.repositories.task_repo import create_task, create_task_dependency
from app.domain.timeline_service import TimelineService


def test_today_view_orders_plan_when_no_due_tasks(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Website polish", priority=5)
    create_task(db_session, user_id=user.id, title="Portfolio cleanup", priority=4)
    db_session.commit()

    service = TimelineService()
    text = service.build_today_view(db_session, user.id, user.timezone)
    lowered = text.lower()
    assert "today plan" in lowered
    assert "1. website polish" in lowered


def test_today_view_prioritizes_unlocking_prerequisite_before_other_work(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)

    prerequisite = create_task(db_session, user_id=user.id, title="Fix website", priority=3)
    blocked_deadline = create_task(
        db_session,
        user_id=user.id,
        title="Send recruiter email",
        priority=5,
        deadline_at=now + timedelta(hours=5),
        blocked_reason="need website fixed first",
    )
    blocked_deadline.status = TaskStatus.blocked
    create_task_dependency(
        db_session,
        user_id=user.id,
        predecessor_task_id=prerequisite.id,
        successor_task_id=blocked_deadline.id,
    )
    create_task(db_session, user_id=user.id, title="Laundry", priority=2, deadline_at=now + timedelta(hours=8))
    db_session.commit()

    service = TimelineService()
    text = service.build_today_view(db_session, user.id, user.timezone)
    lowered = text.lower()

    assert lowered.index("fix website") < lowered.index("laundry")
    assert "unlocks send recruiter email" in lowered
    assert "blocked by fix website" in lowered


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
    assert "tomorrow morning plan" in lowered
    assert "submit scout app" in lowered


def test_week_view_orders_unlocking_work_ahead_of_lower_value_items(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)

    prerequisite = create_task(db_session, user_id=user.id, title="Finalize resume PDF", priority=3)
    blocked = create_task(
        db_session,
        user_id=user.id,
        title="Send recruiter email",
        priority=5,
        deadline_at=now + timedelta(days=1),
        blocked_reason="need resume first",
    )
    blocked.status = TaskStatus.blocked
    create_task_dependency(
        db_session,
        user_id=user.id,
        predecessor_task_id=prerequisite.id,
        successor_task_id=blocked.id,
    )
    create_task(db_session, user_id=user.id, title="Read design article", priority=2, deadline_at=now + timedelta(days=4))
    db_session.commit()

    service = TimelineService()
    text = service.build_week_view(db_session, user.id, user.timezone)
    lowered = text.lower()

    assert "this week plan" in lowered
    assert lowered.index("finalize resume pdf") < lowered.index("read design article")


def test_next_hour_recommendation_prefers_unlocking_move(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)

    prerequisite = create_task(db_session, user_id=user.id, title="Fix website", priority=3)
    blocked = create_task(
        db_session,
        user_id=user.id,
        title="Submit application",
        priority=5,
        deadline_at=now + timedelta(hours=4),
        blocked_reason="need website fixed",
    )
    blocked.status = TaskStatus.blocked
    create_task_dependency(
        db_session,
        user_id=user.id,
        predecessor_task_id=prerequisite.id,
        successor_task_id=blocked.id,
    )
    create_task(db_session, user_id=user.id, title="Tidy desk", priority=1)
    db_session.commit()

    service = TimelineService()
    text = service.next_hour_recommendation(db_session, user.id, user.timezone)
    lowered = text.lower()

    assert lowered.startswith("next hour move:")
    assert "fix website" in lowered
    assert "unlocks submit application" in lowered


def test_weekend_view_handles_empty_case(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    service = TimelineService()
    text = service.build_weekend_view(db_session, user.id, user.timezone)
    assert "weekend is clear" in text.lower()

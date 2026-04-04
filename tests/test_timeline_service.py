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
    assert "today" in lowered
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
    assert "tomorrow morning" in lowered
    assert "submit scout app" in lowered


def test_recommend_next_task_waits_on_windowed_email_and_prefers_cad_tonight(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    tomorrow_morning = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    email = create_task(
        db_session,
        user_id=user.id,
        title="Send that email",
        priority=5,
        deadline_at=(now + timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0),
        start_after=tomorrow_morning,
        next_step="draft the email tonight so it's ready to send tomorrow morning",
    )
    cad = create_task(
        db_session,
        user_id=user.id,
        title="Finish the CAD for the enclosure",
        priority=4,
        deadline_at=(now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0),
    )
    db_session.commit()

    service = TimelineService()
    recommended = service.recommend_next_task(db_session, user.id, user.timezone)
    assert recommended is not None
    assert recommended.id == cad.id

    morning_view = service.build_tomorrow_morning_view(db_session, user.id, user.timezone).lower()
    assert "send that email" in morning_view
    assert "for" in morning_view


def test_tonight_view_keeps_real_work_and_excludes_future_reminder_only_items(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    tomorrow_morning = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    monday_morning = (now + timedelta(days=((7 - now.weekday()) % 7 or 7))).replace(hour=9, minute=0, second=0, microsecond=0)

    create_task(
        db_session,
        user_id=user.id,
        title="Send scout followup",
        priority=4,
        deadline_at=monday_morning,
        start_after=monday_morning,
        deadline_source_phrase="monday morning",
        metadata_json={"action_kind": "quick_message"},
    )
    create_task(
        db_session,
        user_id=user.id,
        title="Text roommate back",
        priority=3,
        deadline_at=tomorrow_morning.replace(hour=10),
        start_after=tomorrow_morning,
        deadline_source_phrase="tomorrow morning",
        metadata_json={"action_kind": "quick_message"},
    )
    create_task(
        db_session,
        user_id=user.id,
        title="Pay rent",
        priority=4,
        deadline_at=now.replace(hour=21, minute=0, second=0, microsecond=0),
        deadline_source_phrase="tonight",
        metadata_json={"action_kind": "quick_admin"},
    )
    create_task(
        db_session,
        user_id=user.id,
        title="Finish enclosure CAD",
        priority=5,
        deadline_at=(now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0),
        metadata_json={"action_kind": "project_chunk"},
    )
    db_session.commit()

    service = TimelineService()
    text = service.build_tonight_view(db_session, user.id, user.timezone).lower()

    assert "finish enclosure cad" in text
    assert "pay rent" in text
    assert "send scout followup" not in text
    assert "text roommate back" not in text


def test_recommend_next_task_does_not_push_future_quick_message_over_real_work(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    tomorrow_morning = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)

    create_task(
        db_session,
        user_id=user.id,
        title="Text roommate back",
        priority=3,
        start_after=tomorrow_morning,
        deadline_source_phrase="tomorrow morning",
        metadata_json={"action_kind": "quick_message"},
    )
    cad = create_task(
        db_session,
        user_id=user.id,
        title="Finish enclosure CAD",
        priority=4,
        deadline_at=(now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0),
        next_step="finish the first clean enclosure pass",
        metadata_json={"action_kind": "project_chunk"},
    )
    db_session.commit()

    service = TimelineService()
    recommended = service.recommend_next_task(db_session, user.id, user.timezone)
    assert recommended is not None
    assert recommended.id == cad.id


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

    assert "this week" in lowered
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

    assert lowered.startswith("for the next hour i'd")
    assert "fix website" in lowered
    assert "clears the way for submit application" in lowered


def test_next_hour_recommendation_prefers_real_work_over_broad_weekend_errand(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)

    create_task(
        db_session,
        user_id=user.id,
        title="Book a dentist appointment",
        priority=3,
        deadline_at=(now + timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
        soft_deadline_at=now.replace(hour=10, minute=0, second=0, microsecond=0),
        deadline_source_phrase="this weekend",
        deadline_is_ambiguous=True,
        deadline_granularity="weekend",
        metadata_json={"action_kind": "quick_admin"},
    )
    create_task(
        db_session,
        user_id=user.id,
        title="Finish the enclosure CAD",
        priority=4,
        deadline_at=(now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0),
        metadata_json={"action_kind": "project_chunk"},
    )
    db_session.commit()

    service = TimelineService()
    lowered = service.next_hour_recommendation(db_session, user.id, user.timezone).lower()

    assert "finish the enclosure cad" in lowered
    assert "dentist" not in lowered


def test_week_view_treats_weekend_errand_as_window_and_not_top_pressure(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)

    create_task(
        db_session,
        user_id=user.id,
        title="Book a dentist appointment",
        priority=3,
        deadline_at=(now + timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
        soft_deadline_at=now.replace(hour=10, minute=0, second=0, microsecond=0),
        deadline_source_phrase="this weekend",
        deadline_is_ambiguous=True,
        deadline_granularity="weekend",
        metadata_json={"action_kind": "quick_admin"},
    )
    create_task(
        db_session,
        user_id=user.id,
        title="Finish the enclosure CAD",
        priority=4,
        deadline_at=(now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0),
        metadata_json={"action_kind": "project_chunk"},
    )
    db_session.commit()

    service = TimelineService()
    lowered = service.build_week_view(db_session, user.id, user.timezone).lower()

    assert "this weekend" in lowered
    assert lowered.index("finish the enclosure cad") < lowered.index("book a dentist appointment")


def test_weekend_view_handles_empty_case(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    service = TimelineService()
    text = service.build_weekend_view(db_session, user.id, user.timezone)
    assert "weekend is clear" in text.lower()


def test_project_view_filters_to_matching_tasks(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Finish the CAD for the enclosure", priority=5)
    create_task(db_session, user_id=user.id, title="Send enclosure update email", priority=4)
    create_task(db_session, user_id=user.id, title="Laundry", priority=1)
    db_session.commit()

    service = TimelineService()
    text = service.build_project_view(db_session, user.id, user.timezone, "what's the plan for the enclosure project?")
    lowered = text.lower()
    assert "enclosure" in lowered
    assert "finish the cad for the enclosure" in lowered
    assert "laundry" not in lowered


def test_today_view_prioritizes_crowded_mixed_workload_by_unlocks_windows_and_deadlines(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    tomorrow_morning = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)

    website = create_task(db_session, user_id=user.id, title="Fix portfolio website", priority=4, next_step="ship the broken homepage fixes")
    recruiter = create_task(
        db_session,
        user_id=user.id,
        title="Send recruiter email",
        priority=5,
        deadline_at=now + timedelta(hours=6),
        blocked_reason="need website fixed first",
        next_step="send the recruiter update once the site is clean",
    )
    recruiter.status = TaskStatus.blocked
    create_task_dependency(
        db_session,
        user_id=user.id,
        predecessor_task_id=website.id,
        successor_task_id=recruiter.id,
    )

    rent = create_task(
        db_session,
        user_id=user.id,
        title="Pay rent",
        priority=5,
        deadline_at=now + timedelta(hours=4),
        next_step="send the rent transfer",
    )
    cad = create_task(
        db_session,
        user_id=user.id,
        title="Finish enclosure CAD",
        priority=4,
        deadline_at=(now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0),
        next_step="finish the first clean enclosure pass",
    )
    roommate = create_task(
        db_session,
        user_id=user.id,
        title="Text roommate back",
        priority=3,
        deadline_at=(now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0),
        start_after=tomorrow_morning,
        next_step="send the roommate text in the morning",
    )
    reading = create_task(
        db_session,
        user_id=user.id,
        title="Read design article",
        priority=1,
        deadline_at=now + timedelta(days=3),
    )
    db_session.commit()

    service = TimelineService()
    today = service.build_today_view(db_session, user.id, user.timezone).lower()
    tomorrow = service.build_tomorrow_morning_view(db_session, user.id, user.timezone).lower()
    recommended = service.recommend_next_task(db_session, user.id, user.timezone)

    assert recommended is not None
    assert recommended.id in {rent.id, website.id}
    assert today.index("pay rent") < today.index("finish enclosure cad")
    assert today.index("fix portfolio website") < today.index("read design article")
    assert "unlocks send recruiter email" in today
    assert "text roommate back" in tomorrow
    assert "finish enclosure cad" not in tomorrow

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import (
    ConversationMessage,
    MessageDirection,
    ProfileStyle,
    Reminder,
    ReminderStatus,
    ScheduleBlock,
    Task,
    TaskStatus,
    User,
    UserProfile,
)
from app.db.repositories.task_repo import create_task
from app.db.repositories.user_repo import get_user_by_phone
from app.worker import tasks as worker_tasks


def test_due_reminders_defer_when_recent_inbound_exists(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "reminders.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    user = User(phone_number="+12488290272", name="Test", timezone="America/New_York")
    session.add(user)
    session.flush()
    session.add(
        UserProfile(
            user_id=user.id,
            style=ProfileStyle.casual_cool,
            planning_preferences={},
            bio="",
        )
    )
    now = datetime.now(tz=ZoneInfo(user.timezone))
    session.add(
        ConversationMessage(
            user_id=user.id,
            direction=MessageDirection.inbound,
            body="yo quick update",
            external_id="SM_RECENT",
            created_at=now - timedelta(minutes=3),
        )
    )
    reminder = Reminder(
        user_id=user.id,
        task_id=None,
        kind="checkin",
        scheduled_for=now - timedelta(minutes=1),
        status=ReminderStatus.pending,
        escalation_level=0,
    )
    session.add(reminder)
    session.commit()
    session.close()

    monkeypatch.setattr(worker_tasks, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(
        worker_tasks,
        "get_or_create_primary_user",
        lambda session: get_user_by_phone(session, "+12488290272"),
    )

    sent: list[tuple[str, str]] = []
    sid_counter = {"n": 0}

    class _FakeTransport:
        def send_sms(self, *, to_number: str, body: str) -> str:
            sent.append((to_number, body))
            sid_counter["n"] += 1
            return f"SM_OUT_TEST_{sid_counter['n']}"

    monkeypatch.setattr(worker_tasks, "TwilioTransport", lambda: _FakeTransport())

    result = worker_tasks.send_due_reminders_task()
    assert result["sent"] == 0
    assert result["skipped"] == 1
    assert sent == []

    verify = TestingSessionLocal()
    refreshed = verify.execute(select(Reminder)).scalars().first()
    verify.close()
    assert refreshed is not None
    assert refreshed.status == ReminderStatus.pending
    assert refreshed.cooldown_until is not None


def test_due_reminders_avoid_burst_and_defer_extra_normal_checkins(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "reminders_cap.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    user = User(phone_number="+12488290272", name="Test", timezone="America/New_York")
    session.add(user)
    session.flush()
    session.add(
        UserProfile(
            user_id=user.id,
            style=ProfileStyle.casual_cool,
            planning_preferences={},
            bio="",
        )
    )
    now = datetime.now(tz=ZoneInfo(user.timezone))
    for i in range(4):
        session.add(
            Reminder(
                user_id=user.id,
                task_id=None,
                kind="checkin",
                scheduled_for=now - timedelta(minutes=1),
                status=ReminderStatus.pending,
                escalation_level=0,
                reason=f"r{i}",
            )
        )
    session.commit()
    session.close()

    monkeypatch.setattr(worker_tasks, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(
        worker_tasks,
        "get_or_create_primary_user",
        lambda session: get_user_by_phone(session, "+12488290272"),
    )

    sent: list[tuple[str, str]] = []
    sid_counter = {"n": 0}

    class _FakeTransport:
        def send_sms(self, *, to_number: str, body: str) -> str:
            sent.append((to_number, body))
            sid_counter["n"] += 1
            return f"SM_OUT_TEST_{sid_counter['n']}"

    monkeypatch.setattr(worker_tasks, "TwilioTransport", lambda: _FakeTransport())

    result = worker_tasks.send_due_reminders_task()
    assert result["sent"] == 1
    assert result["skipped"] == 3
    assert len(sent) == 1

    verify = TestingSessionLocal()
    pending = verify.execute(
        select(Reminder).where(Reminder.status == ReminderStatus.pending).order_by(Reminder.scheduled_for.asc())
    ).scalars().all()
    sent_rows = verify.execute(select(Reminder).where(Reminder.status == ReminderStatus.sent)).scalars().all()
    verify.close()

    assert len(sent_rows) == 1
    assert len(pending) == 3
    assert all(reminder.cooldown_until is not None for reminder in pending)


def test_due_reminders_defer_during_active_context_block(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "reminders_context.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    user = User(phone_number="+12488290272", name="Test", timezone="America/New_York")
    session.add(user)
    session.flush()
    session.add(
        UserProfile(
            user_id=user.id,
            style=ProfileStyle.casual_cool,
            planning_preferences={},
            bio="",
        )
    )
    now = datetime.now(tz=ZoneInfo(user.timezone))
    task = create_task(session, user_id=user.id, title="Submit studio reflection", deadline_at=now + timedelta(hours=2))
    reminder = Reminder(
        user_id=user.id,
        task_id=task.id,
        kind="checkin",
        scheduled_for=now - timedelta(minutes=2),
        status=ReminderStatus.pending,
        escalation_level=1,
    )
    session.add(reminder)
    block = ScheduleBlock(
        user_id=user.id,
        block_type="driving",
        starts_at=now - timedelta(minutes=5),
        ends_at=now + timedelta(minutes=25),
        confidence=0.8,
        notes="on the road",
    )
    session.add(block)
    session.commit()
    block_end = block.ends_at
    session.close()

    monkeypatch.setattr(worker_tasks, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(
        worker_tasks,
        "get_or_create_primary_user",
        lambda session: get_user_by_phone(session, "+12488290272"),
    )

    sent: list[tuple[str, str]] = []

    class _FakeTransport:
        def send_sms(self, *, to_number: str, body: str) -> str:
            sent.append((to_number, body))
            return "SM_OUT_TEST_1"

    monkeypatch.setattr(worker_tasks, "TwilioTransport", lambda: _FakeTransport())

    result = worker_tasks.send_due_reminders_task()
    assert result["sent"] == 0
    assert result["skipped"] == 1
    assert sent == []

    verify = TestingSessionLocal()
    refreshed = verify.execute(select(Reminder)).scalars().first()
    refreshed_task = verify.execute(select(Task)).scalars().first()
    verify.close()

    assert refreshed is not None
    assert refreshed.status == ReminderStatus.pending
    assert refreshed.scheduled_for >= block_end + timedelta(minutes=18)
    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.active
    assert refreshed_task.reminder_pause_until == refreshed.scheduled_for


def test_compose_reminder_text_uses_short_slip_and_blocker_followups(tmp_path: Path) -> None:
    db_path = tmp_path / "reminders_text.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    user = User(phone_number="+12488290272", name="Test", timezone="America/New_York")
    session.add(user)
    session.flush()
    session.add(UserProfile(user_id=user.id, style=ProfileStyle.casual_cool, planning_preferences={}, bio=""))
    slip_task = create_task(
        session,
        user_id=user.id,
        title="Finish CAD pass",
        slip_count=1,
        last_slip_reason="got distracted",
    )
    blocked_task = create_task(
        session,
        user_id=user.id,
        title="Send recruiter text",
        blocked_reason="waiting on portfolio PDF",
    )
    blocked_task.status = TaskStatus.blocked
    session.commit()

    assert worker_tasks._compose_reminder_text(task=slip_task, escalation=1) == "checking on 'Finish CAD pass'. what happened?"
    assert (
        worker_tasks._compose_reminder_text(task=blocked_task, escalation=1)
        == "checking on 'Send recruiter text'. what's the blocker?"
    )
    session.close()

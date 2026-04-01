from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import ConversationMessage, MessageDirection, ProfileStyle, Reminder, ReminderStatus, User, UserProfile
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


def test_due_reminders_caps_sends_per_run(tmp_path: Path, monkeypatch) -> None:
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
    assert result["sent"] == 2
    assert result["skipped"] == 2
    assert len(sent) == 2

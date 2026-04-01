from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import ConversationMessage, MessageDirection, ProfileStyle, User, UserProfile
from app.worker import tasks as worker_tasks


def test_inbound_task_persists_messages_when_processing_fails(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "resilience.db"
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
    session.commit()
    session.close()

    monkeypatch.setattr(worker_tasks, "SessionLocal", TestingSessionLocal)

    class _BoomManager:
        def process_inbound(self, session: Session, payload) -> object:  # noqa: ANN001
            raise TimeoutError("compose timeout")

    monkeypatch.setattr(worker_tasks, "ConversationManager", lambda: _BoomManager())

    sent: list[tuple[str, str]] = []

    class _FakeTransport:
        def send_sms(self, to_number: str, body: str) -> str:
            sent.append((to_number, body))
            return "SM_OUT_TEST"

    monkeypatch.setattr(worker_tasks, "TwilioTransport", lambda: _FakeTransport())

    result = worker_tasks.process_inbound_sms_task(
        {
            "From": "+12488290272",
            "To": "+17622516270",
            "Body": "are you there?",
            "MessageSid": "SM_IN_TEST_1",
            "NumMedia": 0,
            "media": [],
        }
    )

    assert "error" in result
    assert sent, "fallback sms should be attempted"

    verify = TestingSessionLocal()
    messages = verify.execute(
        select(ConversationMessage).order_by(ConversationMessage.created_at.asc())
    ).scalars().all()
    verify.close()

    assert any(m.direction == MessageDirection.inbound and m.external_id == "SM_IN_TEST_1" for m in messages)
    assert any(m.direction == MessageDirection.outbound and "processing miss" in m.body for m in messages)

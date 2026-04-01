from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.api.routes.admin import stream_messages
from app.core.config import get_settings
from app.db.models import MessageDirection
from app.db.repositories.message_repo import create_message
from app.db.repositories.user_repo import get_or_create_primary_user
from app.db.session import get_session
from app.main import app


def _configure_test_overrides(db_session, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("USER_PHONE_NUMBER", "+15550001111")
    get_settings.cache_clear()

    def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session


def _cleanup_test_overrides() -> None:
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_live_monitor_requires_token(db_session, monkeypatch) -> None:
    _configure_test_overrides(db_session, monkeypatch)
    try:
        client = TestClient(app)
        response = client.get("/api/admin/live")
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid admin token"
    finally:
        _cleanup_test_overrides()


def test_live_monitor_page_renders_with_query_token(db_session, monkeypatch) -> None:
    _configure_test_overrides(db_session, monkeypatch)
    try:
        client = TestClient(app)
        response = client.get("/api/admin/live?token=test-token")
        assert response.status_code == 200
        assert "Friend Live SMS Monitor" in response.text
        assert "/api/admin/messages/stream?token=" in response.text
    finally:
        _cleanup_test_overrides()


def test_live_stream_emits_messages_snapshot(db_session, monkeypatch) -> None:
    _configure_test_overrides(db_session, monkeypatch)
    try:
        user = get_or_create_primary_user(db_session)
        assert user is not None
        create_message(
            db_session,
            user_id=user.id,
            direction=MessageDirection.inbound,
            body="hey there",
            external_id="SM_STREAM_1",
        )
        create_message(
            db_session,
            user_id=user.id,
            direction=MessageDirection.outbound,
            body="yo i got you",
            external_id="SM_STREAM_2",
        )
        db_session.commit()

        async def _first_stream_chunk() -> str:
            response = await stream_messages(limit=50, poll_seconds=0.25, heartbeat_seconds=5, session=db_session)
            chunk = await response.body_iterator.__anext__()
            await response.body_iterator.aclose()
            return chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)

        first_chunk = asyncio.run(_first_stream_chunk())
        assert "event: messages" in first_chunk
        assert "hey there" in first_chunk
        assert "yo i got you" in first_chunk
    finally:
        _cleanup_test_overrides()

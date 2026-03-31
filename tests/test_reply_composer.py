from __future__ import annotations

from datetime import datetime

from app.llm.conversation_composer import ConversationComposer
from app.schemas.reply import ReplyBrief


class FakeAdapter:
    def __init__(self, responses: list[dict | None], enabled: bool = True) -> None:
        self.responses = responses
        self.enabled = enabled
        self.calls = 0

    def json_completion(self, *, system: str, user: str, model: str | None = None):  # noqa: ANN001
        self.calls += 1
        if not self.responses:
            return None
        return self.responses.pop(0)


def _brief(latest: str, recent_thread: list[str] | None = None) -> ReplyBrief:
    return ReplyBrief(
        response_goal="open_conversation",
        latest_user_message=latest,
        recent_thread=recent_thread or [],
        generated_at=datetime.now(),
    )


def test_open_ended_message_uses_llm_path():
    adapter = FakeAdapter([{"messages": ["yo what's up, i'm here. what's the move tonight?"]}], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("what up tho"))
    assert adapter.calls >= 1
    assert reply.used_fallback is False
    assert reply.messages


def test_fallback_only_on_forced_model_failure():
    adapter = FakeAdapter([None], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("hey"))
    assert adapter.calls >= 1
    assert reply.used_fallback is True
    assert reply.messages


def test_repetition_guard_triggers_regeneration():
    adapter = FakeAdapter(
        [
            {"messages": ["got you. what's the move?"]},
            {"messages": ["say less. what's the main thing tonight?"]},
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    brief = _brief("hi", recent_thread=["assistant: got you. what's the move?"])
    reply = composer.compose(brief)
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "main thing tonight" in " ".join(reply.messages).lower()


from __future__ import annotations

from datetime import datetime

from app.llm.conversation_composer import ConversationComposer
from app.schemas.reply import ReplyBrief


class FakeAdapter:
    def __init__(self, text_responses: list[str | None], enabled: bool = True) -> None:
        self.text_responses = text_responses
        self.enabled = enabled
        self.calls = 0
        self.text_calls = 0

    def json_completion(  # noqa: ANN001
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict | None = None,
        request_timeout_seconds: float | None = None,
    ):
        self.calls += 1
        return None

    def text_completion(  # noqa: ANN001
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict | None = None,
        request_timeout_seconds: float | None = None,
    ):
        self.text_calls += 1
        if not self.text_responses:
            return None
        return self.text_responses.pop(0)


def _brief(latest: str, recent_thread: list[str] | None = None) -> ReplyBrief:
    return ReplyBrief(
        response_goal="open_conversation",
        latest_user_message=latest,
        recent_thread=recent_thread or [],
        generated_at=datetime.now(),
    )


def _answer_brief(latest: str) -> ReplyBrief:
    return ReplyBrief(
        response_goal="answer_question",
        latest_user_message=latest,
        key_facts_to_include=["yes, i'm live right now and i received this message"],
        generated_at=datetime.now(),
    )


def test_open_ended_message_uses_llm_path():
    adapter = FakeAdapter(["yo what's up, i'm here. what's the move tonight?"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("what up tho"))
    assert adapter.text_calls >= 1
    assert reply.used_fallback is False
    assert reply.messages


def test_fallback_only_on_forced_model_failure():
    adapter = FakeAdapter([None], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("hey"))
    assert adapter.text_calls >= 1
    assert reply.used_fallback is True
    assert reply.messages


def test_repetition_guard_triggers_regeneration():
    adapter = FakeAdapter(
        [
            "got you. what's the move?",
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    brief = _brief("hi", recent_thread=["assistant: got you. what's the move?"])
    reply = composer.compose(brief)
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "what's the real next move" in " ".join(reply.messages).lower()


def test_composer_keeps_single_llm_attempt_path():
    adapter = FakeAdapter(["yo i got you\n\nstart with a 20 min pass, then ping me."], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("i'm cooked"))
    assert adapter.text_calls >= 1
    assert adapter.calls == 0
    assert reply.used_fallback is False
    assert len(reply.messages) >= 1


def test_composer_regenerates_when_internal_phrase_leaks():
    adapter = FakeAdapter(
        [
            "open conversational message received",
            "yeah i'm live. i saw your text. what's the move?",
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("is this thing on?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "open conversational message received" not in " ".join(reply.messages).lower()


def test_composer_regenerates_when_output_leaks_context_labels():
    adapter = FakeAdapter(
        [
            "active tasks: finish cad upcoming deadlines: tomorrow night",
            "yep i'm live. i got your update. wanna lock the first 30 min now?",
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("are you actually live now"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "active tasks:" not in " ".join(reply.messages).lower()


def test_composer_regenerates_when_it_parrots_user_message():
    adapter = FakeAdapter(
        [
            "are you actually live now",
            "yeah i'm here rn and tracking. what do you want to knock out first?",
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("are you actually live now"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "what do you want to knock out first?" in " ".join(reply.messages).lower()


def test_composer_regenerates_when_first_bubble_parrots_user():
    adapter = FakeAdapter(
        [
            "are you actually live now\n\nyeah i'm tracking everything.",
            "yep i'm live and tracking. what's the move right now?",
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("are you actually live now"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert not any(msg.strip().lower() == "are you actually live now" for msg in reply.messages)


def test_composer_regenerates_if_answer_goal_starts_with_question():
    adapter = FakeAdapter(
        [
            "are you alive?\n\ni'm here.",
            "yeah i'm live rn. i got your text.",
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_answer_brief("are you actually live now?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert not reply.messages[0].strip().endswith("?")

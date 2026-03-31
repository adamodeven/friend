from app.llm.composer import ReplyComposer
from app.schemas.intent import IntentResult


def test_fallback_general_chat_is_humanized():
    intent = IntentResult(intent="general_chat", confidence=0.5)
    text = ReplyComposer._fallback_text(intent, "clean slate. let's add the next task you care about.")
    assert "clean slate" not in text.lower()
    assert any(token in text.lower() for token in ["got you", "bet", "say less"])


def test_fallback_status_query_is_not_empty():
    intent = IntentResult(intent="status_query", confidence=0.9)
    text = ReplyComposer._fallback_text(intent, "i keep your task graph live")
    assert text.strip()
    assert "task graph" in text.lower()

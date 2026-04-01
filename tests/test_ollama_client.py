from __future__ import annotations

from app.llm.client import OllamaAdapter


def _adapter_for_test() -> OllamaAdapter:
    adapter = object.__new__(OllamaAdapter)
    adapter._enabled = True
    adapter._text_model = "slow-model"
    adapter._fallback_text_model = "fast-model"
    adapter._keep_alive = "30m"
    return adapter


def test_text_completion_falls_back_to_secondary_model():
    adapter = _adapter_for_test()
    calls: list[tuple[str, float | None]] = []

    def fake_generate(payload, *, request_timeout_seconds=None):  # noqa: ANN001
        calls.append((payload["model"], request_timeout_seconds))
        if payload["model"] == "slow-model":
            return None
        return "yo this is live"

    adapter._generate_content = fake_generate  # type: ignore[attr-defined]

    result = adapter.text_completion(system="s", user="u", request_timeout_seconds=30)
    assert result == "yo this is live"
    assert [model for model, _ in calls] == ["slow-model", "fast-model"]
    assert calls[0][1] == 15.0
    assert calls[1][1] == 15.0


def test_json_completion_falls_back_and_parses():
    adapter = _adapter_for_test()
    calls: list[str] = []

    def fake_generate(payload, *, request_timeout_seconds=None):  # noqa: ANN001
        calls.append(payload["model"])
        if payload["model"] == "slow-model":
            return None
        return '{"intent":"general_chat","confidence":0.88}'

    adapter._generate_content = fake_generate  # type: ignore[attr-defined]

    result = adapter.json_completion(system="s", user="u", request_timeout_seconds=24)
    assert result is not None
    assert result["intent"] == "general_chat"
    assert calls == ["slow-model", "fast-model"]

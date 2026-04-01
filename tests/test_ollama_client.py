from __future__ import annotations

from app.llm.client import OllamaAdapter


def _adapter_for_test() -> OllamaAdapter:
    adapter = object.__new__(OllamaAdapter)
    adapter._enabled = True
    adapter._text_model = "slow-model"
    adapter._fallback_text_model = "fast-model"
    adapter._keep_alive = "30m"
    adapter._auto_pull_missing_models = False
    adapter._native_api_available = None
    adapter._openai_compat_available = None
    adapter._base_url = "http://ollama:11434"
    adapter._default_options = {}
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


def test_text_completion_uses_openai_compat_when_native_returns_404():
    adapter = _adapter_for_test()

    def fake_generate(payload, *, request_timeout_seconds=None):  # noqa: ANN001
        return "__generate_404__"

    def fake_openai_chat_completion(  # noqa: ANN001
        *,
        system,
        user,
        model,
        options=None,
        images=None,
        request_timeout_seconds=None,
    ):
        if model == "slow-model":
            return None
        return "yep live now"

    adapter._generate_content = fake_generate  # type: ignore[attr-defined]
    adapter._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]

    result = adapter.text_completion(system="s", user="u", request_timeout_seconds=18)
    assert result == "yep live now"


def test_text_completion_retries_after_model_not_found_when_auto_pull_enabled():
    adapter = _adapter_for_test()
    adapter._auto_pull_missing_models = True
    seen = {"count": 0}

    def fake_generate(payload, *, request_timeout_seconds=None):  # noqa: ANN001
        if payload["model"] == "slow-model":
            return None
        seen["count"] += 1
        if seen["count"] == 1:
            return "__model_not_found__"
        return "ready after pull"

    def fake_pull_model(model):  # noqa: ANN001
        return model == "fast-model"

    adapter._generate_content = fake_generate  # type: ignore[attr-defined]
    adapter._pull_model = fake_pull_model  # type: ignore[attr-defined]

    result = adapter.text_completion(system="s", user="u", request_timeout_seconds=18)
    assert result == "ready after pull"


def test_text_completion_merges_runtime_defaults_with_call_options():
    adapter = _adapter_for_test()
    adapter._default_options = {"num_thread": 4, "low_vram": True}
    captured: dict[str, object] = {}

    def fake_generate(payload, *, request_timeout_seconds=None):  # noqa: ANN001
        captured["options"] = payload.get("options")
        return "ok"

    adapter._generate_content = fake_generate  # type: ignore[attr-defined]

    result = adapter.text_completion(
        system="s",
        user="u",
        options={"temperature": 0.4, "num_predict": 20},
        request_timeout_seconds=10,
    )

    assert result == "ok"
    assert captured["options"] == {
        "num_thread": 4,
        "low_vram": True,
        "temperature": 0.4,
        "num_predict": 20,
    }

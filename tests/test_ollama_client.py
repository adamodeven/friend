from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

from app.llm import client as llm_client
from app.llm.client import OllamaAdapter


def _adapter_for_test() -> OllamaAdapter:
    OllamaAdapter._shared_openai_rate_limited_until = None
    OllamaAdapter._shared_openai_cooldown_reason = None
    adapter = object.__new__(OllamaAdapter)
    adapter._enabled = True
    adapter._provider = "ollama"
    adapter._openai_fallback_to_ollama = True
    adapter._text_model = "slow-model"
    adapter._fallback_text_model = "fast-model"
    adapter._ollama_text_model = "slow-model"
    adapter._ollama_fallback_text_model = "fast-model"
    adapter._ollama_vision_model = "vision-model"
    adapter._keep_alive = "30m"
    adapter._auto_pull_missing_models = False
    adapter._native_api_available = None
    adapter._openai_compat_available = None
    adapter._base_url = "http://ollama:11434"
    adapter._ollama_base_url = "http://ollama:11434"
    adapter._api_key = ""
    adapter._default_options = {}
    adapter._ollama_default_options = {}
    adapter._timeout = None
    adapter._openai_rate_limit_cooldown = timedelta(seconds=300)
    adapter._openai_insufficient_quota_cooldown = timedelta(seconds=3600)
    adapter._openai_rate_limited_until = None
    adapter._openai_cooldown_reason = None
    return adapter


def _openai_adapter_for_test() -> OllamaAdapter:
    OllamaAdapter._shared_openai_rate_limited_until = None
    OllamaAdapter._shared_openai_cooldown_reason = None
    adapter = object.__new__(OllamaAdapter)
    adapter._enabled = True
    adapter._provider = "openai"
    adapter._openai_fallback_to_ollama = True
    adapter._text_model = "openai-slow"
    adapter._fallback_text_model = "openai-fast"
    adapter._vision_model = "openai-vision"
    adapter._base_url = "https://api.openai.com/v1"
    adapter._api_key = "sk-test"
    adapter._keep_alive = "30m"
    adapter._auto_pull_missing_models = False
    adapter._native_api_available = None
    adapter._openai_compat_available = None
    adapter._ollama_base_url = "http://ollama:11434"
    adapter._ollama_text_model = "ollama-main"
    adapter._ollama_fallback_text_model = "ollama-fallback"
    adapter._ollama_vision_model = "ollama-vision"
    adapter._ollama_default_options = {}
    adapter._default_options = {}
    adapter._timeout = None
    adapter._openai_rate_limit_cooldown = timedelta(seconds=300)
    adapter._openai_insufficient_quota_cooldown = timedelta(seconds=3600)
    adapter._openai_rate_limited_until = None
    adapter._openai_cooldown_reason = None
    return adapter


def test_text_completion_falls_back_to_secondary_model():
    adapter = _adapter_for_test()
    calls: list[tuple[str, float | None]] = []

    def fake_generate(base_url, payload, *, request_timeout_seconds=None):  # noqa: ANN001
        calls.append((payload["model"], request_timeout_seconds))
        if payload["model"] == "slow-model":
            return None
        return "yo this is live"

    adapter._generate_content_for_base = fake_generate  # type: ignore[attr-defined]

    result = adapter.text_completion(system="s", user="u", request_timeout_seconds=30)
    assert result == "yo this is live"
    assert [model for model, _ in calls] == ["slow-model", "fast-model"]
    assert calls[0][1] == 15.0
    assert calls[1][1] == 15.0


def test_json_completion_falls_back_and_parses():
    adapter = _adapter_for_test()
    calls: list[str] = []

    def fake_generate(base_url, payload, *, request_timeout_seconds=None):  # noqa: ANN001
        calls.append(payload["model"])
        if payload["model"] == "slow-model":
            return None
        return '{"intent":"general_chat","confidence":0.88}'

    adapter._generate_content_for_base = fake_generate  # type: ignore[attr-defined]

    result = adapter.json_completion(system="s", user="u", request_timeout_seconds=24)
    assert result is not None
    assert result["intent"] == "general_chat"
    assert calls == ["slow-model", "fast-model"]


def test_text_completion_retries_after_model_not_found_when_auto_pull_enabled():
    adapter = _adapter_for_test()
    adapter._auto_pull_missing_models = True
    seen = {"count": 0}

    def fake_generate(base_url, payload, *, request_timeout_seconds=None):  # noqa: ANN001
        if payload["model"] == "slow-model":
            return None
        seen["count"] += 1
        if seen["count"] == 1:
            return "__model_not_found__"
        return "ready after pull"

    def fake_pull_model(model, *, base_url=None):  # noqa: ANN001
        return model == "fast-model"

    adapter._generate_content_for_base = fake_generate  # type: ignore[attr-defined]
    adapter._pull_model = fake_pull_model  # type: ignore[attr-defined]

    result = adapter.text_completion(system="s", user="u", request_timeout_seconds=18)
    assert result == "ready after pull"


def test_text_completion_merges_runtime_defaults_with_call_options():
    adapter = _adapter_for_test()
    adapter._default_options = {"num_thread": 4, "low_vram": True}
    adapter._ollama_default_options = {"num_thread": 4, "low_vram": True}
    captured: dict[str, object] = {}

    def fake_generate(base_url, payload, *, request_timeout_seconds=None):  # noqa: ANN001
        captured["options"] = payload.get("options")
        return "ok"

    adapter._generate_content_for_base = fake_generate  # type: ignore[attr-defined]

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


def test_openai_provider_text_completion_uses_model_fallback_chain():
    adapter = _openai_adapter_for_test()
    calls: list[str] = []

    def fake_openai_chat_completion(  # noqa: ANN001
        *,
        system,
        user,
        model,
        options=None,
        images=None,
        request_timeout_seconds=None,
    ):
        calls.append(model)
        if model == "openai-slow":
            return None
        return "openai live reply"

    adapter._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]

    result = adapter.text_completion(system="s", user="u", request_timeout_seconds=20)
    assert result == "openai live reply"
    assert calls == ["openai-slow", "openai-fast"]


def test_openai_provider_json_completion_parses_payload():
    adapter = _openai_adapter_for_test()

    def fake_openai_chat_completion(  # noqa: ANN001
        *,
        system,
        user,
        model,
        options=None,
        images=None,
        request_timeout_seconds=None,
    ):
        return '{"intent":"add_task","confidence":0.81}'

    adapter._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]

    result = adapter.json_completion(system="s", user="u", request_timeout_seconds=20)
    assert result is not None
    assert result["intent"] == "add_task"


def test_openai_provider_falls_back_to_ollama_when_rate_limited():
    adapter = _openai_adapter_for_test()

    def fake_openai_chat_completion(  # noqa: ANN001
        *,
        system,
        user,
        model,
        options=None,
        images=None,
        request_timeout_seconds=None,
    ):
        return "__rate_limited__"

    def fake_ollama_text_completion(**kwargs):  # noqa: ANN003,ANN201
        return "ollama fallback reply"

    adapter._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]
    adapter._ollama_text_completion = fake_ollama_text_completion  # type: ignore[attr-defined]

    result = adapter.text_completion(system="s", user="u", request_timeout_seconds=20)
    assert result == "ollama fallback reply"


def test_openai_rate_limit_sets_cooldown_and_skips_openai_next_call():
    adapter = _openai_adapter_for_test()
    openai_calls = {"count": 0}

    def fake_openai_chat_completion(  # noqa: ANN001
        *,
        system,
        user,
        model,
        options=None,
        images=None,
        request_timeout_seconds=None,
    ):
        openai_calls["count"] += 1
        return "__rate_limited__"

    def fake_ollama_text_completion(**kwargs):  # noqa: ANN003,ANN201
        return "ollama fallback reply"

    adapter._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]
    adapter._ollama_text_completion = fake_ollama_text_completion  # type: ignore[attr-defined]

    first = adapter.text_completion(system="s", user="u")
    second = adapter.text_completion(system="s", user="u again")
    assert first == "ollama fallback reply"
    assert second == "ollama fallback reply"
    assert openai_calls["count"] == 1


def test_openai_cooldown_is_shared_across_adapter_instances():
    adapter_a = _openai_adapter_for_test()
    adapter_b = _openai_adapter_for_test()
    openai_calls = {"count": 0}

    def fake_openai_chat_completion(  # noqa: ANN001
        *,
        system,
        user,
        model,
        options=None,
        images=None,
        request_timeout_seconds=None,
    ):
        openai_calls["count"] += 1
        return "__rate_limited__"

    def fake_ollama_text_completion(**kwargs):  # noqa: ANN003,ANN201
        return "ollama fallback reply"

    adapter_a._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]
    adapter_a._ollama_text_completion = fake_ollama_text_completion  # type: ignore[attr-defined]
    adapter_b._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]
    adapter_b._ollama_text_completion = fake_ollama_text_completion  # type: ignore[attr-defined]

    first = adapter_a.text_completion(system="s", user="u")
    second = adapter_b.text_completion(system="s", user="u2")
    assert first == "ollama fallback reply"
    assert second == "ollama fallback reply"
    assert openai_calls["count"] == 1


def test_openai_insufficient_quota_skips_ollama_fallback_and_stays_in_cooldown():
    adapter = _openai_adapter_for_test()
    openai_calls = {"count": 0}
    ollama_calls = {"count": 0}

    def fake_openai_chat_completion(  # noqa: ANN001
        *,
        system,
        user,
        model,
        options=None,
        images=None,
        request_timeout_seconds=None,
    ):
        openai_calls["count"] += 1
        return "__insufficient_quota__"

    def fake_ollama_text_completion(**kwargs):  # noqa: ANN003,ANN201
        ollama_calls["count"] += 1
        return "ollama fallback reply"

    adapter._openai_chat_completion = fake_openai_chat_completion  # type: ignore[attr-defined]
    adapter._ollama_text_completion = fake_ollama_text_completion  # type: ignore[attr-defined]

    first = adapter.text_completion(system="s", user="u")
    second = adapter.text_completion(system="s", user="u again")

    assert first is None
    assert second is None
    assert openai_calls["count"] == 1
    assert ollama_calls["count"] == 0
    assert adapter._openai_cooldown_reason == "insufficient_quota"


def test_openai_chat_completion_uses_max_completion_tokens_for_num_predict(monkeypatch) -> None:
    adapter = _openai_adapter_for_test()
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, *, timeout=None) -> None:  # noqa: ANN001
            self.timeout = timeout

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001,ANN204
            return False

        def post(self, url: str, json: dict, headers: dict):  # noqa: ANN001,ANN201
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            def _raise_for_status() -> None:
                return None
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "ok"}}]},
                raise_for_status=_raise_for_status,
            )

    monkeypatch.setattr(llm_client.httpx, "Client", _Client)

    result = adapter._openai_chat_completion(
        system="s",
        user="u",
        model="gpt-5.4-mini",
        options={"num_predict": 77},
    )

    assert result == "ok"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload.get("max_completion_tokens") == 160
    assert "max_tokens" not in payload


def test_openai_chat_completion_uses_higher_floor_for_json_format(monkeypatch) -> None:
    adapter = _openai_adapter_for_test()
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, *, timeout=None) -> None:  # noqa: ANN001
            self.timeout = timeout

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001,ANN204
            return False

        def post(self, url: str, json: dict, headers: dict):  # noqa: ANN001,ANN201
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers

            def _raise_for_status() -> None:
                return None

            return SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "{\"intent\":\"general_chat\"}"}}]},
                raise_for_status=_raise_for_status,
            )

    monkeypatch.setattr(llm_client.httpx, "Client", _Client)

    result = adapter._openai_chat_completion(
        system="s",
        user="u",
        model="gpt-5.4-nano",
        options={"num_predict": 96, "format": "json"},
    )

    assert result is not None
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload.get("max_completion_tokens") == 512
    assert payload.get("response_format") == {"type": "json_object"}

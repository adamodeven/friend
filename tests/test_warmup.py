from __future__ import annotations

from types import SimpleNamespace

from app.llm import warmup as warmup_module


class _FakeAdapter:
    def __init__(self, *, enabled: bool = True, text: str | None = "ok") -> None:
        self.enabled = enabled
        self._text = text
        self.calls = 0

    def text_completion(self, **_kwargs):  # noqa: ANN003
        self.calls += 1
        return self._text


def test_warmup_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(
        warmup_module,
        "get_settings",
        lambda: SimpleNamespace(ollama_warmup_on_startup=False),
    )
    called = {"adapter": 0}

    def _adapter_factory():
        called["adapter"] += 1
        return _FakeAdapter(enabled=True)

    monkeypatch.setattr(warmup_module, "OllamaAdapter", _adapter_factory)
    assert warmup_module.warmup_ollama_text_model() is False
    assert called["adapter"] == 0


def test_warmup_runs_when_enabled(monkeypatch):
    monkeypatch.setattr(
        warmup_module,
        "get_settings",
        lambda: SimpleNamespace(ollama_warmup_on_startup=True),
    )
    adapter = _FakeAdapter(enabled=True, text="ok")
    monkeypatch.setattr(warmup_module, "OllamaAdapter", lambda: adapter)
    assert warmup_module.warmup_ollama_text_model() is True
    assert adapter.calls == 1

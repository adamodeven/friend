from __future__ import annotations

import logging

from app.core.config import get_settings
from app.llm.client import OllamaAdapter


def warmup_ollama_text_model(*, logger: logging.Logger | None = None) -> bool:
    settings = get_settings()
    if not settings.ollama_warmup_on_startup:
        return False

    adapter = OllamaAdapter()
    if not adapter.enabled:
        return False

    log = logger or logging.getLogger(__name__)
    try:
        text = adapter.text_completion(
            system="You are a warmup probe. Reply with exactly: ok",
            user="ok",
            options={"temperature": 0.0, "num_predict": 4, "num_ctx": 128},
            request_timeout_seconds=90,
        )
        warmed = bool(text and text.strip())
        log.info("ollama warmup complete success=%s", warmed)
        return warmed
    except Exception as exc:  # pragma: no cover
        log.warning("ollama warmup failed: %s", exc)
        return False

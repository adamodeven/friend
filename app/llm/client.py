from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OllamaAdapter:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._text_model = settings.ollama_text_model
        self._vision_model = settings.ollama_vision_model
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._timeout = float(settings.ollama_timeout_seconds)
        self._enabled = settings.llm_provider.lower() == "ollama"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def json_completion(self, *, system: str, user: str, model: str | None = None) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        payload = {
            "model": model or self._text_model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        content = self._chat_content(payload)
        if not content:
            return None
        parsed = self._parse_json(content)
        if isinstance(parsed, dict):
            return parsed
        return None

    def text_completion(self, *, system: str, user: str, model: str | None = None) -> str | None:
        if not self._enabled:
            return None
        payload = {
            "model": model or self._text_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        return self._chat_content(payload)

    def vision_json(
        self,
        *,
        system: str,
        user_prompt: str,
        image_url: str,
    ) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        image_b64 = self._download_image_as_base64(image_url)
        if not image_b64:
            return None
        payload = {
            "model": self._vision_model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
        }
        content = self._chat_content(payload)
        if not content:
            return None
        parsed = self._parse_json(content)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _chat_content(self, payload: dict[str, Any]) -> str | None:
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(f"{self._base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                return (data.get("message") or {}).get("content")
            except Exception as exc:  # pragma: no cover
                last_exc = exc
        if last_exc:  # pragma: no cover
            logger.exception("ollama chat failed: %s", last_exc)
        return None

    def _download_image_as_base64(self, image_url: str) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(image_url, follow_redirects=True)
                resp.raise_for_status()
                content = resp.content
            return base64.b64encode(content).decode("utf-8")
        except Exception as exc:  # pragma: no cover
            logger.exception("failed downloading image for vision parse: %s", exc)
            return None

    @staticmethod
    def _parse_json(text: str) -> Any:
        content = text.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

from __future__ import annotations

import base64
import json
import logging
from typing import Any
from datetime import datetime, timedelta, timezone

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OllamaAdapter:
    _availability_cache: dict[str, tuple[datetime, bool]] = {}

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._text_model = settings.ollama_text_model
        self._vision_model = settings.ollama_vision_model
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._timeout = float(settings.ollama_timeout_seconds)
        self._enabled = settings.llm_provider.lower() == "ollama" and self._is_available()

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
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
            return (data.get("message") or {}).get("content")
        except Exception as exc:  # pragma: no cover
            logger.exception("ollama chat failed: %s", exc)
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

    def _is_available(self) -> bool:
        cached = self._availability_cache.get(self._base_url)
        now = datetime.now(tz=timezone.utc)
        if cached:
            checked_at, available = cached
            if now - checked_at <= timedelta(seconds=30):
                return available
        try:
            with httpx.Client(timeout=httpx.Timeout(0.4, connect=0.4)) as client:
                response = client.get(f"{self._base_url}/api/tags")
                if response.status_code != 200:
                    available = False
                else:
                    data = response.json()
                    model_names = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
                    available = any(name == self._text_model for name in model_names)
        except Exception:
            available = False
        self._availability_cache[self._base_url] = (now, available)
        return available

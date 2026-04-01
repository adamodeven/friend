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
        self._timeout_seconds = float(settings.ollama_timeout_seconds)
        self._keep_alive = settings.ollama_keep_alive
        self._timeout = httpx.Timeout(
            connect=min(5.0, self._timeout_seconds),
            read=self._timeout_seconds,
            write=20.0,
            pool=5.0,
        )
        self._enabled = settings.llm_provider.lower() == "ollama"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def json_completion(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        payload_generate = {
            "model": model or self._text_model,
            "stream": False,
            "format": "json",
            "prompt": f"{system}\n\n{user}",
            "keep_alive": self._keep_alive,
        }
        if options:
            payload_generate["options"] = options
        content = self._generate_content(payload_generate, request_timeout_seconds=request_timeout_seconds)
        if content == "__generate_404__":
            content = None
        if content:
            parsed = self._parse_json(content)
            if isinstance(parsed, dict):
                return parsed
        return None

    def text_completion(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
    ) -> str | None:
        if not self._enabled:
            return None
        payload_generate = {
            "model": model or self._text_model,
            "stream": False,
            "prompt": f"{system}\n\n{user}",
            "keep_alive": self._keep_alive,
        }
        if options:
            payload_generate["options"] = options
        content = self._generate_content(payload_generate, request_timeout_seconds=request_timeout_seconds)
        if content and content != "__generate_404__":
            return content
        return None

    def vision_json(
        self,
        *,
        system: str,
        user_prompt: str,
        image_url: str,
        request_timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        image_b64 = self._download_image_as_base64(image_url, request_timeout_seconds=request_timeout_seconds)
        if not image_b64:
            return None
        payload = {
            "model": self._vision_model,
            "stream": False,
            "format": "json",
            "keep_alive": self._keep_alive,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
        }
        content = self._chat_content(payload, request_timeout_seconds=request_timeout_seconds)
        if content == "__chat_404__":
            payload_generate = {
                "model": self._vision_model,
                "stream": False,
                "format": "json",
                "prompt": f"{system}\n\n{user_prompt}",
                "images": [image_b64],
            }
            content = self._generate_content(payload_generate, request_timeout_seconds=request_timeout_seconds)
        if not content:
            return None
        parsed = self._parse_json(content)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _chat_content(self, payload: dict[str, Any], *, request_timeout_seconds: float | None = None) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout_for(request_timeout_seconds)) as client:
                response = client.post(f"{self._base_url}/api/chat", json=payload)
                if response.status_code == 404:
                    return "__chat_404__"
                response.raise_for_status()
                data = response.json()
            return (data.get("message") or {}).get("content")
        except Exception as exc:  # pragma: no cover
            logger.exception("ollama chat failed: %s", exc)
            return None

    def _generate_content(self, payload: dict[str, Any], *, request_timeout_seconds: float | None = None) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout_for(request_timeout_seconds)) as client:
                response = client.post(f"{self._base_url}/api/generate", json=payload)
                if response.status_code == 404:
                    return "__generate_404__"
                response.raise_for_status()
                data = response.json()
            return data.get("response")
        except Exception as exc:  # pragma: no cover
            logger.exception("ollama generate failed: %s", exc)
            return None

    def _download_image_as_base64(
        self,
        image_url: str,
        *,
        request_timeout_seconds: float | None = None,
    ) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout_for(request_timeout_seconds)) as client:
                resp = client.get(image_url, follow_redirects=True)
                resp.raise_for_status()
                content = resp.content
            return base64.b64encode(content).decode("utf-8")
        except Exception as exc:  # pragma: no cover
            logger.exception("failed downloading image for vision parse: %s", exc)
            return None

    def _timeout_for(self, request_timeout_seconds: float | None) -> httpx.Timeout:
        if request_timeout_seconds is None:
            return self._timeout
        total = max(1.0, float(request_timeout_seconds))
        return httpx.Timeout(
            connect=min(3.0, total),
            read=total,
            write=min(8.0, total),
            pool=3.0,
        )

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

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = bool(settings.openai_api_key)
        self._text_model = settings.openai_text_model
        self._vision_model = settings.openai_vision_model
        self._client = OpenAI(api_key=settings.openai_api_key) if self._enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def json_completion(self, *, system: str, user: str, model: str | None = None) -> dict[str, Any] | None:
        if not self._client:
            return None
        try:
            response = self._client.responses.create(
                model=model or self._text_model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_object"}},
            )
            content = response.output_text
            if not content:
                return None
            return json.loads(content)
        except Exception as exc:  # pragma: no cover
            logger.exception("openai json completion failed: %s", exc)
            return None

    def text_completion(self, *, system: str, user: str, model: str | None = None) -> str | None:
        if not self._client:
            return None
        try:
            response = self._client.responses.create(
                model=model or self._text_model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.output_text
        except Exception as exc:  # pragma: no cover
            logger.exception("openai text completion failed: %s", exc)
            return None

    def vision_json(
        self,
        *,
        system: str,
        user_prompt: str,
        image_url: str,
    ) -> dict[str, Any] | None:
        if not self._client:
            return None
        try:
            response = self._client.responses.create(
                model=self._vision_model,
                input=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    },
                ],
                text={"format": {"type": "json_object"}},
            )
            content = response.output_text
            if not content:
                return None
            return json.loads(content)
        except Exception as exc:  # pragma: no cover
            logger.exception("openai vision completion failed: %s", exc)
            return None


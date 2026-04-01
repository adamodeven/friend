from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OllamaAdapter:
    _shared_openai_rate_limited_until: datetime | None = None
    _shared_openai_cooldown_reason: str | None = None

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._provider = settings.llm_provider.lower().strip()
        self._keep_alive = settings.ollama_keep_alive
        self._auto_pull_missing_models = settings.ollama_auto_pull_missing_models
        self._openai_fallback_to_ollama = bool(getattr(settings, "openai_fallback_to_ollama", True))
        self._openai_rate_limit_cooldown = timedelta(
            seconds=max(0, int(getattr(settings, "openai_rate_limit_cooldown_seconds", 300)))
        )
        self._openai_insufficient_quota_cooldown = timedelta(
            seconds=max(0, int(getattr(settings, "openai_insufficient_quota_cooldown_seconds", 3600)))
        )
        self._openai_rate_limited_until: datetime | None = self.__class__._shared_openai_rate_limited_until
        self._openai_cooldown_reason: str | None = self.__class__._shared_openai_cooldown_reason
        self._native_api_available: bool | None = None
        self._openai_compat_available: bool | None = None
        self._ollama_base_url = settings.ollama_base_url.rstrip("/")
        self._ollama_text_model = settings.ollama_text_model
        self._ollama_fallback_text_model = settings.ollama_fallback_text_model.strip()
        self._ollama_vision_model = settings.ollama_vision_model
        self._ollama_timeout_seconds = float(settings.ollama_timeout_seconds)
        self._ollama_default_options = self._build_default_options(settings)

        if self._provider == "openai":
            self._text_model = settings.openai_text_model
            self._fallback_text_model = settings.openai_fallback_text_model.strip()
            self._vision_model = settings.openai_vision_model
            self._base_url = settings.openai_base_url.rstrip("/")
            self._timeout_seconds = float(settings.openai_timeout_seconds)
            self._api_key = settings.openai_api_key.strip()
            self._default_options: dict[str, Any] = {}
            self._enabled = bool(self._api_key)
            if not self._enabled:
                logger.warning("LLM provider=openai but OPENAI_API_KEY is empty; LLM responses are disabled")
        else:
            self._text_model = self._ollama_text_model
            self._fallback_text_model = self._ollama_fallback_text_model
            self._vision_model = self._ollama_vision_model
            self._base_url = self._ollama_base_url
            self._timeout_seconds = self._ollama_timeout_seconds
            self._api_key = ""
            self._default_options = self._ollama_default_options
            self._enabled = self._provider == "ollama"

        self._timeout = httpx.Timeout(
            connect=min(5.0, self._timeout_seconds),
            read=self._timeout_seconds,
            write=20.0,
            pool=5.0,
        )

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
        if self._provider == "openai":
            if self._is_openai_in_cooldown():
                if self._openai_fallback_to_ollama and self._cooldown_allows_ollama_fallback():
                    logger.warning("openai cooldown active; using ollama json completion")
                    return self._ollama_json_completion(
                        system=system,
                        user=user,
                        model=None,
                        options=options,
                        request_timeout_seconds=request_timeout_seconds,
                    )
                return None
            candidates = self._text_model_candidates(model)
            per_attempt_timeout = self._per_attempt_timeout(request_timeout_seconds, len(candidates))
            rate_limited = False
            insufficient_quota = False
            for candidate in candidates:
                content = self._openai_chat_completion(
                    system=system,
                    user=user,
                    model=candidate,
                    options={**(options or {}), "format": "json"},
                    request_timeout_seconds=per_attempt_timeout,
                )
                if content == "__model_not_found__":
                    continue
                if content == "__insufficient_quota__":
                    insufficient_quota = True
                    self._mark_openai_rate_limited(reason="insufficient_quota")
                    break
                if content == "__rate_limited__":
                    rate_limited = True
                    self._mark_openai_rate_limited(reason="rate_limited")
                    break
                if content:
                    parsed = self._parse_json(content)
                    if isinstance(parsed, dict):
                        self._clear_openai_rate_limit()
                        return parsed
            if self._openai_fallback_to_ollama and not insufficient_quota:
                if rate_limited:
                    logger.warning("openai rate-limited; falling back to ollama json completion")
                return self._ollama_json_completion(
                    system=system,
                    user=user,
                    model=None,
                    options=options,
                    request_timeout_seconds=request_timeout_seconds,
                )
            return None
        return self._ollama_json_completion(
            system=system,
            user=user,
            model=model,
            options=options,
            request_timeout_seconds=request_timeout_seconds,
        )

    def _ollama_json_completion(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        candidates = self._ollama_text_model_candidates(model)
        per_attempt_timeout = self._per_attempt_timeout(request_timeout_seconds, len(candidates))
        for candidate in candidates:
            payload_generate = {
                "model": candidate,
                "stream": False,
                "format": "json",
                "prompt": f"{system}\n\n{user}",
                "keep_alive": self._keep_alive,
            }
            merged_options = self._merge_options_with_defaults(options, self._ollama_default_options)
            if merged_options:
                payload_generate["options"] = merged_options
            content = self._generate_content_for_base(
                self._ollama_base_url,
                payload_generate,
                request_timeout_seconds=per_attempt_timeout,
            )
            if content == "__model_not_found__":
                if self._auto_pull_missing_models and self._pull_model(candidate, base_url=self._ollama_base_url):
                    retry_content = self._generate_content_for_base(
                        self._ollama_base_url,
                        payload_generate,
                        request_timeout_seconds=per_attempt_timeout,
                    )
                    if retry_content and retry_content not in {"__generate_404__", "__model_not_found__"}:
                        parsed = self._parse_json(retry_content)
                        if isinstance(parsed, dict):
                            return parsed
                continue
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
        if self._provider == "openai":
            if self._is_openai_in_cooldown():
                if self._openai_fallback_to_ollama and self._cooldown_allows_ollama_fallback():
                    logger.warning("openai cooldown active; using ollama text completion")
                    return self._ollama_text_completion(
                        system=system,
                        user=user,
                        model=None,
                        options=options,
                        request_timeout_seconds=request_timeout_seconds,
                    )
                return None
            candidates = self._text_model_candidates(model)
            per_attempt_timeout = self._per_attempt_timeout(request_timeout_seconds, len(candidates))
            rate_limited = False
            insufficient_quota = False
            for candidate in candidates:
                content = self._openai_chat_completion(
                    system=system,
                    user=user,
                    model=candidate,
                    options=options,
                    request_timeout_seconds=per_attempt_timeout,
                )
                if content == "__model_not_found__":
                    continue
                if content == "__insufficient_quota__":
                    insufficient_quota = True
                    self._mark_openai_rate_limited(reason="insufficient_quota")
                    break
                if content == "__rate_limited__":
                    rate_limited = True
                    self._mark_openai_rate_limited(reason="rate_limited")
                    break
                if content:
                    self._clear_openai_rate_limit()
                    return content
            if self._openai_fallback_to_ollama and not insufficient_quota:
                if rate_limited:
                    logger.warning("openai rate-limited; falling back to ollama text completion")
                return self._ollama_text_completion(
                    system=system,
                    user=user,
                    model=None,
                    options=options,
                    request_timeout_seconds=request_timeout_seconds,
                )
            return None
        return self._ollama_text_completion(
            system=system,
            user=user,
            model=model,
            options=options,
            request_timeout_seconds=request_timeout_seconds,
        )

    def _ollama_text_completion(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
    ) -> str | None:
        candidates = self._ollama_text_model_candidates(model)
        per_attempt_timeout = self._per_attempt_timeout(request_timeout_seconds, len(candidates))
        for candidate in candidates:
            payload_generate = {
                "model": candidate,
                "stream": False,
                "prompt": f"{system}\n\n{user}",
                "keep_alive": self._keep_alive,
            }
            merged_options = self._merge_options_with_defaults(options, self._ollama_default_options)
            if merged_options:
                payload_generate["options"] = merged_options
            content = self._generate_content_for_base(
                self._ollama_base_url,
                payload_generate,
                request_timeout_seconds=per_attempt_timeout,
            )
            if content == "__model_not_found__":
                if self._auto_pull_missing_models and self._pull_model(candidate, base_url=self._ollama_base_url):
                    retry_content = self._generate_content_for_base(
                        self._ollama_base_url,
                        payload_generate,
                        request_timeout_seconds=per_attempt_timeout,
                    )
                    if retry_content and retry_content not in {"__generate_404__", "__model_not_found__"}:
                        return retry_content
                continue
            if content:
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
        if self._provider == "openai":
            if self._is_openai_in_cooldown():
                if not self._openai_fallback_to_ollama or not self._cooldown_allows_ollama_fallback():
                    return None
                logger.warning("openai cooldown active; using ollama vision completion")
            else:
                candidates = self._text_model_candidates(self._vision_model)
                per_attempt_timeout = self._per_attempt_timeout(request_timeout_seconds, len(candidates))
                rate_limited = False
                insufficient_quota = False
                for candidate in candidates:
                    content = self._openai_chat_completion(
                        system=system,
                        user=user_prompt,
                        model=candidate,
                        options={"format": "json"},
                        images=[image_b64],
                        request_timeout_seconds=per_attempt_timeout,
                    )
                    if content == "__model_not_found__":
                        continue
                    if content == "__insufficient_quota__":
                        insufficient_quota = True
                        self._mark_openai_rate_limited(reason="insufficient_quota")
                        break
                    if content == "__rate_limited__":
                        rate_limited = True
                        self._mark_openai_rate_limited(reason="rate_limited")
                        break
                    if content:
                        parsed = self._parse_json(content)
                        if isinstance(parsed, dict):
                            self._clear_openai_rate_limit()
                            return parsed
                if not self._openai_fallback_to_ollama or insufficient_quota:
                    return None
                if rate_limited:
                    logger.warning("openai rate-limited; falling back to ollama vision completion")
        payload = {
            "model": self._ollama_vision_model,
            "stream": False,
            "format": "json",
            "keep_alive": self._keep_alive,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
        }
        merged_options = self._merge_options_with_defaults(None, self._ollama_default_options)
        if merged_options:
            payload["options"] = merged_options
        content = self._chat_content_for_base(self._ollama_base_url, payload, request_timeout_seconds=request_timeout_seconds)
        if content == "__model_not_found__" and self._auto_pull_missing_models and self._pull_model(
            self._ollama_vision_model, base_url=self._ollama_base_url
        ):
            content = self._chat_content_for_base(self._ollama_base_url, payload, request_timeout_seconds=request_timeout_seconds)
        if content == "__chat_404__":
            payload_generate = {
                "model": self._ollama_vision_model,
                "stream": False,
                "format": "json",
                "prompt": f"{system}\n\n{user_prompt}",
                "images": [image_b64],
            }
            if merged_options:
                payload_generate["options"] = merged_options
            content = self._generate_content_for_base(self._ollama_base_url, payload_generate, request_timeout_seconds=request_timeout_seconds)
            if content == "__model_not_found__" and self._auto_pull_missing_models and self._pull_model(
                self._ollama_vision_model, base_url=self._ollama_base_url
            ):
                content = self._generate_content_for_base(
                    self._ollama_base_url,
                    payload_generate,
                    request_timeout_seconds=request_timeout_seconds,
                )
            if content == "__generate_404__":
                content = self._openai_chat_completion(
                    system=system,
                    user=user_prompt,
                    model=self._ollama_vision_model,
                    options={"format": "json"},
                    images=[image_b64],
                    request_timeout_seconds=request_timeout_seconds,
                )
        if not content:
            return None
        parsed = self._parse_json(content)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _chat_content(self, payload: dict[str, Any], *, request_timeout_seconds: float | None = None) -> str | None:
        return self._chat_content_for_base(self._base_url, payload, request_timeout_seconds=request_timeout_seconds)

    def _chat_content_for_base(
        self,
        base_url: str,
        payload: dict[str, Any],
        *,
        request_timeout_seconds: float | None = None,
    ) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout_for(request_timeout_seconds)) as client:
                response = client.post(f"{base_url}/api/chat", json=payload)
                if response.status_code == 404:
                    try:
                        error = (response.json() or {}).get("error", "")
                    except Exception:
                        error = response.text
                    if "model" in str(error).lower() and "not found" in str(error).lower():
                        return "__model_not_found__"
                    return "__chat_404__"
                response.raise_for_status()
                data = response.json()
            return (data.get("message") or {}).get("content")
        except Exception as exc:  # pragma: no cover
            logger.exception("ollama chat failed: %s", exc)
            return None

    def _generate_content(self, payload: dict[str, Any], *, request_timeout_seconds: float | None = None) -> str | None:
        return self._generate_content_for_base(self._base_url, payload, request_timeout_seconds=request_timeout_seconds)

    def _generate_content_for_base(
        self,
        base_url: str,
        payload: dict[str, Any],
        *,
        request_timeout_seconds: float | None = None,
    ) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout_for(request_timeout_seconds)) as client:
                response = client.post(f"{base_url}/api/generate", json=payload)
                if response.status_code == 404:
                    try:
                        error = (response.json() or {}).get("error", "")
                    except Exception:
                        error = response.text
                    if "model" in str(error).lower() and "not found" in str(error).lower():
                        return "__model_not_found__"
                    return "__generate_404__"
                response.raise_for_status()
                data = response.json()
            return data.get("response")
        except Exception as exc:  # pragma: no cover
            logger.exception("ollama generate failed: %s", exc)
            return None

    def _openai_chat_completion(
        self,
        *,
        system: str,
        user: str,
        model: str,
        options: dict[str, Any] | None = None,
        images: list[str] | None = None,
        request_timeout_seconds: float | None = None,
    ) -> str | None:
        url = self._openai_chat_url()
        messages: list[dict[str, Any]]
        if images:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": user}]
            for image_b64 in images:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "num_predict" in options:
                payload["max_completion_tokens"] = options["num_predict"]
            if "format" in options and options["format"] == "json":
                payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self._provider == "openai":
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            with httpx.Client(timeout=self._timeout_for(request_timeout_seconds)) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code == 429:
                    self._openai_compat_available = True
                    try:
                        body = response.json() or {}
                        err = body.get("error") if isinstance(body, dict) else None
                        code = (err.get("code", "") if isinstance(err, dict) else "").lower()
                        err_type = (err.get("type", "") if isinstance(err, dict) else "").lower()
                        if code == "insufficient_quota" or err_type == "insufficient_quota":
                            logger.warning("openai insufficient quota detected; entering cooldown window")
                            return "__insufficient_quota__"
                    except Exception:
                        pass
                    return "__rate_limited__"
                if response.status_code in (400, 404):
                    self._openai_compat_available = False
                    try:
                        error_payload = response.json() or {}
                        error_obj = error_payload.get("error", {})
                        error = error_obj.get("message", "") if isinstance(error_obj, dict) else str(error_obj)
                    except Exception:
                        error = response.text
                    if "model" in str(error).lower() and "not found" in str(error).lower():
                        return "__model_not_found__"
                    logger.warning("openai chat request rejected (%s): %s", response.status_code, error)
                    return None
                response.raise_for_status()
                data = response.json()
            self._openai_compat_available = True
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                merged = "\n".join(t for t in texts if t.strip()).strip()
                return merged or None
            return None
        except Exception as exc:  # pragma: no cover
            logger.exception("chat completion failed provider=%s: %s", self._provider, exc)
            return None

    def _openai_chat_url(self) -> str:
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/chat/completions"
        return f"{self._base_url}/v1/chat/completions"

    def _pull_model(self, model: str, *, base_url: str | None = None) -> bool:
        try:
            target_base = base_url or self._base_url
            with httpx.Client(timeout=self._timeout_for(120)) as client:
                response = client.post(
                    f"{target_base}/api/pull",
                    json={"model": model, "stream": False},
                )
                if response.status_code == 404:
                    return False
                response.raise_for_status()
            return True
        except Exception as exc:  # pragma: no cover
            logger.exception("ollama model auto-pull failed model=%s error=%s", model, exc)
            return False

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

    def _mark_openai_rate_limited(self, *, reason: str = "rate_limited") -> None:
        cooldown = self._openai_rate_limit_cooldown
        if reason == "insufficient_quota":
            cooldown = self._openai_insufficient_quota_cooldown
        if cooldown.total_seconds() <= 0:
            return
        until = datetime.now(tz=timezone.utc) + cooldown
        self._openai_rate_limited_until = until
        self._openai_cooldown_reason = reason
        self.__class__._shared_openai_rate_limited_until = until
        self.__class__._shared_openai_cooldown_reason = reason

    def _clear_openai_rate_limit(self) -> None:
        self._openai_rate_limited_until = None
        self._openai_cooldown_reason = None
        self.__class__._shared_openai_rate_limited_until = None
        self.__class__._shared_openai_cooldown_reason = None

    def _is_openai_in_cooldown(self) -> bool:
        shared = self.__class__._shared_openai_rate_limited_until
        if shared is not None:
            self._openai_rate_limited_until = shared
        shared_reason = self.__class__._shared_openai_cooldown_reason
        if shared_reason is not None:
            self._openai_cooldown_reason = shared_reason
        if not self._openai_rate_limited_until:
            return False
        return datetime.now(tz=timezone.utc) < self._openai_rate_limited_until

    def _cooldown_allows_ollama_fallback(self) -> bool:
        return (self._openai_cooldown_reason or "").lower() != "insufficient_quota"

    def _text_model_candidates(self, model: str | None) -> list[str]:
        primary = (model or self._text_model).strip()
        candidates = [primary]
        fallback = self._fallback_text_model.strip() if self._fallback_text_model else ""
        if fallback and fallback not in candidates:
            candidates.append(fallback)
        return candidates

    def _ollama_text_model_candidates(self, model: str | None) -> list[str]:
        requested = (model or "").strip()
        # Ignore OpenAI model names when falling back into Ollama.
        if not requested or requested.startswith("gpt-"):
            primary = self._ollama_text_model.strip()
        else:
            primary = requested
        candidates = [primary]
        fallback = self._ollama_fallback_text_model.strip()
        if fallback and fallback not in candidates:
            candidates.append(fallback)
        return candidates

    @staticmethod
    def _per_attempt_timeout(total_timeout: float | None, attempts: int) -> float | None:
        if total_timeout is None:
            return None
        safe_attempts = max(1, attempts)
        return max(8.0, float(total_timeout) / safe_attempts)

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

    @staticmethod
    def _build_default_options(settings: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        option_pairs = (
            ("num_gpu", getattr(settings, "ollama_option_num_gpu", None)),
            ("main_gpu", getattr(settings, "ollama_option_main_gpu", None)),
            ("num_thread", getattr(settings, "ollama_option_num_thread", None)),
            ("num_batch", getattr(settings, "ollama_option_num_batch", None)),
        )
        for key, value in option_pairs:
            if value is not None:
                defaults[key] = value
        if bool(getattr(settings, "ollama_option_low_vram", False)):
            defaults["low_vram"] = True
        return defaults

    def _merge_options(self, options: dict[str, Any] | None) -> dict[str, Any] | None:
        merged = dict(getattr(self, "_default_options", {}) or {})
        if options:
            merged.update({key: value for key, value in options.items() if value is not None})
        return merged or None

    @staticmethod
    def _merge_options_with_defaults(
        options: dict[str, Any] | None,
        defaults: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        merged = dict(defaults or {})
        if options:
            merged.update({key: value for key, value in options.items() if value is not None})
        return merged or None

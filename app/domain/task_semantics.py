from __future__ import annotations

import re


ACTION_KIND_QUICK_MESSAGE = "quick_message"
ACTION_KIND_QUICK_ADMIN = "quick_admin"
ACTION_KIND_WORK_BLOCK = "work_block"
ACTION_KIND_PROJECT_CHUNK = "project_chunk"


def infer_action_kind(title: str, *, deadline_text: str | None = None, start_after=None, metadata: dict | None = None) -> str:
    if metadata and isinstance(metadata.get("action_kind"), str):
        return str(metadata["action_kind"])

    lowered = title.lower().strip()
    if not lowered:
        return ACTION_KIND_WORK_BLOCK

    if lowered.startswith(("text ", "call ", "reply ", "dm ", "ping ")):
        return ACTION_KIND_QUICK_MESSAGE
    if lowered.startswith("send "):
        if any(token in lowered for token in (" text", " message", " dm", " ping", " reminder", "check-in", "check in")):
            return ACTION_KIND_QUICK_MESSAGE
        if any(token in lowered for token in ("email", "application", "proposal", "update")):
            return ACTION_KIND_WORK_BLOCK
    if lowered.startswith(("pay ", "book ", "schedule ", "renew ", "cancel ", "buy ")):
        return ACTION_KIND_QUICK_ADMIN
    if any(token in lowered for token in ("cad", "portfolio", "website", "design", "prototype", "model", "board", "slides", "draft")):
        return ACTION_KIND_PROJECT_CHUNK
    if lowered.startswith(("finish ", "build ", "prepare ", "write ", "design ", "model ", "fix ", "clean up ", "outline ")):
        return ACTION_KIND_PROJECT_CHUNK
    if start_after is not None and deadline_text:
        return ACTION_KIND_WORK_BLOCK
    return ACTION_KIND_WORK_BLOCK


def default_next_step(title: str, *, action_kind: str | None = None) -> str:
    title = title.strip()
    lowered = title.lower()
    kind = action_kind or infer_action_kind(title)

    if kind in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}:
        if not title:
            return "handle it"
        return title[0].lower() + title[1:]

    if lowered.startswith("submit "):
        return f"do a final proofread, then {title[0].lower() + title[1:]}"
    if "submit" in lowered:
        return f"do a final proofread, then submit {title}"
    if lowered.startswith("send "):
        return title
    if "send" in lowered or "email" in lowered or "text" in lowered:
        return f"send {title}"
    if lowered.startswith("finish "):
        return title
    if "finish" in lowered:
        return f"finish the first complete pass on {title}"
    if lowered.startswith("fix "):
        return title
    if lowered.startswith("review "):
        return title
    return f"get a real first pass going on {title}"


def is_soft_later_phrase(phrase: str | None) -> bool:
    if not phrase:
        return False
    lowered = phrase.lower().strip()
    return lowered in {"later", "sometime", "eventually"} or lowered.startswith("later ")


def humanize_window_phrase(phrase: str | None) -> str | None:
    if not phrase:
        return None
    lowered = re.sub(r"\s+", " ", phrase.lower()).strip()
    replacements = {
        "tomorrow morning": "tomorrow morning",
        "tmr morning": "tomorrow morning",
        "tomorrow night": "tomorrow night",
        "tmr night": "tomorrow night",
        "tonight": "tonight",
        "this weekend": "this weekend",
        "weekend": "this weekend",
    }
    return replacements.get(lowered, lowered)


def is_broad_window_phrase(phrase: str | None) -> bool:
    lowered = humanize_window_phrase(phrase)
    if not lowered:
        return False
    return lowered in {
        "this weekend",
        "weekend",
        "later",
        "sometime",
        "eventually",
        "after class",
        "before studio",
    }

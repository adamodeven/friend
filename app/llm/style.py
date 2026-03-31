from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleProfile:
    name: str
    system_hint: str


STYLE_PROFILES: dict[str, StyleProfile] = {
    "casual_cool": StyleProfile(
        name="casual_cool",
        system_hint=(
            "sound like a smart, socially fluent texting friend. concise, casual, natural slang only when it fits. "
            "stay competent and focused. no cringe. no corporate tone."
        ),
    ),
    "direct": StyleProfile(
        name="direct",
        system_hint="be short, clear, decisive, and practical. little fluff.",
    ),
    "more_serious": StyleProfile(
        name="more_serious",
        system_hint="calm, serious, concise, execution-focused, still human.",
    ),
}


def get_style_profile(name: str) -> StyleProfile:
    return STYLE_PROFILES.get(name, STYLE_PROFILES["casual_cool"])


def chunk_sms(text: str, max_chars: int = 320) -> list[str]:
    stripped = " ".join(text.split())
    if len(stripped) <= max_chars:
        return [stripped]

    chunks: list[str] = []
    current = []
    current_len = 0
    for sentence in stripped.split(". "):
        part = sentence.strip()
        if not part:
            continue
        if not part.endswith("."):
            part += "."
        if current_len + len(part) + 1 > max_chars and current:
            chunks.append(" ".join(current).strip())
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += len(part) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


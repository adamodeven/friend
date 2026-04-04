from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleProfile:
    name: str
    system_hint: str
    max_sms_chars: int
    soft_chunk_chars: int
    max_chunks: int
    guardrails: tuple[str, ...]


STYLE_PROFILES: dict[str, StyleProfile] = {
    "casual_cool": StyleProfile(
        name="casual_cool",
        system_hint=(
            "sound like a sharp, socially fluent friend who happens to be very good at keeping things moving. "
            "concise, current, competent, and lightly casual. lowercase is fine when it feels natural. "
            "slang stays light and earned."
        ),
        max_sms_chars=320,
        soft_chunk_chars=150,
        max_chunks=3,
        guardrails=(
            "keep replies short by default",
            "sound human and current, not try-hard",
            "do not turn every message into a demand for the next task",
            "push toward one concrete next move only when the user is vague, overloaded, or clearly asking for direction",
            "urgency should feel real, not hypey",
            "no corny lines, therapy language, or corporate phrasing",
        ),
    ),
    "direct": StyleProfile(
        name="direct",
        system_hint="be terse, decisive, and practical. say the useful thing fast.",
        max_sms_chars=260,
        soft_chunk_chars=120,
        max_chunks=2,
        guardrails=(
            "default to one short text unless a second is necessary",
            "low slang and minimal filler",
            "lead with the decision or next action",
            "stay firm without sounding robotic",
        ),
    ),
    "more_serious": StyleProfile(
        name="more_serious",
        system_hint="be calm, grounded, concise, and execution-focused while still sounding like a real person.",
        max_sms_chars=300,
        soft_chunk_chars=145,
        max_chunks=3,
        guardrails=(
            "keep a calm texting cadence",
            "stay concise and low-fluff",
            "sound steady under urgency",
            "avoid dashboards, lectures, and fake warmth",
        ),
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

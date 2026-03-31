from __future__ import annotations

import re
from difflib import SequenceMatcher


class RepetitionGuard:
    def __init__(self, *, similarity_threshold: float = 0.88) -> None:
        self.similarity_threshold = similarity_threshold

    def is_too_similar(self, candidate: str, recent_messages: list[str]) -> bool:
        norm_candidate = self._normalize(candidate)
        if not norm_candidate:
            return False
        for previous in recent_messages[-8:]:
            norm_previous = self._normalize(previous)
            if not norm_previous:
                continue
            if SequenceMatcher(a=norm_candidate, b=norm_previous).ratio() >= self.similarity_threshold:
                return True
        return False

    def avoid_phrases(self, recent_messages: list[str], limit: int = 3) -> list[str]:
        phrases: list[str] = []
        for msg in recent_messages[-limit:]:
            words = msg.strip().split()
            if not words:
                continue
            phrases.append(" ".join(words[: min(6, len(words))]))
        return phrases

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        return " ".join(lowered.split())


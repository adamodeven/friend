from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import dateparser


def now_tz(timezone: str) -> datetime:
    return datetime.now(tz=ZoneInfo(timezone))


def parse_human_time(
    text: str,
    *,
    timezone: str,
    relative_base: datetime | None = None,
) -> tuple[datetime | None, float]:
    base = relative_base or now_tz(timezone)
    dt = dateparser.parse(
        text,
        settings={
            "TIMEZONE": timezone,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": base,
            "PREFER_DATES_FROM": "future",
        },
    )
    if not dt:
        return None, 0.0
    confidence = 0.9
    lowered = text.lower()
    ambiguous = ("later", "sometime", "eventually", "after class")
    if any(token in lowered for token in ambiguous):
        confidence = 0.45
    elif "tomorrow morning" in lowered or "tonight" in lowered or "eod" in lowered:
        confidence = 0.7
    return dt, confidence


def time_window_for_context(context_text: str, timezone: str) -> tuple[datetime, datetime]:
    now = now_tz(timezone)
    lowered = context_text.lower()
    if "class" in lowered:
        return now, now + timedelta(hours=2)
    if "driving" in lowered:
        return now, now + timedelta(minutes=45)
    if "dinner" in lowered or "social" in lowered:
        return now, now + timedelta(hours=2)
    if "all nighter" in lowered:
        return now, now + timedelta(hours=8)
    return now, now + timedelta(hours=1)


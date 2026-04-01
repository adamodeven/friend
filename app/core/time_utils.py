from __future__ import annotations

from datetime import datetime, timedelta
import re
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
    lowered = text.lower().strip()

    special = _parse_special_time_phrase(lowered, base=base)
    if special:
        return special

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
    ambiguous = ("later", "sometime", "eventually", "after class", "before studio", "this weekend")
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


def _parse_special_time_phrase(text: str, *, base: datetime) -> tuple[datetime, float] | None:
    lowered = text.lower()
    local_tz = base.tzinfo or ZoneInfo("UTC")

    if "tomorrow morning" in lowered:
        return _at_local(base, days=1, hour=9, minute=0, tz=local_tz), 0.72
    if "tomorrow night" in lowered:
        return _at_local(base, days=1, hour=21, minute=0, tz=local_tz), 0.72
    if "tonight" in lowered:
        candidate = _at_local(base, days=0, hour=21, minute=0, tz=local_tz)
        if candidate <= base:
            candidate = _at_local(base, days=1, hour=21, minute=0, tz=local_tz)
        return candidate, 0.68
    if "by eod" in lowered or re_match_word(lowered, "eod"):
        candidate = _at_local(base, days=0, hour=17, minute=0, tz=local_tz)
        if candidate <= base:
            candidate = _at_local(base, days=1, hour=17, minute=0, tz=local_tz)
        return candidate, 0.64
    if "this weekend" in lowered:
        # Use Saturday 10am local as a conservative anchor and ask follow-up in state layer if needed.
        days_until_sat = (5 - base.weekday()) % 7
        target = _at_local(base, days=days_until_sat, hour=10, minute=0, tz=local_tz)
        if target <= base:
            target = _at_local(base, days=days_until_sat + 7, hour=10, minute=0, tz=local_tz)
        return target, 0.5
    if "after class" in lowered:
        return base + timedelta(hours=2), 0.4
    if "before studio" in lowered:
        candidate = _at_local(base, days=1, hour=8, minute=30, tz=local_tz)
        return candidate, 0.42
    if re_match_word(lowered, "later"):
        return base + timedelta(hours=4), 0.35
    return None


def _at_local(base: datetime, *, days: int, hour: int, minute: int, tz) -> datetime:
    local = base.astimezone(tz)
    shifted = local + timedelta(days=days)
    return shifted.replace(hour=hour, minute=minute, second=0, microsecond=0)


def re_match_word(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None

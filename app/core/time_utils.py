from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Mapping
from zoneinfo import ZoneInfo

import dateparser

from app.schemas.intent import ParsedDeadline


_LATER_PHRASES = {"later", "sometime", "eventually"}
_TASK_TIME_PHRASES = (
    "tonight",
    "tomorrow morning",
    "tomorrow night",
    "this weekend",
    "after class",
    "before studio",
    "by eod",
    "eod",
)


def now_tz(timezone: str) -> datetime:
    return datetime.now(tz=ZoneInfo(timezone))


def interpret_time_reference(
    text: str,
    *,
    timezone: str,
    relative_base: datetime | None = None,
    context_anchors: Mapping[str, datetime | tuple[datetime, datetime]] | None = None,
) -> ParsedDeadline:
    zone = ZoneInfo(timezone)
    base = _normalize_base(relative_base=relative_base, zone=zone)
    raw_text = text.strip()
    lowered = raw_text.lower()

    special = _parse_special_time_phrase(
        lowered,
        base=base,
        timezone=timezone,
        context_anchors=context_anchors,
        source_phrase=raw_text,
    )
    if special is not None:
        return special

    dt = dateparser.parse(
        raw_text,
        settings={
            "TIMEZONE": timezone,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": base,
            "PREFER_DATES_FROM": "future",
        },
    )
    if not dt:
        return ParsedDeadline(source_phrase=raw_text or None, timezone=timezone)

    dt = dt.astimezone(zone)
    confidence = 0.9
    granularity = _infer_general_granularity(raw_text)
    return ParsedDeadline(
        source_phrase=raw_text,
        deadline_at=dt,
        timezone=timezone,
        confidence=confidence,
        is_ambiguous=False,
        granularity=granularity,
    )


def parse_human_time(
    text: str,
    *,
    timezone: str,
    relative_base: datetime | None = None,
    context_anchors: Mapping[str, datetime | tuple[datetime, datetime]] | None = None,
) -> tuple[datetime | None, float]:
    parsed = interpret_time_reference(
        text,
        timezone=timezone,
        relative_base=relative_base,
        context_anchors=context_anchors,
    )
    return parsed.deadline_at or parsed.soft_deadline_at, parsed.confidence


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


def _parse_special_time_phrase(
    text: str,
    *,
    base: datetime,
    timezone: str,
    context_anchors: Mapping[str, datetime | tuple[datetime, datetime]] | None,
    source_phrase: str,
) -> ParsedDeadline | None:
    if "tomorrow morning" in text:
        start = _at_local(base, days=1, hour=8, minute=0)
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            deadline_at=_at_local(base, days=1, hour=11, minute=0),
            soft_deadline_at=start,
            confidence=0.78,
            granularity="part_of_day",
        )
    if "tomorrow night" in text:
        start = _at_local(base, days=1, hour=18, minute=0)
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            deadline_at=_at_local(base, days=1, hour=21, minute=0),
            soft_deadline_at=start,
            confidence=0.76,
            granularity="part_of_day",
        )
    if "tonight" in text:
        start = _at_local(base, days=0, hour=18, minute=0)
        end = _at_local(base, days=0, hour=21, minute=0)
        if end <= base:
            start = _at_local(base, days=1, hour=18, minute=0)
            end = _at_local(base, days=1, hour=21, minute=0)
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            deadline_at=end,
            soft_deadline_at=max(start, base + timedelta(minutes=30)),
            confidence=0.74,
            granularity="part_of_day",
        )
    if "by eod" in text or re_match_word(text, "eod"):
        end = _at_local(base, days=0, hour=17, minute=0)
        if end <= base:
            end = _at_local(base, days=1, hour=17, minute=0)
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            deadline_at=end,
            soft_deadline_at=end - timedelta(hours=2),
            confidence=0.73,
            granularity="day",
        )
    if "this weekend" in text:
        start = _next_weekend_anchor(base, day_index=5, hour=10, minute=0)
        end = _next_weekend_anchor(base, day_index=6, hour=18, minute=0)
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            deadline_at=end,
            soft_deadline_at=start,
            confidence=0.52,
            is_ambiguous=True,
            ambiguity_reason="weekend window is broad",
            granularity="weekend",
        )
    if "after class" in text:
        class_end = _resolve_anchor(context_anchors, "class")
        if class_end is None:
            return _build_deadline(
                source_phrase=source_phrase,
                timezone=timezone,
                soft_deadline_at=base + timedelta(hours=2),
                confidence=0.34,
                is_ambiguous=True,
                ambiguity_reason="missing class schedule anchor",
                granularity="unknown",
            )
        soft_at = class_end + timedelta(minutes=15)
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            deadline_at=class_end + timedelta(hours=2),
            soft_deadline_at=soft_at,
            confidence=0.58,
            is_ambiguous=True,
            ambiguity_reason="after class depends on the real class end",
            granularity="hour",
        )
    if "before studio" in text:
        studio_start = _resolve_anchor(context_anchors, "studio")
        if studio_start is None:
            return _build_deadline(
                source_phrase=source_phrase,
                timezone=timezone,
                soft_deadline_at=_at_local(base, days=1, hour=7, minute=30),
                confidence=0.38,
                is_ambiguous=True,
                ambiguity_reason="missing studio schedule anchor",
                granularity="unknown",
            )
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            deadline_at=studio_start,
            soft_deadline_at=studio_start - timedelta(hours=2),
            confidence=0.69,
            granularity="hour",
        )
    if any(re_match_word(text, phrase) for phrase in _LATER_PHRASES):
        return _build_deadline(
            source_phrase=source_phrase,
            timezone=timezone,
            soft_deadline_at=base + timedelta(hours=3),
            confidence=0.3,
            is_ambiguous=True,
            ambiguity_reason="later is underspecified",
            granularity="unknown",
        )
    return None


def _normalize_base(*, relative_base: datetime | None, zone: ZoneInfo) -> datetime:
    if relative_base is None:
        return now_tz(zone.key)
    if relative_base.tzinfo is None:
        return relative_base.replace(tzinfo=zone)
    return relative_base.astimezone(zone)


def _build_deadline(
    *,
    source_phrase: str,
    timezone: str,
    deadline_at: datetime | None = None,
    soft_deadline_at: datetime | None = None,
    confidence: float,
    is_ambiguous: bool = False,
    ambiguity_reason: str | None = None,
    granularity: str = "unknown",
) -> ParsedDeadline:
    return ParsedDeadline(
        source_phrase=source_phrase,
        deadline_at=deadline_at,
        soft_deadline_at=soft_deadline_at,
        timezone=timezone,
        confidence=confidence,
        is_ambiguous=is_ambiguous,
        ambiguity_reason=ambiguity_reason,
        granularity=granularity,
    )


def _resolve_anchor(
    context_anchors: Mapping[str, datetime | tuple[datetime, datetime]] | None,
    prefix: str,
) -> datetime | None:
    if not context_anchors:
        return None

    for key in (f"{prefix}_end", f"{prefix}_start", prefix, f"next_{prefix}", f"next_{prefix}_start"):
        candidate = context_anchors.get(key)
        if isinstance(candidate, tuple):
            for item in reversed(candidate):
                if isinstance(item, datetime):
                    return item
        elif isinstance(candidate, datetime):
            return candidate
    return None


def _next_weekend_anchor(base: datetime, *, day_index: int, hour: int, minute: int) -> datetime:
    days_until = (day_index - base.weekday()) % 7
    target = _at_local(base, days=days_until, hour=hour, minute=minute)
    if target <= base:
        target = _at_local(base, days=days_until + 7, hour=hour, minute=minute)
    return target


def _at_local(base: datetime, *, days: int, hour: int, minute: int) -> datetime:
    local = base.astimezone(base.tzinfo or ZoneInfo("UTC"))
    shifted = local + timedelta(days=days)
    return shifted.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _infer_general_granularity(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in _TASK_TIME_PHRASES):
        return "part_of_day"
    if re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", lowered):
        return "exact"
    if any(word in lowered for word in ("week", "weekend")):
        return "weekend" if "weekend" in lowered else "week"
    return "day"


def re_match_word(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None

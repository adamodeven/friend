from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.time_utils import parse_human_time


def test_parse_human_time_handles_required_phrases():
    base = datetime(2026, 4, 1, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    cases = [
        "tonight",
        "tomorrow morning",
        "after class",
        "later",
        "this weekend",
        "by eod",
        "before studio",
    ]
    for phrase in cases:
        parsed, confidence = parse_human_time(phrase, timezone="America/New_York", relative_base=base)
        assert parsed is not None, phrase
        assert confidence > 0.0, phrase


def test_parse_human_time_marks_ambiguous_phrases_lower_confidence():
    base = datetime(2026, 4, 1, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    parsed, confidence = parse_human_time("later", timezone="America/New_York", relative_base=base)
    assert parsed is not None
    assert confidence < 0.6

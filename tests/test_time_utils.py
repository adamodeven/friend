from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.time_utils import interpret_time_reference, parse_human_time


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


def test_interpret_time_reference_builds_structured_windows_for_common_phrases():
    base = datetime(2026, 4, 1, 14, 0, tzinfo=ZoneInfo("America/New_York"))

    tonight = interpret_time_reference("tonight", timezone="America/New_York", relative_base=base)
    assert tonight.soft_deadline_at == datetime(2026, 4, 1, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    assert tonight.deadline_at == datetime(2026, 4, 1, 21, 0, tzinfo=ZoneInfo("America/New_York"))
    assert tonight.granularity == "part_of_day"
    assert tonight.is_ambiguous is False

    tomorrow_morning = interpret_time_reference("tomorrow morning", timezone="America/New_York", relative_base=base)
    assert tomorrow_morning.soft_deadline_at == datetime(2026, 4, 2, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    assert tomorrow_morning.deadline_at == datetime(2026, 4, 2, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    assert tomorrow_morning.granularity == "part_of_day"

    weekend = interpret_time_reference("this weekend", timezone="America/New_York", relative_base=base)
    assert weekend.soft_deadline_at == datetime(2026, 4, 4, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert weekend.deadline_at == datetime(2026, 4, 5, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    assert weekend.is_ambiguous is True
    assert weekend.granularity == "weekend"


def test_interpret_time_reference_uses_contextual_anchors_for_class_and_studio():
    base = datetime(2026, 4, 1, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    anchors = {
        "class_end": datetime(2026, 4, 1, 15, 15, tzinfo=ZoneInfo("America/New_York")),
        "studio_start": datetime(2026, 4, 2, 13, 30, tzinfo=ZoneInfo("America/New_York")),
    }

    after_class = interpret_time_reference(
        "after class",
        timezone="America/New_York",
        relative_base=base,
        context_anchors=anchors,
    )
    assert after_class.soft_deadline_at == datetime(2026, 4, 1, 15, 30, tzinfo=ZoneInfo("America/New_York"))
    assert after_class.deadline_at == datetime(2026, 4, 1, 17, 15, tzinfo=ZoneInfo("America/New_York"))
    assert after_class.is_ambiguous is True

    before_studio = interpret_time_reference(
        "before studio",
        timezone="America/New_York",
        relative_base=base,
        context_anchors=anchors,
    )
    assert before_studio.soft_deadline_at == datetime(2026, 4, 2, 11, 30, tzinfo=ZoneInfo("America/New_York"))
    assert before_studio.deadline_at == datetime(2026, 4, 2, 13, 30, tzinfo=ZoneInfo("America/New_York"))
    assert before_studio.is_ambiguous is False


def test_interpret_time_reference_keeps_missing_context_phrases_soft_and_ambiguous():
    base = datetime(2026, 4, 1, 14, 0, tzinfo=ZoneInfo("America/New_York"))

    after_class = interpret_time_reference("after class", timezone="America/New_York", relative_base=base)
    assert after_class.deadline_at is None
    assert after_class.soft_deadline_at == datetime(2026, 4, 1, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    assert after_class.is_ambiguous is True

    before_studio = interpret_time_reference("before studio", timezone="America/New_York", relative_base=base)
    assert before_studio.deadline_at is None
    assert before_studio.soft_deadline_at == datetime(2026, 4, 2, 7, 30, tzinfo=ZoneInfo("America/New_York"))
    assert before_studio.is_ambiguous is True


def test_parse_human_time_normalizes_relative_base_to_requested_timezone():
    base = datetime(2026, 4, 1, 23, 30, tzinfo=ZoneInfo("UTC"))
    parsed, confidence = parse_human_time(
        "tomorrow morning",
        timezone="America/Los_Angeles",
        relative_base=base,
    )
    assert parsed == datetime(2026, 4, 2, 11, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert confidence >= 0.7

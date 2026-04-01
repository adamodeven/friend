from app.llm.extraction import IntentExtractor


class _CountingAdapter:
    def __init__(self) -> None:
        self.json_calls = 0

    def json_completion(self, **kwargs):  # noqa: ANN003,ANN201
        self.json_calls += 1
        return None


class _PayloadAdapter:
    def __init__(self, payload):
        self.payload = payload
        self.json_calls = 0

    def json_completion(self, **kwargs):  # noqa: ANN003,ANN201
        self.json_calls += 1
        return self.payload


def test_extract_add_task_from_plain_text():
    extractor = IntentExtractor()
    result = extractor.extract("yo I need to finish the CAD for the enclosure by tomorrow night", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert "cad" in result.task.title.lower()


def test_extract_context_signal():
    extractor = IntentExtractor()
    result = extractor.extract("in class rn", "America/New_York")
    assert result.intent == "context_signal"


def test_extract_timeline_query():
    extractor = IntentExtractor()
    result = extractor.extract("what do I have due this week", "America/New_York")
    assert result.intent == "timeline_query"


def test_extract_status_query():
    extractor = IntentExtractor()
    result = extractor.extract("what do you do?", "America/New_York")
    assert result.intent == "status_query"


def test_extract_meta_architecture_query_as_status_query():
    extractor = IntentExtractor()
    result = extractor.extract("are these canned responses or live ai generated?", "America/New_York")
    assert result.intent == "status_query"


def test_extract_live_status_check_as_status_query():
    extractor = IntentExtractor()
    result = extractor.extract("are you actually live now?", "America/New_York")
    assert result.intent == "status_query"


def test_fallback_task_title_removes_temporal_prefix_noise():
    extractor = IntentExtractor()
    result = extractor.extract(
        "and then tmr morning I need to submit my scout job application",
        "America/New_York",
    )
    assert result.intent == "add_task"
    assert result.task is not None
    assert "tmr" not in result.task.title.lower()
    assert "tomorrow" not in result.task.title.lower()
    assert "submit" in result.task.title.lower()


def test_fallback_task_title_removes_leading_i_need_to_noise():
    extractor = IntentExtractor()
    result = extractor.extract(
        "i need to submit my scout job application tomorrow morning",
        "America/New_York",
    )
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.title.lower().startswith("i ") is False
    assert result.task.title.lower().startswith("need to ") is False
    assert "submit my scout job application" in result.task.title.lower()


def test_high_confidence_add_task_still_attempts_llm_before_fallback():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract(
        "i need to submit the application tomorrow morning and then prepare slides for class",
        "America/New_York",
    )
    assert result.intent == "add_task"
    assert adapter.json_calls >= 1


def test_simple_single_task_short_circuits_llm_for_latency():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("i need to submit the application tomorrow morning", "America/New_York")
    assert result.intent == "add_task"
    assert adapter.json_calls == 0


def test_high_confidence_context_signal_short_circuits_llm_for_latency():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("in class rn", "America/New_York")
    assert result.intent == "context_signal"
    assert adapter.json_calls == 0


def test_simple_checkin_short_circuits_llm():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("yo", "America/New_York")
    assert result.intent == "general_chat"
    assert adapter.json_calls == 0


def test_llm_extracted_task_title_gets_sanitized():
    payload = {
        "intent": "add_task",
        "confidence": 0.6,
        "task": {
            "title": "i need to submit my scout job application tomorrow morning",
            "description": None,
            "project": None,
            "deadline_text": "tomorrow morning",
            "priority": 2,
            "confidence": 0.7,
            "next_step": None,
        },
    }
    adapter = _PayloadAdapter(payload)
    extractor = IntentExtractor(adapter=adapter)
    result = extractor._extract_with_llm("submit my scout job app tomorrow morning", "America/New_York")
    assert adapter.json_calls == 1
    assert result is not None
    assert result.task is not None
    assert result.task.title.lower().startswith("i ") is False
    assert "tomorrow morning" not in result.task.title.lower()


def test_fallback_detects_dependency_blocker_language():
    extractor = IntentExtractor()
    result = extractor.extract("i keep getting distracted because i need to fix the website first", "America/New_York")
    assert result.intent in {"update_task", "reflection"}
    assert result.blockers or result.intent == "reflection"


def test_bulk_clear_task_language_maps_to_update_bulk_action():
    extractor = IntentExtractor()
    result = extractor.extract("alright we're getting there. can you clear all tasks?", "America/New_York")
    assert result.intent == "update_task"
    assert result.task_updates.get("bulk_action") == "clear_active_tasks"


def test_ambiguous_time_sets_clarification_hint():
    extractor = IntentExtractor()
    result = extractor.extract("need to send that email later", "America/New_York")
    assert result.intent == "add_task"
    assert result.time_reference is not None
    assert result.time_confidence <= 0.6
    assert result.needs_clarification is True


def test_timeline_query_fallback_wins_over_bad_llm_add_task_guess():
    payload = {
        "intent": "add_task",
        "confidence": 0.76,
        "task": {
            "title": "what do i need to get done tonight",
            "description": None,
            "project": None,
            "deadline_text": None,
            "priority": 2,
            "confidence": 0.6,
            "next_step": None,
        },
    }
    adapter = _PayloadAdapter(payload)
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("what do i need to get done tonight", "America/New_York")
    assert result.intent == "timeline_query"

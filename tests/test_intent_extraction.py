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


def test_high_confidence_add_task_skips_llm_intent_call():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("i need to submit the application tomorrow morning", "America/New_York")
    assert result.intent == "add_task"
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

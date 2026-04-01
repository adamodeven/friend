from app.llm.extraction import IntentExtractor


class _CountingAdapter:
    def __init__(self) -> None:
        self.json_calls = 0

    def json_completion(self, **kwargs):  # noqa: ANN003,ANN201
        self.json_calls += 1
        return None


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


def test_high_confidence_add_task_skips_llm_intent_call():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("i need to submit the application tomorrow morning", "America/New_York")
    assert result.intent == "add_task"
    assert adapter.json_calls == 0

from app.llm.extraction import IntentExtractor


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

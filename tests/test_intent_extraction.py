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


def test_extract_multitask_message_splits_plain_and_connector_into_two_tasks():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("finish cad tonight and submit app tomorrow morning", "America/New_York")
    assert result.intent == "add_task"
    assert len(result.tasks) == 2
    assert "finish cad" in result.tasks[0].title.lower()
    assert result.tasks[0].deadline_text == "tonight"
    assert result.tasks[0].deadline is not None
    assert result.tasks[0].deadline.granularity == "part_of_day"
    assert "submit app" in result.tasks[1].title.lower()
    assert result.tasks[1].deadline_text == "tomorrow morning"
    assert result.tasks[1].deadline is not None
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


def test_project_plan_query_routes_to_timeline_query_not_task_add():
    extractor = IntentExtractor()
    result = extractor.extract("what's the plan for the enclosure project?", "America/New_York")
    assert result.intent == "timeline_query"
    assert result.task is None
    assert result.tasks == []


def test_attachment_reference_message_does_not_create_task():
    extractor = IntentExtractor()
    result = extractor.extract("here's the assignment screenshot", "America/New_York")
    assert result.intent == "general_chat"
    assert result.task is None
    assert result.tasks == []


def test_delete_task_language_maps_to_archive_update_action():
    extractor = IntentExtractor()
    result = extractor.extract("delete the website task", "America/New_York")
    assert result.intent == "update_task"
    assert result.task_updates.get("action") == "archive"


def test_followup_time_correction_maps_to_update_not_new_task():
    extractor = IntentExtractor()
    result = extractor.extract("actually monday morning", "America/New_York")
    assert result.intent == "update_task"
    assert result.time_reference == "monday morning"
    assert result.task_updates.get("action") == "reschedule"


def test_dont_let_me_forget_intake_creates_reminder_style_task():
    extractor = IntentExtractor()
    result = extractor.extract("yo dont let me forget to email the scout recruiter tmr morning", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.title.lower() == "email the scout recruiter"
    assert result.task.deadline_text == "tmr morning"
    assert result.task.action_kind == "quick_message"


def test_fallback_multitask_split_keeps_clean_titles_with_trailing_preposition_removed():
    extractor = IntentExtractor()
    result = extractor.extract(
        "yo i need to finish the CAD for the enclosure by tomorrow night and send that email tomorrow morning",
        "America/New_York",
    )
    assert result.intent == "add_task"
    assert len(result.tasks) == 2
    assert result.tasks[0].title.lower() == "finish the cad for the enclosure"
    assert result.tasks[1].title.lower() == "send that email"


def test_ambiguous_later_stays_soft_without_forcing_clarification():
    extractor = IntentExtractor()
    result = extractor.extract("need to send that email later", "America/New_York")
    assert result.intent == "add_task"
    assert result.time_reference is not None
    assert result.time_confidence <= 0.6
    assert result.needs_clarification is False
    assert result.task is not None
    assert result.task.deadline is not None
    assert result.task.deadline.deadline_at is None
    assert result.task.deadline.soft_deadline_at is not None
    assert result.task.deadline.is_ambiguous is True


def test_llm_later_deadline_does_not_force_clarification():
    payload = {
        "intent": "add_task",
        "confidence": 0.78,
        "needs_clarification": True,
        "clarification_question": "quick clarify: for 'Fix the website', what exact time should i use for 'later'?",
        "task": {
            "title": "fix the website later",
            "description": None,
            "project": None,
            "deadline_text": "later",
            "priority": 2,
            "confidence": 0.7,
            "next_step": None,
        },
    }
    adapter = _PayloadAdapter(payload)
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("need to fix the website later", "America/New_York")
    assert result.intent == "add_task"
    assert result.needs_clarification is False
    assert result.clarification_question is None


def test_send_email_tomorrow_morning_becomes_windowed_action_not_immediate_deadline():
    extractor = IntentExtractor()
    result = extractor.extract("need to send that email tomorrow morning", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.title == "Send that email"
    assert result.task.start_after is not None
    assert result.task.deadline is not None
    assert result.task.deadline.source_phrase == "tomorrow morning"
    assert result.task.next_step is not None
    assert "ready to send tomorrow morning" in result.task.next_step.lower()


def test_text_roommate_tomorrow_morning_is_quick_message_not_fake_project():
    extractor = IntentExtractor()
    result = extractor.extract("need to text my roommate tomorrow morning", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.action_kind == "quick_message"
    assert result.task.start_after is not None
    assert result.task.next_step is None


def test_leading_time_phrase_before_quick_message_still_extracts_task():
    extractor = IntentExtractor()
    result = extractor.extract("tomorrow morning text my roommate back", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.action_kind == "quick_message"
    assert "roommate" in result.task.title.lower()


def test_pay_rent_tonight_is_quick_admin_not_work_block():
    extractor = IntentExtractor()
    result = extractor.extract("need to pay rent tonight", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.action_kind == "quick_admin"


def test_mixed_assignment_and_context_creates_placeholder_task_and_context_signal():
    extractor = IntentExtractor()
    result = extractor.extract("prof just dropped another assignment and i'm in class rn", "America/New_York")
    assert result.intent == "add_task"
    assert result.context_signal is not None
    assert result.task is not None
    assert result.task.title == "New assignment from professor"


def test_generic_new_assignment_from_studio_uses_placeholder_task():
    extractor = IntentExtractor()
    result = extractor.extract("just got another assignment from studio", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.title == "New assignment from studio"


def test_mixed_assignment_context_fallback_wins_over_literal_llm_task_title():
    payload = {
        "intent": "add_task",
        "confidence": 0.78,
        "task": {
            "title": "prof just dropped another assignment and i'm in class rn",
            "description": None,
            "project": None,
            "deadline_text": None,
            "priority": 2,
            "confidence": 0.6,
            "next_step": None,
        },
        "context_signal": "in class rn",
    }
    adapter = _PayloadAdapter(payload)
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("prof just dropped another assignment and i'm in class rn", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.title == "New assignment from professor"


def test_weekend_time_stays_ambiguous_without_forcing_clarification():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("need to outline the concept this weekend", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.deadline is not None
    assert result.task.deadline.is_ambiguous is True
    assert result.task.deadline.granularity == "weekend"
    assert result.needs_clarification is False


def test_before_studio_without_anchor_stays_soft_and_ambiguous():
    adapter = _CountingAdapter()
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("need to print boards before studio", "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.deadline is not None
    assert result.task.deadline.deadline_at is None
    assert result.task.deadline.soft_deadline_at is not None
    assert result.task.deadline.is_ambiguous is True


def test_llm_tasks_array_gets_sanitized_and_deadlines_hydrated():
    payload = {
        "intent": "add_task",
        "confidence": 0.78,
        "tasks": [
            {
                "title": "finish cad tonight",
                "description": None,
                "project": None,
                "deadline_text": "tonight",
                "priority": 2,
                "confidence": 0.7,
                "next_step": None,
            },
            {
                "title": "submit app later",
                "description": None,
                "project": None,
                "deadline_text": "later",
                "priority": 2,
                "confidence": 0.6,
                "next_step": None,
            },
        ],
    }
    adapter = _PayloadAdapter(payload)
    extractor = IntentExtractor(adapter=adapter)
    result = extractor.extract("finish cad tonight and submit app later", "America/New_York")
    assert result.intent == "add_task"
    assert len(result.tasks) == 2
    assert result.tasks[0].title.lower() == "finish cad"
    assert result.tasks[0].deadline is not None
    assert result.tasks[0].deadline.deadline_at is not None
    assert result.tasks[1].title.lower() == "submit app"
    assert result.tasks[1].deadline is not None
    assert result.tasks[1].deadline.soft_deadline_at is not None
    assert result.tasks[1].deadline.is_ambiguous is True


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

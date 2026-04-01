from __future__ import annotations

from datetime import datetime

from app.llm.conversation_composer import ConversationComposer
from app.schemas.reply import ReplyBrief


class FakeAdapter:
    def __init__(
        self,
        text_responses: list[str | None],
        *,
        enabled: bool = True,
        json_responses: list[dict | None] | None = None,
    ) -> None:
        self.text_responses = text_responses
        self.json_responses = json_responses or []
        self.enabled = enabled
        self.calls = 0
        self.text_calls = 0

    def json_completion(  # noqa: ANN001
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict | None = None,
        request_timeout_seconds: float | None = None,
    ):
        self.calls += 1
        if self.json_responses:
            return self.json_responses.pop(0)
        return None

    def text_completion(  # noqa: ANN001
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        options: dict | None = None,
        request_timeout_seconds: float | None = None,
    ):
        self.text_calls += 1
        if self.text_responses:
            return self.text_responses.pop(0)
        # Simulate second-attempt regeneration by returning a clean default when queue is exhausted.
        lowered_user = user.lower()
        if "reply goal: acknowledge_new_task" in lowered_user:
            return "captured. next move: do a focused first pass right now and ping me when it's done."
        if "reply goal: answer_question" in lowered_user:
            return "yep, these are live-generated responses right now."
        if "short checkin: yes" in lowered_user:
            return "yo i'm here. what's the move right now?"
        return "yeah, i'm live and tracking this. what's the move right now?"


def _brief(latest: str, recent_thread: list[str] | None = None, *, short_checkin: bool = False) -> ReplyBrief:
    return ReplyBrief(
        response_goal="open_conversation",
        latest_user_message=latest,
        recent_thread=recent_thread or [],
        is_short_checkin=short_checkin,
        generated_at=datetime.now(),
    )


def _answer_brief(latest: str) -> ReplyBrief:
    return ReplyBrief(
        response_goal="answer_question",
        latest_user_message=latest,
        key_facts_to_include=["yes, i'm live right now and i received this message"],
        generated_at=datetime.now(),
    )


def test_open_ended_message_uses_llm_path():
    adapter = FakeAdapter(["yo what's up, i'm here. what's the move tonight?"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("what up tho"))
    assert adapter.text_calls >= 1
    assert reply.used_fallback is False
    assert reply.messages


def test_fallback_only_on_forced_model_failure():
    adapter = FakeAdapter([None], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("hey"))
    assert adapter.text_calls >= 1
    assert reply.used_fallback is True
    assert reply.messages


def test_repetition_guard_triggers_regeneration():
    adapter = FakeAdapter(
        [
            "got you. what's the move?",
        ],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    brief = _brief("hi", recent_thread=["assistant: got you. what's the move?"])
    reply = composer.compose(brief)
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "got you. what's the move?" not in " ".join(reply.messages).lower()


def test_composer_keeps_single_llm_attempt_path():
    adapter = FakeAdapter(["yo i got you\n\nstart with a 20 min pass, then ping me."], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("i'm cooked"))
    assert adapter.text_calls == 1
    assert adapter.calls >= 1
    assert reply.used_fallback is False
    assert len(reply.messages) >= 1


def test_composer_repairs_when_internal_phrase_leaks():
    adapter = FakeAdapter(["open conversational message received"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("is this thing on?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert adapter.text_calls >= 1
    assert "open conversational message received" not in " ".join(reply.messages).lower()


def test_composer_repairs_when_output_leaks_context_labels():
    adapter = FakeAdapter(["active tasks: finish cad upcoming deadlines: tomorrow night"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("are you actually live now"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert adapter.text_calls >= 1
    assert "active tasks:" not in " ".join(reply.messages).lower()


def test_composer_repairs_when_output_leaks_lightweight_key_value_dump():
    adapter = FakeAdapter(["user_message: yo whatup tasks=(none) deadlines=(tonight) next_step=finish_bot"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("yo whatup"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "user_message:" not in lowered
    assert "tasks=(" not in lowered
    assert "next_step=" not in lowered


def test_composer_repairs_when_output_uses_markdown_scaffolding():
    adapter = FakeAdapter(["Here's the response:\n\n**Checkpoint 1** Hey, I see where you're at!"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("no why would i break that up"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "here's the response" not in lowered
    assert "checkpoint 1" not in lowered


def test_composer_repairs_when_output_uses_smart_quote_scaffolding():
    adapter = FakeAdapter(["Here’s the response:\n\n**Checkpoint 1** Hey, I see where you're at!"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("no it's just one thing in the morning"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "here’s the response" not in lowered
    assert "checkpoint 1" not in lowered


def test_composer_repairs_when_output_leaks_actual_response_wrapper():
    adapter = FakeAdapter(["Here is the actual response from me:\n\ni'm here assistant. what's on your mind?"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("no why would i break that up"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "actual response from me" not in lowered
    assert "i'm here assistant" not in lowered
    assert "what's on your mind?" not in lowered


def test_composer_repairs_when_output_uses_support_bot_assist_language():
    adapter = FakeAdapter(["no hey i'm available to help you with your scout job application. how can i assist?"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("no why would i break that up"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "how can i assist" not in lowered
    assert "available to help" not in lowered


def test_composer_repairs_when_output_leaks_parenthesized_key_value_dump():
    adapter = FakeAdapter(["tasks=(none) deadlines=(tonight) next_step=finish_bot"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("yo whatup"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "tasks=(" not in lowered
    assert "deadlines=(" not in lowered
    assert "next_step=" not in lowered


def test_composer_repairs_when_output_is_truncated_tail():
    adapter = FakeAdapter(["hey, i see where you're at and it's just one thing at"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("no it's just one thing in the morning"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert not " ".join(reply.messages).lower().rstrip().endswith(" at")


def test_composer_repairs_when_output_leaks_status_priority_dump():
    adapter = FakeAdapter(["i submit my scout job application tomorrow morning status active priority 2 due no deadline"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(
        ReplyBrief(
            response_goal="acknowledge_new_task",
            latest_user_message="and then tmr morning i need to submit my scout job application",
            suggested_next_step="do a final proofread, then submit scout job application",
            generated_at=datetime.now(),
        )
    )
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "status active" not in lowered
    assert "priority 2" not in lowered


def test_composer_repairs_when_output_leaks_markdown_quote_list():
    adapter = FakeAdapter(
        ["i keep getting distracted, so i'll stay on track. here are my thoughts:\n\n* \"ngl i need help prioritizing tasks\""],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("no why would i break that up its just one thing in the morning"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "here are my thoughts" not in lowered
    assert '* "' not in lowered


def test_composer_repairs_when_output_leaks_internal_colon_prefix():
    adapter = FakeAdapter(["internal: task_manager: what's up yo whatup"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("yo whatup", short_checkin=True))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "internal:" not in lowered
    assert "task_manager:" not in lowered


def test_short_checkin_repair_rejects_task_timeline_drift():
    adapter = FakeAdapter(["hey yo whatup add a few things before tomorrow morning to make sure everything is good for review"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("yo whatup", short_checkin=True))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "tomorrow morning" not in lowered
    assert "review" not in lowered


def test_short_checkin_repair_rejects_routine_drift_question():
    adapter = FakeAdapter(["yo whatup why did u wanna change my routine?"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("yo whatup", short_checkin=True))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "routine" not in lowered


def test_acknowledge_new_task_repair_requires_capture_or_next_move_markers():
    adapter = FakeAdapter(["submit your scout job app before morning"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(
        ReplyBrief(
            response_goal="acknowledge_new_task",
            latest_user_message="i need to submit my scout job application tomorrow morning",
            suggested_next_step="do a final proofread, then submit scout job application",
            generated_at=datetime.now(),
        )
    )
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "next move:" in lowered or "got it" in lowered or "locked in" in lowered or "captured" in lowered


def test_acknowledge_new_task_repairs_conjunction_lead_and_submit_duplication():
    adapter = FakeAdapter(["and Submit your scout job application for review. Submit i need to finish proofreading."], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(
        ReplyBrief(
            response_goal="acknowledge_new_task",
            latest_user_message="and then tmr morning i need to submit my scout job application",
            suggested_next_step="do a final proofread, then submit scout job application",
            generated_at=datetime.now(),
        )
    )
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert lowered.startswith("and ") is False
    assert "submit i submit" not in lowered
    assert "next move:" in lowered or "captured" in lowered


def test_acknowledge_new_task_repairs_repeated_next_move_clause():
    adapter = FakeAdapter(
        ["i got you. captured. next move: do a final proofread, then submit scout job application next move: do a final proofread, then submit scout job application"],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(
        ReplyBrief(
            response_goal="acknowledge_new_task",
            latest_user_message="and then tmr morning i need to submit my scout job application",
            suggested_next_step="do a final proofread, then submit scout job application",
            generated_at=datetime.now(),
        )
    )
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert lowered.count("next move:") <= 1


def test_composer_repairs_when_output_leaks_instructional_phrase():
    adapter = FakeAdapter(["be direct about whether this reply is live-generated right now"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_answer_brief("are these canned responses or live ai generated?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "be direct about whether this reply is live-generated right now" not in " ".join(reply.messages).lower()


def test_composer_repairs_when_output_looks_like_internal_task_dump():
    adapter = FakeAdapter(["status active p2 due negative"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("you online?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "status active p2 due negative" not in " ".join(reply.messages).lower()


def test_composer_repairs_when_it_parrots_user_message():
    adapter = FakeAdapter(["are you actually live now"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("are you actually live now"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "are you actually live now" not in " ".join(reply.messages).lower()


def test_composer_repairs_when_first_bubble_parrots_user():
    adapter = FakeAdapter(["are you actually live now\n\nyeah i'm tracking everything."], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("are you actually live now"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert not any(msg.strip().lower() == "are you actually live now" for msg in reply.messages)


def test_composer_repairs_when_first_bubble_repeats_user_prefix():
    adapter = FakeAdapter(["lowk making good progress you actually respond now so thats good, nice work"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("lowk making good progress you actually respond now so thats good"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert not reply.messages[0].lower().startswith("lowk making good progress")


def test_composer_regenerates_if_answer_goal_starts_with_question():
    adapter = FakeAdapter(["are you alive?\n\ni'm here."], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_answer_brief("are you actually live now?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert not reply.messages[0].strip().endswith("?")


def test_answer_postprocess_drops_leading_question_if_still_present():
    adapter = FakeAdapter(["are you actually live now?\n\nI'm live right now. I received your message."], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_answer_brief("are you actually live now?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "live" in reply.messages[0].lower()
    assert not reply.messages[0].strip().endswith("?")


def test_answer_quality_guard_repairs_run_on_direct_answer():
    adapter = FakeAdapter(
        ["are we just having one conversation these are live responses from me not canned templates yet i'm on"],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_answer_brief("are these canned responses or live ai generated?"))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    assert "not canned" in " ".join(reply.messages).lower() or "live" in " ".join(reply.messages).lower()


def test_short_checkin_repair_rejects_nonsequitur_context_bleed():
    adapter = FakeAdapter(
        ["I'm back.\n\nhey, i saw where things are going - your responses got a bit scattered so what's on your mind?"],
        enabled=True,
    )
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("yo whatup", short_checkin=True))
    assert reply.used_fallback is False
    assert reply.regenerated_for_repetition is True
    lowered = " ".join(reply.messages).lower()
    assert "scattered" not in lowered
    assert "out of hand" not in lowered
    assert len(reply.messages) <= 2


def test_short_checkin_merges_tiny_lead_bubble():
    adapter = FakeAdapter(["i'm back.\n\nyo i'm here. what's the move right now?"], enabled=True)
    composer = ConversationComposer(adapter=adapter)
    reply = composer.compose(_brief("hey", short_checkin=True))
    assert reply.used_fallback is False
    assert len(reply.messages) == 1
    assert "what's the move right now?" in reply.messages[0].lower()

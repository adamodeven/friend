from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from app.db.models import Task, User
from app.db.models import MessageDirection
from app.db.repositories.message_repo import create_message
from app.db.repositories.task_repo import create_task
from app.domain.state_engine import StateEngine
from app.domain.timeline_service import TimelineService
from app.llm.extraction import IntentExtractor
from app.schemas.intent import ExtractedTask, IntentResult


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("yo i need to finish the CAD for the enclosure by tomorrow night", "add_task"),
        ("prof just dropped another assignment", "add_task"),
        ("in class rn", "context_signal"),
        ("gonna pull an all nighter tn", "context_signal"),
        ("i keep getting distracted because i need to fix the website first", "update_task"),
        ("what do i have due this week", "timeline_query"),
        ("need to send that email tomorrow morning", "add_task"),
        ("just finished the first draft", "complete_task"),
        ("my bad i underestimated this", "reflection"),
        ("here's the assignment screenshot", "general_chat"),
    ],
)
def test_requirement_intent_matrix(message: str, expected_intent: str):
    extractor = IntentExtractor()
    result = extractor.extract(message, "America/New_York")
    assert result.intent == expected_intent


@pytest.mark.parametrize(
    ("message", "expected_kind", "expects_prep"),
    [
        ("need to text my roommate tomorrow morning", "quick_message", False),
        ("need to pay rent tonight", "quick_admin", False),
        ("yo i need to finish the CAD for the enclosure by tomorrow night", "project_chunk", False),
        ("clean up the portfolio bullets before class", "project_chunk", False),
        ("need to send that recruiter email tomorrow morning", "work_block", True),
    ],
)
def test_task_shape_matrix(message: str, expected_kind: str, expects_prep: bool):
    extractor = IntentExtractor()
    result = extractor.extract(message, "America/New_York")
    assert result.intent == "add_task"
    assert result.task is not None
    assert result.task.action_kind == expected_kind
    if expects_prep:
        assert result.task.next_step is not None
    else:
        if expected_kind == "quick_message":
            assert result.task.next_step is None


def test_state_matrix_handles_quick_message_as_reminder_not_project(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tomorrow_morning = (datetime.now(tz=ZoneInfo(user.timezone)) + timedelta(days=1)).replace(
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )
    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.92,
            task=ExtractedTask(
                title="Text roommate back",
                deadline_text="tomorrow morning",
                start_after=tomorrow_morning,
                action_kind="quick_message",
            ),
        ),
        raw_text="need to text my roommate tomorrow morning",
    )
    assert outcome.should_push_for_action is False
    assert outcome.suggested_next_step is None


def test_state_matrix_handles_quick_admin_as_reminder_not_immediate_work_block(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    evening = datetime.now(tz=ZoneInfo(user.timezone)) + timedelta(hours=4)
    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.92,
            task=ExtractedTask(
                title="Pay rent",
                deadline_text="tonight",
                deadline_at=evening,
                action_kind="quick_admin",
            ),
        ),
        raw_text="need to pay rent tonight",
    )
    assert outcome.should_push_for_action is False
    assert outcome.suggested_next_step is None


def test_state_matrix_asks_for_prioritization_when_load_is_wide_and_loose(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.9,
            tasks=[
                ExtractedTask(title="Clean up portfolio bullets"),
                ExtractedTask(title="Pay rent"),
                ExtractedTask(title="Fix the website"),
                ExtractedTask(title="Reply to that club email"),
            ],
        ),
        raw_text="tonight i need to clean up portfolio bullets, pay rent, fix the website, and reply to that club email",
    )
    assert outcome.should_ask_question is True
    assert outcome.question_if_needed == "which one of those actually has the least wiggle room?"


def test_state_matrix_placeholder_assignment_without_details_prompts_for_followup(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.82,
            task=ExtractedTask(title="New assignment from studio"),
        ),
        raw_text="just got another assignment from studio",
    )
    assert outcome.should_ask_question is True
    assert "assignment details" in (outcome.question_if_needed or "")


def test_state_matrix_prefers_real_work_over_future_quick_message_in_same_turn(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tomorrow_morning = (datetime.now(tz=ZoneInfo(user.timezone)) + timedelta(days=1)).replace(
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )
    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.92,
            tasks=[
                ExtractedTask(
                    title="Finish enclosure CAD",
                    deadline_text="tomorrow night",
                    action_kind="project_chunk",
                ),
                ExtractedTask(
                    title="Text roommate back",
                    deadline_text="tomorrow morning",
                    start_after=tomorrow_morning,
                    action_kind="quick_message",
                ),
            ],
        ),
        raw_text="finish the enclosure cad by tomorrow night and text my roommate back tomorrow morning",
    )
    assert outcome.should_push_for_action is True
    assert "enclosure cad" in (outcome.suggested_next_step or "").lower()


def test_timeline_matrix_prefers_real_work_over_future_quick_message(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    tomorrow_morning = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    create_task(
        db_session,
        user_id=user.id,
        title="Text roommate back",
        priority=3,
        start_after=tomorrow_morning,
        deadline_source_phrase="tomorrow morning",
        metadata_json={"action_kind": "quick_message"},
    )
    deep_work = create_task(
        db_session,
        user_id=user.id,
        title="Finish enclosure CAD",
        priority=4,
        deadline_at=(now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0),
        metadata_json={"action_kind": "project_chunk"},
        next_step="finish the first clean enclosure pass",
    )
    db_session.commit()

    service = TimelineService()
    next_task = service.recommend_next_task(db_session, user.id, user.timezone)
    assert next_task is not None
    assert next_task.id == deep_work.id
    tomorrow_view = service.build_tomorrow_morning_view(db_session, user.id, user.timezone).lower()
    assert "text roommate back" in tomorrow_view


def test_behavior_matrix_plain_language_updates_still_work(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Fix website", priority=4)
    db_session.commit()

    engine = StateEngine()
    archive = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(intent="update_task", confidence=0.95, task_updates={"action": "archive"}),
        raw_text="delete the website task",
    )
    db_session.commit()

    archived = db_session.execute(select(Task).where(Task.title == "Fix website")).scalars().one()
    assert archived.status.value == "archived"
    assert archive.response_goal == "confirm_update"


def test_behavior_matrix_followup_reschedule_updates_same_reminder_task(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()

    first_message = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="yo dont let me forget to email the scout recruiter tmr morning",
        external_id="SM_MATRIX_1",
    )
    first_intent = IntentResult(
        intent="add_task",
        confidence=0.92,
        task=ExtractedTask(
            title="Email the scout recruiter",
            deadline_text="tomorrow morning",
            action_kind="quick_message",
        ),
    )
    engine.apply_intent(
        db_session,
        user=user,
        intent=first_intent,
        raw_text="yo dont let me forget to email the scout recruiter tmr morning",
        source_message_id=first_message.id,
    )

    second_message = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="actually monday morning",
        external_id="SM_MATRIX_2",
    )
    second_intent = IntentExtractor().extract("actually monday morning", user.timezone)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=second_intent,
        raw_text="actually monday morning",
        source_message_id=second_message.id,
    )
    db_session.commit()

    tasks = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().all()
    assert len(tasks) == 1
    assert tasks[0].deadline_source_phrase == "monday morning"
    assert outcome.key_facts_to_include == ["moved that to monday morning"]


def test_behavior_matrix_tmr_morning_query_uses_actual_timeline_window(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    monday_morning = (now + timedelta(days=((0 - now.weekday()) % 7 or 7))).replace(hour=8, minute=0, second=0, microsecond=0)
    create_task(
        db_session,
        user_id=user.id,
        title="Email the scout recruiter",
        priority=2,
        start_after=monday_morning,
        deadline_at=monday_morning.replace(hour=11),
        deadline_source_phrase="monday morning",
        metadata_json={"action_kind": "quick_message"},
    )
    db_session.commit()

    result = IntentExtractor().extract("what do i have tmr morning", user.timezone)
    assert result.intent == "timeline_query"

    summary = TimelineService().build_tomorrow_morning_view(db_session, user.id, user.timezone).lower()
    assert "scout recruiter" not in summary

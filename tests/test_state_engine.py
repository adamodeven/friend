from __future__ import annotations

from sqlalchemy import select

from app.db.models import ScheduleBlock, Task, User
from app.domain.state_engine import StateEngine
from app.schemas.intent import ExtractedTask, IntentResult


def test_add_task_creates_task_and_deadline_event(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.9,
        task=ExtractedTask(title="Finish CAD", deadline_text="tomorrow night"),
    )
    outcome = engine.apply_intent(db_session, user=user, intent=intent, raw_text="finish cad tomorrow night")
    db_session.commit()

    task = db_session.execute(select(Task).where(Task.title == "Finish CAD")).scalars().first()
    assert task is not None
    assert outcome.response_goal == "acknowledge_new_task"


def test_context_signal_creates_schedule_block(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="context_signal", confidence=0.8, context_signal="in class rn")
    engine.apply_intent(db_session, user=user, intent=intent, raw_text="in class rn")
    db_session.commit()

    block = db_session.execute(select(ScheduleBlock)).scalars().first()
    assert block is not None
    assert block.block_type == "in_class"


def test_status_query_meta_gets_direct_explanation(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="status_query", confidence=0.9)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="are these canned responses or live ai generated?",
    )
    assert outcome.response_goal == "answer_question"
    assert any("live-generated" in fact or "live" in fact for fact in outcome.key_facts_to_include)


def test_status_query_live_now_includes_direct_status_fact(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="status_query", confidence=0.9)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="are you actually live now?",
    )
    assert any("yes, i'm live right now" in fact for fact in outcome.key_facts_to_include)


def test_general_progress_message_maps_to_progress_response_goal(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="general_chat", confidence=0.7)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="lowk making good progress now",
    )
    assert outcome.response_goal == "react_to_progress"


def test_single_add_task_does_not_force_checkpoint_question(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.9,
        task=ExtractedTask(title="Submit scout job application", deadline_text="tomorrow morning"),
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="tomorrow morning i need to submit my scout job application",
    )
    assert outcome.response_goal == "acknowledge_new_task"
    assert outcome.should_ask_question is False
    assert outcome.question_if_needed is None

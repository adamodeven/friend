from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.db.models import ConversationMessage, MessageDirection, ProfileStyle, Task, TaskStatus, User
from app.domain.reply_brief_builder import ReplyBriefBuilder
from app.schemas.reply import StateOutcome


def test_reply_brief_goal_reflects_outcome(db_session):
    user = db_session.execute(select(User)).scalars().first()
    builder = ReplyBriefBuilder()

    timeline_outcome = StateOutcome(response_goal="timeline_summary", key_facts_to_include=["today summary"])
    timeline_brief = builder.build(
        db_session,
        user=user,
        latest_user_message="what do i have due today",
        outcome=timeline_outcome,
    )

    task_outcome = StateOutcome(
        response_goal="acknowledge_new_task",
        key_facts_to_include=["task captured: website"],
        should_push_for_action=True,
    )
    task_brief = builder.build(
        db_session,
        user=user,
        latest_user_message="need to fix website tonight",
        outcome=task_outcome,
    )

    assert timeline_brief.response_goal == "timeline_summary"
    assert task_brief.response_goal == "acknowledge_new_task"
    assert timeline_brief.max_chunks >= task_brief.max_chunks


def test_meta_question_brief_omits_task_context(db_session):
    user = db_session.execute(select(User)).scalars().first()
    db_session.add(
        Task(
            user_id=user.id,
            title="Finish CAD",
            status=TaskStatus.active,
            priority=2,
        )
    )
    db_session.commit()

    builder = ReplyBriefBuilder()
    outcome = StateOutcome(response_goal="answer_question", key_facts_to_include=["user asked: are you live now?"])
    brief = builder.build(
        db_session,
        user=user,
        latest_user_message="are you actually live now?",
        outcome=outcome,
    )
    assert brief.active_task_context == []
    assert brief.deadline_context == []


def test_open_conversation_brief_omits_task_context_when_message_is_casual(db_session):
    user = db_session.execute(select(User)).scalars().first()
    db_session.add(
        Task(
            user_id=user.id,
            title="Prepare portfolio",
            status=TaskStatus.active,
            priority=1,
        )
    )
    db_session.commit()

    builder = ReplyBriefBuilder()
    outcome = StateOutcome(response_goal="open_conversation")
    brief = builder.build(
        db_session,
        user=user,
        latest_user_message="hey",
        outcome=outcome,
    )
    assert brief.active_task_context == []
    assert brief.deadline_context == []


def test_short_checkin_brief_marks_flag_and_limits_thread(db_session):
    user = db_session.execute(select(User)).scalars().first()
    db_session.add_all(
        [
            ConversationMessage(user_id=user.id, direction=MessageDirection.inbound, body="older one"),
            ConversationMessage(user_id=user.id, direction=MessageDirection.outbound, body="older two"),
            ConversationMessage(user_id=user.id, direction=MessageDirection.inbound, body="older three"),
            ConversationMessage(user_id=user.id, direction=MessageDirection.outbound, body="older four"),
            ConversationMessage(user_id=user.id, direction=MessageDirection.inbound, body="older five"),
        ]
    )
    db_session.commit()

    builder = ReplyBriefBuilder()
    outcome = StateOutcome(response_goal="open_conversation")
    brief = builder.build(
        db_session,
        user=user,
        latest_user_message="yo whatup",
        outcome=outcome,
    )
    assert brief.is_short_checkin is True
    assert len(brief.recent_thread) <= 3


def test_acknowledge_new_task_brief_omits_unrelated_active_context(db_session):
    user = db_session.execute(select(User)).scalars().first()
    db_session.add_all(
        [
            Task(user_id=user.id, title="I get everything working for you by tonight", status=TaskStatus.active, priority=1),
            Task(user_id=user.id, title="Another active item", status=TaskStatus.active, priority=2),
        ]
    )
    db_session.commit()

    builder = ReplyBriefBuilder()
    outcome = StateOutcome(
        response_goal="acknowledge_new_task",
        key_facts_to_include=["task captured: submit scout job application"],
        suggested_next_step="do a final proofread, then submit scout job application",
    )
    brief = builder.build(
        db_session,
        user=user,
        latest_user_message="tmr morning i need to submit my scout job application",
        outcome=outcome,
    )
    assert brief.active_task_context == []
    assert brief.deadline_context == []


def test_reply_brief_uses_style_profile_chunk_limits(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    assert user.profile is not None
    user.profile.style = ProfileStyle.direct
    db_session.commit()

    builder = ReplyBriefBuilder()
    outcome = StateOutcome(
        response_goal="timeline_summary",
        key_facts_to_include=["tonight: finish CAD, then submit the scout job application"],
    )
    brief = builder.build(
        db_session,
        user=user,
        latest_user_message="what do i need to get done tonight",
        outcome=outcome,
    )

    assert brief.style_mode == "direct"
    assert brief.max_chunk_length == 260
    assert brief.max_chunks == 2


def test_ingestion_confirmation_clears_unrelated_task_context(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    db_session.add(Task(user_id=user.id, title="Fix website", status=TaskStatus.active, priority=5))
    db_session.commit()

    builder = ReplyBriefBuilder()
    brief = builder.build(
        db_session,
        user=user,
        latest_user_message="here's the assignment screenshot",
        outcome=StateOutcome(
            response_goal="ingestion_confirmation",
            key_facts_to_include=["screenshot saved, but the concrete task pull was low-confidence"],
            should_ask_question=True,
            question_if_needed="what do you want me to grab from that screenshot?",
        ),
    )

    assert brief.active_task_context == []
    assert brief.deadline_context == []


def test_reschedule_confirmation_omits_unrelated_task_context(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    db_session.add(Task(user_id=user.id, title="Fix website", status=TaskStatus.active, priority=5))
    db_session.commit()

    builder = ReplyBriefBuilder()
    brief = builder.build(
        db_session,
        user=user,
        latest_user_message="actually monday morning",
        outcome=StateOutcome(
            response_goal="confirm_update",
            key_facts_to_include=["okay monday morning"],
        ),
    )

    assert brief.active_task_context == []
    assert brief.deadline_context == []

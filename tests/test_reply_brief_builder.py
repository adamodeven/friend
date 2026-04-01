from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.db.models import Task, TaskStatus, User
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

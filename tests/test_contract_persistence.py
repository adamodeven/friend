from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import ConversationMessage, MessageDirection, Task, User
from app.db.repositories.task_repo import (
    create_task,
    create_task_dependency,
    record_task_progress,
    record_task_slip,
    update_task_reminder_state,
)
from app.schemas.intent import IntentResult


def test_intent_result_supports_multiple_tasks_with_legacy_primary_task_access():
    result = IntentResult.model_validate(
        {
            "intent": "add_task",
            "confidence": 0.92,
            "tasks": [
                {
                    "title": "Finish CAD",
                    "next_step": "open the assembly and fix the interference",
                    "deadline": {
                        "source_phrase": "tonight",
                        "confidence": 0.74,
                        "is_ambiguous": False,
                        "granularity": "part_of_day",
                    },
                },
                {
                    "title": "Send recruiter email",
                    "blockers": ["need final resume PDF"],
                    "subtasks": [{"title": "export latest resume PDF"}],
                },
            ],
        }
    )

    assert len(result.tasks) == 2
    assert result.task is not None
    assert result.task.title == "Finish CAD"
    assert result.tasks[0].deadline is not None
    assert result.tasks[0].deadline_text == "tonight"
    assert result.tasks[1].subtasks[0].title == "export latest resume PDF"


def test_rich_task_contract_persists_progress_slips_and_dependencies(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None

    message = ConversationMessage(
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="finish cad tonight and send recruiter email after class",
        metadata_json={},
    )
    db_session.add(message)
    db_session.flush()

    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    parent = create_task(
        db_session,
        user_id=user.id,
        source_message_id=message.id,
        title="Finish CAD",
        next_step="open CAD and isolate the broken joint",
        deadline_at=now,
        deadline_source_phrase="tonight",
        deadline_confidence=0.74,
        deadline_is_ambiguous=False,
        deadline_granularity="part_of_day",
        deadline_timezone=user.timezone,
        blocker_details_json={"kind": "dependency"},
        metadata_json={"source_text": message.body},
    )
    child = create_task(
        db_session,
        user_id=user.id,
        source_message_id=message.id,
        parent_task_id=parent.id,
        title="Send recruiter email",
        blocked_reason="need final resume PDF",
        blocked_at=now,
    )
    create_task_dependency(
        db_session,
        user_id=user.id,
        predecessor_task_id=parent.id,
        successor_task_id=child.id,
        metadata_json={"reason": "finish work samples first"},
    )

    record_task_progress(parent, at=now, next_step="export screenshots for the recruiter email")
    record_task_slip(child, at=now, reason="resume wasn't ready", escalation_level=2)
    update_task_reminder_state(child, escalation_level=2, reminder_pause_until=now)
    db_session.commit()

    stored = db_session.execute(select(Task).where(Task.id == child.id)).scalars().one()
    assert stored.source_message_id == message.id
    assert stored.parent_task_id == parent.id
    assert stored.slip_count == 1
    assert stored.last_slip_reason == "resume wasn't ready"
    assert stored.reminder_escalation_level == 2
    pause_until = stored.reminder_pause_until
    assert pause_until is not None
    if pause_until.tzinfo is None:
        pause_until = pause_until.replace(tzinfo=now.tzinfo)
    assert pause_until == now
    assert len(parent.subtasks) == 1
    assert len(parent.predecessor_links) == 1
    assert parent.predecessor_links[0].successor_task_id == child.id

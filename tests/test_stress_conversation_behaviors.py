from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.db.models import ConversationMessage, MessageDirection, ScheduleBlock, Task, TaskStatus, User
from app.domain.conversation_manager import ConversationManager
from app.llm.conversation_composer import ConversationComposer
from app.schemas.reply import ComposedReply, ReplyBrief
from app.schemas.transport import InboundSmsPayload


class _StressAdapter:
    def __init__(self) -> None:
        self.enabled = True
        self._turn = 0

    def json_completion(self, **kwargs):  # noqa: ANN003,ANN201
        # Force the text path in this stress harness to exercise repair/regeneration behavior.
        return None

    def text_completion(self, *, user: str, **kwargs):  # noqa: ANN003,ANN201
        self._turn += 1
        # First attempt every turn intentionally returns low-quality leakage,
        # second attempt should be clean and goal-aligned.
        if self._turn % 2 == 1:
            return "user_message: leaked scaffold tasks=(none) next_step=todo"

        lowered = user.lower()
        if "reply goal: acknowledge_new_task" in lowered:
            return "captured. next move: do a focused first pass now and text me when it's done."
        if "reply goal: answer_question" in lowered:
            return "yep, this is live and i got your message."
        if "reply goal: timeline_summary" in lowered:
            return "for tonight: hit the nearest deadline first, then clean up the next blocker."
        if "short checkin: yes" in lowered:
            return "yo i'm here. what's the move right now?"
        return "got you. we can lock the next move now."


def _brief_for(goal: str, message: str) -> ReplyBrief:
    return ReplyBrief(
        response_goal=goal,  # type: ignore[arg-type]
        latest_user_message=message,
        recent_thread=["assistant: got you. what's the move?"],
        key_facts_to_include=["task captured: sample task"],
        suggested_next_step="do a focused first pass now",
        generated_at=datetime.now(),
    )


def test_composer_stress_regenerates_without_fallback():
    composer = ConversationComposer(adapter=_StressAdapter())
    samples = [
        ("open_conversation", "what up tho"),
        ("answer_question", "are these canned responses or live ai generated?"),
        ("acknowledge_new_task", "need to finish cad by tomorrow night"),
        ("timeline_summary", "what do i need to get done tonight"),
        ("replan_blocker", "i keep getting distracted because i need to fix the website first"),
    ]

    for i in range(30):
        goal, msg = samples[i % len(samples)]
        reply = composer.compose(_brief_for(goal, msg))
        assert reply.used_fallback is False
        assert reply.messages
        joined = " ".join(reply.messages).lower()
        assert "user_message:" not in joined
        assert "tasks=(" not in joined
        assert "next_step=" not in joined


class _DeterministicComposer:
    def compose(self, brief: ReplyBrief) -> ComposedReply:
        return ComposedReply(messages=[f"ok. {brief.response_goal}"], used_fallback=False)


def _payload(from_number: str, body: str, sid: str) -> InboundSmsPayload:
    return InboundSmsPayload(
        From=from_number,
        To="+15550002222",
        Body=body,
        MessageSid=sid,
        NumMedia=0,
        media=[],
    )


def test_end_to_end_conversation_sequence_updates_state(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    manager = ConversationManager(composer=_DeterministicComposer())

    sequence = [
        ("SM_STRESS_001", "yo i need to finish the cad for the enclosure by tomorrow night"),
        ("SM_STRESS_002", "and then tmr morning i need to submit my scout job application"),
        ("SM_STRESS_003", "lowkey i keep getting distracted because i need to fix the website first"),
        ("SM_STRESS_004", "what do i have due this week"),
        ("SM_STRESS_005", "in class rn"),
        ("SM_STRESS_006", "just finished the first draft"),
    ]

    for sid, body in sequence:
        result = manager.process_inbound(db_session, _payload(user.phone_number, body, sid))
        assert result.skipped_duplicate is False
        assert result.outgoing_messages
        db_session.commit()

    tasks = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().all()
    assert len(tasks) >= 2
    assert any(task.status in {TaskStatus.active, TaskStatus.blocked, TaskStatus.completed} for task in tasks)

    blocks = db_session.execute(select(ScheduleBlock).where(ScheduleBlock.user_id == user.id)).scalars().all()
    assert any(block.block_type == "in_class" for block in blocks)

    outbound = (
        db_session.execute(
            select(ConversationMessage).where(
                ConversationMessage.user_id == user.id,
                ConversationMessage.direction == MessageDirection.outbound,
            )
        )
        .scalars()
        .all()
    )
    assert len(outbound) >= len(sequence)

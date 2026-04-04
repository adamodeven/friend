from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.db.models import Attachment, ConversationMessage, ExtractedArtifact, MessageDirection, PlanningNote, ScheduleBlock, Task, TaskDependency, TaskStatus, User
from app.db.repositories.task_repo import create_task
from app.domain.conversation_manager import ConversationManager
from app.llm.conversation_composer import ConversationComposer
from app.schemas.reply import ComposedReply, ReplyBrief
from app.schemas.transport import InboundMedia, InboundSmsPayload


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
            return "bet, got it. i'd start with a focused first pass now and text me when it's done."
        if "reply goal: answer_question" in lowered:
            return "yep, this is live and i got your message."
        if "reply goal: timeline_summary" in lowered:
            return "for tonight, hit the nearest deadline first, then clean up the next blocker."
        if "short checkin: yes" in lowered:
            return "yo i'm here."
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


class _StressAttachmentService:
    def save_attachment(self, session, *, user_id, message_id, media_url: str, content_type: str | None):  # noqa: ANN001
        attachment = Attachment(
            user_id=user_id,
            message_id=message_id,
            media_url=media_url,
            media_content_type=content_type,
            status="received",
        )
        session.add(attachment)
        session.flush()
        return attachment

    def download_attachment(self, attachment: Attachment):  # noqa: ANN201
        attachment.status = "downloaded"
        return None

    def process_assignment_image(self, session, *, attachment: Attachment, timezone: str):  # noqa: ANN001,ARG002
        artifact = ExtractedArtifact(
            user_id=attachment.user_id,
            source_attachment_id=attachment.id,
            title="Prepare fabrication checklist",
            context="screenshot from class portal",
            raw_text="Prepare fabrication checklist due this weekend",
            confidence=0.81,
        )
        session.add(artifact)
        session.flush()
        task = create_task(
            session,
            user_id=attachment.user_id,
            title="Prepare fabrication checklist",
            next_step="outline the checklist sections and gather missing measurements",
            deadline_source_phrase="this weekend",
            extraction_confidence=0.81,
            metadata_json={"source_attachment_id": str(attachment.id)},
        )
        task.source = "attachment_ingestion"
        artifact.created_task_id = task.id
        attachment.status = "processed"
        attachment.analysis = {"title": artifact.title, "due_text": "this weekend"}
        session.flush()
        return artifact, task


def _payload(
    from_number: str,
    body: str,
    sid: str,
    *,
    media: list[InboundMedia] | None = None,
) -> InboundSmsPayload:
    return InboundSmsPayload(
        From=from_number,
        To="+15550002222",
        Body=body,
        MessageSid=sid,
        NumMedia=len(media or []),
        media=media or [],
    )


def test_end_to_end_conversation_sequence_updates_state(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    manager = ConversationManager(
        composer=_DeterministicComposer(),
        attachment_service=_StressAttachmentService(),
    )

    sequence = [
        _payload(
            user.phone_number,
            "yo i need to finish the cad for the enclosure by tomorrow night and submit my scout job application tomorrow morning",
            "SM_STRESS_001",
        ),
        _payload(
            user.phone_number,
            "the scout job application is blocked because i need to fix the portfolio website first",
            "SM_STRESS_002",
        ),
        _payload(
            user.phone_number,
            "also pull this from the screenshot",
            "SM_STRESS_003",
            media=[InboundMedia(media_url="https://example.com/assignment.png", content_type="image/png")],
        ),
        _payload(user.phone_number, "in class rn", "SM_STRESS_004"),
        _payload(user.phone_number, "what do i have due this week", "SM_STRESS_005"),
        _payload(user.phone_number, "i'm behind on the cad for the enclosure because i got distracted again", "SM_STRESS_006"),
        _payload(user.phone_number, "just finished fix the portfolio website", "SM_STRESS_007"),
    ]

    for payload in sequence:
        result = manager.process_inbound(db_session, payload)
        assert result.skipped_duplicate is False
        assert result.outgoing_messages
        db_session.commit()

    tasks = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().all()
    assert len(tasks) >= 4
    assert any(task.title == "Prepare fabrication checklist" for task in tasks)
    assert any("cad for the enclosure" in task.title.lower() for task in tasks)
    assert any("scout job application" in task.title.lower() for task in tasks)
    assert any("portfolio website" in task.title.lower() for task in tasks)
    assert any(task.status in {TaskStatus.active, TaskStatus.blocked, TaskStatus.completed} for task in tasks)

    dependency = db_session.execute(select(TaskDependency)).scalars().first()
    assert dependency is not None

    blocks = db_session.execute(select(ScheduleBlock).where(ScheduleBlock.user_id == user.id)).scalars().all()
    assert any(block.block_type == "in_class" for block in blocks)

    notes = db_session.execute(select(PlanningNote).where(PlanningNote.user_id == user.id)).scalars().all()
    assert any(note.note_type == "slip_reason" for note in notes)

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

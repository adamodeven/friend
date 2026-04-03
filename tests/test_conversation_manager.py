from sqlalchemy import select

import pytest
from app.db.models import Attachment, ConversationMessage, ExtractedArtifact, JobStatus, MessageDirection, ProcessingJob, Reminder, Task, User
from app.db.repositories.task_repo import create_task
from app.domain.conversation_manager import ConversationManager
from app.ingestion.attachments import AttachmentIngestionService
from app.schemas.reply import ComposedReply
from app.schemas.intent import IntentResult
from app.schemas.transport import InboundMedia
from app.schemas.transport import InboundSmsPayload


class _RecordingComposer:
    def __init__(self) -> None:
        self.called = False
        self.last_brief = None

    def compose(self, brief):  # noqa: ANN001
        self.called = True
        self.last_brief = brief
        return ComposedReply(messages=[f"ok. {brief.response_goal}"], used_fallback=False)


class _StaticIntentExtractor:
    def __init__(self, result: IntentResult) -> None:
        self.result = result

    def extract(self, text: str, timezone: str) -> IntentResult:  # noqa: ARG002
        return self.result


class _FakeAttachmentService:
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
            title="Prepare studio board",
            context="canvas assignment screenshot",
            raw_text="Prepare studio board due tomorrow morning with process shots",
            confidence=0.84,
        )
        session.add(artifact)
        session.flush()

        task = create_task(
            session,
            user_id=attachment.user_id,
            title="Prepare studio board",
            next_step="collect the process shots and lay out the board",
            deadline_source_phrase="tomorrow morning",
            extraction_confidence=0.84,
            metadata_json={"source_attachment_id": str(attachment.id)},
        )
        task.source = "attachment_ingestion"
        artifact.created_task_id = task.id
        attachment.status = "processed"
        attachment.analysis = {"title": artifact.title, "due_text": "tomorrow morning"}
        session.flush()
        return artifact, task


def test_duplicate_message_is_skipped(db_session):
    user = db_session.execute(select(User)).scalars().first()
    manager = ConversationManager()
    payload = InboundSmsPayload(
        From=user.phone_number,
        To="+15550002222",
        Body="need to send that email tomorrow morning",
        MessageSid="SM_DUPLICATE_TEST",
        NumMedia=0,
        media=[],
    )

    first = manager.process_inbound(db_session, payload)
    db_session.commit()
    second = manager.process_inbound(db_session, payload)

    assert first.skipped_duplicate is False
    assert second.skipped_duplicate is True

    outbound = (
        db_session.execute(select(ConversationMessage).where(ConversationMessage.direction == MessageDirection.outbound))
        .scalars()
        .all()
    )
    assert outbound


def test_open_ended_message_uses_composer_path(db_session):
    user = db_session.execute(select(User)).scalars().first()
    composer = _RecordingComposer()
    manager = ConversationManager(composer=composer)
    payload = InboundSmsPayload(
        From=user.phone_number,
        To="+15550002222",
        Body="what up tho",
        MessageSid="SM_OPEN_ENDED_TEST",
        NumMedia=0,
        media=[],
    )
    result = manager.process_inbound(db_session, payload)
    db_session.commit()

    assert composer.called is True
    assert result.outgoing_messages


def test_attachment_task_sets_ingestion_goal_and_schedules_reminder(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    composer = _RecordingComposer()
    manager = ConversationManager(
        composer=composer,
        intent_extractor=_StaticIntentExtractor(IntentResult(intent="general_chat", confidence=0.7)),
        attachment_service=_FakeAttachmentService(),
    )
    payload = InboundSmsPayload(
        From=user.phone_number,
        To="+15550002222",
        Body="look at this screenshot",
        MessageSid="SM_IMAGE_FLOW_TEST",
        NumMedia=1,
        media=[InboundMedia(media_url="https://example.com/board.png", content_type="image/png")],
    )

    result = manager.process_inbound(db_session, payload)
    db_session.commit()

    assert result.outgoing_messages
    assert composer.last_brief is not None
    assert composer.last_brief.response_goal == "ingestion_confirmation"
    assert any("screenshot captured task: Prepare studio board" in fact for fact in composer.last_brief.key_facts_to_include)
    task = db_session.execute(select(Task).where(Task.title == "Prepare studio board")).scalars().first()
    assert task is not None
    reminder = db_session.execute(select(Reminder).where(Reminder.task_id == task.id)).scalars().first()
    assert reminder is not None


def test_attachment_failure_does_not_fail_text_turn(db_session):
    class BrokenAttachmentService(_FakeAttachmentService):
        def process_assignment_image(self, session, *, attachment: Attachment, timezone: str):  # noqa: ANN001,ARG002
            raise RuntimeError("vision parse failed")

    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    composer = _RecordingComposer()
    manager = ConversationManager(
        composer=composer,
        attachment_service=BrokenAttachmentService(),
    )
    payload = InboundSmsPayload(
        From=user.phone_number,
        To="+15550002222",
        Body="need to finish the cad tonight",
        MessageSid="SM_IMAGE_FAIL_SOFT",
        NumMedia=1,
        media=[InboundMedia(media_url="https://example.com/fail.png", content_type="image/png")],
    )

    result = manager.process_inbound(db_session, payload)
    db_session.commit()

    assert result.outgoing_messages
    assert composer.last_brief is not None
    assert composer.last_brief.response_goal == "acknowledge_new_task"
    assert any("attachment processing hit 1 issue" in fact for fact in composer.last_brief.key_facts_to_include)
    attachment = db_session.execute(select(Attachment).order_by(Attachment.created_at.desc())).scalars().first()
    assert attachment is not None
    assert attachment.status == "failed"
    latest_job = db_session.execute(select(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(1)).scalars().first()
    assert latest_job is not None
    assert latest_job.status == JobStatus.done


def test_processing_job_marked_failed_when_pipeline_raises(db_session):
    class BoomExtractor:
        def extract(self, text: str, timezone: str):  # noqa: ANN001
            raise RuntimeError("intent extraction exploded")

    user = db_session.execute(select(User)).scalars().first()
    manager = ConversationManager(intent_extractor=BoomExtractor())
    payload = InboundSmsPayload(
        From=user.phone_number,
        To="+15550002222",
        Body="hey",
        MessageSid="SM_FAIL_STATUS_TEST",
        NumMedia=0,
        media=[],
    )

    with pytest.raises(RuntimeError):
        manager.process_inbound(db_session, payload)

    latest_job = (
        db_session.execute(select(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(1))
        .scalars()
        .first()
    )
    assert latest_job is not None
    assert latest_job.status == JobStatus.failed


def test_attachment_ingestion_bounds_long_artifact_text():
    service = AttachmentIngestionService()
    bounded = service._bounded_artifact_text("x" * 400)

    assert bounded is not None
    assert len(bounded) == 255
    assert bounded.endswith("…")

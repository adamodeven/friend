from sqlalchemy import select

import pytest
from app.db.models import ConversationMessage, JobStatus, MessageDirection, ProcessingJob, User
from app.domain.conversation_manager import ConversationManager
from app.schemas.reply import ComposedReply
from app.schemas.transport import InboundSmsPayload


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
    class FakeComposer:
        def __init__(self) -> None:
            self.called = False

        def compose(self, brief):  # noqa: ANN001
            self.called = True
            return ComposedReply(messages=["yeah, i'm live. what's up?"], used_fallback=False)

    user = db_session.execute(select(User)).scalars().first()
    composer = FakeComposer()
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

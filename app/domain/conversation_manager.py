from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.db.models import JobStatus, MessageDirection, ProcessingJob
from app.db.repositories.message_repo import create_message, inbound_message_exists
from app.db.repositories.user_repo import get_user_by_phone, get_or_create_primary_user
from app.domain.reply_brief_builder import ReplyBriefBuilder
from app.domain.state_engine import StateEngine
from app.ingestion.attachments import AttachmentIngestionService
from app.llm.conversation_composer import ConversationComposer
from app.llm.extraction import IntentExtractor
from app.schemas.transport import InboundSmsPayload

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    user_id: str
    outgoing_messages: list[str]
    skipped_duplicate: bool = False


class ConversationManager:
    def __init__(
        self,
        *,
        intent_extractor: IntentExtractor | None = None,
        state_engine: StateEngine | None = None,
        brief_builder: ReplyBriefBuilder | None = None,
        composer: ConversationComposer | None = None,
        attachment_service: AttachmentIngestionService | None = None,
    ) -> None:
        self.intent_extractor = intent_extractor or IntentExtractor()
        self.state_engine = state_engine or StateEngine()
        self.brief_builder = brief_builder or ReplyBriefBuilder()
        self.conversation_composer = composer or ConversationComposer()
        self.attachment_service = attachment_service or AttachmentIngestionService()

    def process_inbound(self, session: Session, payload: InboundSmsPayload) -> ProcessResult:
        started = time.monotonic()
        if inbound_message_exists(session, payload.message_sid):
            return ProcessResult(user_id="", outgoing_messages=[], skipped_duplicate=True)

        user = get_user_by_phone(session, payload.from_number) or get_or_create_primary_user(session)
        job = ProcessingJob(
            user_id=user.id,
            job_type="inbound_message_processing",
            status=JobStatus.running,
            input_payload={"sid": payload.message_sid, "body": payload.body, "num_media": payload.num_media},
        )
        session.add(job)
        session.flush()

        try:
            inbound_msg = create_message(
                session,
                user_id=user.id,
                direction=MessageDirection.inbound,
                body=payload.body,
                external_id=payload.message_sid,
                metadata_json={"from": payload.from_number, "to": payload.to_number, "num_media": payload.num_media},
            )

            attachments_created = []
            for media in payload.media:
                att = self.attachment_service.save_attachment(
                    session,
                    user_id=user.id,
                    message_id=inbound_msg.id,
                    media_url=media.media_url,
                    content_type=media.content_type,
                )
                attachments_created.append(att)

            # Persist inbound capture immediately so it is visible in admin logs
            # even if downstream LLM/state steps time out or fail.
            session.commit()
            logger.info("inbound persisted sid=%s elapsed=%.2fs", payload.message_sid, time.monotonic() - started)

            intent = self.intent_extractor.extract(payload.body, timezone=user.timezone)
            logger.info("intent extracted sid=%s elapsed=%.2fs", payload.message_sid, time.monotonic() - started)
            state_outcome = self.state_engine.apply_intent(
                session,
                user=user,
                intent=intent,
                raw_text=payload.body,
            )
            logger.info("state applied sid=%s elapsed=%.2fs", payload.message_sid, time.monotonic() - started)

            if attachments_created:
                for att in attachments_created:
                    self.attachment_service.download_attachment(att)
                    artifact, task = self.attachment_service.process_assignment_image(
                        session,
                        attachment=att,
                        timezone=user.timezone,
                    )
                    if artifact and task:
                        due = task.deadline_at.astimezone(ZoneInfo(user.timezone)).strftime("%a %-m/%-d %-I:%M%p").lower() if task.deadline_at else "no deadline seen"
                        state_outcome.response_goal = "ingestion_confirmation"
                        state_outcome.key_facts_to_include.append(f"image ingestion captured task: {task.title}")
                        state_outcome.key_facts_to_include.append(f"image-derived due context: {due}")
                        state_outcome.mention_deadline = bool(task.deadline_at)

            brief = self.brief_builder.build(
                session,
                user=user,
                latest_user_message=payload.body,
                outcome=state_outcome,
            )
            reply = self.conversation_composer.compose(brief)
            logger.info("reply composed sid=%s elapsed=%.2fs", payload.message_sid, time.monotonic() - started)

            outgoing = []
            for chunk in reply.messages:
                create_message(
                    session,
                    user_id=user.id,
                    direction=MessageDirection.outbound,
                    body=chunk,
                    external_id=None,
                    metadata_json={
                        "in_reply_to": payload.message_sid,
                        "response_goal": brief.response_goal,
                        "used_fallback": reply.used_fallback,
                    },
                )
                outgoing.append(chunk)

            job.status = JobStatus.done
            job.output_payload = {
                "outgoing_messages": outgoing,
                "response_goal": brief.response_goal,
                "used_fallback": reply.used_fallback,
                "regenerated_for_repetition": reply.regenerated_for_repetition,
            }
            session.flush()
            logger.info("inbound complete sid=%s outgoing=%s elapsed=%.2fs", payload.message_sid, len(outgoing), time.monotonic() - started)
            return ProcessResult(user_id=str(user.id), outgoing_messages=outgoing)
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
            session.commit()
            logger.exception("inbound failed sid=%s elapsed=%.2fs", payload.message_sid, time.monotonic() - started)
            raise

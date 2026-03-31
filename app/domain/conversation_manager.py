from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.db.models import JobStatus, MessageDirection, ProcessingJob
from app.db.repositories.message_repo import create_message, inbound_message_exists, list_recent_messages
from app.db.repositories.user_repo import get_user_by_phone, get_or_create_primary_user
from app.domain.state_engine import StateEngine
from app.ingestion.attachments import AttachmentIngestionService
from app.llm.composer import ReplyComposer
from app.llm.extraction import IntentExtractor
from app.schemas.transport import InboundSmsPayload


@dataclass
class ProcessResult:
    user_id: str
    outgoing_messages: list[str]
    skipped_duplicate: bool = False


class ConversationManager:
    def __init__(self) -> None:
        self.intent_extractor = IntentExtractor()
        self.state_engine = StateEngine()
        self.reply_composer = ReplyComposer()
        self.attachment_service = AttachmentIngestionService()

    def process_inbound(self, session: Session, payload: InboundSmsPayload) -> ProcessResult:
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

            intent = self.intent_extractor.extract(payload.body, timezone=user.timezone)
            action_summary = self.state_engine.apply_intent(
                session,
                user=user,
                intent=intent,
                raw_text=payload.body,
            )

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
                        action_summary += f" | image parsed: {task.title} ({due})"

            state_summary = self._recent_state_summary(session, user.id)
            reply = self.reply_composer.compose(
                style=user.profile.style.value if user.profile else "casual_cool",
                intent=intent,
                state_summary=state_summary,
                action_summary=action_summary,
                timezone=user.timezone,
            )

            outgoing = []
            for chunk in reply.messages:
                create_message(
                    session,
                    user_id=user.id,
                    direction=MessageDirection.outbound,
                    body=chunk,
                    external_id=None,
                    metadata_json={"in_reply_to": payload.message_sid},
                )
                outgoing.append(chunk)

            job.status = JobStatus.done
            job.output_payload = {"outgoing_messages": outgoing, "action_summary": action_summary}
            session.flush()
            return ProcessResult(user_id=str(user.id), outgoing_messages=outgoing)
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
            session.flush()
            raise

    @staticmethod
    def _recent_state_summary(session: Session, user_id) -> str:
        msgs = list_recent_messages(session, user_id, limit=12)
        lines = []
        for msg in msgs[-6:]:
            prefix = "u" if msg.direction == MessageDirection.inbound else "a"
            lines.append(f"{prefix}:{msg.body[:100]}")
        return " | ".join(lines)

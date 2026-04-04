from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.db.models import Attachment, JobStatus, MessageDirection, ProcessingJob, Task
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


@dataclass
class AttachmentEffects:
    processed_count: int = 0
    failed_count: int = 0
    artifacts_without_task: int = 0
    tasks: list[Task] = field(default_factory=list)
    reminder_labels: list[str] = field(default_factory=list)
    ambiguous_tasks: list[Task] = field(default_factory=list)


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
                source_message_id=inbound_msg.id,
            )
            logger.info("state applied sid=%s elapsed=%.2fs", payload.message_sid, time.monotonic() - started)

            if attachments_created:
                effects = self._ingest_attachments(
                    session,
                    attachments=attachments_created,
                    user=user,
                )
                self._merge_attachment_effects(
                    session,
                    raw_text=payload.body,
                    user=user,
                    state_outcome=state_outcome,
                    effects=effects,
                )

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

    def _ingest_attachments(self, session: Session, *, attachments: list[Attachment], user) -> AttachmentEffects:
        effects = AttachmentEffects()
        for attachment in attachments:
            attachment_id = attachment.id
            try:
                with session.begin_nested():
                    self.attachment_service.download_attachment(attachment)
                    artifact, task = self.attachment_service.process_assignment_image(
                        session,
                        attachment=attachment,
                        timezone=user.timezone,
                    )
                effects.processed_count += 1
                if task is not None:
                    if task.user is None:
                        task.user = user
                    reminder = self.state_engine.reminders.schedule_for_task(session, task)
                    effects.tasks.append(task)
                    if reminder is not None:
                        label = reminder.scheduled_for.astimezone(ZoneInfo(user.timezone)).strftime("%-I:%M%p").lower()
                        effects.reminder_labels.append(label)
                    if task.deadline_source_phrase and task.deadline_at is None:
                        effects.ambiguous_tasks.append(task)
                elif artifact is not None and (artifact.title or artifact.raw_text or artifact.context):
                    effects.artifacts_without_task += 1
            except Exception as exc:  # pragma: no cover - production hardening
                attachment.status = "failed"
                attachment.analysis = {"error": str(exc)}
                effects.failed_count += 1
                logger.exception("attachment ingestion failed attachment_id=%s", attachment_id)
        return effects

    def _merge_attachment_effects(
        self,
        session: Session,
        *,
        raw_text: str,
        user,
        state_outcome,
        effects: AttachmentEffects,
    ) -> None:
        if effects.failed_count:
            noun = "issue" if effects.failed_count == 1 else "issues"
            state_outcome.key_facts_to_include.append(
                f"attachment processing hit {effects.failed_count} {noun}, but the text update still went through"
            )

        if not effects.tasks and not effects.artifacts_without_task:
            return

        if effects.tasks:
            titles = [task.title for task in effects.tasks[:3]]
            if len(effects.tasks) == 1:
                task = effects.tasks[0]
                due = self._format_due(task.deadline_at, user.timezone) or task.deadline_source_phrase or "no deadline seen"
                state_outcome.key_facts_to_include.append(f"from the screenshot i pulled {task.title}")
                state_outcome.key_facts_to_include.append(f"due read looks like {due}")
                if effects.reminder_labels:
                    state_outcome.key_facts_to_include.append(f"check-in scheduled around {effects.reminder_labels[0]}")
            else:
                state_outcome.key_facts_to_include.append(f"from the screenshot i pulled {len(effects.tasks)} tasks")
                state_outcome.key_facts_to_include.append("looks like " + ", ".join(titles))

            if any(task.deadline_at or task.deadline_source_phrase for task in effects.tasks):
                state_outcome.mention_deadline = True

            if state_outcome.response_goal in {"open_conversation", "answer_question", "acknowledge_context"}:
                state_outcome.response_goal = "ingestion_confirmation"
                state_outcome.emotional_tone = "direct"
                state_outcome.should_push_for_action = True

            if state_outcome.response_goal == "timeline_summary":
                summary = self._timeline_summary_for_text(
                    session,
                    user_id=user.id,
                    timezone=user.timezone,
                    raw_text=raw_text,
                )
                state_outcome.key_facts_to_include = [summary] + [
                    fact for fact in state_outcome.key_facts_to_include if not self._looks_like_timeline_summary(fact)
                ]

            recommended = self.state_engine.timeline.recommend_next_task(session, user.id, user.timezone)
            if recommended is not None:
                state_outcome.suggested_next_step = self.state_engine._next_step_for_task(recommended)
                state_outcome.should_push_for_action = True

            if effects.ambiguous_tasks and not state_outcome.should_ask_question:
                task = effects.ambiguous_tasks[0]
                state_outcome.should_ask_question = True
                state_outcome.question_if_needed = self.state_engine._time_clarification_question(
                    task_title=task.title,
                    time_reference=task.deadline_source_phrase or "that screenshot due time",
                )
        elif effects.artifacts_without_task:
            if state_outcome.response_goal in {"open_conversation", "answer_question", "acknowledge_context"}:
                state_outcome.response_goal = "ingestion_confirmation"
                state_outcome.emotional_tone = "direct"
            state_outcome.key_facts_to_include.append("screenshot saved, but the concrete task pull was low-confidence")
            if not state_outcome.should_ask_question:
                state_outcome.should_ask_question = True
                state_outcome.question_if_needed = "what do you want me to grab from that screenshot?"

    def _timeline_summary_for_text(self, session: Session, *, user_id, timezone: str, raw_text: str) -> str:
        lowered = raw_text.lower()
        if "weekend" in lowered:
            return self.state_engine.timeline.build_weekend_view(session, user_id, timezone)
        if "tomorrow morning" in lowered:
            return self.state_engine.timeline.build_tomorrow_morning_view(session, user_id, timezone)
        if "tonight" in lowered:
            return self.state_engine.timeline.build_tonight_view(session, user_id, timezone)
        if "next hour" in lowered:
            return self.state_engine.timeline.next_hour_recommendation(session, user_id, timezone)
        if "week" in lowered:
            return self.state_engine.timeline.build_week_view(session, user_id, timezone)
        return self.state_engine.timeline.build_today_view(session, user_id, timezone)

    @staticmethod
    def _looks_like_timeline_summary(value: str) -> bool:
        lowered = value.lower()
        return any(
            token in lowered
            for token in (
                "today\n",
                "tonight\n",
                "tomorrow morning\n",
                "this week\n",
                "weekend\n",
                "for the next hour",
            )
        )

    @staticmethod
    def _format_due(value: datetime | None, timezone_name: str) -> str | None:
        if value is None:
            return None
        due = value if value.tzinfo else value.replace(tzinfo=ZoneInfo(timezone_name))
        return due.astimezone(ZoneInfo(timezone_name)).strftime("%a %-m/%-d %-I:%M%p").lower()

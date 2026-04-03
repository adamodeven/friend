from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Attachment, ExtractedArtifact, Task, TaskStatus
from app.db.repositories.task_repo import create_task
from app.llm.extraction import ImageAssignmentExtractor


class AttachmentIngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.extractor = ImageAssignmentExtractor()

    def save_attachment(
        self,
        session: Session,
        *,
        user_id,
        message_id,
        media_url: str,
        content_type: str | None,
    ) -> Attachment:
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

    def download_attachment(self, attachment: Attachment) -> Path | None:
        if attachment.local_path:
            existing = Path(attachment.local_path)
            if existing.exists():
                attachment.status = "downloaded"
                return existing
        if not attachment.media_url:
            attachment.status = "failed"
            attachment.analysis = {"error": "attachment missing media_url"}
            return None
        with httpx.Client(timeout=20.0) as client:
            response = client.get(attachment.media_url, follow_redirects=True)
            response.raise_for_status()
            data = response.content
        sha = hashlib.sha256(data).hexdigest()
        suffix = ".jpg"
        if attachment.media_content_type:
            if "png" in attachment.media_content_type:
                suffix = ".png"
            elif "pdf" in attachment.media_content_type:
                suffix = ".pdf"
        target = self.settings.attachments_path / f"{attachment.id}{suffix}"
        target.write_bytes(data)
        attachment.local_path = str(target)
        attachment.sha256 = sha
        attachment.status = "downloaded"
        return target

    def process_assignment_image(self, session: Session, *, attachment: Attachment, timezone: str) -> tuple[ExtractedArtifact | None, Task | None]:
        image_source = attachment.media_url
        result = self.extractor.extract(image_source, timezone)
        analysis = result.model_dump(mode="json")
        analysis["source_type"] = "attachment_image"
        if attachment.local_path:
            analysis["local_path"] = attachment.local_path
        attachment.analysis = analysis
        attachment.status = "processed" if self._has_extraction_signal(result=result) else "processed_no_signal"

        artifact = ExtractedArtifact(
            user_id=attachment.user_id,
            source_attachment_id=attachment.id,
            title=result.title,
            context=result.context,
            due_at=result.due_at,
            raw_text=result.raw_text,
            structured_data={
                **analysis,
                "attachment_id": str(attachment.id),
            },
            confidence=result.confidence,
        )
        session.add(artifact)
        session.flush()

        task = self._create_or_update_task_from_result(
            session,
            attachment=attachment,
            artifact=artifact,
            timezone=timezone,
        )
        if task is not None:
            artifact.created_task_id = task.id
        session.flush()
        return artifact, task

    @staticmethod
    def _has_extraction_signal(*, result) -> bool:  # noqa: ANN001
        return bool(
            result.title
            or result.context
            or result.due_text
            or result.due_at
            or result.deliverables
            or result.raw_text
        )

    def _create_or_update_task_from_result(
        self,
        session: Session,
        *,
        attachment: Attachment,
        artifact: ExtractedArtifact,
        timezone: str,
    ) -> Task | None:
        analysis = attachment.analysis or {}
        title = self._task_title_from_analysis(analysis)
        if not title:
            return None

        existing = self._find_matching_task(session, user_id=attachment.user_id, title=title)
        metadata = {
            "source_attachment_id": str(attachment.id),
            "artifact_id": str(artifact.id),
            "source": "attachment_ingestion",
            "deliverables": list(analysis.get("deliverables") or []),
            "due_text": analysis.get("due_text"),
            "context": analysis.get("context"),
        }
        description = self._build_task_description(analysis)
        due_at = artifact.due_at
        due_text = analysis.get("due_text")
        confidence = max(float(artifact.confidence or 0.0), 0.35 if due_at or due_text else 0.0)

        if existing is not None:
            existing.description = existing.description or description
            existing.extraction_confidence = max(existing.extraction_confidence or 0.0, float(artifact.confidence or 0.0))
            existing.metadata_json = {**(existing.metadata_json or {}), **metadata}
            if due_at is not None and (
                existing.deadline_at is None
                or existing.deadline_confidence < confidence
                or due_at < existing.deadline_at
            ):
                existing.deadline_at = due_at
                existing.deadline_source_phrase = due_text
                existing.deadline_confidence = confidence
                existing.deadline_timezone = timezone
            if not existing.next_step:
                existing.next_step = self._next_step_for_task(title=existing.title, deliverables=analysis.get("deliverables") or [])
            existing.source = "attachment_ingestion"
            return existing

        task = create_task(
            session,
            user_id=attachment.user_id,
            title=title,
            description=description,
            next_step=self._next_step_for_task(title=title, deliverables=analysis.get("deliverables") or []),
            deadline_at=due_at,
            deadline_source_phrase=due_text,
            deadline_confidence=confidence,
            deadline_is_ambiguous=bool(due_text and due_at is None),
            deadline_timezone=timezone if due_at or due_text else None,
            priority=self._priority_for_task(due_at=due_at, due_text=due_text, confidence=float(artifact.confidence or 0.0)),
            extraction_confidence=float(artifact.confidence or 0.0),
            metadata_json=metadata,
        )
        task.source = "attachment_ingestion"
        return task

    @staticmethod
    def _task_title_from_analysis(analysis: dict) -> str | None:
        raw_title = (analysis.get("title") or "").strip()
        if raw_title:
            return raw_title
        deliverables = [item.strip() for item in analysis.get("deliverables") or [] if str(item).strip()]
        if len(deliverables) == 1:
            return deliverables[0]
        return None

    @staticmethod
    def _build_task_description(analysis: dict) -> str | None:
        lines: list[str] = ["captured from screenshot ingestion"]
        context = (analysis.get("context") or "").strip()
        if context:
            lines.append(f"context: {context}")
        due_text = (analysis.get("due_text") or "").strip()
        if due_text:
            lines.append(f"due text: {due_text}")
        deliverables = [item.strip() for item in analysis.get("deliverables") or [] if str(item).strip()]
        if deliverables:
            lines.append("deliverables: " + "; ".join(deliverables[:4]))
        raw_text = (analysis.get("raw_text") or "").strip()
        if raw_text:
            lines.append(f"ocr: {raw_text[:220]}")
        return "\n".join(lines)

    @staticmethod
    def _next_step_for_task(*, title: str, deliverables: list[str]) -> str:
        for item in deliverables:
            cleaned = str(item).strip()
            if cleaned and cleaned.lower() != title.lower():
                return cleaned
        lowered = title.lower()
        if lowered.startswith("submit "):
            return f"do a final proofread, then {title[0].lower() + title[1:]}"
        if lowered.startswith(("send ", "email ", "text ", "upload ", "export ", "finish ", "review ", "fix ")):
            return title
        return f"start a focused first pass on {title}"

    @staticmethod
    def _priority_for_task(*, due_at, due_text: str | None, confidence: float) -> int:  # noqa: ANN001
        if due_at is not None:
            return 4
        if due_text:
            return 3
        if confidence >= 0.75:
            return 3
        return 2

    @staticmethod
    def _find_matching_task(session: Session, *, user_id, title: str) -> Task | None:
        normalized = AttachmentIngestionService._normalize_title(title)
        tasks = session.execute(
            select(Task).where(
                Task.user_id == user_id,
                Task.status.in_((TaskStatus.active, TaskStatus.blocked)),
            )
        ).scalars().all()
        for task in tasks:
            existing = AttachmentIngestionService._normalize_title(task.title)
            if existing == normalized:
                return task
            if len(normalized) >= 12 and (normalized in existing or existing in normalized):
                return task
        return None

    @staticmethod
    def _normalize_title(value: str) -> str:
        return " ".join(value.lower().split())

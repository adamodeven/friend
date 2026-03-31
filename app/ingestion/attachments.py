from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Attachment, ExtractedArtifact, Task
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
        if not attachment.media_url:
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
        result = self.extractor.extract(attachment.media_url, timezone)
        attachment.analysis = result.model_dump(mode="json")
        attachment.status = "processed"

        artifact = ExtractedArtifact(
            user_id=attachment.user_id,
            source_attachment_id=attachment.id,
            title=result.title,
            context=result.context,
            due_at=result.due_at,
            raw_text=result.raw_text,
            structured_data=result.model_dump(mode="json"),
            confidence=result.confidence,
        )
        session.add(artifact)
        session.flush()

        task = None
        if result.title:
            task = create_task(
                session,
                user_id=attachment.user_id,
                title=result.title,
                description="from screenshot ingestion",
                deadline_at=result.due_at,
                extraction_confidence=result.confidence,
                metadata_json={"source_attachment_id": str(attachment.id)},
            )
            artifact.created_task_id = task.id
        session.flush()
        return artifact, task


from __future__ import annotations

from app.ingestion.attachments import AttachmentIngestionService


def test_generic_logging_title_does_not_become_task():
    service = AttachmentIngestionService()
    title = service._task_title_from_analysis(  # noqa: SLF001
        {"title": "No clear assignment/title visible; looks like a text conversation/logging task"}
    )
    assert title is None

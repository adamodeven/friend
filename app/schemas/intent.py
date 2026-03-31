from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


IntentName = Literal[
    "add_task",
    "update_task",
    "complete_task",
    "status_query",
    "timeline_query",
    "context_signal",
    "reflection",
    "general_chat",
]


class ExtractedTask(BaseModel):
    title: str
    description: str | None = None
    project: str | None = None
    deadline_text: str | None = None
    deadline_at: datetime | None = None
    soft_deadline_at: datetime | None = None
    priority: int = Field(default=2, ge=1, le=5)
    confidence: float = Field(default=0.6, ge=0, le=1)
    next_step: str | None = None


class IntentResult(BaseModel):
    intent: IntentName
    confidence: float = Field(default=0.5, ge=0, le=1)
    needs_clarification: bool = False
    clarification_question: str | None = None
    time_reference: str | None = None
    time_confidence: float = Field(default=0.0, ge=0, le=1)
    context_signal: str | None = None
    blockers: list[str] = Field(default_factory=list)
    task: ExtractedTask | None = None
    task_updates: dict = Field(default_factory=dict)
    summary: str | None = None


class ImageExtractionResult(BaseModel):
    title: str | None = None
    due_text: str | None = None
    due_at: datetime | None = None
    context: str | None = None
    deliverables: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class ConversationReply(BaseModel):
    messages: list[str]
    internal_summary: str | None = None


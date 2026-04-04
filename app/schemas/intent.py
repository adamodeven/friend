from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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

DeadlineGranularity = Literal["exact", "hour", "part_of_day", "day", "weekend", "week", "month", "unknown"]
DependencyRelation = Literal["blocked_by", "blocks", "subtask_of", "related_to"]
ActionKind = Literal["quick_message", "quick_admin", "work_block", "project_chunk"]


class ParsedDeadline(BaseModel):
    source_phrase: str | None = None
    deadline_at: datetime | None = None
    soft_deadline_at: datetime | None = None
    timezone: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    is_ambiguous: bool = False
    ambiguity_reason: str | None = None
    granularity: DeadlineGranularity = "unknown"


class ExtractedDependency(BaseModel):
    title: str
    relation: DependencyRelation = "blocked_by"
    confidence: float = Field(default=0.6, ge=0, le=1)
    notes: str | None = None


class ExtractedTask(BaseModel):
    title: str
    description: str | None = None
    project: str | None = None
    deadline_text: str | None = None
    deadline_at: datetime | None = None
    soft_deadline_at: datetime | None = None
    start_after: datetime | None = None
    priority: int = Field(default=2, ge=1, le=5)
    confidence: float = Field(default=0.6, ge=0, le=1)
    action_kind: ActionKind | None = None
    next_step: str | None = None
    blockers: list[str] = Field(default_factory=list)
    dependencies: list[ExtractedDependency] = Field(default_factory=list)
    subtasks: list["ExtractedTask"] = Field(default_factory=list)
    progress_timestamp: datetime | None = None
    slip_reason: str | None = None
    deadline: ParsedDeadline | None = None

    @model_validator(mode="after")
    def _sync_deadline_fields(self) -> "ExtractedTask":
        if self.deadline is None and (self.deadline_text or self.deadline_at or self.soft_deadline_at):
            self.deadline = ParsedDeadline(
                source_phrase=self.deadline_text,
                deadline_at=self.deadline_at,
                soft_deadline_at=self.soft_deadline_at,
                confidence=self.confidence,
            )
            return self

        if self.deadline is not None:
            if not self.deadline_text and self.deadline.source_phrase:
                self.deadline_text = self.deadline.source_phrase
            if self.deadline_at is None:
                self.deadline_at = self.deadline.deadline_at
            if self.soft_deadline_at is None:
                self.soft_deadline_at = self.deadline.soft_deadline_at
        return self


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
    tasks: list[ExtractedTask] = Field(default_factory=list)
    task_updates: dict = Field(default_factory=dict)
    summary: str | None = None

    @model_validator(mode="after")
    def _sync_task_fields(self) -> "IntentResult":
        if self.task is None and self.tasks:
            self.task = self.tasks[0]
        elif self.task is not None and not self.tasks:
            self.tasks = [self.task]
        return self


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


ExtractedTask.model_rebuild()

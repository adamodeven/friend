from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ReplyGoal = Literal[
    "acknowledge_new_task",
    "confirm_update",
    "react_to_progress",
    "timeline_summary",
    "answer_question",
    "replan_blocker",
    "acknowledge_context",
    "followup_on_slip",
    "ingestion_confirmation",
    "open_conversation",
]

UrgencyLevel = Literal["low", "medium", "high", "critical"]
EmotionalTone = Literal["casual", "neutral", "direct", "supportive", "urgent", "calm"]


class StateOutcome(BaseModel):
    response_goal: ReplyGoal
    key_facts_to_include: list[str] = Field(default_factory=list)
    urgency_level: UrgencyLevel = "low"
    should_push_for_action: bool = False
    should_ask_question: bool = False
    question_if_needed: str | None = None
    emotional_tone: EmotionalTone = "casual"
    mention_deadline: bool = False
    mention_dependency: bool = False
    mention_progress: bool = False
    suggested_next_step: str | None = None
    avoid_topics: list[str] = Field(default_factory=list)
    operational_reason: str | None = None


class ReplyBrief(BaseModel):
    response_goal: ReplyGoal
    key_facts_to_include: list[str] = Field(default_factory=list)
    urgency_level: UrgencyLevel = "low"
    should_push_for_action: bool = False
    should_ask_question: bool = False
    question_if_needed: str | None = None
    emotional_tone: EmotionalTone = "casual"
    style_mode: str = "casual_cool"
    max_chunks: int = 2
    max_chunk_length: int = 320
    mention_deadline: bool = False
    mention_dependency: bool = False
    mention_progress: bool = False
    suggested_next_step: str | None = None
    avoid_topics: list[str] = Field(default_factory=list)
    thread_context_summary: str = ""
    active_task_context: list[str] = Field(default_factory=list)
    deadline_context: list[str] = Field(default_factory=list)
    memory_notes: list[str] = Field(default_factory=list)
    current_state_flags: list[str] = Field(default_factory=list)
    latest_user_message: str
    recent_thread: list[str] = Field(default_factory=list)
    operational_reason: str | None = None
    is_short_checkin: bool = False
    generated_at: datetime


class ComposedReply(BaseModel):
    messages: list[str]
    used_fallback: bool = False
    regenerated_for_repetition: bool = False

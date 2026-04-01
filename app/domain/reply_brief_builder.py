from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import PlanningNote, ScheduleBlock
from app.db.repositories.message_repo import list_recent_messages
from app.db.repositories.task_repo import list_active_tasks, list_upcoming_deadlines
from app.schemas.reply import ReplyBrief, StateOutcome


class ReplyBriefBuilder:
    def __init__(self) -> None:
        self.settings = get_settings()

    def build(
        self,
        session: Session,
        *,
        user,
        latest_user_message: str,
        outcome: StateOutcome,
    ) -> ReplyBrief:
        now = datetime.now(tz=ZoneInfo(user.timezone))
        recent = list_recent_messages(session, user.id, limit=16)
        recent_thread = [
            f"{'assistant' if msg.direction.value == 'outbound' else 'user'}: {msg.body[:220]}"
            for msg in recent[-10:]
        ]
        thread_summary = " | ".join(recent_thread[-6:])

        active_tasks = list_active_tasks(session, user.id)[:6]
        active_task_context = []
        for task in active_tasks:
            due = "no deadline"
            if task.deadline_at:
                due = task.deadline_at.astimezone(ZoneInfo(user.timezone)).strftime("%a %-m/%-d %-I:%M%p").lower()
            active_task_context.append(f"{task.title} (status {task.status.value}, priority {task.priority}, due {due})")

        upcoming = list_upcoming_deadlines(session, user.id, within_days=7)[:6]
        deadline_context = []
        for task in upcoming:
            if not task.deadline_at:
                continue
            due = task.deadline_at.astimezone(ZoneInfo(user.timezone)).strftime("%a %-m/%-d %-I:%M%p").lower()
            deadline_context.append(f"{task.title} due {due}")

        # Keep conversational/meta replies clean and avoid unnecessary task-context bleed.
        lowered_message = latest_user_message.lower()
        context_tokens = ["task", "deadline", "due", "plan", "project", "week", "today", "tonight", "tomorrow", "hour"]
        wants_task_context = any(token in lowered_message for token in context_tokens)
        if outcome.response_goal in {"answer_question", "open_conversation", "acknowledge_context"} and not wants_task_context:
            active_task_context = []
            deadline_context = []

        memory_notes = self._recent_memory_notes(session, user.id)
        state_flags = self._current_state_flags(session, user.id, now)

        max_chunks = 2
        if outcome.response_goal in {"timeline_summary", "answer_question", "replan_blocker"}:
            max_chunks = 3

        return ReplyBrief(
            response_goal=outcome.response_goal,
            key_facts_to_include=outcome.key_facts_to_include,
            urgency_level=outcome.urgency_level,
            should_push_for_action=outcome.should_push_for_action,
            should_ask_question=outcome.should_ask_question,
            question_if_needed=outcome.question_if_needed,
            emotional_tone=outcome.emotional_tone,
            style_mode=user.profile.style.value if user.profile else self.settings.default_style,
            max_chunks=max_chunks,
            max_chunk_length=self.settings.max_sms_chars,
            mention_deadline=outcome.mention_deadline,
            mention_dependency=outcome.mention_dependency,
            mention_progress=outcome.mention_progress,
            suggested_next_step=outcome.suggested_next_step,
            avoid_topics=outcome.avoid_topics,
            thread_context_summary=thread_summary,
            active_task_context=active_task_context,
            deadline_context=deadline_context,
            memory_notes=memory_notes,
            current_state_flags=state_flags,
            latest_user_message=latest_user_message,
            recent_thread=recent_thread,
            operational_reason=outcome.operational_reason,
            generated_at=now,
        )

    @staticmethod
    def _recent_memory_notes(session: Session, user_id) -> list[str]:
        notes = (
            session.execute(
                select(PlanningNote)
                .where(PlanningNote.user_id == user_id)
                .order_by(PlanningNote.created_at.desc())
                .limit(5)
            )
            .scalars()
            .all()
        )
        return [n.content[:220] for n in notes]

    @staticmethod
    def _current_state_flags(session: Session, user_id, now: datetime) -> list[str]:
        blocks = (
            session.execute(
                select(ScheduleBlock)
                .where(ScheduleBlock.user_id == user_id, ScheduleBlock.starts_at <= now, ScheduleBlock.ends_at >= now)
                .order_by(ScheduleBlock.ends_at.asc())
            )
            .scalars()
            .all()
        )
        return [f"{b.block_type} until {b.ends_at.isoformat()}" for b in blocks]

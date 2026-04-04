from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import PlanningNote, ScheduleBlock
from app.db.repositories.message_repo import list_recent_messages
from app.db.repositories.task_repo import list_active_tasks, list_upcoming_deadlines
from app.llm.style import get_style_profile
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
        short_checkin = self._is_short_checkin_message(latest_user_message)
        style_mode = user.profile.style.value if user.profile else self.settings.default_style
        style_profile = get_style_profile(style_mode)
        recent = list_recent_messages(session, user.id, limit=16)
        recent_thread = [
            f"{'assistant' if msg.direction.value == 'outbound' else 'user'}: {msg.body[:220]}"
            for msg in recent[-10:]
        ]
        if short_checkin:
            recent_thread = recent_thread[-3:]
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
        if outcome.response_goal in {"acknowledge_new_task", "ingestion_confirmation"}:
            # Avoid contaminating acknowledgements with unrelated active-task context.
            # The brief already has the newly captured task fact + next step.
            active_task_context = []
            deadline_context = []
        if outcome.response_goal == "timeline_summary":
            # The state layer already built the requested window/plan summary.
            # Leaving general task/deadline context in here can make the composer
            # drag unrelated items back into a narrow question like tomorrow morning.
            active_task_context = []
            deadline_context = []
        if outcome.response_goal == "confirm_update" and self._looks_like_clean_reschedule(outcome, latest_user_message):
            active_task_context = []
            deadline_context = []
        if short_checkin:
            active_task_context = []
            deadline_context = []

        profile_notes = self._profile_bio_notes(user.profile.bio if user.profile else None)
        recent_notes = self._recent_memory_notes(session, user.id)
        memory_notes = (profile_notes + recent_notes)[:6]
        state_flags = self._current_state_flags(session, user.id, now)

        max_chunks = 2
        if outcome.response_goal in {"timeline_summary", "answer_question", "replan_blocker"}:
            max_chunks = 3
        max_chunks = min(max_chunks, style_profile.max_chunks)

        return ReplyBrief(
            response_goal=outcome.response_goal,
            key_facts_to_include=outcome.key_facts_to_include,
            urgency_level=outcome.urgency_level,
            should_push_for_action=outcome.should_push_for_action,
            should_ask_question=outcome.should_ask_question,
            question_if_needed=outcome.question_if_needed,
            emotional_tone=outcome.emotional_tone,
            style_mode=style_mode,
            max_chunks=max_chunks,
            max_chunk_length=min(self.settings.max_sms_chars, style_profile.max_sms_chars),
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
            is_multi_task_turn=outcome.is_multi_task_turn,
            is_short_checkin=short_checkin,
            generated_at=now,
        )

    @staticmethod
    def _is_short_checkin_message(text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False
        words = lowered.split()
        if len(words) > 6:
            return False
        if any(token in lowered for token in ["need to", "have to", "deadline", "due", "assignment", "project", "task"]):
            return False
        checkin_tokens = {
            "yo",
            "hey",
            "hi",
            "sup",
            "whatup",
            "whatsup",
            "hello",
            "ping",
            "test",
            "you",
            "there",
            "work",
            "working",
            "online",
            "on",
            "back",
            "cooking",
        }
        return bool(set(words).intersection(checkin_tokens))

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
    def _profile_bio_notes(bio: str | None) -> list[str]:
        if not bio:
            return []
        notes: list[str] = []
        for raw in bio.splitlines():
            line = raw.strip()
            if not line or not line.startswith("-"):
                continue
            clean = line.lstrip("-").strip()
            if clean:
                notes.append(clean[:180])
            if len(notes) >= 3:
                break
        return notes

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

    @staticmethod
    def _looks_like_clean_reschedule(outcome: StateOutcome, latest_user_message: str) -> bool:
        if outcome.should_ask_question or outcome.should_push_for_action:
            return False
        lowered_message = latest_user_message.lower().strip()
        short_timing_shift = (
            lowered_message.startswith(("actually ", "wait ", "nah ", "make that ", "change it to "))
            or lowered_message in {"tonight", "tomorrow", "tomorrow morning", "tmr morning", "monday morning", "later"}
        )
        if short_timing_shift and len(outcome.key_facts_to_include) <= 2:
            return True
        return any(
            fact.lower().startswith(("moved that to ", "okay tomorrow", "okay tmr", "okay monday", "okay tonight", "okay this "))
            for fact in outcome.key_facts_to_include
        )

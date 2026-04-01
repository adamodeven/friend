from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.time_utils import parse_human_time, time_window_for_context
from app.db.models import DeadlineEvent, PlanningNote, Project, ScheduleBlock, Task, TaskStatus, UserProfile
from app.db.repositories.task_repo import create_task, find_active_task_by_title, list_active_tasks, mark_task_complete
from app.domain.reminder_engine import ReminderEngine
from app.domain.timeline_service import TimelineService
from app.schemas.intent import IntentResult
from app.schemas.reply import StateOutcome


class StateEngine:
    def __init__(self) -> None:
        self.reminders = ReminderEngine()
        self.timeline = TimelineService()

    def apply_intent(
        self,
        session: Session,
        *,
        user,
        intent: IntentResult,
        raw_text: str,
    ) -> StateOutcome:
        outcome = StateOutcome(
            response_goal="open_conversation",
            operational_reason=f"intent={intent.intent}",
        )

        if intent.intent == "add_task" and intent.task:
            project_id = None
            if intent.task.project:
                project = self._get_or_create_project(session, user.id, intent.task.project)
                project_id = project.id
                outcome.key_facts_to_include.append(f"attached to project {project.title}")

            task = create_task(
                session,
                user_id=user.id,
                title=intent.task.title,
                description=intent.task.description,
                project_id=project_id,
                deadline_at=intent.task.deadline_at,
                priority=intent.task.priority,
                extraction_confidence=intent.task.confidence,
                metadata_json={"source_text": raw_text, "next_step": intent.task.next_step},
            )
            outcome.response_goal = "acknowledge_new_task"
            outcome.key_facts_to_include.append(f"task captured: {task.title}")
            outcome.mention_progress = True

            if task.deadline_at:
                deadline = DeadlineEvent(
                    user_id=user.id,
                    task_id=task.id,
                    title=f"Deadline: {task.title}",
                    due_at=task.deadline_at,
                    source="message_parse",
                    confidence=intent.time_confidence or intent.task.confidence,
                )
                session.add(deadline)
                due_text = task.deadline_at.astimezone(ZoneInfo(user.timezone)).strftime("%a %-m/%-d %-I:%M%p").lower()
                outcome.key_facts_to_include.append(f"due {due_text}")
                outcome.mention_deadline = True
                outcome.urgency_level = self._urgency_from_deadline(task.deadline_at, user.timezone)

            reminder = self.reminders.schedule_for_task(session, task)
            if reminder:
                scheduled = reminder.scheduled_for.astimezone(ZoneInfo(user.timezone)).strftime("%-I:%M%p").lower()
                outcome.key_facts_to_include.append(f"check-in scheduled around {scheduled}")

            outcome.should_push_for_action = True
            outcome.suggested_next_step = intent.task.next_step or self._default_next_step(task.title)
            if self._should_offer_checkpoints(task_title=task.title, raw_text=raw_text, suggested_next_step=intent.task.next_step):
                outcome.question_if_needed = "want me to break that into 2 quick checkpoints?"
                outcome.should_ask_question = True
            else:
                outcome.question_if_needed = None
                outcome.should_ask_question = False
            outcome.emotional_tone = "direct"

        elif intent.intent == "complete_task":
            matched = self._match_task_from_text(session, user.id, raw_text)
            outcome.response_goal = "react_to_progress"
            outcome.mention_progress = True
            outcome.emotional_tone = "supportive"
            if matched:
                mark_task_complete(matched)
                outcome.key_facts_to_include.append(f"marked complete: {matched.title}")
                next_task = self._next_task(session, user.id)
                if next_task:
                    outcome.key_facts_to_include.append(f"next likely focus: {next_task.title}")
                    outcome.suggested_next_step = f"take a first pass on {next_task.title}"
                    outcome.should_push_for_action = True
            else:
                outcome.key_facts_to_include.append("completion noted, but task match was uncertain")
                outcome.should_ask_question = True
                outcome.question_if_needed = "which task should i mark done exactly?"

        elif intent.intent == "timeline_query":
            outcome.response_goal = "timeline_summary"
            outcome.mention_deadline = True
            outcome.emotional_tone = "direct"
            lowered = raw_text.lower()
            if "week" in lowered:
                summary = self.timeline.build_week_view(session, user.id, user.timezone)
                outcome.key_facts_to_include.append(summary)
            elif "next hour" in lowered:
                move = self.timeline.next_hour_recommendation(session, user.id, user.timezone)
                outcome.key_facts_to_include.append(move)
            else:
                today = self.timeline.build_today_view(session, user.id, user.timezone)
                outcome.key_facts_to_include.append(today)
            outcome.should_push_for_action = True

        elif intent.intent == "context_signal":
            starts, ends = time_window_for_context(intent.context_signal or raw_text, user.timezone)
            block = ScheduleBlock(
                user_id=user.id,
                block_type=self._normalize_context_type(intent.context_signal or raw_text),
                starts_at=starts,
                ends_at=ends,
                confidence=intent.confidence,
                notes=raw_text,
            )
            session.add(block)
            outcome.response_goal = "acknowledge_context"
            outcome.emotional_tone = "calm"
            outcome.key_facts_to_include.append(f"availability updated: {block.block_type}")
            outcome.key_facts_to_include.append(
                f"pause active until {ends.astimezone(ZoneInfo(user.timezone)).strftime('%-I:%M%p').lower()}"
            )
            outcome.avoid_topics.append("hard-pressure push while unavailable")

        elif intent.intent == "reflection":
            note = PlanningNote(
                user_id=user.id,
                note_type="slip_reason",
                content=raw_text,
                weight=0.7,
            )
            session.add(note)
            outcome.response_goal = "replan_blocker"
            outcome.emotional_tone = "supportive"
            outcome.key_facts_to_include.append("blocker pattern captured in memory")
            outcome.should_push_for_action = True
            outcome.should_ask_question = True
            outcome.question_if_needed = "what's the smallest next move that would unstick this?"

        elif intent.intent == "update_task":
            matched = self._match_task_from_text(session, user.id, raw_text)
            outcome.response_goal = "confirm_update"
            outcome.emotional_tone = "direct"
            if matched:
                if intent.time_reference:
                    parsed, conf = parse_human_time(intent.time_reference, timezone=user.timezone)
                    if parsed:
                        matched.deadline_at = parsed
                        matched.extraction_confidence = max(matched.extraction_confidence, conf)
                        outcome.key_facts_to_include.append(
                            f"deadline updated for {matched.title} -> {parsed.astimezone(ZoneInfo(user.timezone)).strftime('%a %-m/%-d %-I:%M%p').lower()}"
                        )
                        outcome.mention_deadline = True
                if intent.task_updates.get("status") == "blocked":
                    matched.status = TaskStatus.blocked
                    matched.blocked_reason = ", ".join(intent.blockers) if intent.blockers else raw_text
                    outcome.key_facts_to_include.append(f"task blocked: {matched.title}")
                    outcome.mention_dependency = True
                    outcome.response_goal = "replan_blocker"
                    outcome.should_ask_question = True
                    outcome.question_if_needed = "what has to happen first before this can move?"
                self.reminders.schedule_for_task(session, matched)
            else:
                outcome.key_facts_to_include.append("update noted, but task match was uncertain")
                outcome.should_ask_question = True
                outcome.question_if_needed = "which task do you want updated?"

        elif intent.intent == "status_query":
            outcome.response_goal = "answer_question"
            outcome.emotional_tone = "casual"
            lowered = raw_text.lower()
            if "canned" in lowered or "live" in lowered or "generated" in lowered:
                outcome.key_facts_to_include.append("these replies are live-generated right now, not canned templates")
            if "work" in lowered or "working" in lowered:
                outcome.key_facts_to_include.append("system is up and processing your messages")
            if any(token in lowered for token in ["live", "working", "online", "on now"]):
                outcome.key_facts_to_include.append("yes, i'm live right now and i received this message")
            outcome.should_ask_question = False

        else:
            lowered = raw_text.lower()
            if any(token in lowered for token in ["progress", "made progress", "moving now", "locked in", "momentum"]):
                outcome.response_goal = "react_to_progress"
                outcome.emotional_tone = "supportive"
                next_task = self._next_task(session, user.id)
                if next_task:
                    title = next_task.title.strip()
                    if len(title) > 60:
                        title = f"{title[:57].rstrip()}..."
                    outcome.suggested_next_step = f"take a first pass on '{title}'"
            else:
                outcome.response_goal = "answer_question" if "?" in raw_text else "open_conversation"
            outcome.emotional_tone = "casual"
            outcome.should_push_for_action = False

        self._update_profile_memory(session, user.id, raw_text)
        return outcome

    @staticmethod
    def _normalize_context_type(text: str) -> str:
        lowered = text.lower()
        if "class" in lowered:
            return "in_class"
        if "driving" in lowered:
            return "driving"
        if "dinner" in lowered or "social" in lowered:
            return "social_event"
        if "all nighter" in lowered:
            return "all_nighter"
        if "sleep" in lowered:
            return "sleeping"
        return "busy"

    @staticmethod
    def _default_next_step(task_title: str) -> str:
        return f"start a 20-min first pass on {task_title}"

    @staticmethod
    def _should_offer_checkpoints(*, task_title: str, raw_text: str, suggested_next_step: str | None) -> bool:
        if suggested_next_step:
            return False
        lowered = raw_text.lower()
        if any(token in lowered for token in ["just one", "one thing", "just need to", "just have to", "single thing"]):
            return False
        action_count = len(re.findall(r"\b(need to|have to|gotta|must)\b", lowered))
        if action_count > 1:
            return True
        if len(task_title.split()) >= 12:
            return True
        return False

    @staticmethod
    def _urgency_from_deadline(deadline_at: datetime, timezone_name: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone_name))
        if deadline_at.tzinfo is None:
            deadline = deadline_at.replace(tzinfo=ZoneInfo(timezone_name))
        else:
            deadline = deadline_at.astimezone(ZoneInfo(timezone_name))
        delta = deadline - now
        if delta <= timedelta(hours=3):
            return "critical"
        if delta <= timedelta(hours=18):
            return "high"
        if delta <= timedelta(days=2):
            return "medium"
        return "low"

    @staticmethod
    def _get_or_create_project(session: Session, user_id, title: str) -> Project:
        stmt = select(Project).where(Project.user_id == user_id, Project.title.ilike(title.strip()))
        project = session.execute(stmt).scalars().first()
        if project:
            return project
        project = Project(user_id=user_id, title=title.strip())
        session.add(project)
        session.flush()
        return project

    @staticmethod
    def _match_task_from_text(session: Session, user_id, text: str) -> Task | None:
        pieces = [segment.strip() for segment in text.lower().split() if len(segment.strip()) > 3]
        for piece in pieces[:8]:
            found = find_active_task_by_title(session, user_id, piece)
            if found:
                return found
        return None

    @staticmethod
    def _next_task(session: Session, user_id) -> Task | None:
        tasks = list_active_tasks(session, user_id)
        return tasks[0] if tasks else None

    @staticmethod
    def _update_profile_memory(session: Session, user_id, raw_text: str) -> None:
        lowered = raw_text.lower()
        if any(token in lowered for token in ["underestimated", "distracted", "switching", "conflict"]):
            note = PlanningNote(
                user_id=user_id,
                note_type="behavior_pattern",
                content=raw_text,
                weight=0.8,
            )
            session.add(note)

        profile = session.execute(select(UserProfile).where(UserProfile.user_id == user_id)).scalars().first()
        if profile and profile.planning_preferences is not None:
            prefs = dict(profile.planning_preferences)
            prefs["last_update_at"] = datetime.now(tz=timezone.utc).isoformat()
            profile.planning_preferences = prefs

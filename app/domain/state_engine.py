from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.time_utils import parse_human_time, time_window_for_context
from app.db.models import DeadlineEvent, PlanningNote, Project, ScheduleBlock, Task, TaskStatus, UserProfile
from app.db.repositories.task_repo import create_task, find_active_task_by_title, list_active_tasks, mark_task_complete
from app.domain.reminder_engine import ReminderEngine
from app.domain.timeline_service import TimelineService
from app.schemas.intent import IntentResult


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
    ) -> str:
        action_summary = ""

        if intent.intent == "add_task" and intent.task:
            project_id = None
            if intent.task.project:
                project = self._get_or_create_project(session, user.id, intent.task.project)
                project_id = project.id
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
            self.reminders.schedule_for_task(session, task)
            action_summary = f"added '{task.title}'"
            if task.deadline_at:
                due_text = task.deadline_at.astimezone(ZoneInfo(user.timezone)).strftime("%a %-m/%-d %-I:%M%p").lower()
                action_summary += f" due {due_text}"
            if intent.task.next_step:
                action_summary += f". next move: {intent.task.next_step}"

        elif intent.intent == "complete_task":
            matched = self._match_task_from_text(session, user.id, raw_text)
            if matched:
                mark_task_complete(matched)
                action_summary = f"marked '{matched.title}' complete"
                next_task = self._next_task(session, user.id)
                if next_task:
                    action_summary += f". next up: {next_task.title}"
            else:
                action_summary = "couldn't match the exact task yet, tell me which one and i'll mark it done"

        elif intent.intent == "timeline_query":
            lowered = raw_text.lower()
            if "week" in lowered:
                action_summary = self.timeline.build_week_view(session, user.id, user.timezone)
            elif "next hour" in lowered:
                action_summary = self.timeline.next_hour_recommendation(session, user.id, user.timezone)
            else:
                action_summary = self.timeline.build_today_view(session, user.id, user.timezone)

        elif intent.intent == "status_query":
            action_summary = (
                "i keep your task graph live over text: deadlines, blockers, dependencies, reminders, and replanning. "
                "drop anything you need done and i'll track it + push follow-through."
            )

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
            action_summary = f"noted you're busy ({block.block_type}). i'll back off until around {ends.astimezone(ZoneInfo(user.timezone)).strftime('%-I:%M%p').lower()}"

        elif intent.intent == "reflection":
            note = PlanningNote(
                user_id=user.id,
                note_type="slip_reason",
                content=raw_text,
                weight=0.7,
            )
            session.add(note)
            action_summary = "logged that pattern. let's reduce scope on the next sprint and add tighter checkpoints."

        elif intent.intent == "update_task":
            matched = self._match_task_from_text(session, user.id, raw_text)
            if matched:
                if intent.time_reference:
                    parsed, conf = parse_human_time(intent.time_reference, timezone=user.timezone)
                    if parsed:
                        matched.deadline_at = parsed
                        matched.extraction_confidence = max(matched.extraction_confidence, conf)
                if intent.task_updates.get("status") == "blocked":
                    matched.status = TaskStatus.blocked
                    matched.blocked_reason = ", ".join(intent.blockers) if intent.blockers else raw_text
                self.reminders.schedule_for_task(session, matched)
                action_summary = f"updated '{matched.title}'"
            else:
                action_summary = "i can update it, just name the task directly once."
        else:
            action_summary = self._general_chat_reply(raw_text, session, user)

        self._update_profile_memory(session, user.id, raw_text)
        return action_summary

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
        return "busy"

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

    def _general_chat_reply(self, raw_text: str, session: Session, user) -> str:
        lowered = raw_text.lower().strip()
        if lowered in {"hey", "yo", "sup", "hiya", "hello", "hi", "test"}:
            return "yo, i'm here. text what you need done + due date and i'll handle the tracking."
        return self.timeline.next_hour_recommendation(session, user.id, user.timezone)

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.time_utils import interpret_time_reference, parse_human_time, time_window_for_context
from app.db.models import ConversationMessage, DeadlineEvent, MessageDirection, PlanningNote, Project, Reminder, ReminderStatus, ScheduleBlock, Task, TaskDependency, TaskStatus, UserProfile
from app.db.repositories.message_repo import list_recent_messages
from app.db.repositories.task_repo import (
    create_task,
    create_task_dependency,
    find_active_task_by_title,
    list_active_tasks,
    mark_task_complete,
    record_task_progress,
    record_task_slip,
    set_task_blocked,
)
from app.domain.reminder_engine import ReminderEngine
from app.domain.task_semantics import (
    ACTION_KIND_QUICK_ADMIN,
    ACTION_KIND_QUICK_MESSAGE,
    default_next_step,
    infer_action_kind,
    humanize_window_phrase,
    is_soft_later_phrase,
)
from app.domain.timeline_service import TimelineService
from app.schemas.intent import ExtractedDependency, ExtractedTask, IntentResult
from app.schemas.reply import ReplyTaskContext, StateOutcome


@dataclass(slots=True)
class CapturedTaskEntry:
    extracted: ExtractedTask
    task: Task
    parent_task: Task | None = None


@dataclass(slots=True)
class CaptureBundle:
    entries: list[CapturedTaskEntry] = field(default_factory=list)
    root_tasks: list[Task] = field(default_factory=list)
    dependency_count: int = 0
    subtask_count: int = 0
    reminder_count: int = 0
    created_prerequisites: list[Task] = field(default_factory=list)
    deadline_tasks: list[Task] = field(default_factory=list)
    ambiguous_deadline_task: Task | None = None
    ambiguous_time_reference: str | None = None
    unresolved_blocker_task: Task | None = None
    unresolved_blocker_text: str | None = None


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
        source_message_id=None,
    ) -> StateOutcome:
        outcome = StateOutcome(
            response_goal="open_conversation",
            operational_reason=f"intent={intent.intent}",
        )

        extracted_tasks = intent.tasks or ([intent.task] if intent.task else [])
        if intent.intent == "add_task" and extracted_tasks:
            bundle = self._capture_tasks(
                session,
                user=user,
                extracted_tasks=extracted_tasks,
                raw_text=raw_text,
                needs_time_clarification=intent.needs_clarification,
                time_reference=intent.time_reference,
                time_confidence=intent.time_confidence,
                source_message_id=source_message_id,
            )
            outcome.response_goal = "acknowledge_new_task"
            outcome.emotional_tone = "direct"
            outcome.mention_progress = True
            outcome.mention_dependency = bundle.dependency_count > 0 or bundle.subtask_count > 0
            outcome.is_multi_task_turn = len(bundle.root_tasks) > 1
            outcome.task_contexts = [self._task_context(task) for task in bundle.root_tasks[:6]]
            outcome.should_push_for_action = False
            outcome.operational_reason = (
                f"intent=add_task tasks={len(bundle.root_tasks)} deps={bundle.dependency_count} subtasks={bundle.subtask_count}"
            )

            if len(bundle.root_tasks) == 1:
                task = bundle.root_tasks[0]
                outcome.key_facts_to_include.append(self._new_task_fact(task, user.timezone))
            else:
                outcome.key_facts_to_include.append(f"okay, that's {len(bundle.root_tasks)} different things")
                outcome.key_facts_to_include.append("i'm holding " + ", ".join(task.title for task in bundle.root_tasks[:3]))
            if bundle.subtask_count:
                outcome.key_facts_to_include.append(f"i broke one of those into {bundle.subtask_count} smaller step{'s' if bundle.subtask_count != 1 else ''}")
            if bundle.dependency_count:
                outcome.key_facts_to_include.append("there's a dependency in there too")
            if bundle.created_prerequisites:
                outcome.key_facts_to_include.append(f"real blocker looks like {bundle.created_prerequisites[0].title}")
            actionable_deadline_tasks = [
                task
                for task in bundle.deadline_tasks
                if task.deadline_at is not None
                and (
                    task.start_after is None
                    or self._normalize_dt(task.start_after, user.timezone) <= datetime.now(tz=ZoneInfo(user.timezone))
                )
            ]
            if actionable_deadline_tasks:
                earliest = min(
                    actionable_deadline_tasks,
                    key=lambda task: self._normalize_dt(task.deadline_at, user.timezone) or datetime.max.replace(tzinfo=ZoneInfo(user.timezone)),
                )
                due_text = self._format_due(earliest.deadline_at, user.timezone)
                if due_text:
                    outcome.key_facts_to_include.append(f"deadline coming up {due_text}")
                outcome.mention_deadline = True
                if earliest.deadline_at is not None:
                    outcome.urgency_level = self._urgency_from_deadline(earliest.deadline_at, user.timezone)
            elif bundle.deadline_tasks:
                outcome.mention_deadline = True
            if (
                bundle.reminder_count
                and len(bundle.root_tasks) == 1
                and not intent.context_signal
                and not bundle.root_tasks[0].deadline_is_ambiguous
                and self._task_action_kind(bundle.root_tasks[0]) not in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}
                and (bundle.root_tasks[0].deadline_at is not None or bundle.root_tasks[0].start_after is not None)
            ):
                task = bundle.root_tasks[0]
                pending = self._next_pending_reminder_for_task(session, task.id)
                if pending:
                    scheduled = pending.scheduled_for.astimezone(ZoneInfo(user.timezone)).strftime("%-I:%M%p").lower()
                    outcome.key_facts_to_include.append(f"i'll circle back around {scheduled}")

            preferred_new_focus = self._preferred_focus_from_new_tasks(bundle.root_tasks, user.timezone)
            if preferred_new_focus and self._should_push_new_task_focus(
                preferred_new_focus,
                task_count=len(bundle.root_tasks),
                timezone_name=user.timezone,
                raw_text=raw_text,
            ):
                outcome.suggested_next_step = self._next_step_for_task(preferred_new_focus)
                outcome.should_push_for_action = True

            if intent.context_signal:
                block = self._record_context_block(session, user=user, raw_text=intent.context_signal, confidence=intent.confidence)
                context_label = self._context_ack_text(block.block_type)
                outcome.key_facts_to_include.append(context_label)
                outcome.should_push_for_action = False
                outcome.suggested_next_step = None
            needs_post_context_followup = intent.context_signal and self._looks_like_placeholder_assignment(bundle.root_tasks)

            if bundle.ambiguous_deadline_task and bundle.ambiguous_time_reference:
                outcome.should_ask_question = True
                outcome.question_if_needed = self._time_clarification_question(
                    task_title=bundle.ambiguous_deadline_task.title,
                    time_reference=bundle.ambiguous_time_reference,
                )
            elif bundle.unresolved_blocker_task and bundle.unresolved_blocker_text:
                outcome.should_ask_question = True
                outcome.question_if_needed = self._blocker_followup_question(bundle.unresolved_blocker_task.title)
            elif len(bundle.root_tasks) == 1 and self._should_offer_checkpoints(
                task_title=bundle.root_tasks[0].title,
                raw_text=raw_text,
                suggested_next_step=extracted_tasks[0].next_step,
            ):
                outcome.should_ask_question = True
                outcome.question_if_needed = "want me to break that into 2 quick checkpoints?"
            elif len(bundle.root_tasks) >= 3 and self._should_ask_load_prioritization_question(bundle.root_tasks):
                outcome.should_ask_question = True
                outcome.question_if_needed = "which one of those actually has the least wiggle room?"
                outcome.should_push_for_action = False
                outcome.suggested_next_step = None
            else:
                outcome.should_ask_question = False
                outcome.question_if_needed = None

            if needs_post_context_followup and not outcome.should_ask_question:
                outcome.should_ask_question = True
                outcome.question_if_needed = "when you're out, send me the assignment details and i'll slot it cleanly"
            elif (
                len(bundle.root_tasks) == 1
                and self._looks_like_placeholder_assignment(bundle.root_tasks)
                and not outcome.should_ask_question
            ):
                outcome.should_ask_question = True
                outcome.question_if_needed = "send me the assignment details when you have them and i'll slot it cleanly"
                outcome.should_push_for_action = False
                outcome.suggested_next_step = None

        elif intent.intent == "complete_task":
            matched = self._match_task_from_text(session, user.id, raw_text)
            outcome.response_goal = "react_to_progress"
            outcome.mention_progress = True
            outcome.emotional_tone = "supportive"
            if matched:
                mark_task_complete(matched)
                unlocked = self._refresh_successors(matched)
                session.flush()
                outcome.key_facts_to_include.append(f"{matched.title} is handled")
                if unlocked:
                    outcome.key_facts_to_include.append(f"that frees up {unlocked[0].title}")
                    outcome.mention_dependency = True
                next_task = self.timeline.recommend_next_task(session, user.id, user.timezone)
                if next_task:
                    outcome.key_facts_to_include.append(f"after that, {next_task.title} is the one to hit")
                    outcome.suggested_next_step = self._next_step_for_task(next_task)
                    outcome.should_push_for_action = True
                    outcome.should_ask_question = False
            else:
                outcome.key_facts_to_include.append("completion noted, but task match was uncertain")
                outcome.should_ask_question = True
                outcome.question_if_needed = "which task should i mark done exactly?"

        elif intent.intent == "timeline_query":
            outcome.response_goal = "timeline_summary"
            outcome.mention_deadline = True
            outcome.emotional_tone = "direct"
            lowered = raw_text.lower()
            if "plan for" in lowered:
                summary = self.timeline.build_project_view(session, user.id, user.timezone, raw_text)
            elif "weekend" in lowered:
                summary = self.timeline.build_weekend_view(session, user.id, user.timezone)
            elif "tomorrow morning" in lowered or "tmr morning" in lowered:
                summary = self.timeline.build_tomorrow_morning_view(session, user.id, user.timezone)
            elif "tonight" in lowered:
                summary = self.timeline.build_tonight_view(session, user.id, user.timezone)
            elif "next hour" in lowered:
                summary = self.timeline.next_hour_recommendation(session, user.id, user.timezone)
            elif custom_window := self._timeline_custom_window(raw_text, user.timezone):
                summary = self.timeline.build_window_view(
                    session,
                    user.id,
                    user.timezone,
                    label=custom_window[0],
                    start=custom_window[1],
                    end=custom_window[2],
                )
            elif "week" in lowered:
                summary = self.timeline.build_week_view(session, user.id, user.timezone)
            else:
                summary = self.timeline.build_today_view(session, user.id, user.timezone)
            outcome.key_facts_to_include.append(summary)
            outcome.should_push_for_action = True

        elif intent.intent == "context_signal":
            block = self._record_context_block(session, user=user, raw_text=intent.context_signal or raw_text, confidence=intent.confidence)
            outcome.response_goal = "acknowledge_context"
            outcome.emotional_tone = "calm"
            outcome.key_facts_to_include.append(self._context_ack_text(block.block_type))
            outcome.key_facts_to_include.append("i'll back off for now")
            outcome.avoid_topics.append("hard-pressure push while unavailable")

        elif intent.intent == "reflection":
            target = self._reflection_target(session, user.id, raw_text, user.timezone)
            note = PlanningNote(
                user_id=user.id,
                note_type="slip_reason",
                content=raw_text,
                related_task_id=target.id if target else None,
                weight=0.75,
            )
            session.add(note)
            outcome.response_goal = "replan_blocker"
            outcome.emotional_tone = "supportive"
            outcome.should_push_for_action = True
            if target:
                record_task_slip(target, reason=raw_text, next_step=self._next_step_for_task(target))
                self._refresh_task_block_state(target)
                outcome.key_facts_to_include.append(f"slip noted for {target.title}")
                outcome.task_contexts = [self._task_context(target)]
                outcome.suggested_next_step = self._next_step_for_task(target)
                if target.status == TaskStatus.blocked:
                    outcome.mention_dependency = True
                    outcome.should_ask_question = True
                    outcome.question_if_needed = self._blocker_followup_question(target.title)
                else:
                    outcome.should_ask_question = False
            else:
                outcome.key_facts_to_include.append("blocker pattern captured in memory")
                outcome.should_ask_question = True
                outcome.question_if_needed = "what's the smallest next move that would unstick this?"

        elif intent.intent == "update_task":
            bulk_action = str(intent.task_updates.get("bulk_action", "")).strip().lower()
            action = str(intent.task_updates.get("action", "")).strip().lower()
            if bulk_action == "clear_active_tasks":
                archived_count = self._clear_active_tasks(session, user.id)
                outcome.response_goal = "confirm_update"
                outcome.emotional_tone = "direct"
                outcome.key_facts_to_include.append(f"cleared everything out ({archived_count} archived)")
                outcome.should_push_for_action = archived_count == 0
                if archived_count == 0:
                    outcome.suggested_next_step = "drop the next task you want tracked"
                outcome.should_ask_question = False
            else:
                matched = self._resolve_update_target(
                    session,
                    user.id,
                    raw_text,
                    intent.blockers,
                    time_reference=intent.time_reference,
                    source_message_id=source_message_id,
                )
                outcome.response_goal = "confirm_update"
                outcome.emotional_tone = "direct"
                if matched:
                    applied_change = False
                    if action == "archive":
                        archived_count = self._archive_task_and_skip_pending_reminders(session, matched)
                        session.flush()
                        outcome.key_facts_to_include.append(f"we're off {matched.title}")
                        if archived_count > 1:
                            outcome.key_facts_to_include.append(f"that takes {archived_count - 1} linked subtasks with it")
                        applied_change = True
                    if intent.time_reference:
                        parsed_deadline = interpret_time_reference(intent.time_reference, timezone=user.timezone)
                        if parsed_deadline.deadline_at or parsed_deadline.soft_deadline_at:
                            action_kind = self._task_action_kind(matched)
                            is_window_phrase = (
                                parsed_deadline.granularity in {"hour", "part_of_day"}
                                and not intent.time_reference.lower().strip().startswith(("by ", "before ", "due "))
                            )
                            matched.deadline_at = parsed_deadline.deadline_at
                            matched.soft_deadline_at = parsed_deadline.soft_deadline_at
                            matched.start_after = parsed_deadline.soft_deadline_at if is_window_phrase else None
                            matched.deadline_source_phrase = intent.time_reference
                            matched.deadline_confidence = max(matched.deadline_confidence, parsed_deadline.confidence)
                            matched.extraction_confidence = max(matched.extraction_confidence, parsed_deadline.confidence)
                            matched.deadline_is_ambiguous = parsed_deadline.is_ambiguous
                            matched.deadline_granularity = parsed_deadline.granularity
                            matched.deadline_timezone = parsed_deadline.timezone or user.timezone
                            matched.source_message_id = source_message_id or matched.source_message_id
                            if action_kind in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN} and not is_window_phrase:
                                matched.start_after = None
                            time_phrase = self._reschedule_phrase(parsed_deadline, intent.time_reference, user.timezone)
                            outcome.key_facts_to_include.append(self._reschedule_ack_text(time_phrase))
                            outcome.mention_deadline = True
                            applied_change = True
                    if intent.task_updates.get("status") == "blocked" or intent.blockers:
                        blocker_texts = intent.blockers or [raw_text]
                        resolved = self._apply_blockers_to_task(
                            session,
                            user=user,
                            task=matched,
                            blocker_texts=blocker_texts,
                            title_index={},
                            bundle=None,
                        )
                        self._refresh_task_block_state(matched)
                        outcome.key_facts_to_include.append(f"{matched.title} is blocked for now")
                        outcome.mention_dependency = True
                        outcome.response_goal = "replan_blocker"
                        applied_change = True
                        if resolved:
                            outcome.key_facts_to_include.append(f"real blocker is {resolved[0].title}")
                            outcome.suggested_next_step = self._next_step_for_task(resolved[0])
                            outcome.should_ask_question = False
                        else:
                            outcome.should_ask_question = True
                            outcome.question_if_needed = self._blocker_followup_question(matched.title)
                    if not applied_change:
                        outcome.key_facts_to_include.append(f"matched task: {matched.title}")
                        outcome.should_push_for_action = False
                        outcome.should_ask_question = True
                        outcome.question_if_needed = f"what do you want me to change on '{matched.title}'?"
                    elif action != "archive":
                        self.reminders.schedule_for_task(session, matched)
                else:
                    if intent.blockers:
                        outcome.response_goal = "replan_blocker"
                        outcome.key_facts_to_include.append("blocker noted, but task match was uncertain")
                        outcome.key_facts_to_include.append(f"blocker: {intent.blockers[0]}")
                        outcome.should_ask_question = True
                        outcome.question_if_needed = "which task is blocked by that?"
                    elif action == "archive":
                        outcome.key_facts_to_include.append("drop request noted, but task match was uncertain")
                        outcome.should_ask_question = True
                        outcome.question_if_needed = "which task do you want me to drop exactly?"
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
            if any(
                token in lowered
                for token in [
                    "progress",
                    "made progress",
                    "moving now",
                    "locked in",
                    "momentum",
                    "ready",
                    "all set",
                    "set up",
                ]
            ):
                outcome.response_goal = "react_to_progress"
                outcome.emotional_tone = "supportive"
                matched = self._match_task_from_text(session, user.id, raw_text) or self.timeline.recommend_next_task(
                    session, user.id, user.timezone
                )
                if matched:
                    record_task_progress(matched, next_step=self._next_step_for_task(matched))
                    outcome.suggested_next_step = self._next_step_for_task(matched)
            else:
                outcome.response_goal = "answer_question" if "?" in raw_text else "open_conversation"
                outcome.emotional_tone = "casual"
                outcome.should_push_for_action = False

        self._update_profile_memory(session, user.id, raw_text)
        return outcome

    def _capture_tasks(
        self,
        session: Session,
        *,
        user,
        extracted_tasks: list[ExtractedTask],
        raw_text: str,
        needs_time_clarification: bool,
        time_reference: str | None,
        time_confidence: float,
        source_message_id,
    ) -> CaptureBundle:
        bundle = CaptureBundle()
        title_index: dict[str, Task] = {}
        for extracted in extracted_tasks:
            self._create_task_tree(
                session,
                user=user,
                extracted=extracted,
                raw_text=raw_text,
                bundle=bundle,
                title_index=title_index,
                parent_task=None,
                inherited_project_id=None,
                source_message_id=source_message_id,
            )

        for entry in bundle.entries:
            if entry.parent_task is not None:
                if self._ensure_dependency(
                    session,
                    user_id=user.id,
                    predecessor_task=entry.task,
                    successor_task=entry.parent_task,
                    dependency_type="subtask",
                    metadata_json={"source": "extracted_subtask"},
                ):
                    bundle.dependency_count += 1

            dependency_count = self._apply_declared_dependencies(
                session,
                user=user,
                task=entry.task,
                extracted=entry.extracted,
                title_index=title_index,
                bundle=bundle,
            )
            bundle.dependency_count += dependency_count

            blockers = entry.extracted.blockers
            if blockers:
                resolved = self._apply_blockers_to_task(
                    session,
                    user=user,
                    task=entry.task,
                    blocker_texts=blockers,
                    title_index=title_index,
                    bundle=bundle,
                )
                if not resolved and bundle.unresolved_blocker_task is None:
                    bundle.unresolved_blocker_task = entry.task
                    bundle.unresolved_blocker_text = blockers[0]

            self._refresh_task_block_state(entry.task)

        if needs_time_clarification and bundle.ambiguous_deadline_task is None and bundle.root_tasks:
            bundle.ambiguous_deadline_task = bundle.root_tasks[0]
            bundle.ambiguous_time_reference = time_reference or "that time"
        elif (
            time_reference
            and time_confidence < 0.6
            and self._time_reference_needs_followup(time_reference)
            and bundle.root_tasks
            and bundle.ambiguous_deadline_task is None
        ):
            bundle.ambiguous_deadline_task = bundle.root_tasks[0]
            bundle.ambiguous_time_reference = time_reference

        for entry in bundle.entries:
            reminder = self.reminders.schedule_for_task(session, entry.task)
            if reminder:
                bundle.reminder_count += 1
        return bundle

    def _create_task_tree(
        self,
        session: Session,
        *,
        user,
        extracted: ExtractedTask,
        raw_text: str,
        bundle: CaptureBundle,
        title_index: dict[str, Task],
        parent_task: Task | None,
        inherited_project_id,
        source_message_id,
    ) -> Task:
        project_id = inherited_project_id
        if extracted.project:
            project = self._get_or_create_project(session, user.id, extracted.project)
            project_id = project.id

        deadline_source = extracted.deadline.source_phrase if extracted.deadline else extracted.deadline_text
        deadline_at = extracted.deadline.deadline_at if extracted.deadline and extracted.deadline.deadline_at else extracted.deadline_at
        soft_deadline_at = (
            extracted.deadline.soft_deadline_at if extracted.deadline and extracted.deadline.soft_deadline_at else extracted.soft_deadline_at
        )
        deadline_confidence = extracted.deadline.confidence if extracted.deadline else extracted.confidence
        deadline_is_ambiguous = extracted.deadline.is_ambiguous if extracted.deadline else False
        deadline_granularity = extracted.deadline.granularity if extracted.deadline else "unknown"
        deadline_timezone = extracted.deadline.timezone if extracted.deadline else user.timezone
        action_kind = extracted.action_kind or infer_action_kind(
            extracted.title,
            deadline_text=deadline_source,
            start_after=extracted.start_after,
        )
        next_step = extracted.next_step or self._default_next_step(extracted.title, action_kind=action_kind)
        metadata = {
            "source_text": raw_text,
            "extracted_blockers": extracted.blockers,
            "declared_dependency_titles": [dependency.title for dependency in extracted.dependencies],
            "timing_kind": "windowed_action" if extracted.start_after else "deadline",
            "action_kind": action_kind,
        }

        task = self._find_reusable_task(
            session,
            user_id=user.id,
            title=extracted.title,
            parent_task=parent_task,
            title_index=title_index,
        )
        task_is_new = task is None
        if task is None:
            task = create_task(
                session,
                user_id=user.id,
                title=extracted.title,
                description=extracted.description,
                project_id=project_id,
                source_message_id=source_message_id,
                parent_task_id=parent_task.id if parent_task else None,
                next_step=next_step,
                deadline_at=deadline_at,
                soft_deadline_at=soft_deadline_at,
                start_after=extracted.start_after,
                deadline_source_phrase=deadline_source,
                deadline_confidence=deadline_confidence,
                deadline_is_ambiguous=deadline_is_ambiguous,
                deadline_granularity=deadline_granularity,
                deadline_timezone=deadline_timezone,
                priority=extracted.priority,
                extraction_confidence=extracted.confidence,
                metadata_json=metadata,
            )
            task.user = user
        else:
            self._merge_extracted_into_task(
                task,
                extracted=extracted,
                project_id=project_id,
                next_step=next_step,
                deadline_at=deadline_at,
                soft_deadline_at=soft_deadline_at,
                deadline_source=deadline_source,
                deadline_confidence=deadline_confidence,
                deadline_is_ambiguous=deadline_is_ambiguous,
                deadline_granularity=deadline_granularity,
                deadline_timezone=deadline_timezone,
                action_kind=action_kind,
                metadata=metadata,
            )
            if source_message_id is not None:
                task.source_message_id = source_message_id

        if not any(entry.task.id == task.id and entry.parent_task == parent_task for entry in bundle.entries):
            bundle.entries.append(CapturedTaskEntry(extracted=extracted, task=task, parent_task=parent_task))
        if parent_task is None and not any(existing.id == task.id for existing in bundle.root_tasks):
            bundle.root_tasks.append(task)
        title_index.setdefault(self._normalize_title(task.title), task)

        if task.deadline_at and (task_is_new or not any(existing.id == task.id for existing in bundle.deadline_tasks)):
            session.add(
                DeadlineEvent(
                    user_id=user.id,
                    task_id=task.id,
                    title=f"Deadline: {task.title}",
                    due_at=task.deadline_at,
                    source="message_parse",
                    confidence=deadline_confidence,
                )
            )
            bundle.deadline_tasks.append(task)
        if (
            deadline_is_ambiguous
            and bundle.ambiguous_deadline_task is None
            and deadline_source
            and self._time_reference_needs_followup(deadline_source)
        ):
            bundle.ambiguous_deadline_task = task
            bundle.ambiguous_time_reference = deadline_source or extracted.deadline_text or "that time"

        for subtask in extracted.subtasks:
            bundle.subtask_count += 1
            child = self._create_task_tree(
                session,
                user=user,
                extracted=subtask,
                raw_text=raw_text,
                bundle=bundle,
                title_index=title_index,
                parent_task=task,
                inherited_project_id=project_id,
                source_message_id=source_message_id,
            )
            title_index.setdefault(self._normalize_title(child.title), child)
        return task

    def _apply_declared_dependencies(
        self,
        session: Session,
        *,
        user,
        task: Task,
        extracted: ExtractedTask,
        title_index: dict[str, Task],
        bundle: CaptureBundle,
    ) -> int:
        created_count = 0
        for dependency in extracted.dependencies:
            related = self._resolve_dependency_task(
                session,
                user=user,
                task=task,
                dependency=dependency,
                title_index=title_index,
                bundle=bundle,
            )
            if related is None or related.id == task.id:
                continue
            predecessor = related
            successor = task
            dependency_type = "finish_to_start"
            metadata = {"source": "declared_dependency", "relation": dependency.relation}

            if dependency.relation == "blocks":
                predecessor = task
                successor = related
            elif dependency.relation == "subtask_of":
                task.parent_task_id = related.id
                predecessor = task
                successor = related
                dependency_type = "subtask"
            elif dependency.relation == "related_to":
                dependency_type = "related_to"

            if self._ensure_dependency(
                session,
                user_id=user.id,
                predecessor_task=predecessor,
                successor_task=successor,
                dependency_type=dependency_type,
                metadata_json=metadata,
            ):
                created_count += 1
        return created_count

    def _resolve_dependency_task(
        self,
        session: Session,
        *,
        user,
        task: Task,
        dependency: ExtractedDependency,
        title_index: dict[str, Task],
        bundle: CaptureBundle,
    ) -> Task | None:
        existing = self._lookup_task_by_title(session, user.id, dependency.title, title_index=title_index)
        if existing is not None:
            return existing

        normalized_title = self._canonicalize_dependency_title(dependency.title)
        if normalized_title is None:
            return None
        related = create_task(
            session,
            user_id=user.id,
            title=normalized_title,
            priority=max(3, task.priority),
            next_step=self._default_next_step(
                normalized_title,
                action_kind=infer_action_kind(normalized_title),
            ),
            extraction_confidence=dependency.confidence,
            metadata_json={
                "created_from_dependency": dependency.title,
                "notes": dependency.notes,
                "action_kind": infer_action_kind(normalized_title),
            },
        )
        related.user = user
        title_index[self._normalize_title(related.title)] = related
        bundle.created_prerequisites.append(related)
        return related

    def _apply_blockers_to_task(
        self,
        session: Session,
        *,
        user,
        task: Task,
        blocker_texts: list[str],
        title_index: dict[str, Task],
        bundle: CaptureBundle | None,
    ) -> list[Task]:
        resolved: list[Task] = []
        for blocker_text in blocker_texts:
            prerequisite = self._resolve_prerequisite_task(
                session,
                user=user,
                blocker_text=blocker_text,
                title_index=title_index,
            )
            if prerequisite is None or prerequisite.id == task.id:
                continue
            if self._ensure_dependency(
                session,
                user_id=user.id,
                predecessor_task=prerequisite,
                successor_task=task,
                dependency_type="finish_to_start",
                metadata_json={"source": "blocker_text", "raw_text": blocker_text},
            ):
                if bundle is not None:
                    bundle.dependency_count += 1
            resolved.append(prerequisite)
            if bundle is not None and prerequisite not in bundle.created_prerequisites and prerequisite.created_at == prerequisite.updated_at:
                # Heuristic: freshly created prerequisites have identical timestamps in tests/local DB.
                bundle.created_prerequisites.append(prerequisite)

        unresolved_dependencies = [prereq for prereq in resolved if prereq.status != TaskStatus.completed]
        if unresolved_dependencies:
            set_task_blocked(
                task,
                reason=blocker_texts[0],
                blocker_details_json={
                    "blockers": blocker_texts,
                    "dependency_titles": [prereq.title for prereq in unresolved_dependencies],
                    "dependency_task_ids": [str(prereq.id) for prereq in unresolved_dependencies],
                },
            )
        elif blocker_texts and not resolved:
            set_task_blocked(
                task,
                reason=blocker_texts[0],
                blocker_details_json={"blockers": blocker_texts},
            )
            if bundle is not None and bundle.unresolved_blocker_task is None:
                bundle.unresolved_blocker_task = task
                bundle.unresolved_blocker_text = blocker_texts[0]
        return resolved

    def _resolve_prerequisite_task(
        self,
        session: Session,
        *,
        user,
        blocker_text: str,
        title_index: dict[str, Task],
    ) -> Task | None:
        existing = self._lookup_task_by_title(session, user.id, blocker_text, title_index=title_index)
        if existing is not None:
            return existing

        normalized_title = self._normalize_blocker_to_task_title(blocker_text)
        if normalized_title is None:
            return None

        existing = self._lookup_task_by_title(session, user.id, normalized_title, title_index=title_index)
        if existing is not None:
            return existing

        prerequisite = create_task(
            session,
            user_id=user.id,
            title=normalized_title,
            priority=3,
            next_step=self._default_next_step(
                normalized_title,
                action_kind=infer_action_kind(normalized_title),
            ),
            extraction_confidence=0.72,
            metadata_json={"created_from_blocker": blocker_text, "action_kind": infer_action_kind(normalized_title)},
        )
        prerequisite.user = user
        title_index[self._normalize_title(prerequisite.title)] = prerequisite
        return prerequisite

    @staticmethod
    def _ensure_dependency(
        session: Session,
        *,
        user_id,
        predecessor_task: Task,
        successor_task: Task,
        dependency_type: str,
        metadata_json: dict | None = None,
    ) -> bool:
        existing = session.execute(
            select(TaskDependency).where(
                TaskDependency.user_id == user_id,
                TaskDependency.predecessor_task_id == predecessor_task.id,
                TaskDependency.successor_task_id == successor_task.id,
                TaskDependency.dependency_type == dependency_type,
            )
        ).scalars().first()
        if existing is not None:
            return False
        create_task_dependency(
            session,
            user_id=user_id,
            predecessor_task_id=predecessor_task.id,
            successor_task_id=successor_task.id,
            dependency_type=dependency_type,
            metadata_json=metadata_json or {},
        )
        return True

    @staticmethod
    def _clear_active_tasks(session: Session, user_id) -> int:
        tasks = list_active_tasks(session, user_id)
        for task in tasks:
            task.status = TaskStatus.archived

        if tasks:
            task_ids = [task.id for task in tasks]
            reminder_stmt = select(Reminder).where(
                Reminder.user_id == user_id,
                Reminder.status == ReminderStatus.pending,
                Reminder.task_id.in_(task_ids),
            )
            reminders = list(session.execute(reminder_stmt).scalars().all())
            for reminder in reminders:
                reminder.status = ReminderStatus.skipped
                reminder.last_error = "skipped due to bulk clear action"
        return len(tasks)

    def _archive_task_and_skip_pending_reminders(self, session: Session, task: Task) -> int:
        subtree = self._task_subtree(task)
        task_ids = [node.id for node in subtree]

        reminders = (
            session.execute(
                select(Reminder).where(
                    Reminder.user_id == task.user_id,
                    Reminder.task_id.in_(task_ids),
                    Reminder.status == ReminderStatus.pending,
                )
            )
            .scalars()
            .all()
        )
        for reminder in reminders:
            reminder.status = ReminderStatus.skipped
            reminder.last_error = "skipped due to archive action"

        dependency_links = (
            session.execute(
                select(TaskDependency).where(
                    TaskDependency.user_id == task.user_id,
                    (TaskDependency.predecessor_task_id.in_(task_ids)) | (TaskDependency.successor_task_id.in_(task_ids)),
                )
            )
            .scalars()
            .all()
        )
        successor_ids = {
            link.successor_task_id
            for link in dependency_links
            if link.predecessor_task_id in task_ids and link.successor_task_id not in task_ids
        }
        for link in dependency_links:
            session.delete(link)

        for node in subtree:
            node.status = TaskStatus.archived
            node.blocked_reason = None
            node.blocked_at = None
            node.reminder_pause_until = None
            details = dict(node.blocker_details_json or {})
            details.pop("dependency_titles", None)
            details.pop("dependency_task_ids", None)
            node.blocker_details_json = details

        session.flush()
        for successor_id in successor_ids:
            successor = session.get(Task, successor_id)
            if successor is not None:
                session.expire(successor, ["successor_links"])
                unresolved = [
                    link.predecessor_task
                    for link in successor.successor_links
                    if link.predecessor_task.status != TaskStatus.completed
                ]
                if unresolved:
                    self._refresh_task_block_state(successor)
                else:
                    successor.status = TaskStatus.active
                    successor.blocked_reason = None
                    successor.blocked_at = None
                    details = dict(successor.blocker_details_json or {})
                    details.pop("dependency_titles", None)
                    details.pop("dependency_task_ids", None)
                    successor.blocker_details_json = details
        return len(subtree)

    @staticmethod
    def _task_subtree(task: Task) -> list[Task]:
        nodes: list[Task] = []
        stack = [task]
        seen = set()
        while stack:
            current = stack.pop()
            if current.id in seen:
                continue
            seen.add(current.id)
            nodes.append(current)
            stack.extend(current.subtasks)
        return nodes

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
    def _default_next_step(task_title: str, *, action_kind: str | None = None) -> str:
        return default_next_step(task_title, action_kind=action_kind)

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
    def _should_ask_load_prioritization_question(tasks: list[Task]) -> bool:
        if len(tasks) < 3:
            return False
        timed = sum(1 for task in tasks if task.deadline_at is not None or task.start_after is not None)
        quick_actions = sum(
            1
            for task in tasks
            if StateEngine._task_action_kind(task) in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}
        )
        return timed <= 1 or (len(tasks) >= 4 and timed <= 2) or (len(tasks) >= 4 and quick_actions >= 2)

    @staticmethod
    def _time_clarification_question(*, task_title: str, time_reference: str) -> str:
        cleaned_task = task_title.strip()
        cleaned_ref = time_reference.strip()
        if len(cleaned_task) > 80:
            cleaned_task = f"{cleaned_task[:77].rstrip()}..."
        return f"quick clarify: for '{cleaned_task}', what exact time should i use for '{cleaned_ref}'?"

    @staticmethod
    def _time_reference_needs_followup(time_reference: str) -> bool:
        lowered = time_reference.lower().strip()
        return any(token in lowered for token in ("after class", "before studio"))

    @staticmethod
    def _blocker_followup_question(task_title: str) -> str:
        cleaned_task = task_title.strip()
        if len(cleaned_task) > 70:
            cleaned_task = f"{cleaned_task[:67].rstrip()}..."
        return f"what has to happen first before '{cleaned_task}' can move?"

    @staticmethod
    def _urgency_from_deadline(deadline_at: datetime, timezone_name: str) -> str:
        now = datetime.now(tz=ZoneInfo(timezone_name))
        deadline = deadline_at if deadline_at.tzinfo else deadline_at.replace(tzinfo=ZoneInfo(timezone_name))
        deadline = deadline.astimezone(ZoneInfo(timezone_name))
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

    def _match_task_from_text(self, session: Session, user_id, text: str) -> Task | None:
        lowered = text.lower()
        tasks = list_active_tasks(session, user_id)
        best: Task | None = None
        best_score = 0
        for task in tasks:
            title = task.title.lower()
            score = 0
            if title in lowered:
                score += len(title) + 40
            tokens = [token for token in re.findall(r"[a-z0-9]+", title) if len(token) > 2]
            score += sum(len(token) for token in tokens if token in lowered)
            if score > best_score:
                best = task
                best_score = score
        if best is not None and best_score >= 6:
            return best

        pieces = [segment.strip() for segment in re.findall(r"[a-z0-9]+", lowered) if len(segment.strip()) > 2]
        for piece in pieces[:8]:
            found = find_active_task_by_title(session, user_id, piece)
            if found:
                return found
        return None

    def _resolve_update_target(
        self,
        session: Session,
        user_id,
        raw_text: str,
        blockers: list[str],
        *,
        time_reference: str | None = None,
        source_message_id=None,
    ) -> Task | None:
        matched = self._match_task_from_text(session, user_id, raw_text)
        if blockers and matched is not None:
            blocker_target = self._target_task_for_blocker_update(
                session,
                user_id=user_id,
                matched=matched,
                blockers=blockers,
            )
            if blocker_target is not None:
                matched = blocker_target
        if matched is not None:
            return matched
        if time_reference and self._looks_like_timing_followup(raw_text):
            recent = self._recent_relevant_task(
                session,
                user_id=user_id,
                source_message_id=source_message_id,
                prefer_windowed=True,
            )
            if recent is not None:
                return recent
        active = list_active_tasks(session, user_id)
        if blockers and len(active) == 1:
            return active[0]
        return None

    def _recent_relevant_task(self, session: Session, *, user_id, source_message_id=None, prefer_windowed: bool = False) -> Task | None:
        recent_messages = list_recent_messages(session, user_id, limit=12)
        inbound_messages = [
            msg for msg in reversed(recent_messages)
            if msg.direction == MessageDirection.inbound and msg.id != source_message_id
        ]
        if inbound_messages:
            inbound_ids = [msg.id for msg in inbound_messages[:6]]
            candidates = (
                session.execute(
                    select(Task)
                    .where(
                        Task.user_id == user_id,
                        Task.status.in_([TaskStatus.active, TaskStatus.blocked]),
                        Task.source_message_id.in_(tuple(inbound_ids)),
                    )
                    .order_by(Task.updated_at.desc(), Task.created_at.desc())
                )
                .scalars()
                .all()
            )
            if candidates:
                scored = sorted(
                    candidates,
                    key=lambda task: self._recent_task_score(task, inbound_ids, prefer_windowed=prefer_windowed),
                    reverse=True,
                )
                top = scored[0]
                if len(scored) == 1 or self._recent_task_score(top, inbound_ids, prefer_windowed=prefer_windowed) > self._recent_task_score(scored[1], inbound_ids, prefer_windowed=prefer_windowed) + 2:
                    return top

        active = list_active_tasks(session, user_id)
        if len(active) == 1:
            return active[0]
        return None

    def _recent_task_score(self, task: Task, inbound_ids: list, *, prefer_windowed: bool) -> int:
        score = 0
        try:
            score += max(0, 10 - inbound_ids.index(task.source_message_id)) * 10 if task.source_message_id in inbound_ids else 0
        except ValueError:
            pass
        if prefer_windowed and (task.start_after is not None or task.deadline_source_phrase):
            score += 15
        if self._task_action_kind(task) in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}:
            score += 12
        if task.last_progress_at is not None:
            score += 2
        return score

    @staticmethod
    def _looks_like_timing_followup(text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered or "?" in lowered:
            return False
        if re.search(r"\b(actually|nah|make that|move it to|switch it to|resched(?:ule)?|instead)\b", lowered):
            return True
        bare = re.sub(r"^(actually|nah|make that|move it to|switch it to|instead)\s+", "", lowered).strip()
        return bare in {
            "tomorrow morning",
            "tomorrow night",
            "tonight",
            "monday morning",
            "monday night",
            "tuesday morning",
            "wednesday morning",
            "thursday morning",
            "friday morning",
            "this weekend",
            "later",
        }

    @staticmethod
    def _reschedule_phrase(parsed_deadline, source_phrase: str, timezone_name: str) -> str:
        phrase = humanize_window_phrase(source_phrase)
        if phrase:
            return phrase
        if parsed_deadline.deadline_at is not None:
            return parsed_deadline.deadline_at.astimezone(ZoneInfo(timezone_name)).strftime("%a %-m/%-d %-I:%M%p").lower()
        if parsed_deadline.soft_deadline_at is not None:
            return parsed_deadline.soft_deadline_at.astimezone(ZoneInfo(timezone_name)).strftime("%a %-m/%-d %-I:%M%p").lower()
        return source_phrase.strip()

    @staticmethod
    def _reschedule_ack_text(time_phrase: str) -> str:
        phrase = humanize_window_phrase(time_phrase).strip().lower()
        if not phrase:
            return "okay bet, i moved it"
        if phrase.startswith(("today", "tomorrow", "tmr", "tonight", "this ", "next ")):
            return "okay bet, i moved it"
        weekday_prefixes = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        if phrase.startswith(weekday_prefixes):
            return "okay bet, i moved it"
        return f"okay bet, i moved it to {phrase}"

    def _refresh_task_block_state(self, task: Task) -> bool:
        unresolved = [link.predecessor_task for link in task.successor_links if link.predecessor_task.status != TaskStatus.completed]
        if unresolved:
            task.status = TaskStatus.blocked
            if task.blocked_at is None:
                task.blocked_at = datetime.now(tz=timezone.utc)
            task.blocked_reason = task.blocked_reason or f"waiting on {unresolved[0].title}"
            details = dict(task.blocker_details_json or {})
            details["dependency_titles"] = [dependency.title for dependency in unresolved]
            details["dependency_task_ids"] = [str(dependency.id) for dependency in unresolved]
            task.blocker_details_json = details
            return False

        if task.status == TaskStatus.blocked and (task.successor_links or self._is_dependency_block(task)):
            task.status = TaskStatus.active
            task.blocked_reason = None
            task.blocked_at = None
            details = dict(task.blocker_details_json or {})
            details.pop("dependency_titles", None)
            details.pop("dependency_task_ids", None)
            task.blocker_details_json = details
            return True
        return False

    def _refresh_successors(self, completed_task: Task) -> list[Task]:
        newly_unblocked: list[Task] = []
        for link in completed_task.predecessor_links:
            successor = link.successor_task
            if successor is None or successor.status not in {TaskStatus.active, TaskStatus.blocked}:
                continue
            if self._refresh_task_block_state(successor):
                newly_unblocked.append(successor)
        return newly_unblocked

    @staticmethod
    def _is_dependency_block(task: Task) -> bool:
        details = task.blocker_details_json or {}
        return bool(details.get("dependency_titles") or details.get("dependency_task_ids") or (task.blocked_reason or "").startswith("waiting on "))

    def _reflection_target(self, session: Session, user_id, raw_text: str, timezone_name: str) -> Task | None:
        matched = self._match_task_from_text(session, user_id, raw_text)
        if matched is not None:
            return matched
        return self.timeline.recommend_next_task(session, user_id, timezone_name)

    def _next_step_for_task(self, task: Task) -> str:
        unresolved = [link.predecessor_task for link in task.successor_links if link.predecessor_task.status != TaskStatus.completed]
        if unresolved:
            prerequisite = unresolved[0]
            return prerequisite.next_step or self._default_next_step(
                prerequisite.title,
                action_kind=self._task_action_kind(prerequisite),
            )
        if task.next_step:
            return task.next_step
        active_subtasks = [subtask for subtask in task.subtasks if subtask.status in {TaskStatus.active, TaskStatus.blocked}]
        if active_subtasks:
            subtask = active_subtasks[0]
            return subtask.next_step or self._default_next_step(
                subtask.title,
                action_kind=self._task_action_kind(subtask),
            )
        return self._default_next_step(task.title, action_kind=self._task_action_kind(task))

    @staticmethod
    def _task_action_kind(task: Task) -> str:
        return infer_action_kind(
            task.title,
            deadline_text=task.deadline_source_phrase,
            start_after=task.start_after,
            metadata=task.metadata_json or {},
        )

    def _preferred_focus_from_new_tasks(self, tasks: list[Task], timezone_name: str) -> Task | None:
        if not tasks:
            return None
        now = datetime.now(tz=ZoneInfo(timezone_name))
        ranked: list[tuple[int, Task]] = []
        for task in tasks:
            action_kind = self._task_action_kind(task)
            start_after = self._normalize_dt(task.start_after, timezone_name)
            deadline_at = self._normalize_dt(task.deadline_at, timezone_name)
            score = task.priority * 18
            if deadline_at is not None:
                delta = deadline_at - now
                if delta <= timedelta(hours=8):
                    score += 140
                elif delta <= timedelta(days=1):
                    score += 90
                else:
                    score += 35
            if start_after is not None and start_after > now:
                score -= 170
                if action_kind in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}:
                    score -= 120
            if task.status == TaskStatus.blocked:
                score -= 120
            if is_soft_later_phrase(task.deadline_source_phrase):
                score -= 90
            ranked.append((score, task))

        ranked.sort(
            key=lambda item: (
                -item[0],
                self._normalize_dt(item[1].deadline_at, timezone_name) or datetime.max.replace(tzinfo=ZoneInfo(timezone_name)),
                self._normalize_dt(item[1].created_at, timezone_name),
            )
        )
        focus = ranked[0][1]
        focus_kind = self._task_action_kind(focus)
        focus_start_after = self._normalize_dt(focus.start_after, timezone_name)
        if focus_start_after is not None and focus_start_after > now and focus_kind in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}:
            return None
        if is_soft_later_phrase(focus.deadline_source_phrase):
            return None
        return focus

    @staticmethod
    def _find_reusable_task(
        session: Session,
        *,
        user_id,
        title: str,
        parent_task: Task | None,
        title_index: dict[str, Task],
    ) -> Task | None:
        normalized = StateEngine._normalize_title(title)
        indexed = title_index.get(normalized)
        parent_task_id = parent_task.id if parent_task else None
        if indexed is not None and indexed.parent_task_id == parent_task_id:
            return indexed
        for task in list_active_tasks(session, user_id):
            if StateEngine._normalize_title(task.title) != normalized:
                continue
            if task.parent_task_id != parent_task_id:
                continue
            title_index.setdefault(normalized, task)
            return task
        return None

    def _merge_extracted_into_task(
        self,
        task: Task,
        *,
        extracted: ExtractedTask,
        project_id,
        next_step: str,
        deadline_at: datetime | None,
        soft_deadline_at: datetime | None,
        deadline_source: str | None,
        deadline_confidence: float,
        deadline_is_ambiguous: bool,
        deadline_granularity: str,
        deadline_timezone: str,
        action_kind: str,
        metadata: dict,
    ) -> None:
        if extracted.description and not task.description:
            task.description = extracted.description
        if project_id and task.project_id is None:
            task.project_id = project_id
        task.priority = max(task.priority, extracted.priority)
        task.extraction_confidence = max(task.extraction_confidence, extracted.confidence)
        if next_step and (task.next_step is None or task.next_step == self._default_next_step(task.title, action_kind=self._task_action_kind(task))):
            task.next_step = next_step
        if deadline_at is not None:
            task.deadline_at = deadline_at
        if soft_deadline_at is not None:
            task.soft_deadline_at = soft_deadline_at
        if extracted.start_after is not None:
            task.start_after = extracted.start_after
        if deadline_source:
            task.deadline_source_phrase = deadline_source
        task.deadline_confidence = max(task.deadline_confidence, deadline_confidence)
        task.deadline_is_ambiguous = task.deadline_is_ambiguous or deadline_is_ambiguous
        if deadline_granularity != "unknown":
            task.deadline_granularity = deadline_granularity
        task.deadline_timezone = deadline_timezone or task.deadline_timezone
        merged_metadata = dict(task.metadata_json or {})
        merged_metadata.update({k: v for k, v in metadata.items() if v not in (None, [], "")})
        merged_metadata["action_kind"] = action_kind
        task.metadata_json = merged_metadata

    def _target_task_for_blocker_update(
        self,
        session: Session,
        *,
        user_id,
        matched: Task,
        blockers: list[str],
    ) -> Task | None:
        blocker_text = " ".join(blockers).lower()
        if matched.title.lower() not in blocker_text:
            return matched
        for task in list_active_tasks(session, user_id):
            if task.id == matched.id:
                continue
            if task.title.lower() in blocker_text:
                continue
            return task
        return matched

    def _should_push_new_task_focus(
        self,
        task: Task,
        *,
        task_count: int,
        timezone_name: str,
        raw_text: str,
    ) -> bool:
        kind = self._task_action_kind(task)
        now = datetime.now(tz=ZoneInfo(timezone_name))
        start_after = self._normalize_dt(task.start_after, timezone_name)
        deadline_at = self._normalize_dt(task.deadline_at, timezone_name)
        lowered = raw_text.lower()

        if task.status == TaskStatus.blocked or is_soft_later_phrase(task.deadline_source_phrase):
            return False
        if start_after is not None and start_after > now:
            return False

        if kind in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}:
            if task_count > 1:
                return False
            if any(token in lowered for token in ("right now", "rn", "asap", "before i forget", "real quick")):
                return True
            if deadline_at is not None and deadline_at - now <= timedelta(minutes=90):
                return True
            return False

        return True

    def _new_task_fact(self, task: Task, timezone_name: str) -> str:
        action_kind = self._task_action_kind(task)
        lowered_title = task.title[0].lower() + task.title[1:] if task.title else "it"
        time_phrase = humanize_window_phrase(task.deadline_source_phrase)
        if self._looks_like_placeholder_assignment([task]):
            return f"looks like a {lowered_title} just landed"
        if action_kind == ACTION_KIND_QUICK_MESSAGE and time_phrase:
            return self._quick_reminder_fact(time_phrase, is_admin=False)
        if action_kind == ACTION_KIND_QUICK_ADMIN and time_phrase:
            return self._quick_reminder_fact(time_phrase, is_admin=True)
        if action_kind in {ACTION_KIND_QUICK_MESSAGE, ACTION_KIND_QUICK_ADMIN}:
            return "okay bet, i got you"
        return f"got {task.title}"

    @staticmethod
    def _quick_reminder_fact(time_phrase: str, *, is_admin: bool) -> str:
        phrase = humanize_window_phrase(time_phrase).strip().lower()
        if not phrase:
            return "okay bet, i got you"
        if is_admin:
            return "okay bet, i'll keep track of it"
        return "okay bet, i'll remind you"

    @staticmethod
    def _timeline_custom_window(raw_text: str, timezone_name: str) -> tuple[str, datetime, datetime] | None:
        lowered = raw_text.lower()
        if "what" not in lowered and "due" not in lowered and "have" not in lowered:
            return None
        parsed = interpret_time_reference(raw_text, timezone=timezone_name)
        if parsed.granularity not in {"part_of_day", "day"}:
            return None
        zone = ZoneInfo(timezone_name)
        if parsed.soft_deadline_at is None and parsed.deadline_at is None:
            return None
        label = parsed.source_phrase.lower() if parsed.source_phrase else "that window"
        start = parsed.soft_deadline_at or parsed.deadline_at
        end = parsed.deadline_at or parsed.soft_deadline_at
        if start is None or end is None:
            return None
        start = start.astimezone(zone)
        end = end.astimezone(zone)
        if parsed.granularity == "day":
            start = start.replace(hour=6, minute=0, second=0, microsecond=0)
            end = end.replace(hour=23, minute=59, second=59, microsecond=0)
        return label, start, end

    @staticmethod
    def _quick_task_subject(title: str) -> str:
        lowered = title.strip().lower()
        replacements = (
            ("pay ", ""),
            ("book ", ""),
            ("schedule ", ""),
            ("renew ", ""),
            ("cancel ", ""),
            ("buy ", ""),
        )
        for prefix, replacement in replacements:
            if lowered.startswith(prefix):
                return replacement + lowered[len(prefix) :]
        if lowered.startswith("text "):
            return f"texting {lowered[5:]}"
        if lowered.startswith("call "):
            return f"calling {lowered[5:]}"
        if lowered.startswith("reply "):
            return f"replying to {lowered[6:]}"
        if lowered.startswith("dm "):
            return f"dming {lowered[3:]}"
        if lowered.startswith("ping "):
            return f"pinging {lowered[5:]}"
        return lowered

    @staticmethod
    def _record_context_block(session: Session, *, user, raw_text: str, confidence: float) -> ScheduleBlock:
        starts, ends = time_window_for_context(raw_text, user.timezone)
        block = ScheduleBlock(
            user_id=user.id,
            block_type=StateEngine._normalize_context_type(raw_text),
            starts_at=starts,
            ends_at=ends,
            confidence=confidence,
            notes=raw_text,
        )
        session.add(block)
        return block

    @staticmethod
    def _looks_like_placeholder_assignment(tasks: list[Task]) -> bool:
        if len(tasks) != 1:
            return False
        lowered = tasks[0].title.lower()
        return lowered.startswith("new assignment")

    @staticmethod
    def _context_ack_text(block_type: str) -> str:
        normalized = block_type.replace("_", " ")
        if block_type == "in_class":
            return "okay, class first"
        if block_type == "driving":
            return "okay, drive first"
        if block_type == "social_event":
            return "okay, go do your thing"
        if block_type == "all_nighter":
            return "okay, we're on all-nighter timing"
        if block_type == "sleeping":
            return "okay, you're done for tonight"
        return f"okay, you're tied up with {normalized} rn"

    @staticmethod
    def _task_context(task: Task) -> ReplyTaskContext:
        return ReplyTaskContext(
            title=task.title,
            status=task.status.value,
            deadline_at=task.deadline_at,
            deadline_text=task.deadline_source_phrase,
            next_step=task.next_step,
            blocker=task.blocked_reason,
            reminder_escalation_level=task.reminder_escalation_level,
            slip_count=task.slip_count,
            is_subtask=task.parent_task_id is not None,
        )

    def _lookup_task_by_title(self, session: Session, user_id, title: str, *, title_index: dict[str, Task]) -> Task | None:
        normalized = self._normalize_title(title)
        if normalized in title_index:
            return title_index[normalized]
        found = self._find_task_any_status(session, user_id, title)
        if found is not None:
            title_index.setdefault(normalized, found)
        return found

    @staticmethod
    def _find_task_any_status(session: Session, user_id, title_fragment: str) -> Task | None:
        fragment = f"%{title_fragment.strip().lower()}%"
        stmt = (
            select(Task)
            .where(Task.user_id == user_id, Task.title.ilike(fragment))
            .order_by(Task.updated_at.desc())
        )
        return session.execute(stmt).scalars().first()

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title.strip().lower())

    @staticmethod
    def _canonicalize_dependency_title(title: str) -> str | None:
        cleaned = re.sub(r"\s+", " ", title.strip())
        return cleaned[:1].upper() + cleaned[1:] if cleaned else None

    @staticmethod
    def _normalize_blocker_to_task_title(blocker_text: str) -> str | None:
        cleaned = blocker_text.strip().lower()
        cleaned = re.sub(r"^(i\s+)?(need|have|gotta|must)\s+to\s+", "", cleaned)
        cleaned = re.sub(r"^(i\s+)?need\s+", "", cleaned)
        cleaned = re.sub(r"^(before|until)\s+", "", cleaned)
        cleaned = re.sub(r"\b(first|before this|before that|before i can|before it can)\b", "", cleaned)
        cleaned = cleaned.replace(" rn", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")
        if not cleaned:
            return None

        actionable_verbs = (
            "fix",
            "finish",
            "send",
            "submit",
            "export",
            "make",
            "do",
            "call",
            "email",
            "text",
            "review",
            "update",
            "build",
            "draft",
            "write",
            "upload",
            "get",
            "find",
            "ask",
            "check",
            "schedule",
            "clean",
            "polish",
            "prep",
            "prepare",
        )
        if cleaned.startswith("waiting on ") or cleaned.startswith("wait for "):
            return None
        if cleaned.split()[0] not in actionable_verbs:
            if len(cleaned.split()) <= 5:
                cleaned = f"get {cleaned}"
            else:
                return None
        return cleaned[:1].upper() + cleaned[1:]

    @staticmethod
    def _next_pending_reminder_for_task(session: Session, task_id) -> Reminder | None:
        stmt = (
            select(Reminder)
            .where(Reminder.task_id == task_id, Reminder.status == ReminderStatus.pending)
            .order_by(Reminder.scheduled_for.asc())
        )
        return session.execute(stmt).scalars().first()

    @staticmethod
    def _normalize_dt(value: datetime | None, timezone_name: str) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=ZoneInfo(timezone_name))
        return value.astimezone(ZoneInfo(timezone_name))

    @staticmethod
    def _format_due(value: datetime | None, timezone_name: str) -> str | None:
        if value is None:
            return None
        due = value if value.tzinfo else value.replace(tzinfo=ZoneInfo(timezone_name))
        return due.astimezone(ZoneInfo(timezone_name)).strftime("%a %-m/%-d %-I:%M%p").lower()

    @staticmethod
    def _update_profile_memory(session: Session, user_id, raw_text: str) -> None:
        lowered = raw_text.lower()
        if any(token in lowered for token in ["underestimated", "distracted", "switching", "conflict", "overwhelmed", "avoid"]):
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

from __future__ import annotations

import re
from datetime import datetime

from app.core.time_utils import interpret_time_reference, parse_human_time
from app.core.config import get_settings
from app.llm.client import OllamaAdapter
from app.schemas.intent import ExtractedTask, ImageExtractionResult, IntentName, IntentResult, ParsedDeadline


class IntentExtractor:
    _TASK_START_PATTERN = (
        r"(?:i\s+)?(?:need to|have to|gotta|must|should|want to|submit|finish|send|write|prepare|study|"
        r"review|fix|build|make|do|call|email|text|update|work on|start|complete|wrap|clean|buy|plan|"
        r"schedule|reply|apply|pay|draft|upload|design|model)"
    )

    def __init__(self, adapter: OllamaAdapter | None = None) -> None:
        self.adapter = adapter or OllamaAdapter()
        self.settings = get_settings()
        provider = self.settings.llm_provider.lower().strip()
        if provider == "openai":
            self._intent_model = self.settings.openai_intent_model.strip() or None
        else:
            self._intent_model = self.settings.ollama_intent_model.strip() or None

    def extract(self, text: str, timezone: str) -> IntentResult:
        fallback = self._extract_fallback(text, timezone)
        if self._should_short_circuit_to_fallback(text=text, fallback=fallback):
            return fallback

        llm_result = self._extract_with_llm(text, timezone)
        if llm_result:
            return self._merge_llm_and_fallback(llm_result=llm_result, fallback=fallback, text=text, timezone=timezone)
        return fallback

    def _extract_with_llm(self, text: str, timezone: str) -> IntentResult | None:
        payload = self.adapter.json_completion(
            system=(
                "Classify intent and extract task/deadline fields from a single SMS message. "
                "If the message contains multiple tasks, return them in tasks as separate task objects in message order. "
                "Return JSON keys: intent, confidence, needs_clarification, clarification_question, "
                "time_reference, time_confidence, context_signal, blockers, summary, task, tasks. "
                "Each task.title must be concise and should not include time phrases like tonight/tomorrow/by eod. "
                "Only ask a clarification when timing ambiguity would materially change planning or reminders."
            ),
            user=(
                f"timezone={timezone}\n"
                f"message={text}\n"
                "task objects should include: title, description, project, deadline_text, priority, confidence, next_step."
            ),
            options={
                "temperature": 0.1,
                "num_predict": self.settings.ollama_intent_num_predict,
                "num_ctx": self.settings.ollama_intent_num_ctx,
            },
            model=self._intent_model,
            request_timeout_seconds=8,
        )
        if not payload:
            return None
        try:
            raw_tasks = payload.get("tasks") or ([payload["task"]] if payload.get("task") else [])
            tasks = [self._normalize_task(ExtractedTask.model_validate(raw_task), timezone) for raw_task in raw_tasks]
            task = tasks[0] if tasks else None
            result = IntentResult(
                intent=payload.get("intent", "general_chat"),
                confidence=payload.get("confidence", 0.5),
                needs_clarification=payload.get("needs_clarification", False),
                clarification_question=payload.get("clarification_question"),
                time_reference=payload.get("time_reference"),
                time_confidence=payload.get("time_confidence", 0.0),
                context_signal=payload.get("context_signal"),
                blockers=payload.get("blockers", []),
                summary=payload.get("summary"),
                task=task,
                tasks=tasks,
                task_updates=payload.get("task_updates", {}),
            )
            self._sync_result_timing(result=result, timezone=timezone)
            return result
        except Exception:
            return None

    def _extract_fallback(self, text: str, timezone: str) -> IntentResult:
        lowered = text.lower().strip()
        bulk_action = self._detect_bulk_action(lowered)
        if bulk_action:
            return IntentResult(
                intent="update_task",
                confidence=0.95,
                summary="bulk task list action requested",
                task_updates={"bulk_action": bulk_action},
            )
        timeline_query_cues = [
            "what do i have",
            "what's due",
            "what do i need to get done",
            "deadlines",
            "today",
            "this week",
            "tonight",
            "tomorrow morning",
            "this weekend",
            "weekend",
            "next hour",
        ]
        looks_timeline_query = any(token in lowered for token in timeline_query_cues)
        candidate_tasks = self._extract_tasks_from_text(text, timezone)
        looks_add_task = bool(candidate_tasks) or any(token in lowered for token in ["need to", "have to", "gotta", "assignment"])
        if looks_timeline_query and ("?" in lowered or lowered.startswith("what do i") or lowered.startswith("what's due")):
            looks_add_task = False
        if " due " in f" {lowered} " and not looks_timeline_query:
            looks_add_task = True

        intent: IntentName = "general_chat"
        confidence = 0.55
        context_signal = None
        task: ExtractedTask | None = None

        if any(token in lowered for token in ["in class", "driving", "at dinner", "all nighter", "in a meeting"]):
            intent = "context_signal"
            context_signal = lowered
            confidence = 0.82
        elif self._looks_like_dependency_blocker(lowered):
            intent = "update_task"
            confidence = 0.76
            blocker = self._extract_blocker_phrase(lowered)
            return IntentResult(
                intent=intent,
                confidence=confidence,
                blockers=[blocker] if blocker else [],
                task_updates={"status": "blocked"},
                summary="task appears blocked by prerequisite dependency",
            )
        elif self._is_meta_or_capability_query(lowered):
            intent = "status_query"
            confidence = 0.9
            return IntentResult(intent=intent, confidence=confidence, summary="user asked assistant capabilities")
        elif looks_add_task:
            intent = "add_task"
            tasks = candidate_tasks or self._extract_tasks_from_text(lowered, timezone)
            if not tasks:
                extracted_title = self._simple_task_title(lowered)
                deadline_text = self._extract_deadline_phrase(lowered)
                task = self._build_task_from_segment(
                    lowered,
                    timezone=timezone,
                    fallback_title=extracted_title,
                    deadline_text_override=deadline_text,
                )
                tasks = [task] if task else []
            if not tasks:
                return IntentResult(intent="general_chat", confidence=0.55)

            confidence = 0.84 if len(tasks) > 1 else 0.78
            primary_task = tasks[0]
            primary_deadline = primary_task.deadline or ParsedDeadline(
                source_phrase=primary_task.deadline_text,
                confidence=0.0,
            )
            needs_clarification = self._task_requires_time_clarification(primary_task)
            clarification_question = None
            if needs_clarification and primary_deadline.source_phrase:
                clarification_question = self._clarification_for_task_time(primary_task.title, primary_deadline.source_phrase)
            return IntentResult(
                intent=intent,
                confidence=confidence,
                time_reference=primary_deadline.source_phrase,
                time_confidence=primary_deadline.confidence,
                needs_clarification=needs_clarification,
                clarification_question=clarification_question,
                task=primary_task,
                tasks=tasks,
                summary=f"captured {len(tasks)} task{'s' if len(tasks) != 1 else ''}",
            )
        elif looks_timeline_query:
            intent = "timeline_query"
            confidence = 0.8
        elif any(token in lowered for token in ["finished", "done", "completed", "wrapped"]):
            intent = "complete_task"
            confidence = 0.75
        elif any(token in lowered for token in ["stuck", "distracted", "underestimated", "behind"]):
            intent = "reflection"
            confidence = 0.7

        return IntentResult(intent=intent, confidence=confidence, context_signal=context_signal, task=task)

    def _merge_llm_and_fallback(
        self,
        *,
        llm_result: IntentResult,
        fallback: IntentResult,
        text: str,
        timezone: str,
    ) -> IntentResult:
        merged = llm_result.model_copy(deep=True)

        if self._prefer_fallback(fallback=fallback, llm_result=merged):
            return fallback

        if merged.intent == "add_task":
            if not merged.tasks and fallback.tasks:
                merged.tasks = fallback.tasks
            elif len(fallback.tasks) > len(merged.tasks) and fallback.confidence >= 0.78:
                merged.tasks = fallback.tasks
            if not merged.task and merged.tasks:
                merged.task = merged.tasks[0]
            elif merged.task and not merged.tasks:
                merged.tasks = [merged.task]
            for task in merged.tasks:
                task.title = self._sanitize_task_title(task.title)
            if merged.tasks:
                merged.task = merged.tasks[0]

        if not merged.time_reference and fallback.time_reference:
            merged.time_reference = fallback.time_reference
            merged.time_confidence = max(merged.time_confidence, fallback.time_confidence)

        self._sync_result_timing(result=merged, timezone=timezone)

        if merged.task and self._task_requires_time_clarification(merged.task) and not merged.needs_clarification:
            merged.needs_clarification = True
            merged.clarification_question = (
                merged.clarification_question
                or self._clarification_for_task_time(
                    merged.task.title,
                    merged.task.deadline.source_phrase if merged.task.deadline else merged.time_reference or "that time",
                )
            )

        if merged.intent == "general_chat" and fallback.intent != "general_chat" and fallback.confidence >= 0.78:
            return fallback

        return merged

    @staticmethod
    def _prefer_fallback(*, fallback: IntentResult, llm_result: IntentResult) -> bool:
        if llm_result.confidence < 0.45 and fallback.confidence >= 0.75:
            return True
        if llm_result.intent == "general_chat" and fallback.intent in {"add_task", "timeline_query", "context_signal"} and fallback.confidence >= 0.78:
            return True
        if (
            fallback.intent == "timeline_query"
            and fallback.confidence >= 0.8
            and llm_result.intent == "add_task"
            and llm_result.task is not None
            and llm_result.task.deadline_at is None
        ):
            title = (llm_result.task.title or "").lower()
            if any(token in title for token in ("what do i", "get done", "what's due", "due this week", "tonight", "tomorrow")):
                return True
        if fallback.intent == "add_task" and len(fallback.tasks) > len(llm_result.tasks) and fallback.confidence >= 0.78:
            return True
        if llm_result.intent == "add_task" and not llm_result.task and fallback.task is not None:
            return True
        return False

    @staticmethod
    def _clarification_for_time(time_reference: str) -> str:
        clean = time_reference.strip()
        return f"quick one: when exactly do you want '{clean}' to mean?"

    @staticmethod
    def _clarification_for_task_time(task_title: str, time_reference: str) -> str:
        cleaned_task = task_title.strip()
        cleaned_ref = time_reference.strip()
        return f"quick clarify: for '{cleaned_task}', what exact time should i use for '{cleaned_ref}'?"

    @staticmethod
    def _looks_like_dependency_blocker(text: str) -> bool:
        if "first" not in text:
            return False
        blocker_signals = (
            "because",
            "blocked",
            "stuck",
            "distracted",
            "can't",
            "cannot",
            "need to",
            "have to",
        )
        return any(signal in text for signal in blocker_signals)

    @staticmethod
    def _extract_blocker_phrase(text: str) -> str:
        match = re.search(r"(need to|have to)\s+(.+?)\s+first", text)
        if match:
            phrase = match.group(0).strip()
            return phrase
        return "hidden prerequisite is blocking progress"

    @staticmethod
    def _simple_task_title(text: str) -> str:
        cleaned = re.sub(r"^(yo|hey|ok|okay)\s+", "", text).strip()
        cleaned = cleaned.replace("need to ", "").replace("have to ", "")
        cleaned = re.sub(
            r"\b(and then|then|tmr morning|tomorrow morning|tomorrow night|tonight|this weekend|by eod|eod|later|after class|before studio)\b",
            "",
            cleaned,
        )
        return IntentExtractor._sanitize_task_title(cleaned)

    @staticmethod
    def _sanitize_task_title(title: str) -> str:
        cleaned = (title or "").lower().strip()
        cleaned = cleaned.replace("\n", " ")
        cleaned = re.sub(r"^[\"'`]+|[\"'`]+$", "", cleaned)
        cleaned = re.sub(r"^(and then|and|then)\s+", "", cleaned)
        cleaned = re.sub(r"^(i\s+(need to|have to|gotta|want to|should|must)\s+)", "", cleaned)
        cleaned = re.sub(r"^(need to|have to|gotta|want to|should|must)\s+", "", cleaned)
        cleaned = re.sub(r"^i\s+", "", cleaned)
        cleaned = re.sub(
            r"\b(and then|tmr morning|tomorrow morning|tomorrow night|tonight|this weekend|by eod|eod|later|after class|before studio)\b",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        if len(cleaned) > 90:
            cleaned = cleaned[:90].rsplit(" ", 1)[0]
        if not cleaned:
            return "Task update"
        return cleaned[0].upper() + cleaned[1:]

    @staticmethod
    def _extract_deadline_phrase(text: str) -> str | None:
        patterns = [
            r"\bby eod\b",
            r"\beod\b",
            r"\btomorrow(?: morning| night)?\b",
            r"\btonight\b",
            r"\bthis weekend\b",
            r"\blater\b",
            r"\bafter class\b",
            r"\bbefore studio\b",
            r"\bby [^,.;!?]+",
            r"\bdue [^,.;!?]+",
            r"\bbefore [^,.;!?]+",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_tasks_from_text(self, text: str, timezone: str) -> list[ExtractedTask]:
        tasks: list[ExtractedTask] = []
        seen_titles: set[str] = set()
        for segment in self._split_task_segments(text):
            task = self._build_task_from_segment(segment, timezone=timezone)
            if not task:
                continue
            dedupe_key = f"{task.title.lower()}::{(task.deadline_text or '').lower()}"
            if dedupe_key in seen_titles:
                continue
            seen_titles.add(dedupe_key)
            tasks.append(task)
        return tasks

    def _build_task_from_segment(
        self,
        segment: str,
        *,
        timezone: str,
        fallback_title: str | None = None,
        deadline_text_override: str | None = None,
    ) -> ExtractedTask | None:
        lowered = segment.lower().strip(" .,!?")
        if not lowered:
            return None
        if not fallback_title and not self._looks_like_task_segment(lowered):
            return None

        deadline_text = deadline_text_override or self._extract_deadline_phrase(segment)
        title_source = segment
        if deadline_text:
            title_source = re.sub(re.escape(deadline_text), "", title_source, count=1, flags=re.IGNORECASE)
        title = fallback_title or self._sanitize_task_title(title_source)
        if title == "Task update" and not deadline_text:
            return None

        parsed_deadline = self._parse_deadline(deadline_text, timezone) if deadline_text else None
        task_confidence = 0.74
        if re.search(rf"^(?:yo|hey|ok|okay|alright|and\s+then|then\s+)?{self._TASK_START_PATTERN}\b", lowered):
            task_confidence = 0.82
        if parsed_deadline:
            task_confidence = max(task_confidence, min(0.9, parsed_deadline.confidence + 0.1))

        return ExtractedTask(
            title=title,
            deadline_text=deadline_text,
            deadline_at=parsed_deadline.deadline_at if parsed_deadline else None,
            soft_deadline_at=parsed_deadline.soft_deadline_at if parsed_deadline else None,
            confidence=task_confidence,
            deadline=parsed_deadline,
        )

    def _normalize_task(self, task: ExtractedTask, timezone: str) -> ExtractedTask:
        task.title = self._sanitize_task_title(task.title)
        if not task.deadline_text and task.deadline and task.deadline.source_phrase:
            task.deadline_text = task.deadline.source_phrase
        if task.deadline_text:
            parsed_deadline = self._parse_deadline(task.deadline_text, timezone)
            task.deadline = self._merge_deadlines(task.deadline, parsed_deadline, timezone)
            if task.deadline_at is None:
                task.deadline_at = task.deadline.deadline_at
            if task.soft_deadline_at is None:
                task.soft_deadline_at = task.deadline.soft_deadline_at
        elif task.deadline and not task.deadline.timezone:
            task.deadline.timezone = timezone
        return task

    def _sync_result_timing(self, *, result: IntentResult, timezone: str) -> None:
        if result.task and not result.tasks:
            result.tasks = [result.task]
        if result.tasks and not result.task:
            result.task = result.tasks[0]
        if result.intent != "add_task":
            return

        for index, task in enumerate(result.tasks):
            if not task.deadline_text and result.time_reference and index == 0:
                task.deadline_text = result.time_reference
            result.tasks[index] = self._normalize_task(task, timezone)
        if result.tasks:
            result.task = result.tasks[0]

        primary_task = result.task
        if primary_task and primary_task.deadline:
            result.time_reference = result.time_reference or primary_task.deadline.source_phrase
            result.time_confidence = max(result.time_confidence, primary_task.deadline.confidence)
            if not result.needs_clarification and self._task_requires_time_clarification(primary_task):
                result.needs_clarification = True
                if primary_task.deadline.source_phrase:
                    result.clarification_question = result.clarification_question or self._clarification_for_task_time(
                        primary_task.title,
                        primary_task.deadline.source_phrase,
                    )

    @staticmethod
    def _merge_deadlines(existing: ParsedDeadline | None, parsed: ParsedDeadline, timezone: str) -> ParsedDeadline:
        if existing is None:
            if not parsed.timezone:
                parsed.timezone = timezone
            return parsed
        existing.source_phrase = existing.source_phrase or parsed.source_phrase
        existing.deadline_at = existing.deadline_at or parsed.deadline_at
        existing.soft_deadline_at = existing.soft_deadline_at or parsed.soft_deadline_at
        existing.timezone = existing.timezone or timezone
        existing.confidence = max(existing.confidence, parsed.confidence)
        existing.is_ambiguous = existing.is_ambiguous or parsed.is_ambiguous
        existing.ambiguity_reason = existing.ambiguity_reason or parsed.ambiguity_reason
        if existing.granularity == "unknown":
            existing.granularity = parsed.granularity
        return existing

    @staticmethod
    def _parse_deadline(deadline_text: str, timezone: str) -> ParsedDeadline:
        return interpret_time_reference(deadline_text, timezone=timezone)

    def _split_task_segments(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if not normalized:
            return []

        segments = [normalized]
        split_patterns = [
            r"\s*;\s*",
            r"\s+(?:and then|then|also|plus)\s+",
            rf",\s*(?=(?:then\s+)?{self._TASK_START_PATTERN}\b)",
            rf"\s+and\s+(?=(?:then\s+)?{self._TASK_START_PATTERN}\b)",
        ]
        for pattern in split_patterns:
            next_segments: list[str] = []
            for segment in segments:
                next_segments.extend(part for part in re.split(pattern, segment, flags=re.IGNORECASE) if part.strip())
            segments = next_segments
        return segments

    def _looks_like_task_segment(self, text: str) -> bool:
        if "?" in text:
            return False
        if re.search(rf"^(?:yo|hey|ok|okay|alright|and\s+then|then\s+)?{self._TASK_START_PATTERN}\b", text):
            return True
        if "assignment" in text or "project" in text:
            return True
        deadline_text = self._extract_deadline_phrase(text)
        return deadline_text is not None and bool(
            re.search(r"\b(submit|finish|send|prepare|fix|study|write|review|apply|upload|pay|draft|reply|complete)\b", text),
        )

    @staticmethod
    def _task_requires_time_clarification(task: ExtractedTask) -> bool:
        deadline = task.deadline
        if not deadline or not deadline.source_phrase:
            return False
        if not deadline.is_ambiguous:
            return False
        phrase = deadline.source_phrase.lower()
        if "this weekend" in phrase:
            return False
        if deadline.deadline_at is None:
            return True
        return any(token in phrase for token in ("later", "after class", "before studio"))

    @staticmethod
    def _is_meta_or_capability_query(text: str) -> bool:
        capability_cues = {"what", "how", "help", "do", "can", "are", "is"}
        identity_cues = {
            "you",
            "bot",
            "ai",
            "automated",
            "work",
            "working",
            "live",
            "online",
            "on",
            "responses",
            "generated",
            "capabilities",
        }
        words = set(re.findall(r"[a-z0-9']+", text))
        if not words:
            return False
        if "?" in text and words.intersection(capability_cues) and words.intersection(identity_cues):
            return True
        if re.search(r"\bare you\b.*\b(live|working|online|on)\b", text):
            return True
        return "what do you do" in text or "what can you do" in text

    def _should_short_circuit_to_fallback(self, *, text: str, fallback: IntentResult) -> bool:
        if not bool(getattr(self.adapter, "enabled", True)):
            return True

        lowered = text.lower().strip()
        if fallback.intent in {"context_signal", "timeline_query", "status_query"} and fallback.confidence >= 0.82:
            return True
        if (
            fallback.intent == "add_task"
            and fallback.confidence >= 0.78
            and fallback.task is not None
            and not fallback.needs_clarification
            and self._is_simple_single_task_message(lowered)
        ):
            return True
        if fallback.intent == "general_chat" and fallback.confidence >= 0.55 and self._is_simple_checkin(lowered):
            return True
        return False

    @staticmethod
    def _is_simple_checkin(text: str) -> bool:
        words = re.findall(r"[a-z0-9']+", text)
        if len(words) > 5:
            return False
        if any(token.isdigit() for token in words):
            return False
        checkin_words = {
            "yo",
            "hey",
            "hi",
            "sup",
            "whatup",
            "whatsup",
            "hello",
            "alive",
            "working",
            "online",
            "there",
            "bro",
            "bruh",
            "cookin",
            "cooking",
        }
        return bool(words) and all(word in checkin_words for word in words)

    @staticmethod
    def _is_simple_single_task_message(text: str) -> bool:
        words = re.findall(r"[a-z0-9']+", text)
        if not words:
            return False
        if len(words) > 24:
            return False
        if "?" in text:
            return False
        # Avoid short-circuiting multi-task chains like "and then ..."
        if any(token in text for token in (" and then ", ";", " also ", " plus ")):
            return False
        if re.search(rf"\sand\s+(?=(?:then\s+)?{IntentExtractor._TASK_START_PATTERN}\b)", text):
            return False
        return any(token in text for token in ("need to", "have to", "gotta", "must", "assignment", "submit", "finish", "send"))

    @staticmethod
    def _detect_bulk_action(text: str) -> str | None:
        patterns = (
            r"\b(clear|reset|wipe)\b.*\b(all\s+)?(tasks|todo|to-do)\b",
            r"\b(clean\s+slate)\b",
        )
        for pattern in patterns:
            if re.search(pattern, text):
                return "clear_active_tasks"
        return None


class ImageAssignmentExtractor:
    def __init__(self, adapter: OllamaAdapter | None = None) -> None:
        self.adapter = adapter or OllamaAdapter()

    def extract(self, image_url: str, timezone: str) -> ImageExtractionResult:
        payload = self.adapter.vision_json(
            system=(
                "Extract assignment details from screenshot image. "
                "Return JSON with keys: title, due_text, context, deliverables, raw_text, confidence."
            ),
            user_prompt=f"timezone={timezone}. infer due dates if present. be conservative on confidence.",
            image_url=image_url,
            request_timeout_seconds=25,
        )
        if not payload:
            return ImageExtractionResult(confidence=0.0)
        result = ImageExtractionResult.model_validate(payload)
        if result.due_text and not result.due_at:
            parsed, _ = parse_human_time(result.due_text, timezone=timezone)
            result.due_at = parsed
        return result

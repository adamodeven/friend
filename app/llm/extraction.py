from __future__ import annotations

import re
from datetime import datetime

from app.core.time_utils import parse_human_time
from app.core.config import get_settings
from app.llm.client import OllamaAdapter
from app.schemas.intent import ExtractedTask, IntentResult, IntentName, ImageExtractionResult


class IntentExtractor:
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
                "Return JSON keys: intent, confidence, needs_clarification, clarification_question, "
                "time_reference, time_confidence, context_signal, blockers, summary, task. "
                "task.title must be concise and should not include time phrases like tonight/tomorrow/by eod."
            ),
            user=(
                f"timezone={timezone}\n"
                f"message={text}\n"
                "task should be object with: title, description, project, deadline_text, priority, confidence, next_step."
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
            if payload.get("task"):
                task = ExtractedTask.model_validate(payload["task"])
                task.title = self._sanitize_task_title(task.title)
            else:
                task = None
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
                task_updates=payload.get("task_updates", {}),
            )
            if result.task and result.task.deadline_text and not result.task.deadline_at:
                parsed, conf = parse_human_time(result.task.deadline_text, timezone=timezone)
                result.task.deadline_at = parsed
                result.time_confidence = max(result.time_confidence, conf)
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
        looks_add_task = any(token in lowered for token in ["need to", "have to", "gotta", "assignment"])
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
            confidence = 0.78
            extracted_title = self._simple_task_title(lowered)
            deadline_text = self._extract_deadline_phrase(lowered)
            deadline_at = None
            time_conf = 0.0
            if deadline_text:
                deadline_at, time_conf = parse_human_time(deadline_text, timezone=timezone)
            task = ExtractedTask(
                title=extracted_title,
                deadline_text=deadline_text,
                deadline_at=deadline_at,
                confidence=max(0.55, confidence),
            )
            return IntentResult(
                intent=intent,
                confidence=confidence,
                time_reference=deadline_text,
                time_confidence=time_conf,
                needs_clarification=bool(deadline_text and (deadline_at is None or time_conf < 0.6)),
                task=task,
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
            if not merged.task and fallback.task:
                merged.task = fallback.task
            if merged.task:
                merged.task.title = self._sanitize_task_title(merged.task.title)

        if not merged.time_reference and fallback.time_reference:
            merged.time_reference = fallback.time_reference
            merged.time_confidence = max(merged.time_confidence, fallback.time_confidence)

        if merged.task and merged.task.deadline_text and not merged.task.deadline_at:
            parsed, conf = parse_human_time(merged.task.deadline_text, timezone=timezone)
            merged.task.deadline_at = parsed
            merged.time_confidence = max(merged.time_confidence, conf)

        if merged.time_reference and (not merged.task or not merged.task.deadline_at):
            parsed, conf = parse_human_time(merged.time_reference, timezone=timezone)
            if merged.task and parsed and not merged.task.deadline_at:
                merged.task.deadline_at = parsed
            merged.time_confidence = max(merged.time_confidence, conf)

        if merged.time_reference and merged.time_confidence < 0.6 and not merged.needs_clarification:
            merged.needs_clarification = True
            merged.clarification_question = merged.clarification_question or self._clarification_for_time(merged.time_reference)

        if merged.intent == "general_chat" and fallback.intent != "general_chat" and fallback.confidence >= 0.78:
            return fallback

        return merged

    @staticmethod
    def _prefer_fallback(*, fallback: IntentResult, llm_result: IntentResult) -> bool:
        if llm_result.confidence < 0.45 and fallback.confidence >= 0.75:
            return True
        if llm_result.intent == "general_chat" and fallback.intent in {"add_task", "timeline_query", "context_signal"} and fallback.confidence >= 0.78:
            return True
        if llm_result.intent == "add_task" and not llm_result.task and fallback.task is not None:
            return True
        return False

    @staticmethod
    def _clarification_for_time(time_reference: str) -> str:
        clean = time_reference.strip()
        return f"quick one: when exactly do you want '{clean}' to mean?"

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
            r"\b(and then|then|tmr morning|tomorrow morning|tomorrow night|tonight|this weekend|by eod|eod)\b",
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
            r"\b(and then|tmr morning|tomorrow morning|tomorrow night|tonight|this weekend|by eod|eod)\b",
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
            r"\bby [^,.!?]+",
            r"\bdue [^,.!?]+",
            r"\btomorrow(?: morning| night)?\b",
            r"\btonight\b",
            r"\bthis weekend\b",
            r"\blater\b",
            r"\bafter class\b",
            r"\bbefore studio\b",
            r"\bbefore [^,.!?]+",
            r"\beod\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

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

from __future__ import annotations

import re
from datetime import datetime

from app.core.time_utils import parse_human_time
from app.llm.client import OpenAIAdapter
from app.schemas.intent import ExtractedTask, IntentResult, IntentName, ImageExtractionResult


class IntentExtractor:
    def __init__(self, adapter: OpenAIAdapter | None = None) -> None:
        self.adapter = adapter or OpenAIAdapter()

    def extract(self, text: str, timezone: str) -> IntentResult:
        llm_result = self._extract_with_llm(text, timezone)
        if llm_result:
            return llm_result
        return self._extract_fallback(text, timezone)

    def _extract_with_llm(self, text: str, timezone: str) -> IntentResult | None:
        payload = self.adapter.json_completion(
            system=(
                "Classify intent and extract task/deadline fields from a single SMS message. "
                "Return JSON keys: intent, confidence, needs_clarification, clarification_question, "
                "time_reference, time_confidence, context_signal, blockers, summary, task."
            ),
            user=(
                f"timezone={timezone}\n"
                f"message={text}\n"
                "task should be object with: title, description, project, deadline_text, priority, confidence, next_step."
            ),
        )
        if not payload:
            return None
        try:
            if payload.get("task"):
                task = ExtractedTask.model_validate(payload["task"])
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

        intent: IntentName = "general_chat"
        confidence = 0.55
        context_signal = None
        task: ExtractedTask | None = None

        if any(token in lowered for token in ["in class", "driving", "at dinner", "all nighter", "in a meeting"]):
            intent = "context_signal"
            context_signal = lowered
            confidence = 0.82
        elif any(token in lowered for token in ["what do i have", "what's due", "deadlines", "today", "this week"]):
            intent = "timeline_query"
            confidence = 0.8
        elif any(token in lowered for token in ["finished", "done", "completed", "wrapped"]):
            intent = "complete_task"
            confidence = 0.75
        elif any(token in lowered for token in ["need to", "have to", "gotta", "assignment", "due"]):
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
                task=task,
            )
        elif any(token in lowered for token in ["stuck", "distracted", "underestimated", "behind"]):
            intent = "reflection"
            confidence = 0.7

        return IntentResult(intent=intent, confidence=confidence, context_signal=context_signal, task=task)

    @staticmethod
    def _simple_task_title(text: str) -> str:
        cleaned = re.sub(r"^(yo|hey|ok|okay)\s+", "", text).strip()
        cleaned = cleaned.replace("need to ", "").replace("have to ", "")
        if len(cleaned) > 90:
            cleaned = cleaned[:90].rsplit(" ", 1)[0]
        return cleaned.capitalize()

    @staticmethod
    def _extract_deadline_phrase(text: str) -> str | None:
        patterns = [
            r"\bby [^,.!?]+",
            r"\bdue [^,.!?]+",
            r"\btomorrow(?: morning| night)?\b",
            r"\btonight\b",
            r"\bthis weekend\b",
            r"\bbefore [^,.!?]+",
            r"\beod\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None


class ImageAssignmentExtractor:
    def __init__(self, adapter: OpenAIAdapter | None = None) -> None:
        self.adapter = adapter or OpenAIAdapter()

    def extract(self, image_url: str, timezone: str) -> ImageExtractionResult:
        payload = self.adapter.vision_json(
            system=(
                "Extract assignment details from screenshot image. "
                "Return JSON with keys: title, due_text, context, deliverables, raw_text, confidence."
            ),
            user_prompt=f"timezone={timezone}. infer due dates if present. be conservative on confidence.",
            image_url=image_url,
        )
        if not payload:
            return ImageExtractionResult(confidence=0.0)
        result = ImageExtractionResult.model_validate(payload)
        if result.due_text and not result.due_at:
            parsed, _ = parse_human_time(result.due_text, timezone=timezone)
            result.due_at = parsed
        return result


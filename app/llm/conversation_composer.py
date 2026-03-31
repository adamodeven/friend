from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone

from app.llm.client import OllamaAdapter
from app.llm.message_chunker import MessageChunker
from app.llm.repetition_guard import RepetitionGuard
from app.schemas.reply import ComposedReply, ReplyBrief


class ConversationComposer:
    def __init__(
        self,
        *,
        adapter: OllamaAdapter | None = None,
        chunker: MessageChunker | None = None,
        repetition_guard: RepetitionGuard | None = None,
    ) -> None:
        self.adapter = adapter or OllamaAdapter()
        self.chunker = chunker or MessageChunker()
        self.repetition_guard = repetition_guard or RepetitionGuard()

    def compose(self, brief: ReplyBrief) -> ComposedReply:
        recent_assistant = [line.split(":", 1)[1].strip() for line in brief.recent_thread if line.startswith("assistant:")]
        messages, regenerated = self._compose_with_llm(brief, recent_assistant)
        if messages:
            normalized = self.chunker.normalize_messages(
                messages,
                max_chunk_length=brief.max_chunk_length,
                max_chunks=brief.max_chunks,
            )
            return ComposedReply(messages=normalized, used_fallback=False, regenerated_for_repetition=regenerated)

        fallback = self._fallback_messages(brief)
        return ComposedReply(messages=fallback, used_fallback=True, regenerated_for_repetition=False)

    def _compose_with_llm(self, brief: ReplyBrief, recent_assistant: list[str]) -> tuple[list[str] | None, bool]:
        if not self.adapter.enabled:
            return None, False

        messages = self._generate_messages(brief=brief, avoid_phrases=[])
        if not messages:
            return None, False

        combined = " ".join(messages)
        if self.repetition_guard.is_too_similar(combined, recent_assistant):
            messages[-1] = self._force_distinct_tail(messages[-1], brief)
            return messages, True
        return messages, False

    def _generate_messages(self, *, brief: ReplyBrief, avoid_phrases: list[str]) -> list[str] | None:
        payload = self._model_payload(brief=brief, avoid_phrases=avoid_phrases)
        # Keep compose to one LLM call for predictable latency under local CPU inference.
        text = self.adapter.text_completion(
            system=self._system_prompt(),
            user=payload,
            options={"temperature": 0.65, "num_predict": 120},
            request_timeout_seconds=14,
        )
        return self._extract_messages_from_text(text)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You write outbound SMS replies for a personal execution manager. "
            "Sound like a real person texting: casual, modern, sharp, socially fluent, concise. "
            "Never robotic, corporate, therapist-y, or generic productivity-bot language. "
            "No em dashes, no markdown, no labels, no numbering. "
            "Answer what the user actually said, in context, and keep momentum. "
            "Use 1-3 message bubbles max, each short. "
            "Output plain text only. If multiple bubbles, separate them with one blank line."
        )

    def _model_payload(self, *, brief: ReplyBrief, avoid_phrases: list[str]) -> str:
        compact = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "latest_user_message": brief.latest_user_message,
            "response_goal": brief.response_goal,
            "operational_reason": brief.operational_reason,
            "urgency_level": brief.urgency_level,
            "style_mode": brief.style_mode,
            "key_facts": brief.key_facts_to_include[:4],
            "question_if_needed": brief.question_if_needed,
            "suggested_next_step": brief.suggested_next_step,
            "active_tasks": brief.active_task_context[:4],
            "deadlines": brief.deadline_context[:4],
            "state_flags": brief.current_state_flags[:3],
            "memory_notes": brief.memory_notes[:3],
            "recent_thread": brief.recent_thread[-6:],
            "constraints": {
                "max_chunks": brief.max_chunks,
                "max_chunk_length": brief.max_chunk_length,
                "avoid_phrases": avoid_phrases,
                "must_answer_latest_message": True,
                "should_sound_human": True,
            },
        }
        return json.dumps(compact, ensure_ascii=True)

    @staticmethod
    def _extract_messages_from_text(text: str | None) -> list[str] | None:
        if not text or not text.strip():
            return None
        blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
        if blocks:
            return blocks
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return lines or [text.strip()]

    @staticmethod
    def _force_distinct_tail(last_message: str, brief: ReplyBrief) -> str:
        if brief.question_if_needed:
            return f"{last_message} {brief.question_if_needed}".strip()
        if brief.suggested_next_step:
            return f"{last_message} next move: {brief.suggested_next_step}".strip()
        return f"{last_message} what's the real next move from your side?".strip()

    def _fallback_messages(self, brief: ReplyBrief) -> list[str]:
        # Failure-only safety net; should not be the normal UX path.
        opening = self._fallback_opening(brief.latest_user_message)
        if brief.should_ask_question and brief.question_if_needed:
            base = f"{opening} {brief.question_if_needed}"
        elif brief.suggested_next_step:
            base = f"{opening} next move: {brief.suggested_next_step}"
        elif brief.key_facts_to_include:
            base = f"{opening} i still captured this: {brief.key_facts_to_include[0]}"
        else:
            base = f"{opening} resend that and i got you."
        return self.chunker.chunk(
            base,
            max_chunk_length=brief.max_chunk_length,
            max_chunks=min(brief.max_chunks, 2),
        )

    @staticmethod
    def _fallback_opening(seed: str) -> str:
        options = [
            "my response engine glitched for a sec.",
            "tiny compose hiccup on my side.",
            "i hit a quick generation miss there.",
        ]
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(options)
        return options[idx]

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

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
            normalized = self._postprocess_messages(normalized, brief)
            return ComposedReply(messages=normalized, used_fallback=False, regenerated_for_repetition=regenerated)

        fallback = self._fallback_messages(brief)
        return ComposedReply(messages=fallback, used_fallback=True, regenerated_for_repetition=False)

    def _compose_with_llm(self, brief: ReplyBrief, recent_assistant: list[str]) -> tuple[list[str] | None, bool]:
        if not self.adapter.enabled:
            return None, False

        regenerated = False
        avoid_phrases = self.repetition_guard.avoid_phrases(recent_assistant, limit=4)
        messages = self._generate_messages(brief=brief, avoid_phrases=avoid_phrases, strict=False)
        if not messages:
            return None, False

        combined = " ".join(messages)
        if (
            self._looks_internal_or_robotic(combined)
            or self._looks_low_quality(combined, brief.latest_user_message)
            or self._has_parrot_bubble(messages, brief.latest_user_message)
            or self._first_bubble_asks_question_when_direct_answer_needed(messages, brief)
        ):
            retry = self._generate_messages(
                brief=brief,
                avoid_phrases=avoid_phrases + self._quality_banned_openers(),
                strict=True,
            )
            if retry:
                messages = retry
                combined = " ".join(messages)
                regenerated = True

        combined = " ".join(messages)
        if self.repetition_guard.is_too_similar(combined, recent_assistant):
            messages[-1] = self._force_distinct_tail(messages[-1], brief)
            regenerated = True
        return messages, regenerated

    def _generate_messages(self, *, brief: ReplyBrief, avoid_phrases: list[str], strict: bool) -> list[str] | None:
        payload = self._model_payload(brief=brief, avoid_phrases=avoid_phrases)
        # Keep compose to one LLM call for predictable latency under local CPU inference.
        text = self.adapter.text_completion(
            system=self._system_prompt(strict=strict),
            user=payload,
            options={"temperature": 0.72 if not strict else 0.55, "num_predict": 120},
            request_timeout_seconds=30,
        )
        return self._extract_messages_from_text(text)

    @staticmethod
    def _system_prompt(*, strict: bool) -> str:
        strict_rules = (
            "Do not mention internal labels like intent, response_goal, parser, composer, fallback, or glitches. "
            "Do not say 'open conversational message received' or anything similar. "
            "Do not echo section labels like active tasks, deadlines, key facts, or recent thread. "
            "Do not mirror the user's exact sentence back to them."
            if strict
            else ""
        )
        return (
            "You write outbound SMS replies for a personal execution manager. "
            "Sound like a real person texting: casual, modern, sharp, socially fluent, concise. "
            "Never robotic, corporate, therapist-y, or generic productivity-bot language. "
            "If user asks what you do or whether replies are canned/live, answer directly in plain language first. "
            "For small-talk or quick checks, do not mirror the user's exact words back. "
            "No em dashes, no markdown, no labels, no numbering. "
            "Answer what the user actually said, in context, and keep momentum. "
            "Use 1-3 message bubbles max, each short. "
            "Output plain text only. If multiple bubbles, separate them with one blank line. "
            f"{strict_rules}"
        )

    def _model_payload(self, *, brief: ReplyBrief, avoid_phrases: list[str]) -> str:
        recent_thread = brief.recent_thread[-6:] or ["(no prior thread)"]
        key_facts = brief.key_facts_to_include[:4] or ["(none)"]
        tasks = brief.active_task_context[:3] or ["(none)"]
        deadlines = brief.deadline_context[:3] or ["(none)"]
        flags = brief.current_state_flags[:3] or ["(none)"]
        notes = brief.memory_notes[:2] or ["(none)"]
        avoid = avoid_phrases[:4] or ["(none)"]
        question = brief.question_if_needed or "(none)"
        next_step = brief.suggested_next_step or "(none)"

        def _lines(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        return (
            f"LATEST USER MESSAGE:\n{brief.latest_user_message}\n\n"
            f"REPLY GOAL: {brief.response_goal}\n"
            f"URGENCY: {brief.urgency_level}\n"
            f"TONE MODE: {brief.style_mode}\n"
            f"REASON FOR REPLY: {brief.operational_reason or '(none)'}\n\n"
            f"KEY FACTS TO INCLUDE:\n{_lines(key_facts)}\n\n"
            f"SUGGESTED NEXT STEP:\n{next_step}\n\n"
            f"QUESTION IF NEEDED:\n{question}\n\n"
            f"ACTIVE TASKS:\n{_lines(tasks)}\n\n"
            f"UPCOMING DEADLINES:\n{_lines(deadlines)}\n\n"
            f"CURRENT USER FLAGS:\n{_lines(flags)}\n\n"
            f"MEMORY NOTES:\n{_lines(notes)}\n\n"
            f"RECENT THREAD:\n{_lines(recent_thread)}\n\n"
            f"AVOID REPEATING THESE OPENERS:\n{_lines(avoid)}\n\n"
            f"OUTPUT CONSTRAINTS:\n"
            f"- max_chunks={brief.max_chunks}\n"
            f"- max_chunk_length={brief.max_chunk_length}\n"
            "- answer the latest user message directly\n"
            "- if this is an answer_question goal, first bubble must be a direct answer statement\n"
            "- keep it human and text-like\n"
            "- never expose internal system labels\n"
        )

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

    @classmethod
    def _looks_internal_or_robotic(cls, text: str) -> bool:
        lowered = text.lower()
        return any(flag in lowered for flag in cls._quality_banned_openers())

    @staticmethod
    def _quality_banned_openers() -> list[str]:
        return [
            "open conversational message received",
            "general chat intent",
            "intent=",
            "response_goal",
            "tiny compose hiccup",
            "generation miss",
            "response engine glitched",
            "active tasks:",
            "upcoming deadlines:",
            "key facts",
            "recent thread",
        ]

    @classmethod
    def _looks_low_quality(cls, candidate: str, latest_user_message: str) -> bool:
        lowered = candidate.lower().strip()
        if any(token in lowered for token in cls._quality_banned_openers()):
            return True
        if "user asked:" in lowered:
            return True

        cand_norm = cls._normalize_text(candidate)
        user_norm = cls._normalize_text(latest_user_message)
        if cand_norm and user_norm:
            similarity = SequenceMatcher(a=cand_norm, b=user_norm).ratio()
            # If we mostly parroted the user, force a regeneration.
            if similarity >= 0.86 and len(cand_norm.split()) <= 18:
                return True
        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
        return " ".join(cleaned.split())

    @classmethod
    def _has_parrot_bubble(cls, messages: list[str], latest_user_message: str) -> bool:
        user_norm = cls._normalize_text(latest_user_message)
        if not user_norm:
            return False
        for bubble in messages[:3]:
            bubble_norm = cls._normalize_text(bubble)
            if not bubble_norm:
                continue
            ratio = SequenceMatcher(a=bubble_norm, b=user_norm).ratio()
            if ratio >= 0.88 and len(bubble_norm.split()) <= 14:
                return True
        return False

    @staticmethod
    def _first_bubble_asks_question_when_direct_answer_needed(messages: list[str], brief: ReplyBrief) -> bool:
        if brief.response_goal != "answer_question":
            return False
        if not messages:
            return False
        first = messages[0].strip()
        return first.endswith("?")

    @classmethod
    def _postprocess_messages(cls, messages: list[str], brief: ReplyBrief) -> list[str]:
        if brief.response_goal != "answer_question" or not messages:
            return messages
        first = messages[0].strip()
        user_norm = cls._normalize_text(brief.latest_user_message)
        first_norm = cls._normalize_text(first)
        if first.endswith("?") or (user_norm and first_norm and SequenceMatcher(a=first_norm, b=user_norm).ratio() >= 0.9):
            if len(messages) > 1:
                return messages[1:]
        return messages

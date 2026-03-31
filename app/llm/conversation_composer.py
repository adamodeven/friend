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
        json_result = self.adapter.json_completion(
            system=self._system_prompt(),
            user=payload,
            options={"temperature": 0.65, "num_predict": 260},
        )
        messages = self._extract_messages(json_result)
        if messages:
            return messages

        # Recovery path when strict JSON format is flaky.
        text = self.adapter.text_completion(
            system=self._system_prompt_text(),
            user=payload,
            options={"temperature": 0.65, "num_predict": 260},
        )
        return self._extract_messages_from_text(text)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are composing outbound SMS replies for a personal accountability agent. "
            "Write like a real human texting a friend: casual, modern, socially fluent, concise, and context-aware. "
            "Be naturally warm without fake hype. No corporate tone. No therapist tone. No cringe. "
            "No em dashes. No markdown. Avoid repeating recent wording. "
            "Use short text bubbles that feel native to iMessage. "
            "Respond to what the user actually said, not generic productivity slogans. "
            "Return strict JSON: {\"messages\": [\"...\", \"...\"]}. "
            "1 to 3 messages max, each message should stand alone and be natural."
        )

    @staticmethod
    def _system_prompt_text() -> str:
        return (
            "Compose the outbound SMS replies as plain text. "
            "You may return one or more message bubbles separated by blank lines. "
            "No markdown, no labels, no numbering. "
            "Same voice constraints: casual, modern, concise, human, non-robotic."
        )

    def _model_payload(self, *, brief: ReplyBrief, avoid_phrases: list[str]) -> str:
        compact = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "reply_brief": brief.model_dump(mode="json"),
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
    def _extract_messages(result: dict | None) -> list[str] | None:
        if not isinstance(result, dict):
            return None
        messages = result.get("messages")
        if not isinstance(messages, list):
            single = result.get("message")
            if isinstance(single, str) and single.strip():
                return [single.strip()]
            return None
        cleaned = [str(m).strip() for m in messages if str(m).strip()]
        return cleaned or None

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

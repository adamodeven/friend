from __future__ import annotations

import json
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

        payload = self._model_payload(brief=brief, avoid_phrases=[])
        result = self.adapter.json_completion(system=self._system_prompt(), user=payload)
        messages = self._extract_messages(result)
        if not messages:
            return None, False

        combined = " ".join(messages)
        if self.repetition_guard.is_too_similar(combined, recent_assistant):
            avoid = self.repetition_guard.avoid_phrases(recent_assistant)
            retry_payload = self._model_payload(brief=brief, avoid_phrases=avoid)
            retry_result = self.adapter.json_completion(system=self._system_prompt(), user=retry_payload)
            retry_messages = self._extract_messages(retry_result)
            if retry_messages:
                retry_combined = " ".join(retry_messages)
                if not self.repetition_guard.is_too_similar(retry_combined, recent_assistant):
                    return retry_messages, True
                retry_messages[-1] = self._force_distinct_tail(retry_messages[-1], brief)
                return retry_messages, True
        return messages, False

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
    def _force_distinct_tail(last_message: str, brief: ReplyBrief) -> str:
        if brief.question_if_needed:
            return f"{last_message} {brief.question_if_needed}".strip()
        if brief.suggested_next_step:
            return f"{last_message} next move: {brief.suggested_next_step}".strip()
        return f"{last_message} what's the real next move from your side?".strip()

    def _fallback_messages(self, brief: ReplyBrief) -> list[str]:
        # Failure-only safety net; should not be the normal UX path.
        if brief.should_ask_question and brief.question_if_needed:
            base = f"quick hiccup on my side. {brief.question_if_needed}"
        elif brief.suggested_next_step:
            base = f"quick hiccup on my side, but i still got context. next move: {brief.suggested_next_step}"
        else:
            base = "quick hiccup on my side. resend that and i got you."
        return self.chunker.chunk(
            base,
            max_chunk_length=brief.max_chunk_length,
            max_chunks=min(brief.max_chunks, 2),
        )


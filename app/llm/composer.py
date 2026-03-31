from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from app.core.config import get_settings
from app.llm.client import OllamaAdapter
from app.llm.style import chunk_sms, get_style_profile
from app.schemas.intent import ConversationReply, IntentResult


class ReplyComposer:
    def __init__(self, adapter: OllamaAdapter | None = None) -> None:
        self.adapter = adapter or OllamaAdapter()
        self.settings = get_settings()

    def compose(
        self,
        *,
        style: str,
        intent: IntentResult,
        state_summary: str,
        action_summary: str,
        timezone: str,
    ) -> ConversationReply:
        text = self._compose_with_llm(
            style=style,
            intent=intent,
            state_summary=state_summary,
            action_summary=action_summary,
            timezone=timezone,
        )
        if not text:
            text = self._fallback_text(intent, action_summary)
        return ConversationReply(messages=chunk_sms(text, max_chars=self.settings.max_sms_chars))

    def _compose_with_llm(
        self,
        *,
        style: str,
        intent: IntentResult,
        state_summary: str,
        action_summary: str,
        timezone: str,
    ) -> str | None:
        profile = get_style_profile(style)
        response = self.adapter.text_completion(
            system=(
                f"You are an accountability texting assistant. {profile.system_hint} "
                "Output only the outbound SMS text. keep it short. no markdown."
            ),
            user=(
                f"timezone={timezone}\n"
                f"intent={intent.intent}\n"
                f"message_summary={intent.summary}\n"
                f"action_summary={action_summary}\n"
                f"state_summary={state_summary}\n"
                f"now={datetime.now(tz=dt_timezone.utc).isoformat()}\n"
            ),
        )
        if response:
            return response.strip()
        return None

    @staticmethod
    def _fallback_text(intent: IntentResult, action_summary: str) -> str:
        if intent.needs_clarification and intent.clarification_question:
            return intent.clarification_question
        if intent.intent == "timeline_query":
            return f"bet. here's the plan: {action_summary}"
        if intent.intent == "status_query":
            return f"quick version: {action_summary}"
        if intent.intent == "context_signal":
            return "all good, i’ll chill reminders for now and hit you after that."
        if intent.intent == "complete_task":
            return f"nice. marked that progress. {action_summary}"
        if intent.intent == "add_task":
            return f"locked in. added it. {action_summary}"
        if intent.intent == "reflection":
            return f"noted. let's fix the pattern: {action_summary}"
        return action_summary or "got it. updated and tracking."

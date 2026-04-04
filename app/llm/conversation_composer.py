from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

from app.core.config import get_settings
from app.llm.client import OllamaAdapter
from app.llm.message_chunker import MessageChunker
from app.llm.repetition_guard import RepetitionGuard
from app.llm.style import get_style_profile
from app.schemas.reply import ComposedReply, ReplyBrief


class ConversationComposer:
    def __init__(
        self,
        *,
        adapter: OllamaAdapter | None = None,
        chunker: MessageChunker | None = None,
        repetition_guard: RepetitionGuard | None = None,
    ) -> None:
        settings = get_settings()
        self.adapter = adapter or OllamaAdapter()
        self.chunker = chunker or MessageChunker()
        self.repetition_guard = repetition_guard or RepetitionGuard()
        provider = settings.llm_provider.lower().strip()
        if provider == "openai":
            self._compose_model = settings.openai_composer_model.strip() or settings.openai_text_model.strip() or None
            self._lightweight_compose_model = settings.openai_text_model.strip() or self._compose_model
        else:
            self._compose_model = settings.ollama_composer_model.strip() or settings.ollama_text_model.strip() or None
            self._lightweight_compose_model = settings.ollama_text_model.strip() or self._compose_model

    def compose(self, brief: ReplyBrief) -> ComposedReply:
        recent_assistant = [line.split(":", 1)[1].strip() for line in brief.recent_thread if line.startswith("assistant:")]
        messages, regenerated = self._compose_with_llm(brief, recent_assistant)
        if messages:
            style_profile = get_style_profile(brief.style_mode)
            normalized = self.chunker.normalize_messages(
                messages,
                max_chunk_length=brief.max_chunk_length,
                max_chunks=brief.max_chunks,
                soft_chunk_length=style_profile.soft_chunk_chars,
            )
            normalized = self._postprocess_messages(normalized, brief)
            normalized = self._merge_tiny_lead_bubble(normalized, brief)
            if not normalized or self._is_unacceptable_output(normalized, brief):
                fallback = self._fallback_messages(brief)
                return ComposedReply(messages=fallback, used_fallback=True, regenerated_for_repetition=regenerated)
            return ComposedReply(messages=normalized, used_fallback=False, regenerated_for_repetition=regenerated)

        fallback = self._fallback_messages(brief)
        return ComposedReply(messages=fallback, used_fallback=True, regenerated_for_repetition=False)

    def _compose_with_llm(self, brief: ReplyBrief, recent_assistant: list[str]) -> tuple[list[str] | None, bool]:
        if not self.adapter.enabled:
            return None, False

        regenerated = False
        avoid_phrases = self.repetition_guard.avoid_phrases(recent_assistant, limit=4)
        lightweight = self._should_use_lightweight_compose(brief)
        messages = self._generate_messages_structured(
            brief=brief,
            avoid_phrases=avoid_phrases,
            strict=False,
            lightweight=lightweight,
            quality_errors=[],
        )
        if not messages:
            messages = self._generate_messages(
                brief=brief,
                avoid_phrases=avoid_phrases,
                strict=False,
                lightweight=lightweight,
                quality_errors=[],
            )
        if not messages:
            return None, False

        quality_errors = self._quality_errors(messages, brief)
        if self.repetition_guard.is_too_similar(" ".join(messages), recent_assistant):
            quality_errors.append("too similar to recent assistant phrasing; reword opener and sentence cadence")
        if not quality_errors:
            return messages, regenerated

        regenerated = True
        strict_retry = self._generate_messages_structured(
            brief=brief,
            avoid_phrases=avoid_phrases,
            strict=True,
            lightweight=False,
            quality_errors=quality_errors,
        )
        if not strict_retry:
            strict_retry = self._generate_messages(
                brief=brief,
                avoid_phrases=avoid_phrases,
                strict=True,
                lightweight=False,
                quality_errors=quality_errors,
            )
        if not strict_retry:
            return None, regenerated
        if self._needs_repair(strict_retry, brief):
            return None, regenerated
        if self.repetition_guard.is_too_similar(" ".join(strict_retry), recent_assistant):
            return None, regenerated
        return strict_retry, regenerated

    def _needs_repair(self, messages: list[str], brief: ReplyBrief) -> bool:
        combined = " ".join(messages)
        return (
            self._looks_internal_or_robotic(combined)
            or self._looks_low_quality(combined, brief.latest_user_message)
            or self._has_parrot_bubble(messages, brief.latest_user_message)
            or self._has_nonsequitur_for_short_checkin(messages, brief)
            or self._is_overlong_for_goal(messages, brief)
            or self._goal_alignment_needs_repair(messages, brief)
            or self._first_bubble_asks_question_when_direct_answer_needed(messages, brief)
            or self._answer_quality_needs_repair(messages, brief)
        )

    def _generate_messages(
        self,
        *,
        brief: ReplyBrief,
        avoid_phrases: list[str],
        strict: bool,
        lightweight: bool,
        quality_errors: list[str],
    ) -> list[str] | None:
        payload = self._model_payload(
            brief=brief,
            avoid_phrases=avoid_phrases,
            lightweight=lightweight,
            quality_errors=quality_errors,
        )
        options = {
            "temperature": 0.58 if not strict else 0.42,
            "num_predict": 40 if lightweight else 52,
            "num_ctx": 384 if lightweight else 512,
            "repeat_penalty": 1.15,
        }
        text = self.adapter.text_completion(
            system=self._system_prompt(strict=strict),
            user=payload,
            model=self._select_model(lightweight=lightweight),
            options=options,
            request_timeout_seconds=10 if lightweight else 14,
        )
        return self._extract_messages_from_text(text)

    def _generate_messages_structured(
        self,
        *,
        brief: ReplyBrief,
        avoid_phrases: list[str],
        strict: bool,
        lightweight: bool,
        quality_errors: list[str],
    ) -> list[str] | None:
        payload = self._model_payload(
            brief=brief,
            avoid_phrases=avoid_phrases,
            lightweight=lightweight,
            quality_errors=quality_errors,
        )
        options = {
            "temperature": 0.52 if not strict else 0.35,
            "num_predict": 56 if lightweight else 84,
            "num_ctx": 512 if lightweight else 768,
            "repeat_penalty": 1.15,
        }
        result = self.adapter.json_completion(
            system=self._structured_system_prompt(strict=strict),
            user=payload,
            model=self._select_model(lightweight=lightweight),
            options=options,
            request_timeout_seconds=11 if lightweight else 16,
        )
        if not isinstance(result, dict):
            return None
        messages_raw = result.get("messages")
        if isinstance(messages_raw, list):
            messages = [str(m).strip() for m in messages_raw if str(m).strip()]
            return messages or None
        single = result.get("message")
        if isinstance(single, str) and single.strip():
            return [single.strip()]
        return None

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
            "The vibe is closer to a plugged-in friend with good memory than a nagging task bot. "
            "The actual content has to sound human too, not like task-tracker copy wearing slang. "
            "Never robotic, corporate, therapist-y, or generic productivity-bot language. "
            "Most replies should be one short bubble. Use a second or third only when it clearly helps. "
            "For casual, off-topic, or meta texts, answer socially first and do not force a task pivot. "
            "When the user is vague, overwhelmed, or slipping, narrow it to one concrete next move. "
            "If something is really just a small reminder-type action, treat it like that and do not turn it into a mini project or fake work block. "
            "If a vague new obligation shows up, ask one short follow-up when that would materially sharpen the plan. "
            "Urgency should sound calm and real, never corny, fake inspirational, or hypey. "
            "Avoid stiff database-y verbs like archived, captured, or logged unless the user used that language first. "
            "Prefer natural phrasing like bet, got it, good looks, we're good there, or that clears it when it fits. "
            "If user asks what you do or whether replies are canned/live, answer directly in plain language first. "
            "If latest user message is a short greeting/check-in, keep it present-tense and lightweight. "
            "Do not drag in old thread drama unless the user asked about it in this message. "
            "For small-talk or quick checks, do not mirror the user's exact words back. "
            "Do not keep asking the user to hand you another task unless the message clearly calls for it. "
            "Do not use label-style colons like next up: or next move:, unless it's a real clock time. "
            "When you bring up what comes after progress, make it feel like a fresh thought instead of a task-manager label. "
            "Never start your first bubble with the same 4+ word sequence the user just sent. "
            "No em dashes, no semicolons, no markdown, no labels, no numbering. "
            "Answer what the user actually said, in context, and keep momentum. "
            "Use 1-3 message bubbles max, each short. "
            "Output plain text only. If multiple bubbles, separate them with one blank line. "
            f"{strict_rules}"
        )

    @staticmethod
    def _structured_system_prompt(*, strict: bool) -> str:
        strict_rules = (
            "You must obey all constraints strictly. "
            "Never include labels, markdown, code blocks, headings, or analysis text. "
            "Never output the user's exact sentence as a standalone bubble. "
            if strict
            else ""
        )
        return (
            "You write outbound SMS replies for a personal execution manager. "
            "Output valid JSON only with this exact schema: {\"messages\": [\"bubble 1\", \"bubble 2\"]}. "
            "messages must contain 1 to 3 short text bubbles, each natural and human. "
            "No markdown, no numbering, no bullet points, no labels, no em dash, no semicolons. "
            "Do not use label-style colons like next up: or next move:, unless it's a real clock time. "
            "Default to one short bubble unless extra separation clearly improves clarity. "
            "For casual, off-topic, or meta texts, answer socially first and do not force a task pivot. "
            "If the user is vague or overloaded, reduce cognitive load by choosing one next move. "
            "Reminder-like actions should not be treated like mini projects or given fake first-pass work. "
            "Use a short follow-up question when a vague new obligation needs one detail to plan well. "
            "Urgency should feel clean and real, never corny or fake inspirational. "
            "Do not sound like a database or task tracker. "
            "Answer the latest user message directly and keep context continuity. "
            "If asked whether responses are live/canned, answer that directly in the first bubble. "
            f"{strict_rules}"
        )

    def _model_payload(
        self,
        *,
        brief: ReplyBrief,
        avoid_phrases: list[str],
        lightweight: bool,
        quality_errors: list[str],
    ) -> str:
        style_profile = get_style_profile(brief.style_mode)
        recent_thread = brief.recent_thread[-(2 if lightweight else 5) :] or ["(no prior thread)"]
        key_facts = brief.key_facts_to_include[: (2 if lightweight else 4)] or ["(none)"]
        tasks = (brief.active_task_context[:1] if lightweight else brief.active_task_context[:3]) or ["(none)"]
        deadlines = (brief.deadline_context[:1] if lightweight else brief.deadline_context[:3]) or ["(none)"]
        flags = (brief.current_state_flags[:1] if lightweight else brief.current_state_flags[:3]) or ["(none)"]
        notes = (brief.memory_notes[:1] if lightweight else brief.memory_notes[:2]) or ["(none)"]
        avoid = avoid_phrases[:3] or ["(none)"]
        errors = quality_errors[:5] or ["(none)"]
        avoid_topics = brief.avoid_topics[:2] or ["(none)"]
        style_rules = list(style_profile.guardrails[: (2 if lightweight else 4)]) or ["(none)"]
        question = brief.question_if_needed or "(none)"
        next_step = brief.suggested_next_step or "(none)"
        short_checkin = "yes" if brief.is_short_checkin else "no"
        style_hint = self._style_hint(brief.style_mode)
        push_for_action = "yes" if brief.should_push_for_action else "no"
        ask_question = "yes" if brief.should_ask_question else "no"

        def _lines(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        if lightweight:
            return (
                "INTERNAL CONTEXT (DO NOT QUOTE OR PARAPHRASE):\n"
                f"- user said: {brief.latest_user_message}\n"
                f"- goal: {brief.response_goal}\n"
                f"- tone mode: {brief.style_mode}\n"
                f"- tone hint: {style_hint}\n"
                f"- urgency: {brief.urgency_level}\n"
                f"- reason: {brief.operational_reason or '(none)'}\n"
                f"- push to one next move: {push_for_action}\n"
                f"- ask brief follow-up only if needed: {ask_question}\n"
                f"- short checkin: {short_checkin}\n"
                f"- facts: {' | '.join(key_facts)}\n"
                f"- next step: {next_step}\n"
                f"- question: {question}\n"
                f"- recent thread: {' | '.join(recent_thread)}\n"
                f"- active tasks: {' | '.join(tasks)}\n"
                f"- deadlines: {' | '.join(deadlines)}\n"
                f"- state flags: {' | '.join(flags)}\n"
                f"- memory notes: {' | '.join(notes)}\n"
                f"- style rules: {' | '.join(style_rules)}\n"
                f"- avoid topics: {' | '.join(avoid_topics)}\n"
                f"- avoid repeated openers: {' | '.join(avoid)}\n"
                f"- avoid these quality issues: {' | '.join(errors)}\n"
                f"- max chunks: {brief.max_chunks}\n"
                f"- max chunk length: {brief.max_chunk_length}\n"
                "Write the actual user-facing reply only. 1-3 short text bubbles."
            )

        payload = (
            f"LATEST USER MESSAGE:\n{brief.latest_user_message}\n\n"
            f"REPLY GOAL: {brief.response_goal}\n"
            f"URGENCY: {brief.urgency_level}\n"
            f"TONE MODE: {brief.style_mode}\n"
            f"STYLE HINT: {style_hint}\n"
            f"REASON FOR REPLY: {brief.operational_reason or '(none)'}\n\n"
            f"SHOULD PUSH TO ONE NEXT MOVE: {push_for_action}\n"
            f"SHOULD ASK A BRIEF FOLLOW-UP ONLY IF NEEDED: {ask_question}\n\n"
            f"SHORT CHECKIN: {short_checkin}\n\n"
            f"KEY FACTS TO INCLUDE:\n{_lines(key_facts)}\n\n"
            f"SUGGESTED NEXT STEP:\n{next_step}\n\n"
            f"QUESTION IF NEEDED:\n{question}\n\n"
            f"ACTIVE TASKS:\n{_lines(tasks)}\n\n"
            f"UPCOMING DEADLINES:\n{_lines(deadlines)}\n\n"
            f"CURRENT USER FLAGS:\n{_lines(flags)}\n\n"
            f"MEMORY NOTES:\n{_lines(notes)}\n\n"
            f"RECENT THREAD:\n{_lines(recent_thread)}\n\n"
            f"STYLE RULES:\n{_lines(style_rules)}\n\n"
            f"AVOID TOPICS:\n{_lines(avoid_topics)}\n\n"
            f"AVOID REPEATING THESE OPENERS:\n{_lines(avoid)}\n\n"
            f"QUALITY ISSUES TO AVOID ON THIS ATTEMPT:\n{_lines(errors)}\n\n"
            f"OUTPUT CONSTRAINTS:\n"
            f"- max_chunks={brief.max_chunks}\n"
            f"- max_chunk_length={brief.max_chunk_length}\n"
            "- answer the latest user message directly\n"
            "- keep it short by default\n"
            "- for casual or off-topic texts, stay social first and do not force action\n"
            "- if you push for action, narrow to one next move\n"
            "- if you ask a follow-up, keep it brief and only ask one\n"
            "- if a task is really a reminder-type action, do not treat it like a project or suggest a fake work block\n"
            "- if this is an answer_question goal, first bubble must be a direct answer statement\n"
            "- keep it human and text-like\n"
            "- no semicolons in the user-facing reply\n"
            "- no label-style colons like next up: or next move: unless it's a real clock time\n"
            "- if progress just happened, bring up what comes next like a fresh thought, not a label\n"
            "- never expose internal system labels\n"
        )
        return payload

    @staticmethod
    def _style_hint(style_mode: str) -> str:
        profile = get_style_profile(style_mode)
        rules = "; ".join(profile.guardrails[:3])
        return f"{profile.system_hint} {rules}"

    @staticmethod
    def _should_use_lightweight_compose(brief: ReplyBrief) -> bool:
        if brief.response_goal in {"open_conversation", "acknowledge_context"}:
            return True
        if brief.response_goal in {"confirm_update", "react_to_progress"}:
            return True
        if brief.response_goal == "answer_question" and len(brief.key_facts_to_include) <= 3:
            return True
        if brief.is_short_checkin:
            return True
        return False

    @staticmethod
    def _extract_messages_from_text(text: str | None) -> list[str] | None:
        if not text or not text.strip():
            return None
        cleaned_text = ConversationComposer._clean_candidate_text(text)
        if not cleaned_text:
            return None
        blocks = [b.strip() for b in cleaned_text.split("\n\n") if b.strip()]
        if blocks:
            return blocks
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        return lines or [cleaned_text.strip()]

    def _fallback_messages(self, brief: ReplyBrief) -> list[str]:
        # Failure-only safety net; should not be the normal UX path.
        opening = self._fallback_opening(brief.latest_user_message)
        lowered_user = brief.latest_user_message.lower()
        first_fact = self._first_safe_fact(brief)
        if brief.response_goal == "answer_question":
            if any(token in lowered_user for token in ("canned", "live", "generated")):
                base = "not canned, but i'm in backup mode right now while model access is limited."
            elif first_fact:
                base = first_fact
            else:
                base = "yeah, i'm live and i got your message."
        elif brief.response_goal == "timeline_summary":
            summary = brief.key_facts_to_include[0] if brief.key_facts_to_include else "no hard due items right now."
            base = self._flatten_timeline_summary(summary) or "no hard due items right now."
        elif brief.response_goal == "acknowledge_new_task":
            first_fact = first_fact or "got it."
            if brief.should_ask_question and brief.question_if_needed:
                base = f"bet, got it. {first_fact} {brief.question_if_needed}"
            elif brief.suggested_next_step:
                base = f"bet, got it. {first_fact} if you touch it next, i'd start with {brief.suggested_next_step}"
            else:
                base = f"bet, got it. {first_fact}"
        elif brief.response_goal == "react_to_progress":
            fact = first_fact or "good looks, that clears it."
            if brief.suggested_next_step:
                base = f"{fact} if you could also {brief.suggested_next_step} next, that'd be huge."
            else:
                base = fact
        elif brief.response_goal == "replan_blocker":
            if brief.question_if_needed:
                base = f"got it, that shifts the plan. {brief.question_if_needed}"
            else:
                base = "got it, that shifts the plan. what's the blocker i should account for first?"
        elif brief.response_goal == "confirm_update":
            fact = first_fact or "update applied."
            base = f"bet. {fact}"
        elif brief.should_ask_question and brief.question_if_needed:
            base = f"{opening} {brief.question_if_needed}"
        elif brief.suggested_next_step:
            base = f"{opening} if you want to keep it moving, start with {brief.suggested_next_step}"
        elif first_fact:
            base = f"{opening} {first_fact}"
        elif brief.is_short_checkin:
            base = "yo i'm here."
        else:
            base = f"{opening} got the update."
        return self.chunker.chunk(
            base,
            max_chunk_length=brief.max_chunk_length,
            max_chunks=min(brief.max_chunks, 2),
            soft_chunk_length=get_style_profile(brief.style_mode).soft_chunk_chars,
        )

    def _first_safe_fact(self, brief: ReplyBrief) -> str | None:
        for fact in brief.key_facts_to_include:
            safe = self._sanitize_fallback_text(fact)
            if safe:
                return safe
        return None

    def _flatten_timeline_summary(self, summary: str) -> str:
        cleaned = self._sanitize_fallback_text(summary)
        if not cleaned:
            return ""

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return ""

        heading: str | None = None
        items: list[str] = []
        for line in lines:
            bullet = re.sub(r"^\s*[-*]\s*", "", line).strip()
            if not bullet:
                continue
            if line.endswith(":") and not heading and len(lines) > 1:
                heading = bullet.rstrip(":").strip().lower()
                continue
            items.append(bullet)

        if not items:
            return (heading + " is clear right now.") if heading else cleaned

        compact_items = ", ".join(items[:3]).strip()
        if heading:
            return f"{heading}, {compact_items}"
        return compact_items

    @classmethod
    def _sanitize_fallback_text(cls, text: str) -> str:
        candidate = cls._clean_candidate_text(text or "")
        if not candidate:
            return ""
        if cls._looks_hard_structured_leak(candidate):
            return ""
        lowered = candidate.lower()
        if any(token in lowered for token in cls._quality_banned_openers()):
            return ""
        candidate = re.sub(r"\s*:\s*;\s*", ": ", candidate)
        candidate = candidate.replace(";", ",")
        candidate = re.sub(r"[ \t]{2,}", " ", candidate)
        candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip()
        return candidate

    @staticmethod
    def _fallback_opening(seed: str) -> str:
        options = [
            "got your text.",
            "i'm here.",
            "saw that.",
        ]
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(options)
        return options[idx]

    def _is_unacceptable_output(self, messages: list[str], brief: ReplyBrief) -> bool:
        combined = " ".join(messages)
        return (
            self._looks_internal_or_robotic(combined)
            or self._looks_low_quality(combined, brief.latest_user_message)
            or self._has_scaffolding_preface(messages)
            or self._has_parrot_bubble(messages, brief.latest_user_message)
            or self._has_nonsequitur_for_short_checkin(messages, brief)
            or self._is_overlong_for_goal(messages, brief)
            or self._goal_alignment_needs_repair(messages, brief)
            or self._first_bubble_asks_question_when_direct_answer_needed(messages, brief)
            or self._answer_quality_needs_repair(messages, brief)
        )

    def _quality_errors(self, messages: list[str], brief: ReplyBrief) -> list[str]:
        combined = " ".join(messages)
        errors: list[str] = []
        if self._looks_internal_or_robotic(combined):
            errors.append("contains internal labels or scaffolding text")
        if self._looks_low_quality(combined, brief.latest_user_message):
            errors.append("contains malformed output, markdown, or structured key/value leaks")
        if self._has_scaffolding_preface(messages):
            errors.append("starts with scaffolding preface instead of user-facing text")
        if self._has_parrot_bubble(messages, brief.latest_user_message):
            errors.append("parrots user wording too closely")
        if self._has_nonsequitur_for_short_checkin(messages, brief):
            errors.append("short check-in response drifts off-topic")
        if self._is_overlong_for_goal(messages, brief):
            errors.append("reply is too long or too many bubbles for this goal")
        if self._goal_alignment_needs_repair(messages, brief):
            errors.append("response does not match the reply goal")
        if self._first_bubble_asks_question_when_direct_answer_needed(messages, brief):
            errors.append("first bubble asks a question when a direct answer is required")
        if self._answer_quality_needs_repair(messages, brief):
            errors.append("first bubble lacks a direct answer for status/meta question")
        return errors

    @classmethod
    def _looks_internal_or_robotic(cls, text: str) -> bool:
        lowered = text.lower()
        return any(flag in lowered for flag in cls._quality_banned_openers())

    @staticmethod
    def _quality_banned_openers() -> list[str]:
        return [
            "user_message:",
            "goal=",
            "tone=",
            "urgency=",
            "reason=",
            "facts=",
            "next_step=",
            "question=",
            "thread=",
            "tasks=",
            "deadlines=",
            "flags=",
            "notes=",
            "avoid_openers=",
            "max_chunks=",
            "max_chunk_length=",
            "open conversational message received",
            "general chat intent",
            "intent=",
            "internal:",
            "task_manager:",
            "response_goal",
            "confirm update",
            "timeline summary",
            "answer question",
            "open conversation",
            "tiny compose hiccup",
            "generation miss",
            "response engine glitched",
            "you said \"",
            "active tasks:",
            "upcoming deadlines:",
            "key facts",
            "recent thread",
            "internal context",
            "here's the response",
            "here is the actual response from me",
            "checkpoint 1",
            "i've been noticing your responses",
            "here are my thoughts:",
            "be direct about whether this reply is live-generated right now",
            "confirm current system status plainly",
            "i'm here assistant",
            "i'm here to help",
            "i am here to help",
            "what's on your mind?",
            "how can i assist",
            "i'm available to help",
            "under control",
            "can i unblock it for you",
        ]

    @staticmethod
    def _looks_hard_structured_leak(text: str) -> bool:
        lowered = text.lower()
        if "status=" in lowered or "[status" in lowered or "due=-" in lowered:
            return True
        if re.search(
            r"\b(user_message|goal|tone|urgency|reason|facts|next_step|question|thread|tasks|deadlines|flags|notes|avoid_openers|max_chunks|max_chunk_length)\s*[:=]",
            lowered,
        ):
            return True
        return False

    @classmethod
    def _looks_low_quality(cls, candidate: str, latest_user_message: str) -> bool:
        lowered = candidate.lower().strip()
        folded = lowered.replace("’", "'").replace("`", "'")
        if any(token in folded for token in cls._quality_banned_openers()):
            return True
        if re.search(r"\bhere.?s\s+the\s+response\b", folded):
            return True
        if cls._looks_hard_structured_leak(folded):
            return True
        if re.search(r"(^|\n)\s*(#{1,6}|\*\*|\*\s+|-\s+|\d+\.)", candidate):
            return True
        if re.search(r"^\s*[a-z_]+\s*:\s*[a-z_]+\s*:", lowered):
            return True
        if re.search(r"\b[a-z_]+\s*=\s*\([^)]{0,80}\)", lowered):
            return True
        if re.search(r"\b(user_message|tasks|deadlines|next_step|goal|tone|urgency)\s*=", lowered):
            return True
        if "```" in candidate:
            return True
        if re.search(r'\*\s*".+?"', candidate):
            return True
        if re.search(r"\bstatus\s+[a-z_]+\s+p[0-9]\b", lowered):
            return True
        if re.search(r"\bstatus\s+(active|blocked|completed|archived)\b", lowered):
            return True
        if re.search(r"\bpriority\s+[0-9]\b", lowered):
            return True
        if "due no deadline" in lowered:
            return True
        if "due negative" in lowered:
            return True
        if "user asked:" in lowered:
            return True
        if "|" in candidate:
            return True
        if "actual response from me" in lowered:
            return True
        if "canned template" in lowered:
            return True
        if "i'm here to help you" in lowered or "i am here to help you" in lowered:
            return True
        if lowered.count("next move:") > 1:
            return True
        if cls._has_repeated_long_phrase(lowered):
            return True
        tail_word = lowered.rstrip(" .!?").split(" ")[-1] if lowered else ""
        if not re.search(r"[.!?]['\"]?$", candidate.strip()) and tail_word in {
            "at",
            "to",
            "on",
            "for",
            "with",
            "and",
            "or",
            "but",
            "of",
            "the",
            "a",
            "an",
        }:
            return True

        cand_norm = cls._normalize_text(candidate)
        user_norm = cls._normalize_text(latest_user_message)
        if cand_norm and user_norm:
            similarity = SequenceMatcher(a=cand_norm, b=user_norm).ratio()
            # If we mostly parroted the user, force a regeneration.
            if similarity >= 0.94:
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
        user_words = user_norm.split()
        short_checkin_reply_allowlist = {
            "yo i m here",
            "hey i m here",
            "hi i m here",
            "sup i m here",
            "whatup i m here",
            "whatsup i m here",
        }
        for bubble in messages[:3]:
            bubble_norm = cls._normalize_text(bubble)
            if not bubble_norm:
                continue
            if bubble_norm == user_norm:
                return True
            leading_overlap = cls._leading_overlap_words(bubble_norm, user_norm)
            if len(user_words) >= 6 and leading_overlap >= 4:
                return True
            if len(user_words) < 6 and leading_overlap >= 3:
                return True
            ratio = SequenceMatcher(a=bubble_norm, b=user_norm).ratio()
            if len(user_words) >= 8 and ratio >= 0.92:
                return True
            if len(user_words) < 8 and ratio >= 0.97:
                return True
            overlap_ratio = cls._lexical_overlap_ratio(bubble_norm, user_norm)
            if len(user_words) >= 6 and overlap_ratio >= 0.78:
                return True
            if (
                len(user_words) <= 3
                and user_norm in bubble_norm
                and len(bubble_norm.split()) <= 4
                and bubble_norm not in short_checkin_reply_allowlist
            ):
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

    @staticmethod
    def _answer_quality_needs_repair(messages: list[str], brief: ReplyBrief) -> bool:
        if brief.response_goal != "answer_question" or not messages:
            return False
        first = messages[0].strip()
        lowered = first.lower()
        punctuation = first.count(".") + first.count("?") + first.count("!")
        if punctuation == 0 and len(first.split()) >= 16:
            return True
        direct_markers = ("yes", "yeah", "yep", "no", "nah", "live", "canned", "not canned")
        if not any(marker in lowered for marker in direct_markers):
            return True
        if "live response canned template" in lowered or "canned template" in lowered:
            return True
        return False

    @classmethod
    def _postprocess_messages(cls, messages: list[str], brief: ReplyBrief) -> list[str]:
        if not messages:
            return messages
        messages = [cls._strip_wrapping_quotes(cls._clean_candidate_text(m)) for m in messages]
        messages = [m.replace(";", ",") for m in messages]
        messages = [cls._soften_label_colons(m) for m in messages]
        messages = [cls._soften_tracker_phrases(m) for m in messages]
        messages = cls._drop_scaffolding_preface(messages)
        messages = [m for m in messages if m and not cls._is_scaffolding_preface(m) and not cls._looks_hard_structured_leak(m)]
        if not messages:
            return messages
        first = messages[0].strip()
        user_norm = cls._normalize_text(brief.latest_user_message)
        first_norm = cls._normalize_text(first)
        is_first_echo = user_norm and first_norm and SequenceMatcher(a=first_norm, b=user_norm).ratio() >= 0.9
        if is_first_echo:
            if len(messages) > 1:
                return messages[1:]
            return []
        if brief.response_goal == "answer_question" and first.endswith("?") and len(messages) > 1:
            return messages[1:]
        return messages

    @classmethod
    def _drop_scaffolding_preface(cls, messages: list[str]) -> list[str]:
        if not messages:
            return messages
        first = messages[0].strip()
        if cls._is_scaffolding_preface(first):
            return messages[1:]
        return messages

    @staticmethod
    def _is_scaffolding_preface(text: str) -> bool:
        lowered = text.lower().strip().replace("’", "'").replace("`", "'")
        compact = re.sub(r"[^a-z0-9\s:]", " ", lowered)
        compact = " ".join(compact.split())
        if not compact:
            return False
        if re.fullmatch(r"here'?s the response:?", compact):
            return True
        if re.fullmatch(r"(response|reply):?", compact):
            return True
        return False

    @staticmethod
    def _strip_wrapping_quotes(text: str) -> str:
        stripped = text.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            return stripped[1:-1].strip()
        return stripped

    @staticmethod
    def _soften_label_colons(text: str) -> str:
        softened = text
        replacements = (
            (r"\bnext up:\s*", "and after that "),
            (r"\bnext move:\s*", "i'd start with "),
            (r"\bfor today:\s*", "for today, "),
            (r"\bfor tonight:\s*", "for tonight, "),
            (r"\bfor tomorrow morning:\s*", "for tomorrow morning, "),
            (r"\bfor tomorrow:\s*", "for tomorrow, "),
            (r"\bfor this week:\s*", "for this week, "),
            (r"\btoday plan:\s*", "today\n"),
            (r"\btonight plan:\s*", "tonight\n"),
            (r"\btomorrow morning plan:\s*", "tomorrow morning\n"),
            (r"\bthis week plan:\s*", "this week\n"),
            (r"\bweekend plan:\s*", "weekend\n"),
        )
        for pattern, replacement in replacements:
            softened = re.sub(pattern, replacement, softened, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", softened).strip()

    @staticmethod
    def _soften_tracker_phrases(text: str) -> str:
        softened = text
        replacements = (
            (r"\boff the board\b", "handled"),
            (r"\bon the board\b", "in the mix"),
            (r"\bon deck\b", "in the mix"),
            (r"\bgot (\d+) things from that(?: text)?\b", r"that's \1 things on your plate"),
            (r"\bright now i'm tracking\b", "i've got"),
            (r"\bnext up is\b", "if you can, hit"),
            (r"\bnext up\b", "after that"),
        )
        for pattern, replacement in replacements:
            softened = re.sub(pattern, replacement, softened, flags=re.IGNORECASE)
        softened = re.sub(r"\bi deleted ([^.?!]+)", r"i took \1 out", softened, flags=re.IGNORECASE)
        softened = re.sub(r"\bi archived ([^.?!]+)", r"i took \1 out", softened, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", softened).strip()

    @staticmethod
    def _clean_candidate_text(text: str) -> str:
        cleaned_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            stripped = re.sub(r"^(assistant|outbound|reply|response)\s*:\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"^here(?:'|’)s the response:?\s*$", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"^here is the actual response from me:?\s*$", "", stripped, flags=re.IGNORECASE)
            if stripped:
                cleaned_lines.append(stripped)
        collapsed = "\n".join(cleaned_lines).strip()
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed

    def _select_model(self, *, lightweight: bool) -> str | None:
        if lightweight and self._lightweight_compose_model:
            return self._lightweight_compose_model
        return self._compose_model

    @classmethod
    def _merge_tiny_lead_bubble(cls, messages: list[str], brief: ReplyBrief) -> list[str]:
        if len(messages) < 2:
            return messages
        if not brief.is_short_checkin:
            return messages
        first = messages[0].strip()
        if len(first) > 18 or len(first.split()) > 3:
            return messages
        merged = [f"{first} {messages[1].strip()}".strip()]
        if len(messages) > 2:
            merged.extend(messages[2:])
        return merged

    @classmethod
    def _has_nonsequitur_for_short_checkin(cls, messages: list[str], brief: ReplyBrief) -> bool:
        if not brief.is_short_checkin:
            return False
        combined = " ".join(messages).lower()
        if len(combined.split()) > 34:
            return True
        markers = [
            "you're right",
            "out of hand",
            "scattered",
            "still captured this",
            "as discussed",
            "checkpoint",
            "deadline",
            "task graph",
        ]
        return any(marker in combined for marker in markers)

    @classmethod
    def _has_scaffolding_preface(cls, messages: list[str]) -> bool:
        if not messages:
            return False
        return cls._is_scaffolding_preface(messages[0])

    @staticmethod
    def _is_overlong_for_goal(messages: list[str], brief: ReplyBrief) -> bool:
        if not messages:
            return False
        total_words = len(" ".join(messages).split())
        bubble_count = len(messages)
        max_words = 40
        if brief.is_short_checkin:
            max_words = 18
        elif brief.response_goal in {"open_conversation", "acknowledge_context", "confirm_update", "react_to_progress"}:
            max_words = 32
        elif brief.response_goal in {"acknowledge_new_task", "replan_blocker", "followup_on_slip", "ingestion_confirmation"}:
            max_words = 40
        elif brief.response_goal == "answer_question":
            max_words = 38
        elif brief.response_goal == "timeline_summary":
            max_words = 55
        if total_words > max_words:
            return True
        if brief.response_goal not in {"timeline_summary", "answer_question"} and bubble_count > 2:
            return True
        if brief.is_short_checkin and bubble_count > 1 and any(len(message.split()) > 8 for message in messages):
            return True
        return False

    @classmethod
    def _goal_alignment_needs_repair(cls, messages: list[str], brief: ReplyBrief) -> bool:
        if not messages:
            return True
        combined = " ".join(messages).lower().strip()
        if not combined:
            return True
        if "here are my thoughts:" in combined or re.search(r'\*\s*".+?"', combined):
            return True
        if re.search(r"\bstatus\s+(active|blocked|completed|archived)\b", combined):
            return True
        if re.search(r"\bpriority\s+[0-9]\b", combined):
            return True

        if brief.response_goal == "acknowledge_new_task":
            if not any(
                marker in combined
                for marker in (
                    "got it",
                    "bet",
                    "locked in",
                    "added",
                    "noted",
                    "tracking",
                    "right now i'm tracking",
                    "start with",
                    "i'd start with",
                )
            ):
                return True
            if "what's on your mind?" in combined:
                return True
            if re.search(r"^\s*[a-z_]+\s*:\s*[a-z_]+\s*:", combined):
                return True
            if combined.startswith("and "):
                return True
            if re.search(r"\bsubmit\s+i\s+submit\b", combined):
                return True
            if brief.should_push_for_action and brief.suggested_next_step and not any(
                marker in combined for marker in ("start with", "touch it next", "i'd start with", "if you could", "if you can")
            ):
                return True
            if not any(
                marker in combined
                for marker in ("tracking", "right now i'm tracking", "start with", "i'd start with", "got it", "bet", "locked in")
            ):
                return True

        if brief.response_goal == "confirm_update":
            if not any(
                marker in combined
                for marker in ("updated", "cleared", "noted", "applied", "done", "marked", "bet", "dropped", "took", "we're good", "is out", "clears it")
            ):
                return True
            if "under control" in combined:
                return True

        if brief.response_goal == "react_to_progress":
            if not any(
                marker in combined
                for marker in ("done", "good looks", "nice", "bet", "that clears", "huge", "if you could", "after that")
            ):
                return True

        if brief.response_goal == "replan_blocker":
            if not any(marker in combined for marker in ("blocker", "unblock", "unstick", "next move", "first", "shifts the plan", "clear")):
                return True

        if brief.response_goal == "timeline_summary":
            if not any(marker in combined for marker in ("today", "tonight", "tomorrow", "week", "due", "for the next hour")):
                return True
            if combined.count(":") > 2:
                return True

        if brief.response_goal == "open_conversation":
            if "status active" in combined:
                return True
            if brief.is_short_checkin:
                if len(combined.split()) > 18:
                    return True
                if any(token in combined for token in ["tomorrow", "deadline", "submit", "review", "application"]):
                    return True
                if cls._short_checkin_semantic_drift(response_text=combined, user_text=brief.latest_user_message):
                    return True
            lowered_user = brief.latest_user_message.lower()
            if "what i do" in combined and not any(token in lowered_user for token in ["what do", "what can", "are you", "do you"]):
                return True
            if "i'm here to help" in combined or "under control" in combined:
                return True

        if brief.should_push_for_action and brief.suggested_next_step:
            normalized_step = set(cls._normalize_text(brief.suggested_next_step).split())
            normalized_reply = set(cls._normalize_text(combined).split())
            shared_step_words = len(normalized_step.intersection(normalized_reply))
            action_markers = ("next move", "first", "start", "do ", "take ", "send ", "finish ", "knock out", "try ")
            if shared_step_words < 2 and not any(marker in combined for marker in action_markers):
                return True

        if brief.should_ask_question and brief.question_if_needed:
            if combined.count("?") > 1:
                return True

        return False

    @classmethod
    def _short_checkin_semantic_drift(cls, *, response_text: str, user_text: str) -> bool:
        lowered = response_text.lower()
        if any(token in lowered for token in ["routine", "operations", "company", "submission", "application"]):
            return True
        if lowered.startswith("why "):
            return True

        user_words = set(cls._normalize_text(user_text).split())
        response_words = cls._normalize_text(response_text).split()
        if not response_words:
            return True
        allowlist = {
            "yo",
            "hey",
            "hi",
            "sup",
            "whatup",
            "whatsup",
            "there",
            "here",
            "live",
            "online",
            "locked",
            "in",
            "got",
            "you",
            "im",
            "i",
            "m",
            "what",
            "s",
            "the",
            "move",
            "right",
            "now",
            "good",
            "all",
            "set",
        }
        novel = [word for word in response_words if word not in user_words and word not in allowlist]
        if len(response_words) <= 12 and len(novel) >= 4:
            return True
        if len(response_words) <= 20 and len(novel) >= 6:
            return True
        return False

    @staticmethod
    def _leading_overlap_words(text_a: str, text_b: str) -> int:
        words_a = text_a.split()
        words_b = text_b.split()
        overlap = 0
        for left, right in zip(words_a, words_b):
            if left != right:
                break
            overlap += 1
        return overlap

    @staticmethod
    def _lexical_overlap_ratio(text_a: str, text_b: str) -> float:
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        if not words_a or not words_b:
            return 0.0
        shared = words_a.intersection(words_b)
        return len(shared) / float(min(len(words_a), len(words_b)))

    @staticmethod
    def _has_repeated_long_phrase(text: str) -> bool:
        words = [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w]
        if len(words) < 10:
            return False
        for window in (6, 5):
            if len(words) < window * 2:
                continue
            for i in range(0, len(words) - (window * 2) + 1):
                if words[i : i + window] == words[i + window : i + (window * 2)]:
                    return True
        return False

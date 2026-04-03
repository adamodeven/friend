from __future__ import annotations

import re


class MessageChunker:
    def chunk(
        self,
        text: str,
        *,
        max_chunk_length: int,
        max_chunks: int,
        soft_chunk_length: int | None = None,
    ) -> list[str]:
        normalized = self._normalize(text)
        if not normalized:
            return []
        target_length = min(soft_chunk_length or max_chunk_length, max_chunk_length)
        if len(normalized) <= target_length:
            return [normalized]

        semantic_parts = self._semantic_parts(normalized)
        chunks: list[str] = []
        current_parts: list[str] = []
        current = ""
        for part in semantic_parts:
            if len(part) > max_chunk_length:
                oversized = self._split_oversized_part(part, max_chunk_length=max_chunk_length)
                if current:
                    chunks.append(current.strip())
                    current_parts = []
                    current = ""
                chunks.extend(oversized[:-1])
                current_parts = [oversized[-1]] if oversized else []
                current = current_parts[0] if current_parts else ""
                continue
            if not current:
                current_parts = [part]
                current = part
                continue
            next_len = len(current) + 1 + len(part)
            if next_len <= max_chunk_length and not self._should_soft_break(
                current,
                part,
                target_length=target_length,
                max_chunk_length=max_chunk_length,
            ):
                current_parts.append(part)
                current = " ".join(current_parts).strip()
            else:
                chunks.append(current.strip())
                current_parts = [part]
                current = part
        if current:
            chunks.append(current.strip())

        if len(chunks) > max_chunks:
            head = chunks[: max_chunks - 1]
            tail = " ".join(chunks[max_chunks - 1 :]).strip()
            merged = head + ([tail] if tail else [])
            return merged[:max_chunks]
        return chunks

    def normalize_messages(
        self,
        messages: list[str],
        *,
        max_chunk_length: int,
        max_chunks: int,
        soft_chunk_length: int | None = None,
    ) -> list[str]:
        if not messages:
            return []
        cleaned = [self._normalize(m) for m in messages if self._normalize(m)]
        if not cleaned:
            return []
        if len(cleaned) >= 2 and (len(cleaned[0]) <= 10 or len(cleaned[0].split()) <= 2):
            merged = f"{cleaned[0]} {cleaned[1]}".strip()
            cleaned = [merged, *cleaned[2:]]
        target_length = min(soft_chunk_length or max_chunk_length, max_chunk_length)
        if (
            len(cleaned) <= max_chunks
            and all(len(m) <= max_chunk_length for m in cleaned)
            and not (len(cleaned) == 1 and len(cleaned[0]) > target_length and max_chunks > 1)
        ):
            return cleaned
        return self.chunk(
            " ".join(cleaned),
            max_chunk_length=max_chunk_length,
            max_chunks=max_chunks,
            soft_chunk_length=soft_chunk_length,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.replace("\u2014", "-").split())

    @staticmethod
    def _semantic_parts(text: str) -> list[str]:
        # Prefer natural text-message cadence boundaries.
        blocks = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(blocks) > 1:
            return blocks
        cue_parts = [
            p.strip()
            for p in re.split(
                r"\s+(?=(?:next move:|for today:|for tonight:|for tomorrow:|for this week:|one thing:)\b)",
                text,
                flags=re.IGNORECASE,
            )
            if p.strip()
        ]
        if len(cue_parts) > 1:
            return cue_parts
        sentence_parts = [p.strip() for p in re.split(r"(?<=[.!?;,])\s+", text) if p.strip()]
        if len(sentence_parts) > 1:
            return sentence_parts
        # Handle run-on model output that lacks punctuation but has natural connector boundaries.
        connector_parts = [
            p.strip()
            for p in re.split(
                r"\s+(?=(?:but|so|then|also|anyway|now|first|next|after that)\b)",
                text,
                flags=re.IGNORECASE,
            )
            if p.strip()
        ]
        return connector_parts or [text.strip()]

    @staticmethod
    def _should_soft_break(current: str, next_part: str, *, target_length: int, max_chunk_length: int) -> bool:
        if len(current) + 1 + len(next_part) <= target_length:
            return False
        if len(current) >= max_chunk_length:
            return True
        if current.endswith((".", "!", "?", ";")):
            return True
        lowered_next = next_part.lower()
        return lowered_next.startswith(
            (
                "next move:",
                "for today:",
                "for tonight:",
                "for tomorrow:",
                "for this week:",
                "one thing:",
                "first",
                "then",
                "also",
            )
        )

    @staticmethod
    def _split_oversized_part(part: str, *, max_chunk_length: int) -> list[str]:
        words = part.split()
        if not words:
            return []
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word]).strip()
            if current and len(candidate) > max_chunk_length:
                chunks.append(" ".join(current).strip())
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

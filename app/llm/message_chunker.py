from __future__ import annotations

import re


class MessageChunker:
    def chunk(self, text: str, *, max_chunk_length: int, max_chunks: int) -> list[str]:
        normalized = self._normalize(text)
        if not normalized:
            return []
        if len(normalized) <= max_chunk_length:
            return [normalized]

        semantic_parts = self._semantic_parts(normalized)
        chunks: list[str] = []
        current = ""
        for part in semantic_parts:
            if not current:
                current = part
                continue
            if len(current) + 1 + len(part) <= max_chunk_length:
                current = f"{current} {part}"
            else:
                chunks.append(current.strip())
                current = part
        if current:
            chunks.append(current.strip())

        if len(chunks) > max_chunks:
            head = chunks[: max_chunks - 1]
            tail = " ".join(chunks[max_chunks - 1 :]).strip()
            merged = head + ([tail] if tail else [])
            return merged[:max_chunks]
        return chunks

    def normalize_messages(self, messages: list[str], *, max_chunk_length: int, max_chunks: int) -> list[str]:
        if not messages:
            return []
        cleaned = [self._normalize(m) for m in messages if self._normalize(m)]
        if not cleaned:
            return []
        if len(cleaned) <= max_chunks and all(len(m) <= max_chunk_length for m in cleaned):
            return cleaned
        return self.chunk(" ".join(cleaned), max_chunk_length=max_chunk_length, max_chunks=max_chunks)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.replace("\u2014", "-").split())

    @staticmethod
    def _semantic_parts(text: str) -> list[str]:
        # Prefer natural text-message cadence boundaries.
        blocks = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(blocks) > 1:
            return blocks
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


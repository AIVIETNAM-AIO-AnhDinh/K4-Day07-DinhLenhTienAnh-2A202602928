from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    # Split *after* a sentence-ending mark, keeping the mark with the sentence.
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in self._SENTENCE_BOUNDARY.split(text)]
        sentences = [s for s in sentences if s]
        if not sentences:
            return []

        size = self.max_sentences_per_chunk
        return [
            " ".join(sentences[start : start + size])
            for start in range(0, len(sentences), size)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: nothing left to split.
        if not current_text:
            return []
        # Base case 2: already small enough — keep it whole.
        if len(current_text) <= self.chunk_size:
            return [current_text]
        # Base case 3: out of separators (or the "" catch-all) — cut by length.
        if not remaining_separators or remaining_separators[0] == "":
            return self._hard_split(current_text)

        separator, rest = remaining_separators[0], remaining_separators[1:]
        pieces = current_text.split(separator)
        if len(pieces) == 1:
            # This separator does not occur — try the next, weaker one.
            return self._split(current_text, rest)

        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            candidate = piece if not buffer else buffer + separator + piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue
            if buffer:
                chunks.append(buffer)
                buffer = ""
            if len(piece) <= self.chunk_size:
                buffer = piece
            else:
                # Still too big on its own — recurse with a finer separator.
                chunks.extend(self._split(piece, rest))
        if buffer:
            chunks.append(buffer)
        return [c for c in chunks if c]

    def _hard_split(self, text: str) -> list[str]:
        size = max(1, self.chunk_size)
        return [text[i : i + size] for i in range(0, len(text), size)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(
                chunk_size=chunk_size, overlap=min(50, chunk_size // 10)
            ),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(c) for c in chunks]
            comparison[name] = {
                "chunks": chunks,
                "count": len(chunks),
                "avg_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
            }
        return comparison


class MarkdownSectionChunker:
    """
    Custom chunking strategy for the Markdown policy/notes corpus in ``data/``.

    Design rationale
    ----------------
    Every document in this corpus is Markdown built around ``#``/``##`` headings,
    and each heading marks one self-contained topic ("Chính sách đổi trả",
    "Điều kiện bảo hành", ...). Cutting by character count -- as FixedSizeChunker
    does -- happily slices the middle of a policy clause, so the retrieved chunk
    answers half a question. Splitting on heading boundaries instead keeps each
    answer whole.

    Two decisions matter for retrieval quality:

    1. **Heading breadcrumb is prepended to every chunk.** A body paragraph such
       as "Thời hạn là 7 ngày" is useless on its own -- the embedding has no clue
       what it is 7 days *of*. Prefixing "Chính sách đổi trả > Thời hạn" puts the
       topic words into the very vector being searched, which is what makes the
       chunk retrievable by a question phrased about that topic.
    2. **YAML frontmatter is stripped.** It is machine metadata (``source_url``,
       ``retrieved_at``); embedding it only adds noise that every document shares,
       pushing unrelated documents closer together.

    Sections longer than ``chunk_size`` are delegated to RecursiveChunker so the
    paragraph structure inside a long section is still respected, and the
    breadcrumb is re-attached to each resulting piece.
    """

    _FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
    _HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

    def __init__(self, chunk_size: int = 500, include_heading_path: bool = True) -> None:
        self.chunk_size = chunk_size
        self.include_heading_path = include_heading_path

    def _strip_frontmatter(self, text: str) -> str:
        return self._FRONTMATTER.sub("", text, count=1)

    def _breadcrumb(self, heading_stack: list[tuple[int, str]]) -> str:
        return " > ".join(title for _, title in heading_stack)

    def _emit(self, breadcrumb: str, body_lines: list[str]) -> list[str]:
        body = "\n".join(body_lines).strip()
        if not body and not breadcrumb:
            return []

        prefix = f"{breadcrumb}\n\n" if (breadcrumb and self.include_heading_path) else ""
        if not body:
            return []

        whole = prefix + body
        if len(whole) <= self.chunk_size:
            return [whole]

        # Section too long: split the body, keep the breadcrumb on every piece.
        budget = max(1, self.chunk_size - len(prefix))
        return [prefix + part for part in RecursiveChunker(chunk_size=budget).chunk(body)]

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        body_text = self._strip_frontmatter(text)
        heading_stack: list[tuple[int, str]] = []
        body_lines: list[str] = []
        chunks: list[str] = []

        for line in body_text.splitlines():
            match = self._HEADING.match(line)
            if not match:
                body_lines.append(line)
                continue

            # A new heading closes the section that was being collected.
            chunks.extend(self._emit(self._breadcrumb(heading_stack), body_lines))
            body_lines = []

            level, title = len(match.group(1)), match.group(2)
            # Pop headings at the same or deeper level to keep the path correct.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

        chunks.extend(self._emit(self._breadcrumb(heading_stack), body_lines))
        return chunks

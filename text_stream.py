"""
Incremental sentence assembly for streaming LLM output.

Feed token deltas as they arrive and complete sentences come out as soon as
their boundary is seen, so downstream TTS can start speaking the first
sentence while the LLM is still generating the rest.

A JavaScript port of ``SentenceAssembler`` lives in ``web/dashboard.html``;
keep the splitting rules in sync when changing them.
"""

import re
from typing import Iterator, List, Optional

# Words that commonly end with a period without ending a sentence.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "no", "vol", "approx",
})

# Candidate boundary: terminal punctuation, optional closing quotes/brackets,
# followed by whitespace. End-of-stream without trailing whitespace is
# handled by flush().
_BOUNDARY = re.compile(r'[.!?…]["\')\]]*(?=\s)')

# Token immediately before a period, used to detect false boundaries.
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*|\d+)$")

# Candidates shorter than this merge into the next sentence ("OK." alone
# makes for choppy speech).
_MIN_SENTENCE_CHARS = 12


class SentenceAssembler:
    """Accumulate streamed text deltas and emit complete sentences."""

    def __init__(self, min_chars: int = _MIN_SENTENCE_CHARS):
        self._buf = ""
        self._min_chars = min_chars

    def feed(self, delta: str) -> List[str]:
        """Append ``delta`` to the buffer; return sentences it completed."""
        self._buf += delta
        out: List[str] = []
        while True:
            end = self._next_boundary()
            if end is None:
                break
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> Optional[str]:
        """Return whatever remains at end of stream, or None if empty."""
        rest = self._buf.strip()
        self._buf = ""
        return rest or None

    def _next_boundary(self) -> Optional[int]:
        """Index just past the earliest valid sentence boundary, or None."""
        candidates = []
        for m in _BOUNDARY.finditer(self._buf):
            if m.end() < self._min_chars:
                continue
            if self._is_false_boundary(m.start()):
                continue
            candidates.append(m.end())
            break
        # A blank line is always a boundary (even for a short fragment —
        # the paragraph break marks a real pause).
        para = self._buf.find("\n\n")
        if para != -1 and self._buf[:para].strip():
            candidates.append(para + 2)
        return min(candidates) if candidates else None

    def _is_false_boundary(self, i: int) -> bool:
        """True if the period at ``i`` is an abbreviation/initial/list marker."""
        if self._buf[i] != ".":
            return False
        before = self._buf[:i]
        m = _LAST_WORD.search(before)
        if not m:
            return False
        word = m.group(1)
        if word.isdigit():
            # "1." at the start of a line is a list marker, not a sentence end
            return m.start() == 0 or before[m.start() - 1] == "\n"
        if word.lower() in _ABBREVIATIONS:
            return True
        # Initials and initialisms: "J. Smith", "U.S. economy"
        segments = word.split(".")
        return all(len(s) == 1 for s in segments if s)


def stream_sentences(deltas: Iterator[str]) -> Iterator[str]:
    """Wrap a token-delta iterator; yield sentences as they complete."""
    assembler = SentenceAssembler()
    for delta in deltas:
        yield from assembler.feed(delta)
    rest = assembler.flush()
    if rest:
        yield rest

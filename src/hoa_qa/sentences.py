"""Split a passage into numbered sentences the answer model can cite.

The model cites ``[3]`` and code copies sentence 3's text, so a quote can
never be mistyped and the model spends no output tokens retyping passages.
The split only has to be stable and lossless (every sentence is an exact
piece of the passage, in order); it does not have to be good grammar. HOA
documents are OCR'd legal text, so it leans toward fewer, longer sentences:
a wrong split in the middle of a rule is worse than a long quote.
"""

import re
from functools import lru_cache

# Longer than this and a "sentence" is cut at the last clause break before the
# limit: an OCR'd list can run for a whole page without a full stop.
MAX_SENTENCE_CHARS = 360
# A piece shorter than this is glued to the next one ("B.", "2.2 Noise.").
MIN_SENTENCE_CHARS = 30

_CANDIDATE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9§])")
# A full stop after one of these does not end a sentence: abbreviations, a
# single letter (initials, list letters), and list or section numbers.
_NOT_AN_END = re.compile(
    r"(?:\b(?:No|Nos|Sec|Art|Para|pp?|e\.g|i\.e|etc|Inc|Co|Corp|Ltd|Mr|Mrs|Ms|Dr|St"
    r"|vs|v|a\.m|p\.m|[A-Za-z])|\b\d{1,3}(?:\.\d{1,3})*)\.[\"')\]]*$"
)
_CLAUSE_BREAK = re.compile(r"[;:,]\s+|\s+-\s+|\s+")


def _cut_long(piece: str) -> list[str]:
    parts: list[str] = []
    while len(piece) > MAX_SENTENCE_CHARS:
        window = piece[:MAX_SENTENCE_CHARS]
        cut = 0
        for pattern in (r"[;:]\s+", r",\s+", r"\s+"):
            matches = list(re.finditer(pattern, window))
            # Not too early, or the remainder is no shorter.
            usable = [m for m in matches if m.end() > MAX_SENTENCE_CHARS // 3]
            if usable:
                cut = usable[-1].end()
                break
        if cut == 0:
            cut = MAX_SENTENCE_CHARS
        parts.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        parts.append(piece)
    return parts


@lru_cache(maxsize=4096)
def split_sentences(text: str) -> tuple[str, ...]:
    """The passage as sentences, numbered from 1 by position in the tuple.

    Every sentence is a verbatim piece of ``text`` (so it passes the quote
    check as-is) and none is empty. Joined with spaces they give back the
    passage with its whitespace collapsed, except where a run longer than
    MAX_SENTENCE_CHARS has no space to cut at.
    """
    text = " ".join(text.split())
    if not text:
        return ()
    pieces: list[str] = []
    start = 0
    for match in _CANDIDATE.finditer(text):
        head = text[start : match.start()]
        if _NOT_AN_END.search(head) or len(head) < MIN_SENTENCE_CHARS:
            continue
        pieces.append(head)
        start = match.end()
    tail = text[start:]
    if pieces and len(tail) < MIN_SENTENCE_CHARS:
        pieces[-1] = f"{pieces[-1]} {tail}"
    else:
        pieces.append(tail)
    return tuple(part for piece in pieces for part in _cut_long(piece))


def number_sentences(text: str) -> str:
    """The passage as the answer model sees it: one ``[n] sentence`` per line."""
    return "\n".join(
        f"[{number}] {sentence}"
        for number, sentence in enumerate(split_sentences(text), start=1)
    )

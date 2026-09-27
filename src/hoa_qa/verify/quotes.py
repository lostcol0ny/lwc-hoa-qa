"""Code-only citation check: the quote must really appear in the cited passage."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hoa_qa.answer.provider import DraftCitation
from hoa_qa.models import Chunk

_QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        # Models often retype these; folding them never joins separate words.
        "…": "...",
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
    }
)
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_SPACE = re.compile(r"\s*([;,:()\[\]{}/])\s*")


def normalize(text: str) -> str:
    """Fold typography and punctuation spacing without joining words or digits."""
    text = _WHITESPACE.sub(" ", text.translate(_QUOTE_MAP)).strip().casefold()
    return _PUNCTUATION_SPACE.sub(r"\1", text)


@dataclass(frozen=True)
class CheckedCitation:
    chunk: Chunk
    quote: str


def check_quotes(
    citations: Sequence[DraftCitation], passages: Mapping[str, Chunk]
) -> list[CheckedCitation]:
    """Keep citations whose chunk was provided and whose quote is in its text.

    Empty quotes are dropped (the empty string is a substring of everything),
    and repeated (chunk_id, quote) pairs are kept once.
    """
    kept: list[CheckedCitation] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        chunk = passages.get(citation.chunk_id)
        quote = normalize(citation.quote)
        if chunk is None or not quote or quote not in normalize(chunk.text_clean):
            continue
        if (chunk.id, quote) in seen:
            continue
        seen.add((chunk.id, quote))
        kept.append(CheckedCitation(chunk=chunk, quote=citation.quote.strip()))
    return kept

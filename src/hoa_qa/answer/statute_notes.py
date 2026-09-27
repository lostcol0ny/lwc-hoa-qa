"""Fixed text code adds to answers that cite Illinois law (addendum §6).

The model never decides whether these lines appear. Any answer citing a
statute gets ``STATUTE_DISCLAIMER``; one citing the Common Interest Community
Association Act also gets the applicability note, whose figures come from
configured corpus chunks and whose arithmetic is done here. If a configured
chunk is missing or no longer states its figure (the dues change, say), the
corpus build and the asker both fail rather than show a stale number.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from hoa_qa.models import Chunk, Citation, citation_url
from hoa_qa.verify.quotes import normalize

STATUTE_DISCLAIMER = (
    "This quotes Illinois law and is not legal advice. Whether a provision "
    "applies to your situation can depend on the facts; consult an attorney "
    "for advice."
)

CICAA_DOC_ID = "cicaa"
EXEMPTION_CHUNK_ID = "cicaa-1-75"
EXEMPTION_QUOTE = "annual budgeted assessments of $100,000 or less"


@dataclass(frozen=True)
class Evidence:
    """A figure the note uses, and the corpus chunk that states it."""

    chunk_id: str
    # Copied verbatim from the chunk; shown as the note's citation quote.
    quote: str
    # Extracts the figure from the quote.
    pattern: str

    def value(self, chunks: dict[str, Chunk]) -> int:
        chunk = chunks.get(self.chunk_id)
        if chunk is None:
            raise ValueError(f"applicability evidence {self.chunk_id} is missing")
        if normalize(self.quote) not in normalize(chunk.text_clean):
            raise ValueError(
                f"applicability evidence {self.chunk_id} no longer says "
                f"{self.quote!r}; update hoa_qa/answer/statute_notes.py"
            )
        match = re.search(self.pattern, self.quote)
        assert match is not None, "the pattern must match its own quote"
        return int(match[1].replace(",", ""))


HOMES = Evidence(
    chunk_id="amendments-committee-intro",
    quote="492 of 735 homes",
    pattern=r"of ([\d,]+) homes",
)
DUES = Evidence(
    chunk_id="home-1",
    quote="$452 year",
    pattern=r"\$([\d,]+) year",
)


@dataclass(frozen=True)
class ApplicabilityNote:
    text: str
    citations: tuple[Citation, ...]


def applicability_note(chunks: Iterable[Chunk]) -> ApplicabilityNote:
    """The CICAA applicability note, computed from the corpus.

    Raises ValueError if any evidence chunk, or the §1-75 exemption text, is
    missing or changed.
    """
    by_id = {chunk.id: chunk for chunk in chunks}
    homes = HOMES.value(by_id)
    dues = DUES.value(by_id)
    exemption = Evidence(EXEMPTION_CHUNK_ID, EXEMPTION_QUOTE, r"\$([\d,]+)")
    threshold = exemption.value(by_id)
    estimate = round(homes * dues, -3)
    if estimate <= threshold:
        # The wording below says "above that threshold"; never state it falsely.
        raise ValueError("estimated assessments no longer exceed the threshold")
    text = (
        "CICAA exempts associations with 10 or fewer units or annual budgeted "
        f"assessments of ${threshold:,} or less (765 ILCS 160/1-75). Lakewood "
        f"Creek's website lists {homes:,} homes and annual dues of ${dues:,}, "
        f"which suggests assessments of roughly ${estimate:,}, above that "
        "threshold. Confirm with the Board or an attorney."
    )
    citations = tuple(
        Citation(
            chunk_id=evidence.chunk_id,
            citation_label=by_id[evidence.chunk_id].citation_label,
            url=citation_url(by_id[evidence.chunk_id]),
            quote=evidence.quote,
        )
        for evidence in (exemption, HOMES, DUES)
    )
    return ApplicabilityNote(text=text, citations=citations)


def cites_cicaa(chunks: Iterable[Chunk]) -> bool:
    return any(chunk.doc_id == CICAA_DOC_ID for chunk in chunks)

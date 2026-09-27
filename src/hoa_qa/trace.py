"""Opt-in pipeline trace for eval diagnostics.

``QAAsker.ask_traced`` reports what each stage saw and decided to an
``AskTrace``: the gate score, every swept chunk's relevance, the passages sent
to the answer model, and each draft claim's verification. The production path
(``QAAsker.__call__``) passes ``NO_TRACE``, whose methods do nothing, so the
web app never records or logs any of it. Only the eval runner, whose
questions are committed fixtures, installs a recorder.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from hoa_qa.models import Authority, Chunk
from hoa_qa.retrieval.sweep import ScoredChunk


@dataclass(frozen=True)
class CitationTrace:
    chunk_id: str
    quote: str
    authority: Authority | None  # None: the chunk was not a provided passage
    quote_ok: bool  # the quote was found in a provided passage
    used: bool  # passed to the support check (and shown if the claim is kept)


@dataclass(frozen=True)
class ClaimTrace:
    statement: str
    kind: str
    essential: bool
    citations: tuple[CitationTrace, ...]
    support: float | None  # None: not support-checked
    kept: bool
    reason: str  # "ok" or why the claim failed


class AskTrace(Protocol):
    def gate(self, probability: float, passed: bool) -> None: ...

    def sweep(
        self, scores: Sequence[ScoredChunk], selected: Sequence[ScoredChunk]
    ) -> None: ...

    def passages(self, chunks: Sequence[Chunk]) -> None: ...

    def attempt(
        self,
        number: int,
        claims: Sequence[ClaimTrace] | None,
        note: str | None = None,
    ) -> None:
        """One answer-model attempt; ``claims`` is None for an invalid draft.

        ``note`` is the provider's parse note (see ``ProviderResult.note``).
        """
        ...


class NoTrace:
    """The default: records nothing."""

    def gate(self, probability: float, passed: bool) -> None:
        pass

    def sweep(
        self, scores: Sequence[ScoredChunk], selected: Sequence[ScoredChunk]
    ) -> None:
        pass

    def passages(self, chunks: Sequence[Chunk]) -> None:
        pass

    def attempt(
        self,
        number: int,
        claims: Sequence[ClaimTrace] | None,
        note: str | None = None,
    ) -> None:
        pass


NO_TRACE: AskTrace = NoTrace()

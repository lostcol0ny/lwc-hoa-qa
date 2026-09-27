"""A canned asker for local development (``HOA_QA_FAKE_ASKER=1``).

It never calls an AI service and is refused when ``VERCEL_ENV=production``.
"""

import uuid
from dataclasses import dataclass

from hoa_qa.models import Answer, Citation, Corpus, Outcome, citation_url
from hoa_qa.web.settings import DISCLAIMER


@dataclass
class FakeAskResult:
    # Not frozen: the wave-2 ``AskResult`` protocol declares plain attributes,
    # which pyright treats as settable, so a frozen class wouldn't match it.
    answer: Answer
    estimated_cost_usd: float


class FakeAsker:
    """Answers every question by citing the corpus's first chunk."""

    def __init__(self, corpus: Corpus, cost_usd: float = 0.001) -> None:
        self._corpus = corpus
        self._cost_usd = cost_usd

    async def __call__(self, question: str) -> FakeAskResult:
        citations: tuple[Citation, ...] = ()
        if self._corpus.chunks:
            chunk = self._corpus.chunks[0]
            citations = (
                Citation(
                    chunk_id=chunk.id,
                    citation_label=chunk.citation_label,
                    url=citation_url(chunk),
                    quote=chunk.text_clean[:120],
                ),
            )
        answer = Answer(
            request_id=uuid.uuid4().hex,
            outcome=Outcome.answered,
            answer_text=(
                "[Dev fake asker] This is a canned answer for local testing. "
                "No AI service was called."
            ),
            citations=citations,
            confidence=0.5,
            conflicts_noted=(),
            disclaimer=DISCLAIMER,
        )
        return FakeAskResult(answer=answer, estimated_cost_usd=self._cost_usd)

"""max_cost_usd is a provable bound: checked against 1-token-per-byte billing.

These fakes never use the pipeline's estimators. The Jev fake bills one token
per byte of the exact wire body the SDK sends (``pydantic_core.to_json``,
the SDK's own serializer); the answer fake bills one token per UTF-8 byte of
the prompt and the full ``max_tokens`` of output on every call.
"""

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic_core import to_json
from qa_fakes import FIXTURE

from hoa_qa.answer.prompt import AnswerPrompt
from hoa_qa.answer.provider import (
    AnswerDraft,
    DraftCitation,
    DraftClaim,
    ProviderResult,
)
from hoa_qa.ask import QAAsker, QASettings, build_asker
from hoa_qa.models import Chunk, Corpus, CorpusManifest, Outcome, load_corpus
from hoa_qa.retrieval.jev import (
    MAX_QUESTION_CHARS,
    WORST_QUESTION,
    NoulBatchResult,
    NoulQuestion,
    conservative_tokens,
    fits,
)
from hoa_qa.retrieval.sweep import SweepPassage, batch_request, plan_passages

ADVERSARIAL = "1234567890 !@#$%^&*() {}[]<>|~ 0.5% $1,000.00 "


def wire_bytes(state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]) -> int:
    body = {
        "state": state,
        "model": "jev-latest",
        "questions": {
            name: {"type": "noul", "instructions": q.instructions}
            | ({"criteria": {"true": q.yes, "false": q.no}} if q.yes or q.no else {})
            for name, q in questions.items()
        },
    }
    return len(to_json(body))


class ByteJev:
    """Always on-topic, everything relevant; claims with "exempt" unsupported."""

    def __init__(self) -> None:
        self.billed: list[int] = []
        self.support_calls = 0

    async def nouls(
        self, state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> NoulBatchResult:
        assert fits(state, questions)  # packing still respects the limits
        tokens = wire_bytes(state, questions)
        self.billed.append(tokens)
        probabilities: dict[str, float] = {}
        for name in questions:
            if name.startswith("c"):
                statement = state["claims"][int(name[1:])]["statement"]
                probabilities[name] = 0.05 if "exempt" in statement else 0.9
            else:
                probabilities[name] = 0.9
        if any(name.startswith("c") for name in questions):
            self.support_calls += 1
        return NoulBatchResult(probabilities, tokens)


class ByteProvider:
    """Cites the prompt's passages with maximal claims; one always fails."""

    model = "claude-haiku-4-5"

    def __init__(self, corpus: Corpus, max_tokens: int = 2048) -> None:
        self.max_tokens = max_tokens
        self._by_id = {c.id: c for c in corpus.chunks}
        self.calls = 0

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult:
        self.calls += 1
        ids = re.findall(r'<passage chunk_id="([^"]+)"', prompt.user)
        claims = []
        for i in range(8):
            cited = [ids[(i + j) % len(ids)] for j in range(3)]
            statement = f"{i} " + "\U0001d538" * 380
            if i == 7:
                statement = "Everyone is exempt. " + "\U0001d538" * 370
            claims.append(
                DraftClaim(
                    statement=statement,
                    kind="answer",
                    essential=i != 7,
                    citations=[
                        DraftCitation(chunk_id=c, quote=self._by_id[c].text_clean[:30])
                        for c in cited
                    ],
                )
            )
        draft = AnswerDraft(claims=claims, confidence=0.5, refer_to_board=True)
        billed_in = len((prompt.system + prompt.user).encode("utf-8"))
        return ProviderResult(draft, self.model, billed_in, self.max_tokens)


# --- corpora ----------------------------------------------------------------


def chunk(doc: str, i: int, text: str, heading: str = "Section") -> Chunk:
    return Chunk.model_validate(
        {
            "id": f"{doc}-{i}",
            "doc_id": doc,
            "doc_title": f"Document {doc}",
            "source_url": f"https://example.org/{doc}.pdf",
            "page_start": 1,
            "page_end": 1,
            "citation_label": f"{doc} §{i}",
            "heading_path": [heading],
            "text_clean": text,
            "text_raw": text,
            "authority": "rules",
            "effective_date": "2023-01-01",
            "published_date": None,
            "superseded_by": None,
            "token_estimate": 1,
        }
    )


def corpus_of(chunks: Sequence[Chunk]) -> Corpus:
    return Corpus(
        manifest=CorpusManifest(
            build_time="2026-09-27T00:00:00Z",  # type: ignore[arg-type]
            source_hashes={c.doc_id: "0" * 64 for c in chunks},
            chunk_count=len(chunks),
            ocr_fallbacks=(),
        ),
        chunks=tuple(chunks),
    )


def synthetic_74k() -> Corpus:
    """~74K bytes (= tokens under the byte bound), digit/symbol heavy."""
    chunks = []
    for d in range(20):
        for i in range(12):
            body = (ADVERSARIAL * 6)[: 250 + 7 * i] + f" Pool rule {d}.{i} é ✓"
            chunks.append(chunk(f"doc{d:02d}", i, body))
    corpus = corpus_of(chunks)
    total = sum(len(c.text_clean.encode()) for c in corpus.chunks)
    assert 70_000 < total < 80_000
    return corpus


def largest_fitting_length() -> int:
    """Longest adversarial text that fits one request with the longest question."""
    lo, hi = 1, 400_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        probe = SweepPassage(chunk("b", 0, (ADVERSARIAL * 10_000)[:mid]), "")
        probe = SweepPassage(probe.chunk, probe.chunk.text_clean)
        if fits(*batch_request(WORST_QUESTION, [probe])):
            lo = mid
        else:
            hi = mid - 1
    return lo


def boundary_corpus() -> Corpus:
    n = largest_fitting_length()
    text = ADVERSARIAL * 20_000
    chunks = [
        chunk("big", 0, text[: n - 10]),  # just fits: sent whole
        chunk("big", 1, text[: n + 10]),  # just over: split in two
        chunk("big", 2, text[: 3 * n]),  # split into several parts
        chunk("big", 3, "Short pool rule."),
        # A heading so large no piece can fit: skipped for every question.
        chunk("skip", 0, "Tiny text.", heading="H" * 200_000),
        *synthetic_74k().chunks[:24],
    ]
    return corpus_of(chunks)


CORPORA = {
    "mini": lambda: load_corpus(FIXTURE),
    "synthetic-74k": synthetic_74k,
    "boundaries": boundary_corpus,
}
QUESTIONS = {
    "short": "Pool rules?",
    "max-length": "\U0001d538" * MAX_QUESTION_CHARS,
}


def test_heuristic_undercounts_adversarial_text() -> None:
    text = ADVERSARIAL * 100
    assert conservative_tokens(text) < len(text.encode()) / 2


def test_boundary_corpus_exercises_split_and_skip() -> None:
    plan = plan_passages(boundary_corpus().chunks)
    parts = {}
    for p in plan.passages:
        parts[p.chunk.id] = p.parts
    assert parts["big-0"] == 1
    assert parts["big-1"] == 2
    assert parts["big-2"] >= 3
    assert plan.skipped == ("skip-0",)


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("corpus_name", CORPORA)
@pytest.mark.parametrize("max_tokens", [1024, 2048, 4096])
def test_max_cost_bounds_byte_billing(
    corpus_name: str, question: str, max_tokens: int
) -> None:
    corpus = CORPORA[corpus_name]()
    jev = ByteJev()
    provider = ByteProvider(corpus, max_tokens)
    asker: QAAsker = build_asker(corpus, QASettings(), jev=jev, provider=provider)
    result = asyncio.run(asker(QUESTIONS[question]))
    # The worst path ran: both answer attempts. In the boundary corpus the
    # claims cite chunks too large for any support request, so they fail
    # closed (not_found); elsewhere support runs on both attempts.
    assert provider.calls == 2
    if corpus_name != "boundaries":
        assert result.answer.outcome is Outcome.answered
        assert jev.support_calls >= 2
    assert asker.max_cost_usd >= result.estimated_cost_usd > 0

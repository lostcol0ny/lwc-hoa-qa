"""Per-claim support check: batching, bounded concurrency, and failures."""

import asyncio

import pytest
from qa_fakes import FakeJev

from hoa_qa.models import Chunk, Corpus
from hoa_qa.retrieval.jev import JevLimits, PartialCostError, fits
from hoa_qa.verify.quotes import CheckedCitation
from hoa_qa.verify.support import ClaimEvidence, check_support, support_request


def evidence(corpus: Corpus, statement: str, *chunk_ids: str) -> ClaimEvidence:
    by_id = {c.id: c for c in corpus.chunks}
    return ClaimEvidence(
        statement=statement,
        citations=tuple(CheckedCitation(chunk=by_id[i], quote="q") for i in chunk_ids),
    )


def run(jev: FakeJev, claims, concurrency: int = 4, **kw):
    async def go():
        return await check_support(
            jev,
            claims,
            threshold=0.5,
            limit=asyncio.Semaphore(concurrency),
            **kw,
        )

    return asyncio.run(go())


def test_one_noul_per_claim_with_distinct_passages(
    corpus: Corpus, jev: FakeJev
) -> None:
    jev.support = lambda statement, ids: 0.9 if "$125" in statement else 0.2
    claims = [
        evidence(
            corpus, "Second violation is $125.", "rules-2023-fines", "rules-2023-fines"
        ),
        evidence(corpus, "Seniors are exempt.", "rules-2023-fines"),
        evidence(corpus, "No citations survived."),
    ]
    result = run(jev, claims)
    assert result.supported == (True, False, False)
    [(state, questions)] = jev.calls
    # Claim without citations is not sent; duplicate chunk passages collapse.
    assert len(state["claims"]) == 2 and set(questions) == {"c0", "c1"}
    assert len(state["claims"][0]["passages"]) == 1
    assert result.input_tokens == sum(jev.billed)


def test_empty_input_makes_no_calls(jev: FakeJev) -> None:
    result = run(jev, [])
    assert result.supported == () and jev.calls == []


def _big_claims(corpus: Corpus, n: int) -> list[ClaimEvidence]:
    return [
        evidence(corpus, f"Claim {i} about fines " + "x" * 300, "rules-2023-fines")
        for i in range(n)
    ]


def test_multiple_support_batches_respect_limits_and_concurrency(
    corpus: Corpus,
) -> None:
    """B4: several packed support batches never exceed the shared bound."""
    active = peak = 0

    class SlowJev(FakeJev):
        async def nouls(self, state, questions):  # type: ignore[override]
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return await super().nouls(state, questions)

    jev = SlowJev(text_to_id={c.text_clean: c.id for c in corpus.chunks})
    limits = JevLimits(max_request_tokens=1_000, max_state_tokens=1_000)
    result = run(jev, _big_claims(corpus, 8), concurrency=2, limits=limits)
    assert result.requests >= 4
    assert len(jev.calls) == result.requests
    assert peak == 2
    for state, questions in jev.calls:
        assert fits(state, questions, limits=limits)
    assert all(result.supported)


def test_oversized_claim_fails_closed_without_a_call(
    corpus: Corpus, jev: FakeJev, caplog: pytest.LogCaptureFixture
) -> None:
    limits = JevLimits(max_request_tokens=800, max_state_tokens=800)
    big = evidence(corpus, "Big " + "y" * 3_000, "rules-2023-fines")
    small = evidence(corpus, "Small.", "rules-2023-fines")
    result = run(jev, [big, small], limits=limits)
    assert result.supported == (False, True)
    assert all(len(state["claims"]) == 1 for state, _ in jev.calls)
    assert "index=0" in caplog.text and "yyyy" not in caplog.text


def test_failed_support_batch_keeps_sibling_cost(corpus: Corpus, jev: FakeJev) -> None:
    """B2: one failed batch; the rest finish and their tokens are reported."""
    limits = JevLimits(max_request_tokens=1_000, max_state_tokens=1_000)
    jev.fail = lambda state: state["claims"][0]["statement"].startswith("Claim 0 ")
    with pytest.raises(PartialCostError) as info:
        run(jev, _big_claims(corpus, 8), limits=limits)
    assert len(jev.calls) >= 4
    assert info.value.jev_input_tokens == sum(jev.billed) > 0


def test_request_shape(corpus: Corpus) -> None:
    chunk: Chunk = next(c for c in corpus.chunks if c.id == "rules-2023-fines")
    state, questions = support_request(
        [ClaimEvidence("S", (CheckedCitation(chunk, "q"),))]
    )
    assert state == {"claims": [{"statement": "S", "passages": [chunk.text_clean]}]}
    assert "support this specific claim" in questions["c0"].instructions

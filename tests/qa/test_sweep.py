"""Sweep batching per doc_id, Jev limit guarantees, failures, and top-k."""

import asyncio
from collections.abc import Sequence

import pytest
from qa_fakes import FakeJev

from hoa_qa.models import Chunk, Corpus
from hoa_qa.retrieval.jev import (
    QUESTION_OVERHEAD_TOKENS,
    REQUEST_OVERHEAD_TOKENS,
    JevLimits,
    NoulQuestion,
    PartialCostError,
    conservative_tokens,
    fits,
    pack,
    request_tokens,
)
from hoa_qa.retrieval.sweep import SweepPassage, batch_request, plan_batches, sweep

Q = "What are the fines?"


def run_sweep(jev: FakeJev, chunks: Sequence[Chunk], concurrency: int = 2, **kw):
    options = {"top_k": 8, "threshold": 0.3} | kw

    async def go():
        return await sweep(
            jev, Q, chunks, limit=asyncio.Semaphore(concurrency), **options
        )

    return asyncio.run(go())


def make_chunk(i: int, doc: str, text: str) -> Chunk:
    return Chunk.model_validate(
        {
            "id": f"{doc}-{i}",
            "doc_id": doc,
            "doc_title": doc,
            "source_url": "https://example.org/x",
            "page_start": None,
            "page_end": None,
            "citation_label": f"{doc} {i}",
            "heading_path": [],
            "text_clean": text,
            "text_raw": text,
            "authority": "rules",
            "effective_date": "2023-01-01",
            "published_date": None,
            "superseded_by": None,
            "token_estimate": 1,
        }
    )


def words(i: int, doc: str, n: int) -> Chunk:
    return make_chunk(i, doc, " ".join(["word"] * n) + f" #{i}")


def word_tokens(text: str) -> int:
    """Fake tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def sizes(batch: Sequence[SweepPassage], counter=word_tokens) -> tuple[int, int]:
    return request_tokens(*batch_request(Q, batch), counter)


# --- token bound ------------------------------------------------------------


def test_conservative_tokens() -> None:
    assert conservative_tokens("") == 0
    assert conservative_tokens("abcde") == 2  # ASCII: ceil(5 / 2.5)
    assert conservative_tokens("é") == 2  # non-ASCII: one per UTF-8 byte
    assert conservative_tokens("\U0001d538") == 4


def test_request_tokens_include_overheads() -> None:
    q = {"a": NoulQuestion("x")}
    total, state_plus = request_tokens({}, q, word_tokens)
    # "{}" and "ax" are one word each
    assert total == REQUEST_OVERHEAD_TOKENS + 1 + QUESTION_OVERHEAD_TOKENS + 1
    assert state_plus == total


# --- batching per document --------------------------------------------------


def test_one_request_per_document(corpus: Corpus, jev: FakeJev) -> None:
    result = run_sweep(jev, corpus.chunks)
    sweep_calls = [c for c in jev.calls if "on_topic" not in c[1]]
    assert len(sweep_calls) == len({c.doc_id for c in corpus.chunks}) == 7
    assert result.requests == 7
    for state, questions in sweep_calls:
        docs = {
            next(c.doc_id for c in corpus.chunks if c.text_clean == p["text"])
            for p in state["passages"]
        }
        assert len(docs) == 1
        assert len(questions) == len(state["passages"])
        assert state["question"] == Q
    # rules-2023 has two chunks, judged together in one request.
    assert sorted(len(q) for _, q in sweep_calls) == [1, 1, 1, 1, 1, 1, 2]
    assert result.input_tokens == sum(jev.billed)


def test_question_wording_references_each_passage() -> None:
    passages = [
        SweepPassage(make_chunk(i, "d", f"text {i}"), f"text {i}") for i in range(3)
    ]
    state, questions = batch_request(Q, passages)
    assert [p["text"] for p in state["passages"]] == ["text 0", "text 1", "text 2"]
    assert "`passages[2]`" in questions["p2"].instructions
    assert "help answer the question" in questions["p2"].instructions


# --- the two Jev limits, independently ---------------------------------------


def test_total_limit_binds_independently() -> None:
    """Many tiny passages: questions dominate, so only the 64K-style total binds."""
    chunks = [words(i, "d", 1) for i in range(40)]
    limits = JevLimits(max_request_tokens=800, max_state_tokens=100_000)
    plan = plan_batches(Q, chunks, word_tokens, limits)
    assert len(plan.batches) > 1 and not plan.skipped
    for batch in plan.batches:
        total, state_plus = sizes(batch)
        assert total <= 800
    # With only the state limit loose, merging any two batches breaks the total.
    merged = plan.batches[0] + plan.batches[1]
    assert sizes(merged)[0] > 800


def test_state_limit_binds_independently() -> None:
    """Few large passages: the 32K-style state+longest limit binds first."""
    chunks = [words(i, "d", 100) for i in range(4)]
    one = sizes([SweepPassage(chunks[0], chunks[0].text_clean)])[1]
    limits = JevLimits(max_request_tokens=100_000, max_state_tokens=one + 20)
    plan = plan_batches(Q, chunks, word_tokens, limits)
    assert [len(b) for b in plan.batches] == [1, 1, 1, 1]
    for batch in plan.batches:
        total, state_plus = sizes(batch)
        assert state_plus <= limits.max_state_tokens
        assert total <= 100_000
    merged = plan.batches[0] + plan.batches[1]
    assert sizes(merged)[1] > limits.max_state_tokens


def test_large_document_is_split_under_both_limits() -> None:
    chunks = [words(i, "big", 100) for i in range(10)] + [words(0, "small", 5)]
    limits = JevLimits(max_request_tokens=700, max_state_tokens=500)
    plan = plan_batches(Q, chunks, word_tokens, limits)
    assert [p.chunk.doc_id for p in plan.batches[-1]] == ["small"]
    big = [b for b in plan.batches if b[0].chunk.doc_id == "big"]
    assert len(big) > 1
    assert [p.chunk.id for b in big for p in b] == [c.id for c in chunks[:10]]
    for batch in plan.batches:
        assert fits(*batch_request(Q, batch), word_tokens, limits)


def test_oversized_singleton_is_split_into_parts(corpus: Corpus) -> None:
    huge = words(1, "d", 1_000)
    chunks = [words(0, "d", 5), huge, words(2, "d", 5)]
    limits = JevLimits(max_request_tokens=600, max_state_tokens=500)
    plan = plan_batches(Q, chunks, word_tokens, limits)
    assert not plan.skipped
    parts = [p for b in plan.batches for p in b if p.chunk.id == "d-1"]
    assert len(parts) > 1
    assert all(p.parts == len(parts) for p in parts)
    # Parts reassemble to the original text, and every request fits.
    assert "".join(p.text for p in parts) == huge.text_clean
    for batch in plan.batches:
        assert fits(*batch_request(Q, batch), word_tokens, limits)


def test_unsplittable_singleton_is_skipped_not_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # One 400-character "word": halving stops at MIN_SPLIT_CHARS pieces that
    # still exceed a limit this tight, so the chunk is skipped.
    chunk = make_chunk(0, "d", "x" * 400)
    limits = JevLimits(max_request_tokens=300, max_state_tokens=290)
    plan = plan_batches(Q, [chunk], lambda t: len(t), limits)
    assert plan.batches == [] and plan.skipped == ["d-0"]
    assert "chunk_id=d-0" in caplog.text
    assert "x" * 50 not in caplog.text


def test_oversized_real_chunk_is_split_and_scored_by_best_part(
    corpus: Corpus,
) -> None:
    # A ~100K-token chunk under the real limits and conservative bound.
    big_text = " ".join(f"sentence {i} about pool hours." for i in range(12_000))
    big = make_chunk(0, "huge", big_text)
    jev = FakeJev(text_to_id={})

    async def nouls(state, questions):
        jev.calls.append((dict(state), dict(questions)))
        assert fits(state, questions)
        probs = {
            n: (
                0.9
                if "sentence 11999 " in state["passages"][int(n[1:])]["text"]
                else 0.1
            )
            for n in questions
        }
        from hoa_qa.retrieval.jev import NoulBatchResult

        return NoulBatchResult(probs, 1)

    jev.nouls = nouls  # type: ignore[method-assign]
    result = run_sweep(jev, [big])
    assert result.requests > 1
    assert [s.chunk.id for s in result.selected] == ["huge-0"]
    assert result.selected[0].probability == 0.9


def test_real_corpus_fits_default_limits(corpus: Corpus) -> None:
    plan = plan_batches(Q, corpus.chunks)
    assert not plan.skipped
    for batch in plan.batches:
        assert fits(*batch_request(Q, batch))


def test_pack_returns_oversized_separately() -> None:
    def build(group):
        return {"xs": " ".join(group)}, {}

    groups, oversized = pack(
        ["a", "b", "big " * 50, "c"],
        build,
        word_tokens,
        JevLimits(
            max_request_tokens=10_000, max_state_tokens=REQUEST_OVERHEAD_TOKENS + 10
        ),
    )
    assert oversized == ["big " * 50]
    assert all(len(g) >= 1 for g in groups)
    assert [x for g in groups for x in g] == ["a", "b", "c"]


# --- failures ----------------------------------------------------------------


def test_failed_batch_keeps_sibling_cost_and_finishes_all(
    corpus: Corpus, jev: FakeJev
) -> None:
    jev.fail = lambda state: state.get("document") == "LWC Bylaws"
    with pytest.raises(PartialCostError) as info:
        run_sweep(jev, corpus.chunks)
    # All 7 requests ran; the 6 successes were billed and reported.
    assert len(jev.calls) == 7
    assert info.value.jev_input_tokens == sum(jev.billed) > 0
    assert len(jev.billed) == 6
    assert [type(e).__name__ for e in info.value.errors] == ["RuntimeError"]


def test_all_batches_failing_reports_zero_cost(corpus: Corpus, jev: FakeJev) -> None:
    jev.fail = lambda state: True
    with pytest.raises(PartialCostError) as info:
        run_sweep(jev, corpus.chunks)
    assert info.value.jev_input_tokens == 0
    assert len(info.value.errors) == 7


# --- selection ---------------------------------------------------------------


def test_threshold_and_top_k(corpus: Corpus, jev: FakeJev) -> None:
    jev.relevance = {
        "rules-2023-fines": 0.9,
        "rules-2016-fines": 0.7,
        "blog-2022-violations": 0.5,
        "bylaws-3.4": 0.3,  # exactly at threshold: kept
        "rules-2023-2.1": 0.29,  # just below: dropped
    }
    result = run_sweep(jev, corpus.chunks)
    assert [s.chunk.id for s in result.selected] == [
        "rules-2023-fines",
        "rules-2016-fines",
        "blog-2022-violations",
        "bylaws-3.4",
    ]
    top2 = run_sweep(jev, corpus.chunks, top_k=2)
    assert [s.chunk.id for s in top2.selected] == [
        "rules-2023-fines",
        "rules-2016-fines",
    ]


def test_ties_keep_corpus_order(corpus: Corpus, jev: FakeJev) -> None:
    jev.relevance = {c.id: 0.5 for c in corpus.chunks}
    result = run_sweep(jev, corpus.chunks)
    assert [s.chunk.id for s in result.selected] == [c.id for c in corpus.chunks]


def test_concurrency_is_bounded(corpus: Corpus) -> None:
    active = peak = 0

    class SlowJev(FakeJev):
        async def nouls(self, state, questions):  # type: ignore[override]
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return await super().nouls(state, questions)

    slow = SlowJev(text_to_id={c.text_clean: c.id for c in corpus.chunks})
    run_sweep(slow, corpus.chunks, concurrency=2)
    assert peak == 2

"""Sweep batching per doc_id, the Jev token-limit split, and top-k/threshold."""

import asyncio
from collections.abc import Sequence

from qa_fakes import FakeJev

from hoa_qa.models import Chunk, Corpus
from hoa_qa.retrieval.jev import estimate_tokens, fits, pack, request_tokens
from hoa_qa.retrieval.sweep import batch_request, plan_batches, sweep

Q = "What are the fines?"


def run_sweep(jev: FakeJev, chunks: Sequence[Chunk], **kwargs):
    options = {"top_k": 8, "threshold": 0.3, "concurrency": 2} | kwargs
    return asyncio.run(sweep(jev, Q, chunks, **options))


def test_one_request_per_document(corpus: Corpus, jev: FakeJev) -> None:
    result = run_sweep(jev, corpus.chunks)
    doc_ids = {c.doc_id for c in corpus.chunks}
    sweep_calls = jev.calls_of("p")
    assert len(sweep_calls) == len(doc_ids) == result.requests == 7
    for state, questions in (c for c in jev.calls if "on_topic" not in c[1]):
        docs = {
            next(c.doc_id for c in corpus.chunks if c.text_clean == p["text"])
            for p in state["passages"]
        }
        assert len(docs) == 1
        assert len(questions) == len(state["passages"])
        assert state["question"] == Q
    # rules-2023 has two chunks, judged together in one request.
    assert sorted(len(q) for q in sweep_calls) == [1, 1, 1, 1, 1, 1, 2]
    assert result.input_tokens == 7 * 1_000


def test_question_wording_references_each_passage() -> None:
    chunks = [
        Chunk.model_validate(
            {
                "id": f"d-{i}",
                "doc_id": "d",
                "doc_title": "D",
                "source_url": "https://example.org/d",
                "page_start": None,
                "page_end": None,
                "citation_label": f"D {i}",
                "heading_path": [],
                "text_clean": f"text {i}",
                "text_raw": f"text {i}",
                "authority": "rules",
                "effective_date": "2023-01-01",
                "published_date": None,
                "superseded_by": None,
                "token_estimate": 2,
            }
        )
        for i in range(3)
    ]
    state, questions = batch_request(Q, chunks)
    assert [p["text"] for p in state["passages"]] == ["text 0", "text 1", "text 2"]
    assert "`passages[2]`" in questions["p2"].instructions
    assert "help answer the question" in questions["p2"].instructions


def _chunk(i: int, doc: str, words: int) -> Chunk:
    text = " ".join(["word"] * words) + f" #{i}"
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
            "token_estimate": words,
        }
    )


def word_tokens(text: str) -> int:
    """Fake tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def test_large_document_is_split_under_the_limits() -> None:
    chunks = [_chunk(i, "big", 100) for i in range(10)] + [_chunk(0, "small", 5)]
    batches = plan_batches(
        Q, chunks, word_tokens, max_request_tokens=450, max_state_tokens=350
    )
    assert [c.doc_id for c in batches[-1]] == ["small"]
    big = [b for b in batches if b[0].doc_id == "big"]
    assert len(big) > 1
    assert [c.id for b in big for c in b] == [c.id for c in chunks[:10]]
    for batch in batches:
        assert fits(
            *batch_request(Q, batch),
            word_tokens,
            max_request_tokens=450,
            max_state_tokens=350,
        )
        total, longest = request_tokens(*batch_request(Q, batch), word_tokens)
        assert total <= 450 and longest <= 350


def test_state_limit_binds_even_when_total_fits() -> None:
    chunks = [_chunk(i, "big", 100) for i in range(4)]
    loose_total = plan_batches(
        Q, chunks, word_tokens, max_request_tokens=10_000, max_state_tokens=150
    )
    assert len(loose_total) == 4


def test_oversized_single_chunk_gets_its_own_batch() -> None:
    chunks = [_chunk(0, "d", 5), _chunk(1, "d", 1_000), _chunk(2, "d", 5)]
    batches = plan_batches(
        Q, chunks, word_tokens, max_request_tokens=500, max_state_tokens=400
    )
    assert [[c.id for c in b] for b in batches] == [["d-0"], ["d-1"], ["d-2"]]


def test_real_corpus_fits_default_limits(corpus: Corpus) -> None:
    for batch in plan_batches(Q, corpus.chunks):
        assert fits(*batch_request(Q, batch), estimate_tokens)


def test_pack_is_generic() -> None:
    groups = pack(
        list(range(5)),
        lambda g: ({"xs": " ".join("aaaa" for _ in g)}, {}),
        max_request_tokens=100,
        max_state_tokens=5,
    )
    assert [len(g) for g in groups] == [2, 2, 1]


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

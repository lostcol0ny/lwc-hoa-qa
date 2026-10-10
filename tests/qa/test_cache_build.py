"""Building the cache: generated answers, reviewed claims, and the proposal file."""

import asyncio
from pathlib import Path

import pytest
import yaml
from qa_fakes import FINES_DRAFT, FIXTURE, FakeJev, FakeProvider

from hoa_qa.ask import AskResult, QAAsker, QASettings, build_asker
from hoa_qa.cache import CuratedQuestions, load_cache, load_questions
from hoa_qa.cache_build import build_entries, with_claims
from hoa_qa.cli import main
from hoa_qa.models import Answer, Corpus, Outcome, load_corpus

FINE_QUESTION = "What is the fine for a third violation?"
FINE = {
    "statement": "The fine for a third violation is $125.",
    "citations": [{"chunk_id": "rules-2023-fines", "quote": "3rd violation: $125."}],
}
OLD_FINE = {
    "statement": "The superseded 2016 rules set the third fine at $100.",
    "kind": "conflict",
    "citations": [{"chunk_id": "rules-2016-fines", "quote": "3rd offense: $100."}],
}


def questions(*claims: dict, question: str = FINE_QUESTION) -> CuratedQuestions:
    return CuratedQuestions.model_validate(
        {"questions": [{"id": "fine-third", "question": question, "claims": claims}]}
    )


def build(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    curated: CuratedQuestions,
    provider: FakeProvider | None = None,
):
    asker = build_asker(
        corpus, settings, jev=jev, provider=provider or FakeProvider([FINES_DRAFT])
    )
    return asyncio.run(build_entries(curated, asker))


def test_reviewed_claims_are_stored_word_for_word_without_the_answer_model(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([FINES_DRAFT])
    entries, rows = build(corpus, settings, jev, questions(FINE, OLD_FINE), provider)
    assert provider.prompts == [], "reviewed answers never reach the answer model"
    [entry] = entries
    assert entry.answer.answer_text == "The fine for a third violation is $125."
    assert entry.answer.conflicts_noted == (
        "The superseded 2016 rules set the third fine at $100.",
    )
    assert [c.chunk_id for c in entry.answer.citations] == [
        "rules-2023-fines",
        "rules-2016-fines",
    ]
    [row] = rows
    assert (row.source, row.cached, row.problems) == ("reviewed", True, ())
    # Verified like any other claim: the quote check in code, support by Jev.
    assert len(jev.calls_of("c")) == 1 and jev.calls_of("gate") == []


def test_reviewed_claim_may_cite_a_passage_the_sweep_did_not_select(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    dues = {
        "statement": "The 2026 assessment is $452 per year.",
        "citations": [
            {"chunk_id": "website-2026-dues", "quote": "$452 year or $113 per quarter"}
        ],
    }
    assert "website-2026-dues" not in jev.relevance
    entries, _ = build(corpus, settings, jev, questions(dues, question="Dues?"))
    assert [e.answer.answer_text for e in entries] == [dues["statement"]]


@pytest.mark.parametrize(
    ("claim", "support", "problem"),
    [
        (
            {
                **FINE,
                "citations": [
                    {"chunk_id": "rules-2023-fines", "quote": "4th violation: $999."}
                ],
            },
            0.9,
            "claim 1: no_valid_quote",
        ),
        (
            {
                **FINE,
                "citations": [
                    {"chunk_id": "no-such-chunk", "quote": "3rd violation: $125."}
                ],
            },
            0.9,
            "claim 1: no_valid_quote",
        ),
        (FINE, 0.2, "claim 1: unsupported (support 0.20)"),
        (
            # An informal post can't answer next to the current Rules.
            {
                "statement": "The fine for a third offense is $100.",
                "citations": [
                    {
                        "chunk_id": "blog-2022-violations",
                        "quote": "3rd offense - $100.00",
                    }
                ],
            },
            0.9,
            "claim 1: low_authority",
        ),
    ],
)
def test_reviewed_claim_that_fails_verification_is_not_stored(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    claim: dict,
    support: float,
    problem: str,
) -> None:
    jev.support = support
    entries, rows = build(corpus, settings, jev, questions(claim))
    assert entries == ()
    [row] = rows
    assert row.outcome is Outcome.not_found and not row.cached
    assert row.problems == (problem,)


def test_one_failed_claim_rejects_the_whole_reviewed_answer(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """Unlike a model draft, reviewed text is never served with a piece missing."""
    bad = {
        **OLD_FINE,
        "citations": [{"chunk_id": "rules-2016-fines", "quote": "3rd offense: $999."}],
    }
    entries, rows = build(corpus, settings, jev, questions(FINE, bad))
    assert entries == ()
    assert rows[0].problems == ("claim 2: no_valid_quote",)


def test_generated_answer_reports_its_kept_claims(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    curated = questions()
    entries, rows = build(corpus, settings, jev, curated)
    [row] = rows
    assert row.source == "generated" and row.cached
    [claim] = row.claims
    assert claim.statement == "A third violation is a $125 fine under the 2023 Rules."
    assert claim.citations[0].quote == "3rd violation: $125."

    # Committing the proposal makes the next build verify that exact wording.
    proposed = with_claims(curated, rows)
    provider = FakeProvider([None])
    again, _ = build(corpus, settings, jev, proposed, provider)
    assert provider.prompts == []
    assert again[0].answer.answer_text == entries[0].answer.answer_text
    assert again[0].answer.citations == entries[0].answer.citations


def test_asker_without_tracing_can_build_generated_answers_only() -> None:
    answered = Answer(
        request_id="r",
        outcome=Outcome.answered,
        answer_text="Yes.",
        citations=(),
        confidence=0.9,
        conflicts_noted=(),
        disclaimer="d",
    )

    class Plain:
        async def __call__(self, question: str) -> AskResult:
            return AskResult(
                answer=answered,
                estimated_cost_usd=0.02,
                jev_input_tokens=0,
                answer_input_tokens=0,
                answer_output_tokens=0,
                latency_ms=1.0,
            )

    entries, rows = asyncio.run(build_entries(questions(), Plain()))
    assert [e.id for e in entries] == ["fine-third"] and rows[0].claims == ()
    with pytest.raises(ValueError, match="cannot verify reviewed claims"):
        asyncio.run(build_entries(questions(FINE), Plain()))


def run_build(tmp_path: Path, curated: dict, *extra: str) -> tuple[int, Path]:
    def factory(corpus: Corpus, settings: QASettings) -> QAAsker:
        jev = FakeJev(
            relevance={"rules-2023-fines": 0.9},
            text_to_id={c.text_clean: c.id for c in corpus.chunks},
        )
        return build_asker(
            corpus, settings, jev=jev, provider=FakeProvider([FINES_DRAFT])
        )

    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump(curated))
    out = tmp_path / "answer_cache.json"
    code = main(
        [
            "cache",
            "build",
            str(path),
            "--corpus",
            str(FIXTURE),
            "--out",
            str(out),
            *extra,
        ],
        asker_factory=factory,
    )
    return code, out


def test_cli_writes_a_proposal_that_loads_and_rebuilds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    proposal = tmp_path / "proposed.yaml"
    curated = {
        "questions": [
            {"id": "fine-third", "question": FINE_QUESTION},
            {"id": "old-fine", "question": "What was the old fine?", "claims": [FINE]},
        ]
    }
    code, out = run_build(tmp_path, curated, "--propose", str(proposal))
    assert code == 0
    table = capsys.readouterr().out
    assert "generated" in table and "reviewed" in table
    proposed = load_questions(proposal)
    assert all(q.claims for q in proposed.questions)
    assert proposed.questions[1].claims[0].statement == FINE["statement"]

    first = load_cache(out)
    code, out = run_build(tmp_path, yaml.safe_load(proposal.read_text()))
    assert code == 0
    assert [e.answer.answer_text for e in load_cache(out).entries] == [
        e.answer.answer_text for e in first.entries
    ]


def test_cli_fails_when_reviewed_wording_no_longer_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = {
        **FINE,
        "citations": [
            {"chunk_id": "rules-2023-fines", "quote": "3rd violation: $130."}
        ],
    }
    curated = {
        "questions": [
            {"id": "fine-third", "question": FINE_QUESTION},
            {"id": "stale", "question": "What was the fine?", "claims": [stale]},
        ]
    }
    code, out = run_build(tmp_path, curated)
    captured = capsys.readouterr()
    assert code == 1
    assert "claim 1: no_valid_quote" in captured.out
    assert "reviewed answers failed verification: ['stale']" in captured.err
    # The answers that did verify are still written, for inspection.
    assert [e.id for e in load_cache(out).entries] == ["fine-third"]


def test_cli_verify_accepts_only_a_cache_built_from_this_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    curated = {"questions": [{"id": "fine-third", "question": FINE_QUESTION}]}
    _, cache_path = run_build(tmp_path, curated)
    capsys.readouterr()
    verify = ["cache", "verify", "--cache", str(cache_path), "--corpus"]
    assert main([*verify, str(FIXTURE)]) == 0
    assert "answer cache ok: 1 entries" in capsys.readouterr().out

    corpus = load_corpus(FIXTURE)
    first = corpus.chunks[0]
    edited = first.model_copy(update={"text_clean": first.text_clean + " Changed."})
    other = tmp_path / "other_corpus.json"
    other.write_text(
        corpus.model_copy(
            update={"chunks": (edited, *corpus.chunks[1:])}
        ).model_dump_json()
    )
    assert main([*verify, str(other)]) == 1
    assert "different corpus" in capsys.readouterr().err

    cache_path.write_text("{broken")
    assert main([*verify, str(FIXTURE)]) == 1
    assert main([*verify[:-1], "--corpus", str(tmp_path / "missing.json")]) == 1


def test_reviewed_quote_must_be_long_enough_to_mean_something() -> None:
    short = {**FINE, "citations": [{"chunk_id": "rules-2023-fines", "quote": "3"}]}
    with pytest.raises(ValueError, match="at least 8 characters"):
        questions(short)

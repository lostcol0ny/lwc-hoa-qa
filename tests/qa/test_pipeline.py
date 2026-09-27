"""End-to-end ask() behavior with fake Jev and answer providers."""

import asyncio
import uuid

import pytest
from qa_fakes import (
    FINE_CLAIM,
    FINES_DRAFT,
    FakeJev,
    FakeProvider,
    claim,
    draft,
    make_asker,
)

from hoa_qa.ask import (
    BOARD_REFERRAL,
    DISCLAIMER,
    MAX_QUESTION_CHARS,
    OMITTED_NOTE,
    QASettings,
    build_asker,
    clean_question,
)
from hoa_qa.models import Corpus, Outcome, citation_url

QUESTION = "How much is the fine for a third violation?"

INVENTED = claim(
    "Seniors over 65 are exempt from all fines.",
    ("rules-2023-fines", "2nd violation: $75."),
    essential=False,
)


def ask(asker, question: str = QUESTION):
    return asyncio.run(asker(question))


def invented_unsupported(statement: str, ids) -> float:
    return 0.05 if "exempt" in statement else 0.9


@pytest.mark.parametrize(
    "question",
    ["", "   ", "\x00\x07​", "x" * (MAX_QUESTION_CHARS + 1)],
    ids=["empty", "blank", "only-control", "too-long"],
)
def test_invalid_input_makes_no_calls(
    corpus: Corpus, settings: QASettings, jev: FakeJev, question: str
) -> None:
    provider = FakeProvider([FINES_DRAFT])
    result = ask(make_asker(corpus, settings, jev, provider), question)
    assert result.answer.outcome is Outcome.invalid_input
    assert jev.calls == [] and provider.prompts == []
    assert result.estimated_cost_usd == 0


def test_clean_question_strips_controls_and_counts_after_stripping() -> None:
    assert clean_question("  what\tis\n the\x00 fee?‮ ") == "what is the fee?"
    padded = " " * 50 + "q" * MAX_QUESTION_CHARS + "\x00" * 20
    assert clean_question(padded) == "q" * MAX_QUESTION_CHARS


def test_off_topic_is_refused_without_answer_model(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.gate = 0.1
    provider = FakeProvider([FINES_DRAFT])
    result = ask(make_asker(corpus, settings, jev, provider), "Write me a poem")
    assert result.answer.outcome is Outcome.refused_off_topic
    assert settings.documents_url in result.answer.answer_text
    assert provider.prompts == []
    assert len(jev.calls) == 1 and "on_topic" in jev.calls[0][1]
    # The question is passed as data in the state, not in the instructions.
    assert jev.calls[0][0] == {"message": "Write me a poem"}


def test_gate_threshold_is_configurable(corpus: Corpus, jev: FakeJev) -> None:
    jev.gate = 0.6
    strict = QASettings(gate_threshold=0.7)
    result = ask(make_asker(corpus, strict, jev, FakeProvider([FINES_DRAFT])))
    assert result.answer.outcome is Outcome.refused_off_topic


def test_not_found_when_no_chunk_passes_sweep(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.relevance = {"rules-2023-fines": 0.29}
    provider = FakeProvider([FINES_DRAFT])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert "couldn't find" in result.answer.answer_text
    assert "Board" in result.answer.answer_text
    assert result.answer.citations == ()
    assert provider.prompts == []


def test_answered_builds_public_answer(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([FINES_DRAFT])
    result = ask(make_asker(corpus, settings, jev, provider))
    answer = result.answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == FINE_CLAIM.statement
    assert answer.disclaimer == DISCLAIMER
    assert uuid.UUID(answer.request_id).version == 4
    [citation] = answer.citations
    chunk = next(c for c in corpus.chunks if c.id == "rules-2023-fines")
    assert citation.chunk_id == chunk.id
    assert citation.citation_label == chunk.citation_label
    assert citation.url == citation_url(chunk) == chunk.source_url + "#page=15"
    assert answer.confidence == 0.8
    # Only swept passages reach the prompt, best first.
    user = provider.prompts[0].user
    assert user.index('chunk_id="rules-2023-fines"') < user.index(
        'chunk_id="rules-2016-fines"'
    )
    assert 'chunk_id="bylaws-3.4"' not in user


def test_answer_text_is_composed_from_claims_and_code_text(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    conflict = claim(
        "An informal 2022 blog post lists $100 for a 3rd offense.",
        ("blog-2022-violations", "3rd offense - $100.00"),
        kind="conflict",
        essential=False,
    )
    jev.relevance = {**jev.relevance, "blog-2022-violations": 0.6}
    provider = FakeProvider([draft(FINE_CLAIM, conflict, refer_to_board=True)])
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == f"{FINE_CLAIM.statement} {BOARD_REFERRAL}"
    assert answer.conflicts_noted == (conflict.statement,)
    assert [c.chunk_id for c in answer.citations] == [
        "rules-2023-fines",
        "blog-2022-violations",
    ]


def test_request_ids_are_unique(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    asker = make_asker(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    assert ask(asker).answer.request_id != ask(asker).answer.request_id


def test_invented_claim_never_reaches_answer_text(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """B3: a correct fine plus an invented exemption citing a real quote."""
    jev.support = invented_unsupported
    provider = FakeProvider([draft(FINE_CLAIM, INVENTED)])
    result = ask(make_asker(corpus, settings, jev, provider))
    answer = result.answer
    assert "exempt" not in answer.answer_text
    assert "Seniors" not in " ".join(answer.conflicts_noted)
    # It regenerated once (naming the failed claim), then dropped it and noted so.
    assert len(provider.prompts) == 2
    assert INVENTED.statement in provider.prompts[1].user
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == f"{FINE_CLAIM.statement} {OMITTED_NOTE}"
    assert [c.quote for c in answer.citations] == ["3rd violation: $125."]


def test_regenerate_fixes_failed_claim(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = invented_unsupported
    provider = FakeProvider([draft(FINE_CLAIM, INVENTED), FINES_DRAFT])
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == FINE_CLAIM.statement  # no omission note


def test_failed_essential_claim_is_not_found(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = invented_unsupported
    essential_invention = claim(
        "Fines are exempt for first-time owners.",
        ("rules-2023-fines", "2nd violation: $75."),
        essential=True,
    )
    bad = draft(FINE_CLAIM, essential_invention)
    provider = FakeProvider([bad, bad])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert "exempt" not in result.answer.answer_text


def test_only_conflict_claims_surviving_is_not_found(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = lambda statement, ids: 0.9 if "blog" in statement else 0.1
    conflict = claim(
        "An informal blog post lists $100.",
        ("blog-2022-violations", "3rd offense - $100.00"),
        kind="conflict",
        essential=False,
    )
    wrong = claim(
        "A third violation costs $125 and is waived on holidays.",
        ("rules-2023-fines", "3rd violation: $125."),
        essential=False,
    )
    jev.relevance = {**jev.relevance, "blog-2022-violations": 0.6}
    bad = draft(wrong, conflict)
    result = ask(make_asker(corpus, settings, jev, FakeProvider([bad, bad])))
    assert result.answer.outcome is Outcome.not_found


def test_fabricated_quote_fails_claim(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    fake_quote = claim(
        "A third violation is $500.",
        ("rules-2023-fines", "3rd violation: $500."),
        essential=False,
    )
    provider = FakeProvider([draft(FINE_CLAIM, fake_quote)] * 2)
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert "$500" not in answer.answer_text
    assert [c.quote for c in answer.citations] == ["3rd violation: $125."]


def test_chunk_id_outside_provided_passages_fails_claim(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    # bylaws-3.4 exists in the corpus and the quote is real, but the sweep did
    # not select it, so the model was never shown it.
    outside = claim(
        "The Board manages the Association.",
        ("bylaws-3.4", "managed by its Board of Directors"),
        essential=False,
    )
    provider = FakeProvider([draft(FINE_CLAIM, outside)] * 2)
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert [c.chunk_id for c in answer.citations] == ["rules-2023-fines"]
    assert "manages" not in answer.answer_text


def test_one_bad_citation_does_not_sink_a_claim(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    mixed = claim(
        FINE_CLAIM.statement,
        ("rules-2023-fines", "3rd violation: $125."),
        ("rules-2023-fines", "not in the passage"),
    )
    answer = ask(make_asker(corpus, settings, jev, FakeProvider([draft(mixed)]))).answer
    assert answer.outcome is Outcome.answered
    assert [c.quote for c in answer.citations] == ["3rd violation: $125."]


def test_support_is_judged_per_claim(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = invented_unsupported
    ask(make_asker(corpus, settings, jev, FakeProvider([draft(FINE_CLAIM, INVENTED)])))
    state, questions = next(c for c in jev.calls if "c0" in c[1])
    assert [c["statement"] for c in state["claims"]] == [
        FINE_CLAIM.statement,
        INVENTED.statement,
    ]
    assert "`claims[1].statement`" in questions["c1"].instructions


def test_all_claims_fail_regenerates_once_then_not_found(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    bad = draft(claim("Fines are waived.", ("rules-2023-fines", "Fines are waived.")))
    provider = FakeProvider([bad, bad, FINES_DRAFT])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert result.answer.citations == ()
    assert len(provider.prompts) == 2
    assert "rejected" not in provider.prompts[0].user
    assert "rejected" in provider.prompts[1].user
    assert "Fines are waived." in provider.prompts[1].user


def test_no_claims_means_not_found_without_retry(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft()])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert len(provider.prompts) == 1
    assert jev.calls_of("c") == []  # nothing to support-check


def test_invalid_model_output_counts_as_failed_attempt(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([None, None])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert len(provider.prompts) == 2
    assert "not valid" in provider.prompts[1].user


def test_provider_exception_becomes_error_outcome(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    class Boom(FakeProvider):
        async def generate(self, prompt):  # type: ignore[override]
            raise RuntimeError("upstream down")

    result = ask(make_asker(corpus, settings, jev, Boom([])))
    assert result.answer.outcome is Outcome.error
    assert result.answer.citations == ()
    assert result.estimated_cost_usd > 0  # gate + sweep were still spent


def test_settings_from_env() -> None:
    settings = QASettings.from_env(
        {
            "TYPESAFE_API_KEY": "ts-test",
            "ANSWER_MODEL": "claude-sonnet-5",
            "GATE_THRESHOLD": "0.6",
            "SWEEP_TOP_K": "3",
            "SUPPORT_THRESHOLD": "",
        }
    )
    assert settings.answer_model == "claude-sonnet-5"
    assert settings.gate_threshold == 0.6
    assert settings.sweep_top_k == 3
    assert settings.support_threshold == 0.5
    assert settings.anthropic_api_key is None
    assert "ts-test" not in repr(settings)
    with pytest.raises(ValueError):
        QASettings.from_env({"SWEEP_THRESHOLD": "1.5"})


def test_jev_concurrency_env_with_legacy_fallback() -> None:
    assert QASettings.from_env({"JEV_CONCURRENCY": "3"}).jev_concurrency == 3
    assert QASettings.from_env({"SWEEP_CONCURRENCY": "2"}).jev_concurrency == 2
    both = {"JEV_CONCURRENCY": "5", "SWEEP_CONCURRENCY": "2"}
    assert QASettings.from_env(both).jev_concurrency == 5


def test_build_asker_requires_keys_for_real_clients(corpus: Corpus) -> None:
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        build_asker(corpus, QASettings())


def test_worse_retry_falls_back_to_the_verified_first_attempt(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """Live eval: attempt 1 was sound but for a side claim; attempt 2 broke."""
    jev.support = invented_unsupported
    essential_invention = claim(
        "Fines are exempt for first-time owners.",
        ("rules-2023-fines", "2nd violation: $75."),
    )
    provider = FakeProvider([draft(FINE_CLAIM, INVENTED), draft(essential_invention)])
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert len(provider.prompts) == 2
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == f"{FINE_CLAIM.statement} {OMITTED_NOTE}"
    assert "exempt" not in answer.answer_text


@pytest.mark.parametrize("retry", [None, draft()], ids=["invalid", "no-claims"])
def test_unusable_retry_falls_back_to_the_first_attempt(
    corpus: Corpus, settings: QASettings, jev: FakeJev, retry
) -> None:
    jev.support = invented_unsupported
    provider = FakeProvider([draft(FINE_CLAIM, INVENTED), retry])
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == f"{FINE_CLAIM.statement} {OMITTED_NOTE}"


def test_failed_conflict_claim_never_blocks_a_verified_answer(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """Even marked essential, an unsupported conflict note is just dropped."""
    jev.relevance = {**jev.relevance, "blog-2022-violations": 0.6}
    jev.support = lambda statement, ids: 0.2 if "outdated" in statement else 0.9
    conflict = claim(
        "An informal blog post says $100, which is outdated.",
        ("blog-2022-violations", "3rd offense - $100.00"),
        kind="conflict",
        essential=True,
    )
    provider = FakeProvider([draft(FINE_CLAIM, conflict)] * 2)
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == f"{FINE_CLAIM.statement} {OMITTED_NOTE}"
    assert answer.conflicts_noted == ()

"""End-to-end ask() behavior with fake Jev and answer providers."""

import asyncio
import uuid

import pytest
from qa_fakes import FINES_DRAFT, FakeJev, FakeProvider, draft, make_asker

from hoa_qa.ask import DISCLAIMER, MAX_QUESTION_CHARS, QASettings, clean_question
from hoa_qa.models import Corpus, Outcome, citation_url

QUESTION = "How much is the fine for a second violation?"


def ask(asker, question: str = QUESTION):
    return asyncio.run(asker(question))


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
    provider = FakeProvider([FINES_DRAFT])
    result = ask(make_asker(corpus, strict, jev, provider))
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
    assert answer.answer_text == FINES_DRAFT.answer_text
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


def test_request_ids_are_unique(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    asker = make_asker(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    assert ask(asker).answer.request_id != ask(asker).answer.request_id


def test_fabricated_quote_is_dropped(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider(
        [
            draft(
                ("rules-2023-fines", "Second violation: $125."),
                ("rules-2023-fines", "Second violation: $500."),
            )
        ]
    )
    result = ask(make_asker(corpus, settings, jev, provider))
    assert [c.quote for c in result.answer.citations] == ["Second violation: $125."]


def test_chunk_id_outside_provided_passages_is_dropped(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    # bylaws-3.4 exists in the corpus and the quote is real, but the sweep did
    # not select it, so the model was never shown it.
    provider = FakeProvider(
        [
            draft(
                ("bylaws-3.4", "managed by its Board of Directors"),
                ("rules-2023-fines", "First violation: $75."),
            )
        ]
    )
    result = ask(make_asker(corpus, settings, jev, provider))
    assert [c.chunk_id for c in result.answer.citations] == ["rules-2023-fines"]


def test_unsupported_citation_is_dropped(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = lambda chunk_id: 0.2 if chunk_id == "rules-2016-fines" else 0.9
    provider = FakeProvider(
        [
            draft(
                ("rules-2023-fines", "Second violation: $125."),
                ("rules-2016-fines", "Second violation: $100."),
            )
        ]
    )
    result = ask(make_asker(corpus, settings, jev, provider))
    assert [c.chunk_id for c in result.answer.citations] == ["rules-2023-fines"]
    [support_call] = jev.calls_of("c")
    assert len(support_call) == 2  # both survivors judged in one request


def test_all_citations_fail_regenerates_once_then_not_found(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    bad = draft(("rules-2023-fines", "Fines are waived for everyone."))
    provider = FakeProvider([bad, bad, FINES_DRAFT])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert result.answer.citations == ()
    assert len(provider.prompts) == 2
    assert "rejected" not in provider.prompts[0].user
    assert "rejected" in provider.prompts[1].user


def test_regenerate_can_recover(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft(("nope", "x")), FINES_DRAFT])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.answered
    assert len(provider.prompts) == 2


def test_zero_citation_answer_is_never_answered(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft(text="Dues are $0.")])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert jev.calls_of("c") == []  # nothing to support-check


def test_invalid_model_output_counts_as_failed_attempt(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([None, None])
    result = ask(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.not_found
    assert len(provider.prompts) == 2


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


def test_build_asker_requires_keys_for_real_clients(corpus: Corpus) -> None:
    from hoa_qa.ask import build_asker

    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        build_asker(corpus, QASettings())

"""Cost aggregation, partial-failure cost, max_cost_usd, and private logs."""

import asyncio
import logging
from collections.abc import Callable

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

from hoa_qa.answer.pricing import (
    ANSWER_MODEL_PRICES,
    FALLBACK_PRICE,
    JEV_INPUT_USD_PER_MTOK,
    answer_price,
)
from hoa_qa.ask import QASettings, max_cost_usd
from hoa_qa.models import Corpus, Outcome

HAIKU = ANSWER_MODEL_PRICES["claude-haiku-4-5"]


def expected_usd(jev: FakeJev, provider: FakeProvider) -> float:
    return (
        sum(jev.billed) * JEV_INPUT_USD_PER_MTOK
        + sum(provider.input_tokens) * HAIKU.input_usd_per_mtok
        + sum(provider.output_tokens) * HAIKU.output_usd_per_mtok
    ) / 1e6


def run(corpus, settings, jev, provider, question="fines?"):
    return asyncio.run(make_asker(corpus, settings, jev, provider)(question))


def test_answered_cost_sums_jev_and_answer_model(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([FINES_DRAFT])
    result = run(corpus, settings, jev, provider)
    assert result.answer.outcome is Outcome.answered
    # gate + 7 per-doc sweep requests + 1 support request
    assert len(jev.calls) == 9
    assert result.jev_input_tokens == sum(jev.billed)
    assert result.answer_input_tokens == sum(provider.input_tokens)
    assert result.answer_output_tokens == sum(provider.output_tokens)
    assert result.estimated_cost_usd == pytest.approx(expected_usd(jev, provider))


def test_regenerate_cost_counts_both_answer_calls(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    bad = draft(claim("Not in the text.", ("rules-2023-fines", "not in the text")))
    provider = FakeProvider([bad, bad])
    result = run(corpus, settings, jev, provider)
    assert result.answer.outcome is Outcome.not_found
    assert len(provider.input_tokens) == 2
    assert result.estimated_cost_usd == pytest.approx(expected_usd(jev, provider))


def test_gate_refusal_costs_only_the_gate(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.gate = 0.0
    provider = FakeProvider([FINES_DRAFT])
    result = run(corpus, settings, jev, provider, "poem")
    assert len(jev.billed) == 1
    assert result.estimated_cost_usd == pytest.approx(
        jev.billed[0] * JEV_INPUT_USD_PER_MTOK / 1e6
    )
    assert result.answer_input_tokens == result.answer_output_tokens == 0


def test_sweep_partial_failure_cost_reaches_ask_result(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """B2: siblings of a failed sweep batch still count, on the error outcome."""
    jev.fail = lambda state: state.get("document") == "LWC Bylaws"
    provider = FakeProvider([FINES_DRAFT])
    result = run(corpus, settings, jev, provider)
    assert result.answer.outcome is Outcome.error
    assert len(jev.billed) == 1 + 6  # gate + the 6 successful sweep batches
    assert result.jev_input_tokens == sum(jev.billed)
    assert result.estimated_cost_usd == pytest.approx(expected_usd(jev, provider))
    assert provider.prompts == []


def test_support_failure_cost_reaches_ask_result(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.fail = lambda state: "claims" in state
    provider = FakeProvider([FINES_DRAFT])
    result = run(corpus, settings, jev, provider)
    assert result.answer.outcome is Outcome.error
    # gate + sweep billed, the answer call billed, the support call failed.
    assert result.estimated_cost_usd == pytest.approx(expected_usd(jev, provider))
    assert result.answer_input_tokens > 0


def test_price_lookup() -> None:
    assert answer_price("claude-haiku-4-5") == HAIKU
    assert answer_price("claude-haiku-4-5-20251001") == HAIKU
    assert (
        answer_price("claude-opus-5-5-20261101")
        == ANSWER_MODEL_PRICES["claude-opus-5-5"]
    )
    assert answer_price("some-unknown-model") == FALLBACK_PRICE


# --- max_cost_usd (B5) ---------------------------------------------------------

INVENTED = claim(
    "Seniors are exempt from fines.",
    ("rules-2023-fines", "First violation: $75."),
    essential=False,
)
LONG_QUESTION = "What are the fines " + "and the rules " * 34  # ~500 chars


def _setup_answered(jev: FakeJev) -> list:
    return [FINES_DRAFT]


def _setup_regenerate_drop(jev: FakeJev) -> list:
    jev.support = lambda statement, ids: 0.05 if "exempt" in statement else 0.9
    return [draft(FINE_CLAIM, INVENTED)] * 2


def _setup_regenerate_not_found(jev: FakeJev) -> list:
    bad = draft(claim("Nope.", ("rules-2023-fines", "nope")))
    return [bad, bad]


def _setup_everything_relevant(jev: FakeJev) -> list:
    jev.relevance = {cid: 0.9 for cid in jev.text_to_id.values()}
    many = [
        claim(f"{i}: second violation $125.", ("rules-2023-fines", "$125"))
        for i in range(8)
    ]
    return [draft(*many[:7], INVENTED), draft(*many)]


def _setup_invalid(jev: FakeJev) -> list:
    return [None, None]


def _setup_refused(jev: FakeJev) -> list:
    jev.gate = 0.0
    return [FINES_DRAFT]


SCENARIOS: dict[str, Callable[[FakeJev], list]] = {
    "answered": _setup_answered,
    "regenerate-drop": _setup_regenerate_drop,
    "regenerate-not-found": _setup_regenerate_not_found,
    "everything-relevant": _setup_everything_relevant,
    "invalid-output": _setup_invalid,
    "refused": _setup_refused,
}


@pytest.mark.parametrize("question", ["fines?", LONG_QUESTION], ids=["short", "long"])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_max_cost_bounds_estimated_cost(
    corpus: Corpus, settings: QASettings, jev: FakeJev, scenario: str, question: str
) -> None:
    provider = FakeProvider(SCENARIOS[scenario](jev))
    asker = make_asker(corpus, settings, jev, provider)
    result = asyncio.run(asker(question))
    assert result.estimated_cost_usd > 0
    assert asker.max_cost_usd >= result.estimated_cost_usd


def test_max_cost_value_for_mini_corpus(corpus: Corpus, settings: QASettings) -> None:
    bound = max_cost_usd(corpus, settings, settings.answer_model)
    # Two full-length Haiku outputs alone are 2 * 4096 * $5/MTok = $0.04096.
    assert 0.04096 < bound < 0.1
    pricier = max_cost_usd(corpus, settings, "claude-opus-5")
    assert pricier > bound


# --- logging ---------------------------------------------------------------------

SECRET_QUESTION = "My neighbor Zebulon at 12 Quail Run keeps parking a boat, fine?"


@pytest.mark.parametrize("gate", [0.95, 0.0], ids=["answered", "refused"])
def test_logs_never_contain_question_or_answer_text(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
    gate: float,
) -> None:
    jev.gate = gate
    caplog.set_level(logging.DEBUG)
    result = run(corpus, settings, jev, FakeProvider([FINES_DRAFT]), SECRET_QUESTION)
    assert caplog.records, "expected a structured request log"
    rendered = caplog.text + " ".join(str(r.__dict__) for r in caplog.records)
    for secret in ("Zebulon", "Quail Run", "boat", SECRET_QUESTION):
        assert secret not in rendered
    assert result.answer.answer_text not in rendered
    [record] = [r for r in caplog.records if hasattr(r, "hoa_qa")]
    fields = record.__dict__["hoa_qa"]
    assert fields["request_id"] == result.answer.request_id
    assert fields["outcome"] == result.answer.outcome.value
    assert {"latency_ms", "jev_input_tokens", "estimated_cost_usd"} <= fields.keys()


def test_failed_claim_text_is_not_logged(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    jev.support = lambda statement, ids: 0.05 if "exempt" in statement else 0.9
    run(corpus, settings, jev, FakeProvider([draft(FINE_CLAIM, INVENTED)] * 2))
    assert "exempt" not in caplog.text
    assert "claims_failed=1" in caplog.text


def test_error_log_has_type_but_not_message(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Leaky(FakeProvider):
        async def generate(self, prompt):  # type: ignore[override]
            raise RuntimeError(f"bad request body: {prompt.user}")

    caplog.set_level(logging.DEBUG)
    asyncio.run(make_asker(corpus, settings, jev, Leaky([]))(SECRET_QUESTION))
    assert "RuntimeError" in caplog.text
    assert "Zebulon" not in caplog.text


def test_partial_failure_logs_underlying_error_type(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
) -> None:
    jev.fail = lambda state: "passages" in state
    caplog.set_level(logging.DEBUG)
    run(corpus, settings, jev, FakeProvider([FINES_DRAFT]), SECRET_QUESTION)
    assert "error_type=RuntimeError" in caplog.text
    assert "Zebulon" not in caplog.text

"""Cost aggregation, pricing lookup, and privacy-preserving request logs."""

import asyncio
import logging

import pytest
from qa_fakes import (
    ANSWER_INPUT_TOKENS,
    ANSWER_OUTPUT_TOKENS,
    FINES_DRAFT,
    JEV_TOKENS_PER_CALL,
    FakeJev,
    FakeProvider,
    draft,
    make_asker,
)

from hoa_qa.answer.pricing import (
    ANSWER_MODEL_PRICES,
    FALLBACK_PRICE,
    JEV_INPUT_USD_PER_MTOK,
    answer_price,
)
from hoa_qa.ask import QASettings
from hoa_qa.models import Corpus, Outcome

HAIKU = ANSWER_MODEL_PRICES["claude-haiku-4-5"]


def jev_usd(calls: int) -> float:
    return calls * JEV_TOKENS_PER_CALL * JEV_INPUT_USD_PER_MTOK / 1e6


def answer_usd(calls: int) -> float:
    return (
        calls
        * (
            ANSWER_INPUT_TOKENS * HAIKU.input_usd_per_mtok
            + ANSWER_OUTPUT_TOKENS * HAIKU.output_usd_per_mtok
        )
        / 1e6
    )


def test_answered_cost_sums_jev_and_answer_model(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    result = asyncio.run(
        make_asker(corpus, settings, jev, FakeProvider([FINES_DRAFT]))("fines?")
    )
    assert result.answer.outcome is Outcome.answered
    # gate + 7 per-doc sweep requests + 1 support request
    assert len(jev.calls) == 9
    assert result.jev_input_tokens == 9 * JEV_TOKENS_PER_CALL
    assert result.answer_input_tokens == ANSWER_INPUT_TOKENS
    assert result.answer_output_tokens == ANSWER_OUTPUT_TOKENS
    assert result.estimated_cost_usd == pytest.approx(jev_usd(9) + answer_usd(1))


def test_regenerate_cost_counts_both_answer_calls(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    bad = draft(("rules-2023-fines", "not in the text"))
    result = asyncio.run(
        make_asker(corpus, settings, jev, FakeProvider([bad, bad]))("fines?")
    )
    assert result.answer.outcome is Outcome.not_found
    assert result.estimated_cost_usd == pytest.approx(jev_usd(8) + answer_usd(2))


def test_gate_refusal_costs_only_the_gate(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.gate = 0.0
    result = asyncio.run(
        make_asker(corpus, settings, jev, FakeProvider([FINES_DRAFT]))("poem")
    )
    assert result.estimated_cost_usd == pytest.approx(jev_usd(1))
    assert result.answer_input_tokens == result.answer_output_tokens == 0


def test_price_lookup() -> None:
    assert answer_price("claude-haiku-4-5") == HAIKU
    assert answer_price("claude-haiku-4-5-20251001") == HAIKU
    assert (
        answer_price("claude-opus-5-5-20261101")
        == ANSWER_MODEL_PRICES["claude-opus-5-5"]
    )
    assert answer_price("some-unknown-model") == FALLBACK_PRICE


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
    result = asyncio.run(
        make_asker(corpus, settings, jev, FakeProvider([FINES_DRAFT]))(SECRET_QUESTION)
    )
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

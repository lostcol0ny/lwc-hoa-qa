"""Streaming: claims are verified and announced as the model writes them."""

import asyncio
import json
from typing import Any

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
from test_cache import QUIET, make_cache

from hoa_qa.answer.provider import AnswerDraft, ClaimStream
from hoa_qa.ask import OMITTED_NOTE, QASettings, build_asker
from hoa_qa.models import Corpus, Outcome
from hoa_qa.trace import NO_TRACE

SECOND = claim(
    "A second violation is a $75 fine.", ("rules-2023-fines", "2nd violation: $75.")
)
INVENTED = claim(
    "Seniors over 65 are exempt from all fines.",
    ("rules-2023-fines", "2nd violation: $75."),
    essential=False,
)


def unsupported_if_exempt(statement: str, ids: Any) -> float:
    return 0.1 if "exempt" in statement else 0.9


def stream(asker, question: str = "fines?") -> tuple[Any, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    result = asyncio.run(asker.ask_traced(question, NO_TRACE, emit=events.append))
    return result, events


# --- finding claims in partial JSON --------------------------------------------

OUTPUT = json.dumps(
    {
        "claims": [
            {
                "statement": 'A fine of {$125} "applies".',
                "kind": "answer",
                "essential": True,
                "citations": [{"chunk_id": "rules-2023-fines", "sentence": 2}],
            },
            {
                "statement": "Second.",
                "kind": "conflict",
                "essential": False,
                "citations": [
                    {"chunk_id": "x", "sentence": 1},
                    {"chunk_id": "y", "sentence": 3},
                ],
            },
        ],
        "confidence": 0.9,
        "refer_to_board": False,
    }
)


@pytest.mark.parametrize("size", [1, 2, 7, 50, len(OUTPUT)])
def test_claims_are_found_whatever_the_chunking(size: int) -> None:
    claims = ClaimStream()
    found = []
    for start in range(0, len(OUTPUT), size):
        found += claims.feed(OUTPUT[start : start + size])
    # Braces and escaped quotes inside a string do not end a claim early.
    assert [c.statement for c in found] == ['A fine of {$125} "applies".', "Second."]
    assert [len(c.citations) for c in found] == [1, 2]


def test_a_claim_is_announced_before_the_output_is_complete() -> None:
    claims = ClaimStream()
    first_end = OUTPUT.index("}, {") + 1
    assert len(claims.feed(OUTPUT[:first_end])) == 1
    assert claims.feed(OUTPUT[first_end : first_end + 20]) == []


def test_an_invalid_claim_is_skipped_not_raised() -> None:
    broken = json.dumps(
        {"claims": [{"statement": "", "kind": "answer"}, {"oops": 1}], "confidence": 1}
    )
    assert ClaimStream().feed(broken) == []


# --- the pipeline ----------------------------------------------------------------


def test_events_follow_the_pipeline_and_claims_arrive_verified(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft(FINE_CLAIM, SECOND)])
    result, events = stream(make_asker(corpus, settings, jev, provider))
    assert result.answer.outcome is Outcome.answered
    assert [e["type"] for e in events] == ["stage", "stage", "stage", "claim", "claim"]
    assert [e["stage"] for e in events[:3]] == ["gate", "sweep", "draft"]
    assert events[1]["passages"] == len(corpus.chunks) and events[2]["selected"] == 2
    assert [e["statement"] for e in events[3:]] == [
        FINE_CLAIM.statement,
        SECOND.statement,
    ]
    # What was announced is exactly what the answer says.
    assert result.answer.answer_text == f"{FINE_CLAIM.statement} {SECOND.statement}"
    assert provider.streamed == 1


def test_a_streamed_answer_matches_the_unstreamed_one(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = unsupported_if_exempt
    drafts: list[AnswerDraft | None] = [draft(FINE_CLAIM, INVENTED, SECOND)]
    streamed, _ = stream(make_asker(corpus, settings, jev, FakeProvider(drafts)))
    plain = asyncio.run(
        make_asker(corpus, settings, jev, FakeProvider(drafts))("fines?")
    )
    assert streamed.answer.answer_text == plain.answer.answer_text
    assert streamed.answer.citations == plain.answer.citations
    assert streamed.jev_input_tokens == plain.jev_input_tokens
    assert streamed.estimated_cost_usd == pytest.approx(plain.estimated_cost_usd)


def test_a_claim_that_fails_is_never_announced(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = unsupported_if_exempt
    provider = FakeProvider([draft(FINE_CLAIM, INVENTED)])
    result, events = stream(make_asker(corpus, settings, jev, provider))
    announced = [e["statement"] for e in events if e["type"] == "claim"]
    assert announced == [FINE_CLAIM.statement]
    assert result.answer.answer_text == f"{FINE_CLAIM.statement} {OMITTED_NOTE}"


def test_a_draft_that_cannot_stand_is_withdrawn_before_the_retry(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.support = unsupported_if_exempt
    essential = claim(
        "Fines are exempt for new owners.", ("rules-2023-fines", "2nd violation: $75.")
    )
    provider = FakeProvider([draft(FINE_CLAIM, essential), FINES_DRAFT])
    result, events = stream(make_asker(corpus, settings, jev, provider))
    kinds = [e["type"] for e in events]
    assert kinds == ["stage", "stage", "stage", "claim", "reset", "claim"]
    assert result.answer.outcome is Outcome.answered
    assert result.answer.answer_text == FINE_CLAIM.statement
    assert provider.streamed == 2


def test_conflict_claims_are_not_announced_as_answer_text(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    old = claim(
        "The superseded 2016 rules set the third fine at $100.",
        ("rules-2016-fines", "3rd offense: $100."),
        kind="conflict",
        essential=False,
    )
    provider = FakeProvider([draft(FINE_CLAIM, old)])
    result, events = stream(make_asker(corpus, settings, jev, provider))
    assert [e["statement"] for e in events if e["type"] == "claim"] == [
        FINE_CLAIM.statement
    ]
    assert result.answer.conflicts_noted == (old.statement,)


def test_an_early_claim_the_final_draft_lacks_is_ignored(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """The complete output is the authority; an unconfirmed early claim is not
    part of the answer."""
    provider = FakeProvider([FINES_DRAFT], stream_extra=[SECOND])
    result, _ = stream(make_asker(corpus, settings, jev, provider))
    assert result.answer.answer_text == FINE_CLAIM.statement


def test_cache_hit_and_refusal_emit_no_claims(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.match = {QUIET: 0.97}
    asker = build_asker(
        corpus,
        settings,
        jev=jev,
        provider=FakeProvider([FINES_DRAFT]),
        cache=make_cache(corpus),
    )
    result, events = stream(asker, "When are quiet hours?")
    assert result.cache_entry == "quiet-hours"
    assert events == [{"type": "stage", "stage": "gate"}]

    jev.gate = 0.0
    refused, events = stream(
        make_asker(corpus, settings, jev, FakeProvider([FINES_DRAFT])), "poem"
    )
    assert refused.answer.outcome is Outcome.refused_off_topic
    assert events == [{"type": "stage", "stage": "gate"}]


def test_without_emit_the_provider_is_not_streamed(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([FINES_DRAFT])
    asyncio.run(make_asker(corpus, settings, jev, provider)("fines?"))
    assert provider.streamed == 0

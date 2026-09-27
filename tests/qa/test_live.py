"""Optional smoke test against the real Jev and Anthropic APIs (costs ~a cent)."""

import asyncio
import os

import pytest
from qa_fakes import FIXTURE

from hoa_qa.ask import QASettings, build_asker
from hoa_qa.models import Outcome, load_corpus

HAS_KEYS = all(os.environ.get(k) for k in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not HAS_KEYS,
        reason="needs TYPESAFE_API_KEY and ANTHROPIC_API_KEY",
    ),
]


def ask_live(question: str):
    async def run():
        asker = build_asker(load_corpus(FIXTURE), QASettings.from_env())
        try:
            return await asker(question)
        finally:
            await asker.aclose()

    return asyncio.run(run())


def test_second_violation_fine_uses_2023_rules() -> None:
    result = ask_live("How much is the fine for a second violation?")
    assert result.answer.outcome is Outcome.answered
    assert "$125" in result.answer.answer_text
    # $100 (2016 rules, quoted by the 2022 blog) may appear only as a noted
    # conflict; the cited authority must be the 2023 Rules.
    assert "rules-2023-fines" in {c.chunk_id for c in result.answer.citations}


def test_continuing_violation_fine_is_per_day() -> None:
    result = ask_live("What is the fine for a continuing violation?")
    assert result.answer.outcome is Outcome.answered
    assert "$75" in result.answer.answer_text
    assert "per day" in result.answer.answer_text.lower()
    assert "rules-2023-fines" in {c.chunk_id for c in result.answer.citations}

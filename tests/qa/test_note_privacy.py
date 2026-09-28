"""Parse notes are built from code-owned names only: a key the model writes
(an extra field) never reaches a log, the retry prompt, or the answer."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from qa_fakes import FakeJev, make_asker

from hoa_qa.answer.prompt import AnswerPrompt
from hoa_qa.answer.provider import (
    AnthropicAnswerProvider,
    ProviderResult,
    parse_draft_checked,
)
from hoa_qa.ask import QASettings
from hoa_qa.models import Corpus, Outcome

SECRETS = (
    "Zebulon at 12 Quail Run",
    'Zeb "quoted"\nnewline key',
    "LONGSECRET" * 500,
)
QUOTE = "3rd violation: $125."


def citation(**extra: Any) -> dict[str, Any]:
    return {"chunk_id": "rules-2023-fines", "quote": QUOTE, **extra}


def claim(
    *, essential: bool = True, cite: dict | None = None, extra: dict | None = None
) -> dict:
    return {
        "statement": "A third violation is a $125 fine under the 2023 Rules.",
        "kind": "answer",
        "essential": essential,
        "citations": [cite or citation()],
        **(extra or {}),
    }


def output(*claims: dict, **extra: Any) -> str:
    return json.dumps(
        {"claims": list(claims), "confidence": 0.8, "refer_to_board": False, **extra}
    )


def leaky_outputs(secret: str) -> dict[str, str]:
    """An extra key at each level; the first three are invalid, the last is
    salvaged (a non-essential claim with an extra key is dropped)."""
    return {
        "output": output(claim(), **{secret: 1}),
        "claim": output(claim(extra={secret: 1})),
        "citation": output(claim(cite=citation(**{secret: "x"}))),
        "salvaged": output(claim(), claim(essential=False, extra={secret: 1})),
    }


CASES = [
    (secret, level, text)
    for secret in SECRETS
    for level, text in leaky_outputs(secret).items()
]
IDS = [f"{i // 4}-{level}" for i, (_, level, _) in enumerate(CASES)]


def assert_clean(rendered: str, secret: str) -> None:
    for fragment in (secret, "Zebulon", "Quail", "quoted", "LONGSECRET"):
        assert fragment not in rendered


@pytest.mark.parametrize(("secret", "level", "text"), CASES, ids=IDS)
def test_real_parser_notes_hold_no_model_written_keys(
    secret: str, level: str, text: str
) -> None:
    parsed = parse_draft_checked(text)
    assert (parsed.draft is not None) == (level == "salvaged")
    assert parsed.note is not None
    assert "<extra>" in parsed.note and "extra_forbidden" in parsed.note
    assert_clean(parsed.note + repr(parsed.issues), secret)


@dataclass
class ParsingProvider:
    """Runs raw model text through the real parser, like the real provider."""

    texts: list[str]
    model: str = "claude-haiku-4-5"
    max_tokens: int = 2048
    prompts: list[AnswerPrompt] = field(default_factory=list)

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult:
        self.prompts.append(prompt)
        parsed = parse_draft_checked(self.texts[len(self.prompts) - 1])
        return ProviderResult(
            parsed.draft,
            self.model,
            100,
            100,
            note=parsed.note,
            claims_trimmed=parsed.claims_trimmed,
            issues=parsed.issues,
        )


@pytest.mark.parametrize(("secret", "level", "text"), CASES, ids=IDS)
def test_model_written_keys_never_reach_logs_retry_or_answer(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
    secret: str,
    level: str,
    text: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider = ParsingProvider([text, text])
    asker = make_asker(corpus, settings, jev, provider)  # type: ignore[arg-type]
    answer = asyncio.run(asker("fines?")).answer
    attempts = [r for r in caplog.records if hasattr(r, "hoa_qa_attempt")]
    assert attempts and all(r.hoa_qa_attempt["note"] for r in attempts)  # type: ignore[attr-defined]
    assert_clean(caplog.text, secret)
    assert_clean(
        " ".join(r.getMessage() + str(r.__dict__) for r in caplog.records), secret
    )
    assert_clean(" ".join(p.user for p in provider.prompts), secret)
    assert_clean(answer.model_dump_json(), secret)
    if level == "salvaged":
        assert answer.outcome is Outcome.answered
    else:
        assert answer.outcome is Outcome.not_found
        assert len(provider.prompts) == 2
        assert "The previous output was not valid." in provider.prompts[1].user


class FakeMessages:
    def __init__(self, text: str) -> None:
        self.text = text

    async def create(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=10),
        )


@pytest.mark.parametrize("secret", SECRETS, ids=["plain", "quote-newline", "long"])
def test_real_provider_warning_log_holds_no_model_written_keys(
    caplog: pytest.LogCaptureFixture, secret: str
) -> None:
    caplog.set_level(logging.DEBUG)
    for text in leaky_outputs(secret).values():
        provider = AnthropicAnswerProvider(api_key="test-key")
        provider._client = SimpleNamespace(messages=FakeMessages(text))  # type: ignore[assignment]
        result = asyncio.run(provider.generate(AnswerPrompt(system="s", user="u")))
        assert result.note is not None and "<extra>" in result.note
    assert "answer model output unusable" in caplog.text
    assert_clean(caplog.text, secret)
    assert_clean(" ".join(str(r.__dict__) for r in caplog.records), secret)

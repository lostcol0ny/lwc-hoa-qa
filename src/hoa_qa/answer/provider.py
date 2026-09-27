"""Answer-model provider; the only module that imports ``anthropic``."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hoa_qa.answer.prompt import (
    MAX_CITATIONS_PER_CLAIM,
    MAX_CLAIMS,
    MAX_STATEMENT_CHARS,
    AnswerPrompt,
)

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_MODEL = "claude-haiku-4-5"
# Default answer-model output cap (ANSWER_MAX_TOKENS), part of the budget
# bound. A realistic 8-claim answer is ~1K tokens and the schema caps allow
# ~2K, so 2048 avoids truncating valid answers (truncated JSON -> not_found)
# while halving the output term of max_cost_usd versus 4096. Raise it for
# models with adaptive thinking, whose reasoning counts against the cap.
DEFAULT_MAX_OUTPUT_TOKENS = 2048


class DraftCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    quote: str


class DraftClaim(BaseModel):
    """One short factual statement and the passages that back it.

    ``kind`` is "answer" for statements that answer the question and
    "conflict" for disagreements between sources. ``essential`` marks a claim
    the answer cannot stand without; it only ever makes verification stricter.
    """

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=MAX_STATEMENT_CHARS)
    kind: Literal["answer", "conflict"]
    essential: bool
    citations: list[DraftCitation] = Field(
        min_length=1, max_length=MAX_CITATIONS_PER_CLAIM
    )


class AnswerDraft(BaseModel):
    """The answer model's structured output, before verification."""

    model_config = ConfigDict(extra="forbid")

    claims: list[DraftClaim] = Field(max_length=MAX_CLAIMS)
    confidence: float = Field(ge=0, le=1)
    refer_to_board: bool


_CITATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["chunk_id", "quote"],
    "additionalProperties": False,
}

# Structured-output JSON schema. Numeric bounds and array/string length caps
# are enforced by AnswerDraft, since output schemas do not accept them all.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "kind": {"type": "string", "enum": ["answer", "conflict"]},
                    "essential": {"type": "boolean"},
                    "citations": {"type": "array", "items": _CITATION_SCHEMA},
                },
                "required": ["statement", "kind", "essential", "citations"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
        "refer_to_board": {"type": "boolean"},
    },
    "required": ["claims", "confidence", "refer_to_board"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProviderResult:
    """``draft`` is None when the output was missing or failed validation.

    ``note`` says why a draft is missing, or what was trimmed from a salvaged
    one: field paths and error types only, never output text.
    ``claims_trimmed`` is True when salvage dropped a (non-essential) claim,
    so the answer must carry OMITTED_NOTE like any other dropped claim.
    """

    draft: AnswerDraft | None
    model: str
    input_tokens: int
    output_tokens: int
    note: str | None = None
    claims_trimmed: bool = False


class AnswerProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def max_tokens(self) -> int:
        """Output-token cap per call; bounds billed output for max_cost_usd."""
        ...

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult: ...


def parse_draft(text: str) -> AnswerDraft | None:
    return parse_draft_noted(text)[0]


@dataclass(frozen=True)
class ParsedDraft:
    draft: AnswerDraft | None
    note: str | None = None
    claims_trimmed: bool = False


def parse_draft_noted(text: str) -> tuple[AnswerDraft | None, str | None]:
    parsed = parse_draft_checked(text)
    return parsed.draft, parsed.note


def parse_draft_checked(text: str) -> ParsedDraft:
    """Parse the model output; salvage it if only optional content is over cap.

    Over-cap output can be trimmed without weakening any check: citations
    past MAX_CITATIONS_PER_CLAIM are dropped (a claim then needs support from
    fewer passages), and a non-essential claim that is malformed, too long,
    or past MAX_CLAIMS is dropped (never shown). If an essential claim would
    be lost, the draft is invalid, as before.
    """
    try:
        return ParsedDraft(AnswerDraft.model_validate_json(text))
    except ValidationError as exc:
        reason = _describe(exc)
    salvaged = _salvage(text)
    if salvaged is None:
        return ParsedDraft(None, f"invalid: {reason}")
    draft, dropped = salvaged
    return ParsedDraft(draft, f"salvaged: {reason}", claims_trimmed=dropped > 0)


def _describe(exc: ValidationError) -> str:
    """Error locations and types; never the offending values."""
    parts = {
        ".".join("#" if isinstance(x, int) else str(x) for x in e["loc"])
        + f":{e['type']}"
        for e in exc.errors()
    }
    return ",".join(sorted(parts))[:300]


def _salvage(text: str) -> tuple[AnswerDraft, int] | None:
    """The trimmed draft and how many claims were dropped, or None."""
    try:
        raw = json.loads(text)
    except ValueError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("claims"), list):
        return None
    claims: list[DraftClaim] = []
    dropped = 0
    for item in raw["claims"]:
        essential = not isinstance(item, dict) or item.get("essential") is not False
        claim = None
        if isinstance(item, dict):
            citations = item.get("citations")
            if isinstance(citations, list):
                item = {**item, "citations": citations[:MAX_CITATIONS_PER_CLAIM]}
            try:
                claim = DraftClaim.model_validate(item)
            except ValidationError:
                claim = None
        if claim is None or len(claims) >= MAX_CLAIMS:
            if essential:
                return None
            dropped += 1
            continue
        claims.append(claim)
    try:
        return AnswerDraft.model_validate({**raw, "claims": claims}), dropped
    except ValidationError:
        return None


class AnthropicAnswerProvider:
    """``AnswerProvider`` using the Messages API with JSON-schema output."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_ANSWER_MODEL,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        from anthropic import AsyncAnthropic

        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        )
        parsed = ParsedDraft(None, f"invalid: stop={response.stop_reason}")
        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if b.type == "text")
            parsed = parse_draft_checked(text)
        draft, note = parsed.draft, parsed.note
        if draft is None:
            logger.warning(
                "answer model output unusable: stop=%s note=%s",
                response.stop_reason,
                note,
            )
        return ProviderResult(
            draft=draft,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            note=note,
            claims_trimmed=parsed.claims_trimmed,
        )

    async def aclose(self) -> None:
        await self._client.close()

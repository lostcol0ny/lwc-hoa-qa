"""Answer-model provider; the only module that imports ``anthropic``."""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hoa_qa.answer.prompt import AnswerPrompt

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 4096


class DraftCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    quote: str


class AnswerDraft(BaseModel):
    """The answer model's structured output, before verification."""

    model_config = ConfigDict(extra="forbid")

    answer_text: str
    citations: list[DraftCitation]
    confidence: float = Field(ge=0, le=1)
    conflicts_noted: list[str]


# Structured-output JSON schema. Numeric bounds are enforced by AnswerDraft,
# since output schemas do not accept minimum/maximum.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_text": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["chunk_id", "quote"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
        "conflicts_noted": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer_text", "citations", "confidence", "conflicts_noted"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProviderResult:
    """``draft`` is None when the output was missing or failed validation."""

    draft: AnswerDraft | None
    model: str
    input_tokens: int
    output_tokens: int


class AnswerProvider(Protocol):
    @property
    def model(self) -> str: ...

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult: ...


def parse_draft(text: str) -> AnswerDraft | None:
    try:
        return AnswerDraft.model_validate_json(text)
    except ValidationError:
        return None


class AnthropicAnswerProvider:
    """``AnswerProvider`` using the Messages API with JSON-schema output."""

    def __init__(self, *, api_key: str, model: str = DEFAULT_ANSWER_MODEL) -> None:
        from anthropic import AsyncAnthropic

        self._model = model
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        )
        draft = None
        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if b.type == "text")
            draft = parse_draft(text)
        if draft is None:
            logger.warning(
                "answer model output unusable: stop=%s", response.stop_reason
            )
        return ProviderResult(
            draft=draft,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._client.close()

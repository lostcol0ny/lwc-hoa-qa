"""Real SDK adapters exercised offline through httpx2 mock transports."""

import asyncio
import json

import httpx2
from anthropic import AsyncAnthropic
from typesafe_sdk import AsyncTypeSafeClient

from hoa_qa.answer.prompt import AnswerPrompt
from hoa_qa.answer.provider import AnthropicAnswerProvider
from hoa_qa.retrieval.jev import NoulQuestion, TypeSafeJevClient


def test_typesafe_adapter_batches_nouls_in_one_request() -> None:
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 321, "output_tokens": 0},
                "answers": {
                    "p0": {"type": "noul", "noul": 0.8},
                    "p1": {"type": "noul", "noul": 0.1},
                },
            },
        )

    jev = TypeSafeJevClient(api_key="test-key", model="jev-latest")
    jev._client = AsyncTypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handler)
    )
    result = asyncio.run(
        jev.nouls(
            {"question": "q", "passages": [{"text": "a"}, {"text": "b"}]},
            {
                "p0": NoulQuestion("Is `passages[0]` relevant?", yes="y", no="n"),
                "p1": NoulQuestion("Is `passages[1]` relevant?"),
            },
        )
    )
    assert result.probabilities == {"p0": 0.8, "p1": 0.1}
    assert result.input_tokens == 321
    [body] = seen
    assert body["model"] == "jev-latest"
    assert body["questions"]["p0"] == {
        "type": "noul",
        "instructions": "Is `passages[0]` relevant?",
        "criteria": {"true": "y", "false": "n"},
    }
    assert body["questions"]["p1"] == {
        "type": "noul",
        "instructions": "Is `passages[1]` relevant?",
    }


def test_anthropic_adapter_uses_json_schema_output() -> None:
    seen: list[dict] = []
    output = {
        "answer_text": "A second violation is $125.",
        "citations": [{"chunk_id": "rules-2023-fines", "quote": "$125"}],
        "confidence": 0.9,
        "conflicts_noted": [],
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": json.dumps(output)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1500, "output_tokens": 120},
            },
        )

    provider = AnthropicAnswerProvider(api_key="test-key")
    provider._client = AsyncAnthropic(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    result = asyncio.run(provider.generate(AnswerPrompt(system="S", user="U")))
    assert result.draft is not None
    assert result.draft.citations[0].quote == "$125"
    assert (result.model, result.input_tokens, result.output_tokens) == (
        "claude-haiku-4-5",
        1500,
        120,
    )
    [body] = seen
    assert body["model"] == "claude-haiku-4-5"
    assert body["system"] == "S"
    assert body["messages"] == [{"role": "user", "content": "U"}]
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_adapter_rejects_truncated_output() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": '{"answer_text": "A'}],
                "stop_reason": "max_tokens",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 4096},
            },
        )

    provider = AnthropicAnswerProvider(api_key="test-key")
    provider._client = AsyncAnthropic(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    result = asyncio.run(provider.generate(AnswerPrompt(system="S", user="U")))
    assert result.draft is None
    assert result.output_tokens == 4096  # still billed

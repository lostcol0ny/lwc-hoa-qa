"""Model clients wrapped so every request becomes a span (see ``telemetry``).

``build_asker`` wraps whatever Jev client and answer provider it is given, so
real clients and test fakes are traced the same way. With tracing off the
wrappers add nothing but a function call.
"""

from collections.abc import Mapping
from typing import Any

from hoa_qa import telemetry
from hoa_qa.answer.pricing import answer_price, jev_cost_usd
from hoa_qa.answer.prompt import AnswerPrompt
from hoa_qa.answer.provider import AnswerProvider, OnClaim, ProviderResult
from hoa_qa.retrieval.jev import (
    JevClient,
    NoulBatchResult,
    NoulQuestion,
    PartialCostError,
)

# Sweep and support requests carry whole passages. A question sweeps the
# entire corpus, so their text is summarized instead of copied into every
# trace: the passage is identified and sized, and the answer model's span
# holds the full text of the few passages that matter.
SUPPORT_TEXT_CHARS = 300


def _request_name(questions: Mapping[str, NoulQuestion]) -> str:
    first = next(iter(questions), "")
    if first == "on_topic":
        return "jev gate"
    return {
        "k": "jev cache match",
        "p": "jev sweep batch",
        "c": "jev support",
    }.get(first[:1], "jev request")


def _shown_state(state: Mapping[str, Any]) -> dict[str, Any]:
    shown = dict(state)
    passages = state.get("passages")
    if isinstance(passages, Mapping):
        shown["passages"] = {
            name: {"heading": p.get("heading"), "chars": len(p.get("text", ""))}
            for name, p in passages.items()
        }
    claims = state.get("claims")
    if isinstance(claims, list):
        shown["claims"] = [
            {
                **claim,
                "passages": [
                    {**p, "text": p.get("text", "")[:SUPPORT_TEXT_CHARS]}
                    for p in claim.get("passages", [])
                ],
            }
            for claim in claims
        ]
    return shown


class TracedJev:
    """A ``JevClient`` that records each request as a generation span."""

    def __init__(self, inner: JevClient, model: str) -> None:
        self._inner = inner
        self._model = model

    async def nouls(
        self, state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> NoulBatchResult:
        if not telemetry.enabled():
            return await self._inner.nouls(state, questions)
        with telemetry.span(_request_name(questions), "generation") as span:
            span.set(questions=len(questions))
            if telemetry.content_enabled():
                span.input(
                    {
                        "state": _shown_state(state),
                        "questions": {
                            name: q.instructions for name, q in questions.items()
                        },
                    }
                )
            try:
                result = await self._inner.nouls(state, questions)
            except PartialCostError as exc:
                # A billed request that still failed: keep its cost visible.
                self._usage(span, exc.jev_input_tokens)
                raise
            self._usage(span, result.input_tokens)
            # Probabilities keyed by question name hold no text.
            span.output(dict(result.probabilities), text=False)
            return result

    def _usage(self, span: telemetry.Observation, tokens: int) -> None:
        span.model(self._model, input_tokens=tokens, input_usd=jev_cost_usd(tokens))

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


class TracedProvider:
    """An ``AnswerProvider`` that records each draft as a generation span."""

    def __init__(self, inner: AnswerProvider) -> None:
        self._inner = inner

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def max_tokens(self) -> int:
        return self._inner.max_tokens

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult:
        return await self._traced(prompt, None)

    async def generate_streaming(
        self, prompt: AnswerPrompt, on_claim: OnClaim
    ) -> ProviderResult:
        """Stream when the wrapped provider can; otherwise a plain ``generate``."""
        return await self._traced(prompt, on_claim)

    async def _call(
        self, prompt: AnswerPrompt, on_claim: OnClaim | None
    ) -> ProviderResult:
        streaming = getattr(self._inner, "generate_streaming", None)
        if on_claim is not None and streaming is not None:
            return await streaming(prompt, on_claim)
        return await self._inner.generate(prompt)

    async def _traced(
        self, prompt: AnswerPrompt, on_claim: OnClaim | None
    ) -> ProviderResult:
        if not telemetry.enabled():
            return await self._call(prompt, on_claim)
        with telemetry.span("answer model draft", "generation") as span:
            span.input(
                [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ]
            )
            result = await self._call(prompt, on_claim)
            span.set(streamed=on_claim is not None)
            price = answer_price(result.model)
            span.model(
                result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                input_usd=result.input_tokens * price.input_usd_per_mtok / 1e6,
                output_usd=result.output_tokens * price.output_usd_per_mtok / 1e6,
            )
            span.set(
                valid=result.draft is not None,
                claims=len(result.draft.claims) if result.draft else 0,
                note=result.note,
            )
            if result.draft is not None:
                span.output(result.draft.model_dump(mode="json"))
            return result

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

"""Fake Jev and answer providers for the QA tests. No network, no keys."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hoa_qa.answer.prompt import AnswerPrompt
from hoa_qa.answer.provider import AnswerDraft, DraftCitation, ProviderResult
from hoa_qa.ask import QAAsker, QASettings, build_asker
from hoa_qa.models import Corpus
from hoa_qa.retrieval.jev import NoulBatchResult, NoulQuestion

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mini_corpus.json"
JEV_TOKENS_PER_CALL = 1_000
ANSWER_INPUT_TOKENS = 2_000
ANSWER_OUTPUT_TOKENS = 300


@dataclass
class FakeJev:
    """Routes by question name: ``on_topic`` (gate), ``p*`` (sweep), ``c*``."""

    gate: float = 0.95
    relevance: Mapping[str, float] = field(default_factory=dict)
    support: float | Callable[[str], float] = 0.9
    calls: list[tuple[dict[str, Any], dict[str, NoulQuestion]]] = field(
        default_factory=list
    )
    text_to_id: dict[str, str] = field(default_factory=dict)

    async def nouls(
        self, state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> NoulBatchResult:
        self.calls.append((dict(state), dict(questions)))
        probabilities: dict[str, float] = {}
        for name in questions:
            if name == "on_topic":
                probabilities[name] = self.gate
            elif name.startswith("p"):
                text = state["passages"][int(name[1:])]["text"]
                probabilities[name] = self.relevance.get(self.text_to_id[text], 0.0)
            else:
                passage = state["citations"][int(name[1:])]["passage"]
                support = self.support
                probabilities[name] = (
                    support(self.text_to_id[passage]) if callable(support) else support
                )
        return NoulBatchResult(probabilities, JEV_TOKENS_PER_CALL)

    def calls_of(self, kind: str) -> list[dict[str, NoulQuestion]]:
        return [
            q
            for _, q in self.calls
            if (kind == "gate" and "on_topic" in q)
            or (kind != "gate" and next(iter(q)).startswith(kind))
        ]


@dataclass
class FakeProvider:
    drafts: list[AnswerDraft | None]
    model: str = "claude-haiku-4-5"
    prompts: list[AnswerPrompt] = field(default_factory=list)

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult:
        self.prompts.append(prompt)
        draft = self.drafts[min(len(self.prompts), len(self.drafts)) - 1]
        return ProviderResult(
            draft, self.model, ANSWER_INPUT_TOKENS, ANSWER_OUTPUT_TOKENS
        )


def draft(*citations: tuple[str, str], text: str = "Answer.") -> AnswerDraft:
    return AnswerDraft(
        answer_text=text,
        citations=[DraftCitation(chunk_id=c, quote=q) for c, q in citations],
        confidence=0.8,
        conflicts_noted=[],
    )


FINES_DRAFT = draft(
    ("rules-2023-fines", "Second violation: $125."),
    text="A second violation is a $125 fine under the 2023 Rules.",
)


def make_asker(
    corpus: Corpus, settings: QASettings, jev: FakeJev, provider: FakeProvider
) -> QAAsker:
    return build_asker(corpus, settings, jev=jev, provider=provider)

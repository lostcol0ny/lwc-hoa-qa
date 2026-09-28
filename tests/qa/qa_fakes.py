"""Fake Jev and answer providers for the QA tests. No network, no keys.

Both fakes bill realistic token counts (the same conservative bound the
pipeline uses for sizing) so cost and ``max_cost_usd`` tests are meaningful.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from hoa_qa.answer.prompt import AnswerPrompt
from hoa_qa.answer.provider import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnswerDraft,
    DraftCitation,
    DraftClaim,
    DraftIssue,
    ProviderResult,
)
from hoa_qa.ask import QAAsker, QASettings, build_asker
from hoa_qa.models import Authority, Chunk, Corpus, StatuteCompilation
from hoa_qa.retrieval.jev import (
    NoulBatchResult,
    NoulQuestion,
    PartialCostError,
    conservative_tokens,
    fits,
    request_tokens,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mini_corpus.json"

SupportFn = Callable[[str, Sequence[str]], float]


@dataclass
class FakeJev:
    """Routes by question name: ``on_topic`` (gate), ``p*`` (sweep), ``c*``.

    ``support`` is a constant or ``f(statement, cited_chunk_ids)``. ``fail``
    decides, per request, whether to raise instead of answering (unbilled);
    ``malformed`` bills the request and then raises ``PartialCostError``, like
    a billed response that is missing a judgment.
    """

    gate: float = 0.95
    relevance: Mapping[str, float] = field(default_factory=dict)
    support: float | SupportFn = 0.9
    fail: Callable[[Mapping[str, Any]], bool] = lambda state: False
    malformed: Callable[[Mapping[str, Any]], bool] = lambda state: False
    calls: list[tuple[dict[str, Any], dict[str, NoulQuestion]]] = field(
        default_factory=list
    )
    billed: list[int] = field(default_factory=list)
    text_to_id: dict[str, str] = field(default_factory=dict)

    async def nouls(
        self, state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> NoulBatchResult:
        # Every request the pipeline sends must be within Jev's limits.
        assert fits(state, questions)
        self.calls.append((dict(state), dict(questions)))
        if self.fail(state):
            raise RuntimeError("jev unavailable")
        probabilities: dict[str, float] = {}
        for name in questions:
            if name == "on_topic":
                probabilities[name] = self.gate
            elif name.startswith("p"):
                text = state["passages"][name]["text"]
                probabilities[name] = self.relevance.get(self.text_to_id[text], 0.0)
            else:
                claim = state["claims"][int(name[1:])]
                ids = [self.text_to_id[p["text"]] for p in claim["passages"]]
                support = self.support
                probabilities[name] = (
                    support(claim["statement"], ids) if callable(support) else support
                )
        tokens = self.bill(state, questions)
        self.billed.append(tokens)
        if self.malformed(state):
            raise PartialCostError(tokens, [RuntimeError("missing judgment")])
        return NoulBatchResult(probabilities, tokens)

    def bill(
        self, state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> int:
        return request_tokens(state, questions)[0]

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
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    prompts: list[AnswerPrompt] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    # Per call: whether salvage dropped claims (ProviderResult.claims_trimmed).
    trimmed: list[bool] = field(default_factory=list)
    # Per call: why an invalid (None) draft was invalid (ProviderResult.issues).
    issues: list[tuple[DraftIssue, ...]] = field(default_factory=list)

    async def generate(self, prompt: AnswerPrompt) -> ProviderResult:
        self.prompts.append(prompt)
        draft = self.drafts[min(len(self.prompts), len(self.drafts)) - 1]
        tokens_in, tokens_out = self.bill(prompt, draft)
        self.input_tokens.append(tokens_in)
        self.output_tokens.append(tokens_out)
        call = len(self.prompts) - 1
        trimmed = self.trimmed[call] if call < len(self.trimmed) else False
        issues = self.issues[call] if call < len(self.issues) else ()
        return ProviderResult(
            draft,
            self.model,
            tokens_in,
            tokens_out,
            claims_trimmed=trimmed,
            issues=issues,
        )

    def bill(self, prompt: AnswerPrompt, draft: AnswerDraft | None) -> tuple[int, int]:
        return (
            conservative_tokens(prompt.system + prompt.user),
            conservative_tokens(draft.model_dump_json() if draft else "{"),
        )


def claim(
    statement: str,
    *citations: tuple[str, str],
    kind: str = "answer",
    essential: bool = True,
) -> DraftClaim:
    return DraftClaim.model_validate(
        {
            "statement": statement,
            "kind": kind,
            "essential": essential,
            "citations": [
                DraftCitation(chunk_id=c, quote=q).model_dump() for c, q in citations
            ],
        }
    )


def draft(*claims: DraftClaim, refer_to_board: bool = False) -> AnswerDraft:
    return AnswerDraft(
        claims=list(claims), confidence=0.8, refer_to_board=refer_to_board
    )


FINE_CLAIM = claim(
    "A third violation is a $125 fine under the 2023 Rules.",
    ("rules-2023-fines", "3rd violation: $125."),
)
FINES_DRAFT = draft(FINE_CLAIM)


def make_asker(
    corpus: Corpus, settings: QASettings, jev: FakeJev, provider: FakeProvider
) -> QAAsker:
    return build_asker(corpus, settings, jev=jev, provider=provider)


STATUTE_MEETINGS = "(a) The board shall meet at least 4 times annually."
BYLAWS_MEETINGS = (
    "Regular meetings of the Board of Directors shall be held at least twice each year."
)
EXEMPTION = (
    "(a) A common interest community association organized under the General "
    "Not for Profit Corporation Act of 1986 and having either (i) 10 units or "
    "less or (ii) annual budgeted assessments of $100,000 or less shall be "
    "exempt from this Act unless the association affirmatively elects to be "
    "covered by this Act by a majority of its directors or members."
)
HOMES = (
    "The goal of this committee will be to bring new ideas and strategies on how "
    "to successfully get the 67% (492 of 735 homes) vote needed to amend the "
    "Declaration of CC&Rs."
)
DUES = "2026 Assessment Prices: $452 year or $113 per quarter"
COMPILATION = StatuteCompilation(
    through_public_act="104-433", updated_on=date(2025, 11, 21)
)


def with_statutes(corpus: Corpus) -> Corpus:
    """The mini corpus plus two CICAA sections, a Bylaws clause that differs
    from one of them, and the chunks the CICAA applicability note cites."""
    bylaws = next(c for c in corpus.chunks if c.doc_id == "bylaws")
    website = next(c for c in corpus.chunks if c.authority is Authority.website)
    blog = next(c for c in corpus.chunks if c.authority is Authority.informal)

    def statute(section: str, caption: str, text: str) -> Chunk:
        label = f"765 ILCS 160/{section} ({caption})"
        body = f"(765 ILCS 160/{section})\nSec. {section}. {caption}.\n{text}"
        return Chunk(
            id=f"cicaa-{section}",
            doc_id="cicaa",
            doc_title="Common Interest Community Association Act",
            source_url="https://ftp.ilga.gov/ILCS/Ch%200765/Act%200160/"
            f"076501600K{section}.html",
            page_start=None,
            page_end=None,
            citation_label=label,
            heading_path=("Common Interest Community Association Act (765 ILCS 160)",),
            text_clean=body,
            text_raw=body,
            authority=Authority.statute,
            effective_date=date(2024, 1, 1),
            published_date=None,
            superseded_by=None,
            token_estimate=len(body) // 4,
        )

    def copy(chunk: Chunk, chunk_id: str, text: str, **update: object) -> Chunk:
        return chunk.model_copy(
            update={"id": chunk_id, "text_clean": text, "text_raw": text, **update}
        )

    extra = (
        statute("1-30", "Board duties and obligations; records", STATUTE_MEETINGS),
        statute("1-75", "Exemptions for small common interest communities", EXEMPTION),
        copy(bylaws, "bylaws-5.1", BYLAWS_MEETINGS, citation_label="Bylaws, §5.1"),
        copy(blog, "amendments-committee-intro", HOMES),
        copy(website, "home-1", DUES),
    )
    hashes = {**corpus.manifest.source_hashes, "cicaa": "0" * 64}
    manifest = corpus.manifest.model_copy(
        update={
            "chunk_count": len(corpus.chunks) + len(extra),
            "source_hashes": hashes,
            "statute_compilation": COMPILATION,
        }
    )
    return Corpus(manifest=manifest, chunks=(*corpus.chunks, *extra))

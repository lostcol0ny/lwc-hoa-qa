"""Build the curated answer cache: answer each question once and store it.

A curated question either carries reviewed claims (the operator's own wording,
verified here against the corpus) or has none, in which case the answer model
drafts the answer and the claims it kept are reported, so they can be reviewed
and committed for the next build. Kept apart from ``hoa_qa.cache`` because the
pipeline imports that module and this one imports the pipeline.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from hoa_qa.answer.provider import AnswerDraft, DraftCitation, DraftClaim
from hoa_qa.ask import Asker
from hoa_qa.cache import (
    CacheEntry,
    CuratedQuestion,
    CuratedQuestions,
    ReviewedCitation,
    ReviewedClaim,
)
from hoa_qa.eval import AttemptDiag, DiagnosticsRecorder
from hoa_qa.models import Answer, Outcome


@dataclass(frozen=True)
class BuildRow:
    id: str
    source: Literal["reviewed", "generated"]
    outcome: Outcome
    cached: bool
    cost_usd: float
    # The claims behind the stored answer: the reviewed ones, or the ones the
    # answer model wrote and verification kept.
    claims: tuple[ReviewedClaim, ...] = ()
    # Why a reviewed entry was not stored, per claim: codes and scores only.
    problems: tuple[str, ...] = ()


def reviewed_draft(item: CuratedQuestion) -> AnswerDraft:
    """The operator's claims in the shape the verifier takes."""
    return AnswerDraft(
        claims=[
            DraftClaim(
                statement=claim.statement,
                kind=claim.kind,
                # Nothing may be dropped from a reviewed answer, so the flag
                # only mirrors the kind.
                essential=claim.kind == "answer",
                citations=[
                    DraftCitation(chunk_id=c.chunk_id, quote=c.quote)
                    for c in claim.citations
                ],
            )
            for claim in item.claims
        ],
        # Reviewed by a person and verified in full.
        confidence=1.0,
        refer_to_board=item.refer_to_board,
    )


def _problems(attempts: Sequence[AttemptDiag]) -> tuple[str, ...]:
    if not attempts:
        return ("not verified",)
    found = []
    for number, claim in enumerate(attempts[-1].claims, start=1):
        if not claim.kept:
            score = "" if claim.support is None else f" (support {claim.support:.2f})"
            found.append(f"claim {number}: {claim.reason}{score}")
    return tuple(found) or ("no answer claim",)


def _claims_behind(
    answer: Answer, attempts: Sequence[AttemptDiag]
) -> tuple[ReviewedClaim, ...]:
    """The kept claims of the attempt the answer was assembled from.

    Usually the last attempt; the first when a worse retry fell back to it.
    """
    for attempt in reversed(attempts):
        kept = [claim for claim in attempt.claims if claim.kept]
        answers = [c.statement.strip() for c in kept if c.kind == "answer"]
        conflicts = tuple(c.statement.strip() for c in kept if c.kind == "conflict")
        if (
            answers
            and all(statement in answer.answer_text for statement in answers)
            and conflicts == answer.conflicts_noted
        ):
            return tuple(
                ReviewedClaim(
                    statement=claim.statement.strip(),
                    kind="conflict" if claim.kind == "conflict" else "answer",
                    citations=tuple(
                        ReviewedCitation(chunk_id=c.chunk_id, quote=c.quote)
                        for c in claim.citations
                        if c.used
                    ),
                )
                for claim in kept
            )
    return ()


async def build_entries(
    questions: CuratedQuestions, asker: Asker
) -> tuple[tuple[CacheEntry, ...], list[BuildRow]]:
    """Answer every curated question; keep the ones that come out answered.

    ``asker`` must be the full pipeline with no cache attached. Questions run
    one at a time, like the eval, to keep the API rate and spend modest.
    """
    traced = getattr(asker, "ask_traced", None)
    entries: list[CacheEntry] = []
    rows: list[BuildRow] = []
    for item in questions.questions:
        recorder = DiagnosticsRecorder()
        if item.claims:
            if traced is None:
                raise ValueError("this asker cannot verify reviewed claims")
            result = await traced(
                item.question, recorder, reviewed=reviewed_draft(item)
            )
        elif traced is not None:
            result = await traced(item.question, recorder)
        else:
            result = await asker(item.question)
        answer = result.answer
        answered = answer.outcome is Outcome.answered
        if answered:
            entries.append(
                CacheEntry(id=item.id, question=item.question, answer=answer)
            )
        attempts = recorder.data.attempts
        if item.claims:
            claims = item.claims
        else:
            claims = _claims_behind(answer, attempts) if answered else ()
        rows.append(
            BuildRow(
                id=item.id,
                source="reviewed" if item.claims else "generated",
                outcome=answer.outcome,
                cached=answered,
                cost_usd=result.estimated_cost_usd,
                claims=claims,
                problems=_problems(attempts) if item.claims and not answered else (),
            )
        )
    return tuple(entries), rows


def with_claims(
    questions: CuratedQuestions, rows: Sequence[BuildRow]
) -> CuratedQuestions:
    """The question list with every built answer's claims filled in.

    Written out by ``hoa-qa cache build --propose`` for review: committing it
    makes the next build verify that exact wording instead of drafting anew.
    """
    by_id = {row.id: row for row in rows}
    return CuratedQuestions(
        questions=tuple(
            item.model_copy(update={"claims": by_id[item.id].claims or item.claims})
            for item in questions.questions
        )
    )

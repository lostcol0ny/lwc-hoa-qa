"""Golden-set evaluation (spec §7): run the real asker over ``evals/golden.yaml``.

Each case names the expected outcome, chunk ids of which at least one must be
cited, and phrases that must or must not appear (case-insensitive substrings).
The two phrase checks read different text on purpose (see ``GoldenCase``).
Citation quotes are verbatim document text, so they're never scored. The runner
prints a pass/fail table and the total cost, and fails below a minimum pass
rate.

The golden questions are committed fixtures, not user input, so the JSON
report includes the answers for review. Nothing here logs question or answer
text.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from hoa_qa.ask import Asker
from hoa_qa.models import Answer, Corpus, Outcome

DEFAULT_GOLDEN_PATH = Path("evals/golden.yaml")
DEFAULT_IDS_PATH = Path("evals/corpus_ids.txt")
DEFAULT_MIN_PASS = 0.85

Phrase = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PhraseOrAlternatives = Phrase | Annotated[list[Phrase], Field(min_length=1)]
"""A phrase, or a list of alternatives of which any one satisfies the check."""


class GoldenCase(BaseModel):
    """One golden question and what a correct answer looks like.

    - ``must_include``: every entry must appear in ``answer_text`` **or**
      ``conflicts_noted`` (both are shown to the user). An entry that is a list
      of alternatives passes if any one of them appears.
    - ``must_not_include``: no entry may appear in ``answer_text``.
      ``conflicts_noted`` is **not** checked, because reporting a superseded or
      informal figure there ("the 2016 rules said $100") is the correct way to
      flag a conflict, not a wrong answer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
    question: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    expect_outcome: Outcome
    expect_chunk_ids_any: tuple[str, ...] = ()
    must_include: tuple[PhraseOrAlternatives, ...] = ()
    must_not_include: tuple[Phrase, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def citations_only_for_answers(self) -> Self:
        answered = self.expect_outcome == Outcome.answered
        if answered and not self.expect_chunk_ids_any:
            raise ValueError(f"{self.id}: answered cases need expect_chunk_ids_any")
        if not answered and self.expect_chunk_ids_any:
            raise ValueError(f"{self.id}: only answered cases can expect citations")
        return self


class GoldenSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cases: tuple[GoldenCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate case ids")
        return self

    def chunk_ids(self) -> set[str]:
        return {cid for case in self.cases for cid in case.expect_chunk_ids_any}


def load_golden(path: Path) -> GoldenSet:
    # PyYAML ships in the ingest/dev dependency groups, not the web runtime.
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "the eval needs PyYAML: `uv sync --group ingest` (or the dev group)"
        ) from exc
    return GoldenSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def missing_chunk_ids(golden: GoldenSet, known: Iterable[str]) -> list[str]:
    return sorted(golden.chunk_ids() - set(known))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_ids(corpus: Corpus, path: Path) -> None:
    """Commit-safe id manifest: chunk ids only, one per line, corpus order."""
    path.write_text("".join(f"{chunk.id}\n" for chunk in corpus.chunks))


class Check(BaseModel):
    name: str
    passed: bool


class CaseResult(BaseModel):
    id: str
    passed: bool
    checks: list[Check]
    outcome: Outcome
    cited: list[str]
    cost_usd: float
    answer_text: str
    conflicts_noted: list[str]


class EvalReport(BaseModel):
    cases: list[CaseResult]
    passed: int
    total: int
    pass_rate: float
    total_cost_usd: float
    max_cost_usd: float | None


def _shown_text(answer: Answer) -> str:
    return " ".join((answer.answer_text, *answer.conflicts_noted)).casefold()


def _answer_text(answer: Answer) -> str:
    return answer.answer_text.casefold()


def score_case(case: GoldenCase, answer: Answer) -> list[Check]:
    """One check per expectation; the case passes only if all pass."""
    checks = [
        Check(
            name=f"outcome={case.expect_outcome.value}",
            passed=answer.outcome == case.expect_outcome,
        )
    ]
    if case.expect_chunk_ids_any:
        cited = {citation.chunk_id for citation in answer.citations}
        checks.append(
            Check(
                name="cites_any_of_expected",
                passed=bool(cited & set(case.expect_chunk_ids_any)),
            )
        )
    shown = _shown_text(answer)
    for phrase in case.must_include:
        options = [phrase] if isinstance(phrase, str) else phrase
        checks.append(
            Check(
                name="includes " + " | ".join(repr(o) for o in options),
                passed=any(option.casefold() in shown for option in options),
            )
        )
    stated = _answer_text(answer)  # conflicts_noted may cite superseded figures
    for phrase in case.must_not_include:
        checks.append(
            Check(name=f"excludes {phrase!r}", passed=phrase.casefold() not in stated)
        )
    return checks


async def run_eval(golden: GoldenSet, asker: Asker) -> EvalReport:
    """Run every case sequentially (keeps the API rate and spend modest)."""
    results: list[CaseResult] = []
    total_cost = 0.0
    for case in golden.cases:
        result = await asker(case.question)
        answer = result.answer
        checks = score_case(case, answer)
        total_cost += result.estimated_cost_usd
        results.append(
            CaseResult(
                id=case.id,
                passed=all(check.passed for check in checks),
                checks=checks,
                outcome=answer.outcome,
                cited=[citation.chunk_id for citation in answer.citations],
                cost_usd=result.estimated_cost_usd,
                answer_text=answer.answer_text,
                conflicts_noted=list(answer.conflicts_noted),
            )
        )
    passed = sum(r.passed for r in results)
    declared: Any = getattr(asker, "max_cost_usd", None)
    return EvalReport(
        cases=results,
        passed=passed,
        total=len(results),
        pass_rate=passed / len(results),
        total_cost_usd=total_cost,
        max_cost_usd=float(declared) if isinstance(declared, int | float) else None,
    )


def render_table(report: EvalReport) -> str:
    """A plain-text table: ids, outcomes and failed checks, never answer text."""
    width = max(len(r.id) for r in report.cases)
    lines = [f"{'case':<{width}}  result  outcome            cost_usd  failed checks"]
    for r in report.cases:
        failed = "; ".join(c.name for c in r.checks if not c.passed)
        lines.append(
            f"{r.id:<{width}}  {'PASS' if r.passed else 'FAIL':<6}  "
            f"{r.outcome.value:<17}  {r.cost_usd:8.5f}  {failed}"
        )
    lines.append("")
    lines.append(
        f"passed {report.passed}/{report.total} ({report.pass_rate:.0%}); "
        f"total cost ${report.total_cost_usd:.4f}"
        + (
            f"; max_cost_usd per question ${report.max_cost_usd:.4f}"
            if report.max_cost_usd is not None
            else ""
        )
    )
    return "\n".join(lines)


def write_report(report: EvalReport, path: Path) -> None:
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")

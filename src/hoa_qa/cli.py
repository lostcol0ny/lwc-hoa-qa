"""Command-line entry points.

- ``hoa-qa ask "question" --corpus corpus.json``
- ``hoa-qa eval evals/golden.yaml --corpus build/corpus.json [--json out.json]``
- ``hoa-qa eval --write-ids --corpus build/corpus.json`` (refresh the id manifest)
- ``hoa-qa cache build evals/cache_questions.yaml --corpus build/corpus.json
  --out build/answer_cache.json [--propose drafted.yaml]`` (verify the reviewed
  answers; draft any question that has none yet)
- ``hoa-qa cache check evals/cache_cases.yaml --cache build/answer_cache.json``
  (does the matcher hit the paraphrases and miss the near-misses?)
- ``hoa-qa cache verify --cache answer_cache.json --corpus corpus.json`` (was
  the cache built from this corpus? no API calls)
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from hoa_qa import telemetry
from hoa_qa.ask import AskerFactory, AskResult, QASettings, build_asker
from hoa_qa.cache import (
    AnswerCache,
    CuratedQuestions,
    MatchRow,
    check_matches,
    corpus_fingerprint,
    load_cache,
    load_match_cases,
    load_questions,
)
from hoa_qa.cache_build import BuildRow, build_entries, with_claims
from hoa_qa.eval import (
    DEFAULT_GOLDEN_PATH,
    DEFAULT_IDS_PATH,
    DEFAULT_MIN_PASS,
    EvalReport,
    load_golden,
    missing_chunk_ids,
    render_table,
    run_eval,
    write_ids,
    write_report,
)
from hoa_qa.models import Corpus, load_corpus
from hoa_qa.retrieval.jev import JevClient, TypeSafeJevClient

DEFAULT_QUESTIONS_PATH = Path("evals/cache_questions.yaml")
DEFAULT_MATCH_CASES_PATH = Path("evals/cache_cases.yaml")

JevFactory = Callable[[QASettings], JevClient]


def build_jev(settings: QASettings) -> JevClient:
    if settings.typesafe_api_key is None:
        raise ValueError("TYPESAFE_API_KEY is not set")
    return TypeSafeJevClient(
        api_key=settings.typesafe_api_key.get_secret_value(), model=settings.jev_model
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lakewood Creek HOA Q&A")
    parser.add_argument("--version", action="version", version=version("hoa-qa"))
    commands = parser.add_subparsers(dest="command")
    ask = commands.add_parser("ask", help="Answer one question from the corpus")
    ask.add_argument("question")
    ask.add_argument("--corpus", required=True, type=Path)
    ask.add_argument("--json", action="store_true", help="Print the full result")

    ev = commands.add_parser("eval", help="Score the golden set with the real asker")
    ev.add_argument("golden", nargs="?", type=Path, default=DEFAULT_GOLDEN_PATH)
    ev.add_argument("--corpus", required=True, type=Path)
    ev.add_argument("--json", type=Path, dest="json_out", help="Write results here")
    ev.add_argument(
        "--min-pass",
        type=float,
        default=DEFAULT_MIN_PASS,
        help=f"Exit 1 below this pass rate (default {DEFAULT_MIN_PASS})",
    )
    ev.add_argument(
        "--write-ids",
        nargs="?",
        type=Path,
        const=DEFAULT_IDS_PATH,
        metavar="PATH",
        help=f"Only write the corpus's chunk ids (default {DEFAULT_IDS_PATH})",
    )
    cache = commands.add_parser("cache", help="Build or check the answer cache")
    actions = cache.add_subparsers(dest="cache_command", required=True)
    build = actions.add_parser("build", help="Answer the curated questions once")
    build.add_argument(
        "questions", nargs="?", type=Path, default=DEFAULT_QUESTIONS_PATH
    )
    build.add_argument("--corpus", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument(
        "--propose",
        type=Path,
        metavar="PATH",
        help="Also write the question list with each answer's claims filled in",
    )
    check = actions.add_parser("check", help="Score the matcher on known pairs")
    check.add_argument("cases", nargs="?", type=Path, default=DEFAULT_MATCH_CASES_PATH)
    check.add_argument("--cache", required=True, type=Path)
    verify = actions.add_parser(
        "verify", help="Exit 0 if the cache was built from this corpus (no API calls)"
    )
    verify.add_argument("--cache", required=True, type=Path)
    verify.add_argument("--corpus", required=True, type=Path)
    return parser


async def _build_cache(
    factory: AskerFactory, corpus: Corpus, questions: CuratedQuestions
) -> tuple[AnswerCache, list[BuildRow]]:
    settings = QASettings.from_env()
    # No cache is attached: every stored answer comes from the full pipeline.
    asker = factory(corpus, settings)
    try:
        entries, rows = await build_entries(questions, asker)
    finally:
        close = getattr(asker, "aclose", None)
        if close is not None:
            await close()
    cache = AnswerCache(
        corpus_fingerprint=corpus_fingerprint(corpus),
        built_at=datetime.now(UTC),
        jev_model=settings.jev_model,
        answer_model=settings.answer_model,
        entries=entries,
    )
    return cache, rows


def _run_cache_build(args: argparse.Namespace, factory: AskerFactory) -> int:
    try:
        corpus = load_corpus(args.corpus)
        questions = load_questions(args.questions)
        cache, rows = asyncio.run(_build_cache(factory, corpus, questions))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"hoa-qa: {exc}", file=sys.stderr)
        return 2
    width = max(len(row.id) for row in rows)
    print(f"{'question':<{width}}  cached  source     outcome            cost_usd")
    for row in rows:
        print(
            f"{row.id:<{width}}  {'yes' if row.cached else 'NO':<6}  "
            f"{row.source:<9}  {row.outcome.value:<17}  {row.cost_usd:8.5f}"
        )
        # Codes and scores only, like every other log line.
        for problem in row.problems:
            print(f"{'':<{width}}    {problem}")
    total = sum(row.cost_usd for row in rows)
    print(f"\ncached {len(cache.entries)}/{len(rows)}; total cost ${total:.4f}")
    if not cache.entries:
        print("no question was answered; nothing written", file=sys.stderr)
        return 1
    # Write beside the target, then swap, so a reader never sees half a file.
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(cache.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    if args.propose is not None:
        _write_questions(with_claims(questions, rows), args.propose)
    rejected = [row.id for row in rows if row.source == "reviewed" and not row.cached]
    if rejected:
        # Reviewed wording that no longer verifies needs a person to look.
        print(f"reviewed answers failed verification: {rejected}", file=sys.stderr)
        return 1
    return 0


def _run_cache_verify(args: argparse.Namespace) -> int:
    """Would the app use this cache with this corpus? The deploy asks this."""
    try:
        cache = load_cache(args.cache)
        corpus = load_corpus(args.corpus)
    except (OSError, ValueError) as exc:
        # The type only: a validation message can echo file contents.
        print(f"hoa-qa: cache unusable ({type(exc).__name__})", file=sys.stderr)
        return 1
    if cache.corpus_fingerprint != corpus_fingerprint(corpus):
        print("hoa-qa: cache was built from a different corpus", file=sys.stderr)
        return 1
    built = cache.built_at.date().isoformat()
    print(f"answer cache ok: {len(cache.entries)} entries, built {built}")
    return 0


def _write_questions(questions: CuratedQuestions, path: Path) -> None:
    import yaml  # load_questions has already checked that PyYAML is installed

    data = questions.model_dump(mode="json", exclude_defaults=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88),
        encoding="utf-8",
    )


async def _check_cache(
    jev_factory: JevFactory, cache: AnswerCache, cases_path: Path
) -> list[MatchRow]:
    settings = QASettings.from_env()
    jev = jev_factory(settings)
    try:
        return await check_matches(
            jev,
            cache,
            load_match_cases(cases_path),
            threshold=settings.cache_threshold,
            limit=asyncio.Semaphore(settings.jev_concurrency),
        )
    finally:
        close = getattr(jev, "aclose", None)
        if close is not None:
            await close()


def _run_cache_check(args: argparse.Namespace, jev_factory: JevFactory) -> int:
    try:
        cache = load_cache(args.cache)
        rows = asyncio.run(_check_cache(jev_factory, cache, args.cases))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"hoa-qa: {exc}", file=sys.stderr)
        return 2
    cached = {entry.id for entry in cache.entries}
    # A case whose entry was not answered at build time can't be scored.
    scored = [r for r in rows if r.case.expect is None or r.case.expect in cached]
    print("case  result     expected                  got                       score")
    for number, row in enumerate(rows, start=1):
        if row not in scored:
            result = "skipped"
        elif row.false_hit:
            result = "FALSE HIT"
        elif row.missed:
            result = "missed"
        else:
            result = "ok"
        score = "-" if row.probability is None else f"{row.probability:.2f}"
        print(
            f"{number:>4}  {result:<9}  {row.case.expect or '(miss)':<24}  "
            f"{row.got or '(miss)':<24}  {score}"
        )
    false_hits = sum(r.false_hit for r in scored)
    wanted = [r for r in scored if r.case.expect is not None]
    hits = sum(not r.missed and not r.false_hit for r in wanted)
    print(
        f"\nfalse hits {false_hits}; paraphrases matched {hits}/{len(wanted)}; "
        f"skipped {len(rows) - len(scored)}"
    )
    # Only a false hit is unsafe; a missed paraphrase just takes the slow path.
    return 1 if false_hits else 0


def _render(result: AskResult) -> str:
    answer = result.answer
    lines = [answer.answer_text, ""]
    for citation in answer.citations:
        lines.append(f'- {citation.citation_label}: "{citation.quote}"')
        lines.append(f"  {citation.url}")
    if answer.conflicts_noted:
        lines.append("")
        lines.extend(f"Note: {conflict}" for conflict in answer.conflicts_noted)
    if answer.related_documents:
        lines += ["", "Related documents:"]
        for link in answer.related_documents:
            lines += [f"- {link.title}", f"  {link.url}"]
    lines += [
        "",
        answer.disclaimer,
        f"[{answer.outcome.value}] request_id={answer.request_id} "
        f"cost~${result.estimated_cost_usd:.5f}",
    ]
    return "\n".join(lines)


async def _ask(factory: AskerFactory, corpus_path: Path, question: str) -> AskResult:
    asker = factory(load_corpus(corpus_path), QASettings.from_env())
    try:
        return await asker(question)
    finally:
        close = getattr(asker, "aclose", None)
        if close is not None:
            await close()


async def _eval(factory: AskerFactory, corpus: Corpus, golden_path: Path) -> EvalReport:
    golden = load_golden(golden_path)
    missing = missing_chunk_ids(golden, (chunk.id for chunk in corpus.chunks))
    if missing:
        raise ValueError(f"golden cases cite chunk ids not in the corpus: {missing}")
    asker = factory(corpus, QASettings.from_env())
    try:
        return await run_eval(golden, asker)
    finally:
        close = getattr(asker, "aclose", None)
        if close is not None:
            await close()


def _run_eval_command(args: argparse.Namespace, factory: AskerFactory) -> int:
    if not 0 <= args.min_pass <= 1:
        print("hoa-qa: --min-pass must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        corpus = load_corpus(args.corpus)
        if args.write_ids is not None:
            write_ids(corpus, args.write_ids)
            print(f"wrote {len(corpus.chunks)} chunk ids to {args.write_ids}")
            return 0
        report = asyncio.run(_eval(factory, corpus, args.golden))
    except (OSError, ValueError) as exc:
        print(f"hoa-qa: {exc}", file=sys.stderr)
        return 2
    print(render_table(report))
    if args.json_out is not None:
        write_report(report, args.json_out)
    if report.pass_rate < args.min_pass:
        print(
            f"pass rate {report.pass_rate:.0%} is below --min-pass {args.min_pass:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    asker_factory: AskerFactory = build_asker,
    jev_factory: JevFactory = build_jev,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    # Traces go to Langfuse when its keys are set (hoa_qa.telemetry); the
    # provider flushes what is queued when the process exits.
    telemetry.configure(os.environ)
    if args.command == "cache":
        if args.cache_command == "build":
            return _run_cache_build(args, asker_factory)
        if args.cache_command == "verify":
            return _run_cache_verify(args)
        return _run_cache_check(args, jev_factory)
    if args.command == "eval":
        return _run_eval_command(args, asker_factory)
    if args.command != "ask":
        parser.print_help()
        return 2
    try:
        result = asyncio.run(_ask(asker_factory, args.corpus, args.question))
    except (OSError, ValueError) as exc:
        print(f"hoa-qa: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        print(_render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry points.

- ``hoa-qa ask "question" --corpus corpus.json``
- ``hoa-qa eval evals/golden.yaml --corpus build/corpus.json [--json out.json]``
- ``hoa-qa eval --write-ids --corpus build/corpus.json`` (refresh the id manifest)
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from hoa_qa.ask import AskerFactory, AskResult, QASettings, build_asker
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
    return parser


def _render(result: AskResult) -> str:
    answer = result.answer
    lines = [answer.answer_text, ""]
    for citation in answer.citations:
        lines.append(f'- {citation.citation_label}: "{citation.quote}"')
        lines.append(f"  {citation.url}")
    if answer.conflicts_noted:
        lines.append("")
        lines.extend(f"Note: {conflict}" for conflict in answer.conflicts_noted)
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
    argv: Sequence[str] | None = None, *, asker_factory: AskerFactory = build_asker
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
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

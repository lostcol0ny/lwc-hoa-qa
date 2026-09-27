"""Command-line entry point: ``hoa-qa ask "question" --corpus corpus.json``."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from hoa_qa.ask import AskerFactory, AskResult, QASettings, build_asker
from hoa_qa.models import load_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lakewood Creek HOA Q&A")
    parser.add_argument("--version", action="version", version=version("hoa-qa"))
    commands = parser.add_subparsers(dest="command")
    ask = commands.add_parser("ask", help="Answer one question from the corpus")
    ask.add_argument("question")
    ask.add_argument("--corpus", required=True, type=Path)
    ask.add_argument("--json", action="store_true", help="Print the full result")
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


def main(
    argv: Sequence[str] | None = None, *, asker_factory: AskerFactory = build_asker
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
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

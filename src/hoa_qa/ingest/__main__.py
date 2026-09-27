"""python -m hoa_qa.ingest build --out build/ [--no-llm]."""

import argparse
import json
from collections import Counter
from pathlib import Path

from hoa_qa.ingest import OCR_SOURCES, build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--out", type=Path, default=Path("build"))
    parser.add_argument("--sources", type=Path, default=Path("sources.yaml"))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-ocr-fallback-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.max_ocr_fallback_fraction <= 1:
        parser.error("--max-ocr-fallback-fraction must be between 0 and 1")
    warnings: list[str] = []
    corpus = build(args.out, args.sources, no_llm=args.no_llm, warnings=warnings)
    print(
        json.dumps(
            {
                "chunks_per_doc": dict(Counter(c.doc_id for c in corpus.chunks)),
                "chunks": len(corpus.chunks),
                "token_estimate": sum(c.token_estimate for c in corpus.chunks),
                "ocr_fallbacks": list(corpus.manifest.ocr_fallbacks),
                "section_warnings": warnings,
            },
            indent=2,
        )
    )

    eligible = sum(c.doc_id in OCR_SOURCES for c in corpus.chunks)
    if (
        eligible
        and len(corpus.manifest.ocr_fallbacks) / eligible
        > args.max_ocr_fallback_fraction
    ):
        parser.exit(
            1, "OCR fallback threshold exceeded; inspect corpus_manifest.json\n"
        )


if __name__ == "__main__":
    main()

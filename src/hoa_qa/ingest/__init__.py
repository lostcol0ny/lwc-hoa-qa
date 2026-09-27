"""Build the versioned, source-grounded HOA corpus."""

import hashlib
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from hoa_qa.ingest.cleanup import OCRCleanup, clean
from hoa_qa.ingest.extract import extract
from hoa_qa.ingest.fetch import Fetcher
from hoa_qa.ingest.pii import check_pii
from hoa_qa.ingest.section import citation, parts, sections
from hoa_qa.ingest.sources import load_sources
from hoa_qa.models import Chunk, Corpus, CorpusManifest, load_corpus

OCR_SOURCES = {"declaration", "bylaws", "articles", "rules-2016"}


def build(
    out: Path,
    sources_path: Path = Path("sources.yaml"),
    *,
    no_llm: bool = False,
    fetcher: Callable[[str], bytes] | None = None,
    repair: Callable[[str], str] | None = None,
) -> Corpus:
    sources = load_sources(sources_path)
    fetch = fetcher if fetcher is not None else Fetcher(out / "cache")
    cleanup = None if no_llm else (repair if repair is not None else OCRCleanup())
    chunks: list[Chunk] = []
    hashes: dict[str, str] = {}
    fallbacks: list[str] = []
    errors: list[str] = []
    for source in sources:
        if source.exclude:
            continue
        data = fetch(source.url)
        hashes[source.doc_id] = hashlib.sha256(data).hexdigest()
        pages = extract(source, data)
        if not pages:
            raise ValueError(f"{source.doc_id}: no extracted pages")
        for section in sections(source, pages):
            for part in parts(
                section,
                keep_together=(
                    source.authority.value == "board_decision"
                    or source.doc_id == "arch-form"
                ),
            ):
                chunk_id = f"{source.doc_id}-{part.key}"
                raw = part.text
                text = clean(
                    raw,
                    chunk_id,
                    cleanup if source.doc_id in OCR_SOURCES else None,
                    fallbacks,
                )
                try:
                    check_pii(text, source.doc_id)
                except ValueError as exc:
                    errors.append(f"{chunk_id}: {exc}")
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=source.doc_id,
                        doc_title=source.title,
                        source_url=source.url,
                        page_start=part.lines[0][1].number,
                        page_end=part.lines[-1][1].number,
                        citation_label=citation(source, part),
                        heading_path=part.heading,
                        text_clean=text,
                        text_raw=raw,
                        authority=source.authority,
                        effective_date=source.effective_date,
                        published_date=None,
                        superseded_by=source.superseded_by,
                        token_estimate=math.ceil(len(text) / 4),
                    )
                )
    if errors:
        raise ValueError("\n".join(errors))
    corpus = Corpus(
        manifest=CorpusManifest(
            build_time=datetime.now(UTC),
            source_hashes=hashes,
            chunk_count=len(chunks),
            ocr_fallbacks=tuple(fallbacks),
        ),
        chunks=tuple(chunks),
    )
    out.mkdir(parents=True, exist_ok=True)
    # Validation and all privacy checks precede either output replacement.
    temporary = out / "corpus.json.tmp"
    temporary.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")
    load_corpus(temporary)
    temporary.replace(out / "corpus.json")
    manifest = out / "corpus_manifest.json.tmp"
    manifest.write_text(corpus.manifest.model_dump_json(indent=2), encoding="utf-8")
    manifest.replace(out / "corpus_manifest.json")
    return corpus

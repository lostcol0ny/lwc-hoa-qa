# Lakewood Creek HOA Q&A

A public question-and-answer tool grounded in Lakewood Creek HOA documents,
with source citations. **Under construction:** this scaffold provides shared
corpus/answer models and development tooling; the Q&A pipeline is not implemented.

See the [design spec](docs/specs/2026-09-27-hoa-qa-design.md).

## Development

Install Python 3.12+ and uv, then run:

```sh
uv sync --locked
uv run pre-commit install
uv run hoa-qa --version
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest -q
```

Copy `.env.example` to `.env` when configuring providers. Never commit API keys.
Gitleaks runs in pre-commit and CI.

`hoa_qa.models.load_corpus(path)` reads a JSON object with `manifest` and `chunks`.
The manifest contains `build_time`, `source_hashes` (document ID to hash),
`chunk_count`, and `ocr_fallbacks` (diagnostic strings). Authority ranks increase
with precedence (0–5). Models reject extra fields and field reassignment; sequence
fields are tuples (including corpus chunks), so they cannot be mutated in place.
The manifest source_hashes dictionary remains mutable; treat it as read-only.
Superseded chunks must name the replacement document by doc_id in superseded_by.
Corpus loading checks replacement references, source hashes, and chunk counts.
Dates accept calendar dates only; year-only sources use YYYY-01-01 with the plain
year in citation_label. Build timestamps must include a timezone.

`tests/fixtures/mini_corpus.json` is synthetic test data based on the spec, with
illustrative URLs and passages; it is not an authoritative source or a built
production corpus. Production corpus files stay out of git.

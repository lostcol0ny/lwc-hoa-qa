# Corpus ingestion

Run from the repository root:

```sh
uv sync --locked
uv run python -m hoa_qa.ingest build --out build/ --no-llm
```

This fetches the included sources, validates their metadata, extracts and sections
them, checks contact information, and writes `build/corpus.json` (the
`hoa_qa.models.load_corpus` envelope) and `build/corpus_manifest.json`. The CLI prints
chunk counts by document and a token estimate (normalized Unicode characters / 4,
rounded up per chunk). `build/` is ignored by git.

For optional build-time OCR repair, set `ANTHROPIC_API_KEY` in the environment and
omit `--no-llm`. `INGEST_CLEANUP_MODEL` defaults to `claude-haiku-4-5-20251001`, the
small, fast Haiku tier confirmed against [Anthropic's live model table](https://platform.claude.com/docs/en/models/overview)
on 2026-09-27. Runtime Q&A never invokes this cleanup. Only Declaration, Bylaws,
Articles, and historical Rules text goes through it; digital documents and posts
retain their original wording. The repair prompt prohibits rephrasing and treats
the source as untrusted data.

The multiset of numeric tokens, currency expressions, dates, and section
references must survive repair exactly. A mismatch or truncated response uses the
raw text and records the chunk ID in `manifest.ocr_fallbacks`. All text retains a
`text_raw` counterpart. `--no-llm` normalizes whitespace only, requires no key,
and creates no Anthropic client. Provider/network errors fail the build. The CLI
fails if more than 10% of eligible OCR chunks fall back; adjust with
`--max-ocr-fallback-fraction` (0–1). In that case the local outputs remain available
for inspection, but the workflow will not upload them.

## Fetching and refreshing

HTTP requests use a descriptive User-Agent, 45-second timeout, redirects, and up
to four attempts with exponential backoff for transport errors, 408, 429, and 5xx.
Raw bytes are cached in `build/cache/<sha256-of-url>`. Manifest hashes cover the
actual downloaded bytes, not extracted or repaired text. Excluded sources are
neither fetched nor hashed.

Remove the appropriate cache entry (or the cache directory) to refresh a local
source whose URL has not changed. Review `sources.yaml` dates alongside refreshed
website content: website effective dates are the recorded crawl date, not the
time an old cache was read. CI starts with a fresh source cache. New discovered
sources need a reviewed registry entry; ingestion does not silently expand scope.

## Source authority and dates

The inventory's 22 documents comprise seven PDFs and 15 posts, including the
excluded post. Home, FAQ, and About add three registry entries: **25 total, 24
included**. Meeting dates order minutes. Displayed dates order informal posts,
even where metadata reflects later republication. The March 2025 newsletter's
post date is May 29, 2025; its title still identifies its content month. Year-only
dates use January 1, with the year shown in governing-document citations.

| Source ID | Authority | Effective date | Treatment |
|---|---|---|---|
| `declaration` | governing | 2001-08-09 | Included |
| `clubhouse` | rules | 2022-05-12 | Included |
| `arch-form` | form | 2010-01-01 | Included |
| `rules-2023` | rules | 2023-10-24 | Included |
| `articles` | governing | 2001-01-01 | Included |
| `bylaws` | governing | 2001-01-01 | Included |
| `rules-2016` | superseded | 2016-06-07 | Superseded by rules-2023 |
| `newsletter-2025-03` | informal | 2025-05-29 | Included |
| `minutes-2024-12-20` | board_decision | 2024-12-20 | Included |
| `minutes-2024-03-19` | board_decision | 2024-03-19 | Included |
| `minutes-2023-10-12` | board_decision | 2023-10-12 | Included |
| `minutes-2023-06-09` | board_decision | 2023-06-09 | Included |
| `payment-options` | informal | 2024-07-01 | Included |
| `minutes-2022-12-29` | board_decision | 2022-12-29 | Included |
| `minutes-2022-12-06` | board_decision | 2022-12-06 | Included |
| `email-answers` | informal | 2023-06-14 | Excluded: disputed allegations |
| `unsolicited-email` | informal | 2023-06-13 | Included |
| `newsletter-2022-winter` | informal | 2022-11-07 | Included |
| `violation-tips` | informal | 2022-10-20 | Included |
| `amendments-committee` | informal | 2022-08-26 | Included |
| `meeting-recap-2022-08-18` | informal | 2022-08-20 | Included |
| `meeting-notice-2022-08` | informal | 2022-08-01 | Included |
| `home` | website | 2026-09-27 | Included |
| `faq` | website | 2026-09-27 | Included |
| `about` | website | 2026-09-27 | Included |

## Sectioning, citations, and deduplication

PDF page indices are always 1-based physical pages, including excluded pages in
the original numbering. Detectable centered footer pagination is used in citation
labels, while source links still target the PDF index. For example, Declaration
§8.4 begins on PDF page 28, printed page 23, and continues on printed page 24.
Without reliable printed pagination, the label uses the PDF page index.

Article/section headings, numbered paragraphs, agenda items, topic headings, and
individual FAQ questions establish boundaries. Long sections split into stable
`-a`/`-b` parts, never combine different sections, and target at most 800 estimated
tokens. Minutes agenda blocks are kept whole to preserve motions with outcomes;
an exceptionally long future agenda block may exceed the target. The architectural
form stays in one chunk. Repeated exhibit letters receive `-occ2` suffixes.
HTML extraction decodes the nested Draft.js JSON in `_BLOG_DATA.post.fullContent`,
with HTML-body support as well. Missing bodies or changed website selectors fail
loudly. Website selectors retain home assessment facts and FAQ/About main content,
without repeated navigation, banners, or cookie notices.

Declaration PDF **pages 68–79** contain the appended bylaws. They are excluded in
`sources.yaml`; the standalone bylaws remain canonical. Declaration pages 2–5
(table of contents) are also excluded to avoid duplicate section IDs and index
snippets competing with actual provisions. Substantive exhibits, including pages
59–67 and 80–83, remain. Historical Rules page 3 is blank apart from its printed
page number and is excluded.

## Rules exhibit inspection

Rendered Rules 2023 PDF pages 16–19 to `/tmp/rules-16.png` through
`/tmp/rules-19.png` and inspected them visually:

| PDF page | Contents | Decision |
|---|---|---|
| 16 | Exhibit C: 4/6-foot board-on-board (shadow box) and white composite fence styles | Retain existing text layer and original page link |
| 17 | White composite fence illustration and caption | Retain caption; illustration requires opening the PDF |
| 18 | Allowed mailbox models and product descriptions | Retain existing OCR; some product text is noisy |
| 19 | Exhibit D: cedar mailbox illustrations and post/topper note | Retain existing OCR; diagram text is very noisy |

These are scanned images **with existing text layers**, not image-only pages.
No new OCR or system packages were installed, and none of these pages was dropped.
The searchable text cannot express every visual specification. A future image-only
page fails extraction until its OCR or explicit exclusion has been reviewed.
Should a later pass add reviewed transcriptions of the noisy mailbox diagrams?

## Privacy gate

Every cleaned chunk is scanned for US phone numbers, email addresses, and street
addresses. Unexpected matches fail before outputs are replaced, with the document,
chunk ID, and offending match. Do not publish a failure log without reviewing it.
Allowlisted contacts are published organizational endpoints; scoped entries only
apply to their source document. Exact normalization handles phone punctuation,
case, address punctuation, and whitespace. These are the reviewed entries:

| Scope | Normalized contact | Evidence / purpose |
|---|---|---|
| `*` | `6302290092` | HOA clubhouse office |
| `*` | `6302290254` | HOA clubhouse fax (rental form and newsletter) |
| `*` | `2799 oakmont dr` | HOA clubhouse postal address |
| `*` | `2799 oakmont drive` | HOA clubhouse postal address (expanded suffix) |
| `*` | `lakewoodcreek@comcast.net` | HOA public organizational mailbox |
| `declaration` | `2500 w higgins road` | Original HOA principal office, PDF p.56 |
| `declaration` | `222 north lasalle street` | Recording law firm return address, PDF p.57 |
| `declaration` | `3122363003` | Recording law firm office telephone, PDF p.57 |
| `bylaws` | `2500 w higgins road` | Original HOA principal office, Article 1 |
| `articles` | `2500 west niggins road` | OCR of original HOA business office, PDF p.2 |
| `articles` | `2500 west higgina road` | OCR of original HOA business office, PDF p.2 |
| `articles` | `2500 west higgins road` | Original HOA business office, PDF p.2 |
| `articles` | `118 wesp edward street` | OCR of incorporator corporation office, PDF p.3 |
| `articles` | `118 west edward street` | Incorporator corporation office, PDF p.3 |
| `clubhouse` | `6302735547` | Published rental attendant emergency contact, PDF p.2 |
| `faq` | `8446333577` | LRS municipal refuse service customer support |
| `faq` | `montgomery@lrsrecycles.com` | LRS municipal refuse service mailbox |
| `faq` | `4694902805` | MuniCap published special-assessment business contact |
| `faq` | `8666488482` | MuniCap main office |
| `faq` | `6308968080` | Village Planner office (extension 9022) |

Address-shaped phrases “30 RIGHT OF WAY” and “144 Correction by Court” are survey
or section text, not addresses. The detector excludes those linguistic patterns.
The regex gate is not a general name/entity anonymizer and cannot recover contacts
that source OCR has garbled beyond recognition. The disputed post is excluded
before any processing; no resident-specific allowlist was added.

## Workflow and validation

`.github/workflows/build-corpus.yml` runs manually or for relevant pushes to main.
It uses the checkout/setup-uv pins from CI, locked dependencies, read-only
permissions, and disabled persisted checkout credentials. When the Anthropic
secret exists it performs cleanup; otherwise it adds `--no-llm`. It uploads
**`corpus`**, containing exactly **`corpus.json`** and **`corpus_manifest.json`**,
with 90-day retention. This matches deploy's lookup of `build-corpus.yml` on main.
A failed build never uploads an artifact.

Offline tests in `tests/ingest/` cover nested post extraction, sections, motion
outcomes, stable IDs, PDF/printed pages, numeric fallback logging, PII failure,
registry/exclusion validation, retries/cache, and a fake-fetch no-key CLI build.
Run all repository gates:

```sh
uv sync --locked && uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest -q
```

## Local real-run record and limitations

A real HTTP `--no-llm` build on 2026-09-27 fetched all 24 included sources and
passed `load_corpus` and the privacy gate. Subsequent extraction audits rebuilt
from the exact cached bytes. The following counts describe the final extraction:

| Document | Chunks |
|---|---:|
| `declaration` | 161 |
| `clubhouse` | 36 |
| `arch-form` | 1 |
| `rules-2023` | 41 |
| `articles` | 11 |
| `bylaws` | 62 |
| `rules-2016` | 26 |
| `newsletter-2025-03` | 5 |
| `minutes-2024-12-20` | 8 |
| `minutes-2024-03-19` | 8 |
| `minutes-2023-10-12` | 8 |
| `minutes-2023-06-09` | 8 |
| `payment-options` | 1 |
| `minutes-2022-12-29` | 8 |
| `minutes-2022-12-06` | 7 |
| `unsolicited-email` | 1 |
| `newsletter-2022-winter` | 11 |
| `violation-tips` | 2 |
| `amendments-committee` | 1 |
| `meeting-recap-2022-08-18` | 10 |
| `meeting-notice-2022-08` | 1 |
| `home` | 1 |
| `faq` | 12 |
| `about` | 1 |

Total: **431 chunks**, **64,884 estimated tokens**. No LLM calls or OCR fallbacks in this run.

No source failed extraction and no chunk exceeded 800 estimated tokens. Remaining
sectioning limits are source OCR defects: some Declaration numbers appear as
`7.22`, `7.74`, `8.38`, and `12.34`; these are retained rather than silently
renumbered. Missing/unrecognizable headings remain with their enclosing article
or introductory section (notably the Articles registration form and some Bylaws
article labels). The Articles OCR is especially poor. Compact, recognizable
headings such as Declaration `84 Assessments` can supply a §8.4 citation without
rewriting their raw numbers. Better source scans or reviewed heading mappings
would improve retrieval precision; cleanup cannot change protected numbers.

The public minutes archive is incomplete. Proposals in minutes and informal posts
are not automatically adopted policy; authority and original wording are retained.
One large FAQ answer spans two parts. Visual exhibit details remain accessible
through page citations. No live Anthropic cleanup run was performed; its behavior
is covered with offline fakes and the numeric guard.

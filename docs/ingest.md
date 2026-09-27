# Corpus ingestion

Run from the repository root:

```sh
uv sync --locked --group ingest
uv run python -m hoa_qa.ingest build --out build/ --no-llm
```

PyMuPDF, BeautifulSoup, and PyYAML live in the `ingest` dependency group, not in
`[project]` dependencies, so the Vercel runtime never installs them (PyMuPDF is
AGPL-licensed and large). The `dev` group includes `ingest`, so the default
`uv sync --locked` used by CI still type-checks and tests this package; a
`--no-dev` install (deploy's corpus validation, the runtime) does not get it.

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

Code, not the prompt, enforces that the repair only fixes characters
(`cleanup.py`). Any failed check, or a truncated response, uses the raw text and
records the chunk ID in `manifest.ocr_fallbacks`:

1. The multisets of numbers, currency expressions, dates, and section references
   (including subsection letters, e.g. `8.2(d)(4)`) must match exactly.
2. The text must stay near-identical: `difflib` character ratio ≥ 0.9 and word
   count within ±3%.
3. Word-level edits must be character repairs: no whole word inserted or deleted
   (so dropping "not" or adding "only" fails even in a long chunk); each replaced
   span may change at most two characters or a quarter of its length; and a
   replaced span that contains a protected word (number words one…ninety,
   hundred, thousand; `not`/`no`/`shall`/`may`/`must`/`only`/…) must keep exactly
   those words. A garbled source word can still be repaired into one ("rnay" →
   "may").

These guards are deliberately conservative; a real LLM run may show a higher
fallback rate than the old numeric-only guard. All text retains a
`text_raw` counterpart. `--no-llm` normalizes whitespace only, requires no key,
and creates no Anthropic client. Provider/network errors fail the build. The CLI
fails if more than 10% of eligible OCR chunks fall back; adjust with
`--max-ocr-fallback-fraction` (0–1). In that case the local outputs remain available
for inspection, but the workflow will not upload them.

## Fetching and refreshing

HTTP requests use a descriptive User-Agent, a 45-second timeout, and up to four
attempts with exponential backoff for transport errors, 408, 429, and 5xx.
Redirects are followed only to `https://` URLs (checked on every hop), and a
response over 50 MB (declared or streamed) is refused; the largest source is
~8 MB.
Raw bytes are cached in `build/cache/<sha256-of-url>`. Manifest hashes cover the
actual downloaded bytes, not extracted or repaired text. Excluded sources are
neither fetched nor hashed.

Remove the appropriate cache entry (or the cache directory) to refresh a local
source whose URL has not changed. Website pages declare `effective_date: crawl`
and are dated with the build's UTC date, so a local build over an old cache dates
them by the build, not the download. CI starts with a fresh source cache. New discovered
sources need a reviewed registry entry; ingestion does not silently expand scope.

## Source authority and dates

The inventory's 22 documents comprise seven PDFs and 15 posts, including the
excluded post. Home, FAQ, and About add three registry entries: **25 total, 24
included**. `effective_date` is the content date (the meeting date for minutes)
and `published_date` the displayed post date. The March 2025 newsletter is
effective 2025-03-01 and published 2025-05-29; Tips for Responding to HOA
Violations is dated by its displayed 2022-10-20 (its 2025 metadata timestamp is a
republication). Year-only dates use January 1, with the year shown in
governing-document citations.

| Source ID | Authority | Effective date | Treatment |
|---|---|---|---|
| `declaration` | governing | 2001-08-09 | Included |
| `clubhouse` | rules | 2022-05-12 | Included |
| `arch-form` | form | 2010-01-01 | Included |
| `rules-2023` | rules | 2023-10-24 | Included |
| `articles` | governing | 2001-01-01 | Included |
| `bylaws` | governing | 2001-01-01 | Included |
| `rules-2016` | superseded | 2016-06-07 | Superseded by rules-2023 |
| `newsletter-2025-03` | informal | 2025-03-01 (published 2025-05-29) | Included |
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
| `meeting-recap-2022-08-18` | informal | 2022-08-18 (published 2022-08-20) | Included |
| `meeting-notice-2022-08` | informal | 2022-08-01 | Included |
| `home` | website | build crawl date | Included |
| `faq` | website | build crawl date | Included |
| `about` | website | build crawl date | Included |

## Sectioning, citations, and deduplication

PDF page indices are always 1-based physical pages, including excluded pages in
the original numbering; source links target them. Labels use printed page numbers
only when the document has a reliable pagination offset: at least three detected
centered footers, 80% agreeing on one PDF − printed offset (the Declaration is
PDF − 5 throughout its body). That modal offset is applied to every page from
printed page 1 through the last detected footer, which fills footers OCR missed
(Declaration PDF pp. 6–15). A disagreeing footer is logged and overridden. A label
never mixes numbering: if either end of a range has no printed number, the whole
label uses `PDF p./pp.`. Example: Declaration §8.4 is PDF pp. 28–29, labelled
"Declaration, Article 8, §8.4, pp. 23–24".

### Sectioning

- **Declaration, Bylaws, Articles:** `Article N` (OCR-tolerant spellings, and a
  heading OCR merged mid-line after sentence-ending punctuation, e.g. Bylaws
  "…indemnified. Axticle 13 Aysessments") and `N.M[.K]` sections.
- **Rules 2023 / 2016:** upper-case `SECTION N` and `N.M`; sections without
  numbered subsections split on lettered subsections (`A. Intent`, `B. Fines`…),
  keyed and labelled `rules-2023-3.B`, "Rules 2023, Section 3.B (Fines), PDF p. 9".
- **Monotonic numbering:** every numbered heading must be a plausible *next* key
  after the previous one (a sibling up to +3, a first child, the next article, or
  N.1–N.9 right after an article heading). OCR that drops or misreads dots is
  resolved by picking the candidate reading that is next expected: dotless `84`
  → 8.4, `838` → 8.8, `722` → 7.2.2; explicit `7.74` → 7.7.4 and `12.34` → 12.3.4.
  An explicit dot is trusted (`2.23` is never read as 2.3). Anything else
  (survey distances, cross-references wrapped to a line start) is kept as body
  text and reported in the CLI's `section_warnings`.
- **Exhibits:** an `EXHIBIT X` line alone, or an all-caps `EXHIBIT X` line;
  in-prose references ("Exhibit B (hereafter…)") are not headings. Labels are
  normalized to "Exhibit {letter}". Repeated letters get `-occ2` ids.
- **Heading-only sections** (an `Article N` title, optionally with a one-line
  lead-in ending in ":", or a short numbered title such as "59 Fences.") fold into
  the next section: their text is prepended and their heading joins
  `heading_path`.
- **Minutes (`board_decision`):** **one chunk per meeting** (150–310 estimated
  tokens today), id `minutes-YYYY-MM-DD-meeting`, `heading_path = [title,
  "Meeting YYYY-MM-DD"]`, and the citation label includes the meeting date. A
  motion therefore always shares its chunk with its subject and outcome (e.g. the
  2023 budget's "$99 per quarter" and "motion approved"). A meeting over ~800
  tokens would split only at top-level agenda items, and every later part starts
  with a "Title (meeting YYYY-MM-DD)" line.
- **Clubhouse, newsletters, posts:** numbered paragraphs, Q&A lines, and ALL-CAPS
  headings, but an ALL-CAPS line starts a section only after a line that ended a
  sentence, and a run of form blanks (`____`) stays in one chunk.
- **Web pages:** one chunk per FAQ Q&A; one chunk for the banner facts.
- The architectural form stays in one chunk. Long sections split into stable
  `-a`/`-b` parts at sentence boundaries and never combine sections.
- Intro chunks are labelled with the title once (no "Declaration, Declaration").
  Display titles fix source typos ("December 2O" → "December 20", trailing
  spaces); URLs are unchanged.

HTML extraction decodes the nested Draft.js JSON in `_BLOG_DATA.post.fullContent`
(with HTML-body support as well); malformed blocks, missing bodies, or changed
website selectors fail loudly. Website selectors retain home assessment facts and
FAQ/About main content, without navigation, banners, or cookie notices.

### Deduplication

Declaration PDF **pages 68–79** contain the appended bylaws. They are excluded in
`sources.yaml`; the standalone bylaws remain canonical. **Pages 80–83** are a
later-appended copy of the Rules 2023 fence/mailbox exhibits: their text layer is
byte-identical to `rules-2023` PDF pp. 16–19, they have a different page box and
fonts, and they follow the 2001 recording pages. They are excluded so rules
exhibits never carry `governing` authority; the `rules-2023` copy is kept. The
Declaration's own exhibits A (p. 59), B (pp. 60–65), and D (p. 67) are retained.
Declaration pages 2–5 (table of contents) are also excluded to avoid duplicate
section IDs. Historical Rules page 3 is blank apart from its printed page number
and is excluded.

## Rules exhibit inspection

Rendered Rules 2023 PDF pages 16–19 to `/tmp/rules-16.png` through
`/tmp/rules-19.png` and inspected them visually:

| PDF page | Contents | Decision |
|---|---|---|
| 16 | Exhibit C: 4/6-foot board-on-board (shadow box) and white composite fence styles | Retain existing text layer and original page link |
| 17 | White composite fence illustration and caption | Retain caption; illustration requires opening the PDF |
| 18 | Allowed mailbox models and product descriptions | Retain existing OCR; some product text is noisy |
| 19 | Exhibit D: cedar mailbox illustrations and post/topper note | Retain existing OCR; diagram text is very noisy |

These are scanned images **with existing text layers**, not image-only pages. They
are the only copy in the corpus: the byte-identical copy appended to the
Declaration (its PDF pp. 80–83) is excluded, so they carry `rules` authority.
No new OCR or system packages were installed, and none of these pages was dropped.
The searchable text cannot express every visual specification. A future image-only
page fails extraction until its OCR or explicit exclusion has been reviewed.
Should a later pass add reviewed transcriptions of the noisy mailbox diagrams?

## Privacy gate

Every chunk's `text_clean` **and** `text_raw` (both ship in the corpus) are
scanned for US phone numbers (any of space, dot, dash, or no separator, e.g.
`(630)229-0092` or `6302290092`), email addresses, and street addresses
(Dr/St/Ave/Ct/Ln/Rd/Blvd/Way/Pl/Pkwy/Cir/Trl/Ter and long forms). Unexpected matches fail before outputs are replaced, with the document,
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
| `articles` | `1835400709` | IL Secretary of State certificate authentication number (10 digits, not a phone), PDF p.8 |
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

The generated corpus keeps the public source text, including board members'
names in minutes. Committed test fixtures and golden files never contain an
individual's name; excerpts use placeholders such as `[Board Member A]` or omit
names.

## Untrusted PDF parsing

Source PDFs come from a third-party website and are parsed with PyMuPDF (native
MuPDF code), so a malicious or malformed PDF is an attack surface for the build.
Mitigations: sources are a reviewed allowlist in `sources.yaml` (the build never
follows discovered links); fetches are HTTPS-only on every redirect hop and capped
at 50 MB; parsing runs only at build time, in an ephemeral GitHub runner with a
read-only token, no persisted checkout credentials, and only the optional
Anthropic secret; PyMuPDF is in the `ingest` group and is never installed in the
Vercel runtime; manifest hashes record the exact bytes parsed; and the build
emits only validated JSON (`load_corpus`), never source files. Keep PyMuPDF
current (the `<2` bound admits security releases).

## Workflow and validation

`.github/workflows/build-corpus.yml` runs manually, for relevant pushes to main,
and monthly (cron `17 6 1 * *`). The monthly run re-crawls the website pages
(dated by that crawl) and keeps a fresh `corpus` artifact inside its 90-day
retention for deploy. It installs `uv sync --locked --group ingest` and uses the
checkout/setup-uv pins from CI, locked dependencies, read-only
permissions, and disabled persisted checkout credentials. When the Anthropic
secret exists it performs cleanup; otherwise it adds `--no-llm`. It uploads
**`corpus`**, containing exactly **`corpus.json`** and **`corpus_manifest.json`**,
with 90-day retention. This matches deploy's lookup of `build-corpus.yml` on main.
A failed build never uploads an artifact.

Offline tests in `tests/ingest/` cover nested post extraction, sections, one
chunk per meeting, stable IDs, PDF/printed pages and the modal offset, dotless and
mis-dotted headings, mid-line articles, exhibit headings, folding, cleanup guards,
PII formats on raw and clean text, crawl/published dates, registry/exclusion
validation, HTTPS-only redirects and the size cap, retries/cache, and a fake-fetch
no-key CLI build. `test_golden_sections` checks ids and labels for Declaration
Article 8 and Rules 2023 Section 3 against `fixtures/golden_sections.json`,
computed from committed real-text excerpts in `fixtures/golden_pages.json`
(heading lines plus one body line per section, with each page's resolved printed
number). Regenerate both only after reviewing a real build.
Run all repository gates:

```sh
uv sync --locked && uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest -q
```

## Local real-run record and limitations

A real HTTP `--no-llm` build on 2026-09-27 fetched all 24 included sources and
passed `load_corpus` and the privacy gate. The review fixes were rebuilt from the
exact cached bytes:

| Document | Before (PR #9) | After |
|---|---:|---:|
| `declaration` | 161 | 143 |
| `clubhouse` | 36 | 19 |
| `arch-form` | 1 | 1 |
| `rules-2023` | 41 | 41 |
| `articles` | 11 | 11 |
| `bylaws` | 62 | 55 |
| `rules-2016` | 26 | 38 |
| `newsletter-2025-03` | 5 | 5 |
| `minutes-2024-12-20` | 8 | 1 |
| `minutes-2024-03-19` | 8 | 1 |
| `minutes-2023-10-12` | 8 | 1 |
| `minutes-2023-06-09` | 8 | 1 |
| `payment-options` | 1 | 1 |
| `minutes-2022-12-29` | 8 | 1 |
| `minutes-2022-12-06` | 7 | 1 |
| `unsolicited-email` | 1 | 1 |
| `newsletter-2022-winter` | 11 | 11 |
| `violation-tips` | 2 | 2 |
| `amendments-committee` | 1 | 1 |
| `meeting-recap-2022-08-18` | 10 | 10 |
| `meeting-notice-2022-08` | 1 | 1 |
| `home` | 1 | 1 |
| `faq` | 12 | 12 |
| `about` | 1 | 1 |

Total: **360 chunks** (was 431), **64,298 estimated tokens** (was 64,884).
No OCR fallbacks and no `section_warnings`. The largest chunk is
794 estimated tokens.

Remaining limits are source OCR defects. Headings that OCR lost entirely stay in
the enclosing section rather than being invented: Declaration §1.9 ("Owner"),
§5.1, and §6.1 have no number in the text layer, so their text sits in §1.8,
`declaration-article-5`, and `declaration-article-6`. The Articles registration
form (Articles 1–3) has no recognizable headings and stays in the intro chunks.
Rules 2023 carries two exhibit sets (forms A–D, then fence/mailbox diagrams
C/A/D), so the second set has `-occ2` ids. Better scans or reviewed heading
mappings would improve precision; cleanup cannot change protected numbers.

The public minutes archive is incomplete. Proposals in minutes and informal posts
are not automatically adopted policy; authority and original wording are
retained. One large FAQ answer spans two parts. Visual exhibit details remain
accessible through page citations. No live Anthropic cleanup run has been
performed (no `ANTHROPIC_API_KEY` in the development environment); its behavior is
covered with offline fakes and the guards above.

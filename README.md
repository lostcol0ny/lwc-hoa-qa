# Lakewood Creek HOA Q&A

A public web page with one text box. A neighbor asks a question about the
Lakewood Creek HOA (rules, dues, the pool, architectural approvals,
governance) and gets a short answer grounded **only** in the HOA's published
documents, with citations that link to the exact source page. It's an
unofficial tool, not legal advice: the governing documents and the Board are
authoritative.

Each question stands alone (no chat history). Off-topic questions are refused,
questions the documents don't answer get "I couldn't find this", and every
claim in an answer must be backed by a verbatim quote that a second model
confirms supports it.

- Design spec: [docs/specs/2026-09-27-hoa-qa-design.md](docs/specs/2026-09-27-hoa-qa-design.md)
- Corpus build: [docs/ingest.md](docs/ingest.md)
- Question answering: [docs/qa-core.md](docs/qa-core.md)
- Web app, budget and deployment: [docs/web-ops.md](docs/web-ops.md)

## Architecture

```text
 CI (build-corpus.yml, monthly / manual)
   sources.yaml ─▶ fetch ─▶ extract (PyMuPDF) ─▶ OCR cleanup (off by default)
               ─▶ section ─▶ dedup ─▶ PII check ─▶ corpus.json + manifest
                                                        │  (artifact, never committed)
                                                        ▼
 Browser ─▶ Vercel CDN (public/: index.html, app.js)   deploy.yml bundles it
    │
    └─ POST /api/ask ─▶ Vercel Firewall (edge burst limit)
                          ─▶ api/index.py → hoa_qa.web.app (FastAPI)
                               ├─ same-origin check, 500-char cap
                               ├─ per-IP rate limit ───────────┐
                               ├─ reserve R from the monthly ──┤ Upstash Redis
                               │  budget (atomic Lua EVAL)     │ (integer µ$)
                               ├─ QAAsker (hoa_qa.ask)         │
                               │   1. gate: Jev "is this about the HOA?"
                               │   2. sweep: Jev scores every chunk, top 8
                               │   3. answer: Claude, structured claims + quotes
                               │   4. verify: quote in chunk (code) + Jev support
                               └─ reconcile actual cost ───────┘
```

`hoa-qa ask` (CLI) and `hoa-qa eval` call the same `QAAsker` as the web app.

## Local development

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked            # dev tools + the ingest group
uv run pre-commit install   # ruff + gitleaks on commit

# Gates (CI runs the same):
uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest -q
```

Build the corpus (network needed, no API key; OCR cleanup stays off):

```sh
uv run python -m hoa_qa.ingest build --no-llm --out build/
```

Ask one question from the command line (needs both API keys):

```sh
export TYPESAFE_API_KEY=... ANTHROPIC_API_KEY=...
uv run hoa-qa ask "How many guests can I bring to the pool?" --corpus build/corpus.json
```

Run the web app locally:

```sh
# Real pipeline (needs both API keys):
CORPUS_PATH=build/corpus.json uv run uvicorn hoa_qa.web.app:app --reload
# Canned answers, no AI calls and no keys (dev only; refused in production):
HOA_QA_FAKE_ASKER=1 CORPUS_PATH=tests/fixtures/mini_corpus.json \
  uv run uvicorn hoa_qa.web.app:app --reload
```

Then open <http://127.0.0.1:8000/>. Health: <http://127.0.0.1:8000/api/health>.

Run the golden eval (real APIs, a few cents):

```sh
uv run hoa-qa eval evals/golden.yaml --corpus build/corpus.json --json out.json
# After a corpus change, refresh the committed chunk-id manifest:
uv run hoa-qa eval --write-ids --corpus build/corpus.json
```

Copy `.env.example` to `.env` for local keys. Never commit API keys; gitleaks
runs in pre-commit and CI.

## Configuration

Every environment variable, across all units. Empty values mean "use the
default".

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `TYPESAFE_API_KEY` | (required) | QA core | Jev (gate, sweep, support check) |
| `ANTHROPIC_API_KEY` | (required) | QA core; ingest with LLM cleanup | Answer model; optional OCR cleanup |
| `ANSWER_MODEL` | `claude-haiku-4-5` | QA core | Anthropic model for the answer step |
| `ANSWER_MAX_TOKENS` | `2048` | QA core | Output cap per answer call (256–16000); part of `max_cost_usd` |
| `JEV_MODEL` | `jev-latest` | QA core | Pin a version once thresholds are tuned |
| `GATE_THRESHOLD` | `0.5` | QA core | Minimum on-topic probability |
| `SWEEP_THRESHOLD` | `0.3` | QA core | Minimum chunk relevance |
| `SWEEP_TOP_K` | `8` | QA core | Maximum passages sent to the answer model |
| `SUPPORT_THRESHOLD` | `0.5` | QA core | Minimum claim-support probability |
| `JEV_CONCURRENCY` | `4` | QA core | Concurrent Jev requests per question (`SWEEP_CONCURRENCY` is a fallback name) |
| `HOA_DOCUMENTS_URL` | `https://lakewoodcreekhoa.com/` | QA core, web | Link in refusals, not-found and budget pages; must be `https://` |
| `MONTHLY_BUDGET_USD` | `5.0` in dev; **`0` in production if missing or invalid** | web | Hard monthly spend cap (UTC calendar month) |
| `BUDGET_RESERVE_PER_REQUEST_USD` | `0.05` | web | Minimum per-question reservation R |
| `UPSTASH_REDIS_REST_URL` | unset | web | Shared budget and rate-limit counters; **required in production** (the budget fails closed without it) |
| `UPSTASH_REDIS_REST_TOKEN` | unset | web | Read-write REST token for the above |
| `RATE_LIMIT_PER_HOUR` | `10` | web | Per-IP questions per hour |
| `RATE_LIMIT_PER_DAY` | `50` | web | Per-IP questions per day |
| `CORPUS_PATH` | `corpus.json` | web | Corpus file, relative to the working directory |
| `HOA_QA_FAKE_ASKER` | unset | web | `1` = canned dev answers; ignored in production |
| `TYPESAFE_LOG_LEVEL`, `ANTHROPIC_LOG` | unset | web | Never `debug` in production (the SDKs would log questions); the app refuses to start |
| `VERCEL`, `VERCEL_ENV` | set by Vercel | web | `VERCEL=1` trusts `x-forwarded-for`; `VERCEL_ENV=production` enables the fail-closed rules |
| `INGEST_CLEANUP_MODEL` | `claude-haiku-4-5-20251001` | ingest | Model for the optional OCR cleanup |

GitHub Actions secrets: `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY` (eval; and
`ANTHROPIC_API_KEY` for an opt-in LLM cleanup build), `VERCEL_TOKEN`,
`VERCEL_ORG_ID`, `VERCEL_PROJECT_ID` (deploy). Every workflow skips with a
summary when its secrets are missing.

## Cost model

**Typical cost: about $0.01 per answered question** with the defaults
(estimates from `docs/qa-core.md`, to be replaced with the first eval run's
measured `total_cost_usd`):

| Step | Typical cost |
|---|---|
| Jev gate | ~$0.00001 |
| Jev sweep over the whole corpus (~90–110K tokens at $0.042/MTok) | ~$0.004 |
| Jev support check | ~$0.0001 |
| `claude-haiku-4-5` answer (~3–4K in, ~500 out) | ~$0.006 |

A refused off-topic question costs only the gate; invalid, rate-limited and
budget-exhausted requests cost nothing. A regenerated answer can roughly double
the cost.

**The reservation R.** Before calling the pipeline, the web app atomically
reserves `R = max(BUDGET_RESERVE_PER_REQUEST_USD, asker.max_cost_usd)` from the
month's budget, then reconciles to the actual cost afterwards. `max_cost_usd` is
a provable worst case: it charges every Jev request as if the question were 500
four-byte characters and every token a single byte, and both answer attempts at
their largest prompt and the full `ANSWER_MAX_TOKENS` output. For the real
corpus (360 chunks) at the default settings it is **≈ $0.178**, so R ≈ $0.178.

**How `MONTHLY_BUDGET_USD` and `ANSWER_MAX_TOKENS` interact.**

- `ANSWER_MAX_TOKENS` raises R: each extra 1,024 tokens adds ≈ $0.010 with
  Haiku (2 attempts × 1,024 × $5/MTok). R ≈ $0.167 at 1,024, $0.178 at 2,048,
  $0.198 at 4,096.
- The budget must be at least R, or **nothing** is admitted:
  `/api/health` then reports `"budget_config": "budget_below_reservation"` and
  a WARNING is logged.
- At most `floor(MONTHLY_BUDGET_USD / R)` questions can be in flight at once,
  and the last R of the month goes unused (a question is refused while
  `spend + R > budget`). Spend is counted in exact integer micro-dollars.
- Rough capacity: $5/month ≈ 480 typical answers; $20/month ≈ 2,000.
- Provider-side spend limits (step 4 below) are the backstop behind the app's
  cap.

## Launch checklist

Do these in order.

1. Add the GitHub repository secrets `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY`.
2. Run the **Build corpus** workflow (Actions → Build corpus → Run workflow).
   Leave `llm_cleanup` off unless you want the guarded OCR repair.
3. Run the **Eval** workflow and review the uploaded `eval-results` (per-case
   checks, answers, total cost). Tune `GATE_THRESHOLD`, `SWEEP_THRESHOLD`,
   `SUPPORT_THRESHOLD` or `SWEEP_TOP_K` until it passes (default bar 85%), then
   pin `JEV_MODEL`.
4. Set provider-side spend limits in the Anthropic console (and at TypeSafe if
   it offers them).
5. Create the Vercel project (turn off automatic Git deployments) and add
   Upstash Redis from the Vercel Marketplace. If the integration injects
   `KV_REST_API_URL`/`KV_REST_API_TOKEN` (or other `KV_*` names), also set
   `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` to the same values.
6. Set the Vercel environment variables: `TYPESAFE_API_KEY`,
   `ANTHROPIC_API_KEY`, `MONTHLY_BUDGET_USD` (at least R, see above) and
   `HOA_DOCUMENTS_URL` (plus any threshold you tuned in step 3).
7. Add the GitHub secrets `VERCEL_TOKEN`, `VERCEL_ORG_ID` and
   `VERCEL_PROJECT_ID` (the last two from `.vercel/project.json` after
   `vercel link`).
8. Configure the Vercel Firewall rate-limit rule for `/api/ask`
   ([docs/web-ops.md](docs/web-ops.md#vercel-firewall-rule-edge-layer)).
9. Deploy (push to `main`, or re-run **Deploy**) and verify `/api/health`
   returns `"status": "ok"`, the expected `chunk_count`, and
   `"budget_config": "ok"`. Ask one question on the live page.
10. In GitHub settings → Emails, enable **Block command line pushes that expose
    my email**.

### Day-2 operations

Vercel reads environment variables when a deployment is built, so after
changing one, **redeploy**: push to `main`, or open the latest **Deploy** run on
`main` in Actions and choose **Re-run all jobs**.

**Rotate a key.** Create the new key at the provider (Anthropic console,
TypeSafe, Upstash, Vercel account tokens). Update it everywhere it's used:
the GitHub secret (`TYPESAFE_API_KEY`/`ANTHROPIC_API_KEY` for eval and the
opt-in cleanup build, `VERCEL_TOKEN` for deploy) and the Vercel environment
variable (the two API keys, `UPSTASH_REDIS_REST_TOKEN`). Redeploy, check
`/api/health` and ask one question, then revoke the old key.

**The budget tripped** (every question gets `budget_exhausted`):

- To keep answering this month, raise `MONTHLY_BUDGET_USD` in Vercel and
  redeploy. Otherwise do nothing: the counter is per UTC calendar month, so
  answers resume at 00:00 UTC on the 1st.
- To see the spend, open the Upstash database (console → Data Browser, or its
  CLI) and `GET budget_micros:YYYY-MM` (e.g. `budget_micros:2026-10`). The
  value is integer micro-dollars: divide by 1,000,000 for dollars. It includes
  the reservation R of any question in flight.
- **Avoid resetting or deleting that key.** It is the only record of this
  month's spend, so resetting it lets the month spend up to the budget again
  on top of what was already spent (provider limits are then the only cap).
  And questions in flight still reconcile against it afterwards (a negative
  `INCRBY`), which can push it below zero and admit more than the budget. If
  you must correct it, change it with `INCRBY`/`DECRBY` by a known amount while
  no questions are in flight (for example with the bot taken offline, below),
  never `DEL`. Raising `MONTHLY_BUDGET_USD` is almost always the better fix.

**Refresh the corpus.** **Build corpus** runs monthly (06:17 UTC on the 1st),
on pushes to `main` that touch `sources.yaml` or the ingest code, and by hand.
Deploy doesn't run when a new corpus is built. It bundles the latest successful
`corpus` artifact from `main` each time it runs, so trigger a deploy afterwards
(push, or re-run the latest Deploy run as above) and check `chunk_count` and
`corpus_build_time` on `/api/health`. The artifact is kept for 90 days, and
the monthly run keeps a fresh one around. If Deploy says the artifact expired,
run **Build corpus** first. When chunk ids change, refresh
`evals/corpus_ids.txt` (`hoa-qa eval --write-ids --corpus build/corpus.json`).
The eval stops before any spend if a golden case cites an id that's gone.

**If something goes wrong** (bad answers, a spend spike, abuse), take the bot
offline, fastest first:

1. Add a Vercel Firewall custom rule: `Request Path` equals `/api/ask` →
   **Deny**. It takes effect when published, with no redeploy. The page stays
   up, and questions fail before any function runs.
2. Set `MONTHLY_BUDGET_USD=0` and redeploy. Every question then gets the
   friendly `budget_exhausted` answer with the documents link, and nothing is
   spent (`/api/health` shows `budget_below_reservation`).
3. If a key may have leaked, revoke it at the provider (Anthropic console,
   TypeSafe). The pipeline then fails with a generic error and stops spending.
4. Roll back to a known-good deployment (Vercel → Deployments → Instant
   Rollback), or pause or delete the Vercel project.

Undo in reverse order, and check `/api/health` before removing the Firewall
rule.

## Repository layout

```text
src/hoa_qa/
  ingest/      fetch, extract, OCR cleanup guard, sectioning, PII check
  retrieval/   Jev gate + sweep
  answer/      prompt, provider, pricing
  verify/      quote check + Jev support check
  ask.py       QAAsker / build_asker (the pipeline)
  eval.py      golden-set scoring
  budget.py    monthly budget + counter stores
  web/         FastAPI app, security, rate limiting
  cli.py       hoa-qa ask / hoa-qa eval
api/index.py   Vercel entrypoint
public/        the static page
evals/         golden.yaml + corpus_ids.txt
sources.yaml   the document registry
```

The processed corpus (`corpus.json`) is built in CI and never committed.

`tests/fixtures/mini_corpus.json` is test data, not an authoritative source:
`blog-2022-violations` and `declaration-8-assessment` are verbatim excerpts with
real source URLs; `website-2026-dues` is the verbatim banner with a synthetic
URL; the other chunks are synthetic passages on example.org URLs. The synthetic
fine chunks mirror the real tier structure: 2023 Rules, 1st courtesy letter /
2nd $75 / 3rd $125 / 4th and subsequent $75 per day; 2016 rules (and the blog),
written warning / $50 / $100 / $50 per day. Its source hashes are the SHA-256
of each document's chunk texts joined by newlines.

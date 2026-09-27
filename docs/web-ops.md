# Web app and operations

The web layer is a thin FastAPI app (`src/hoa_qa/web/`) in front of the QA core's
asker, plus a static page (`public/`). It adds four things the QA core doesn't
have: a monthly spend cap, per-IP rate limiting, security headers, and the
Vercel deployment.

```
browser ──▶ Vercel CDN ──▶ public/index.html, app.js, styles.css
        └─▶ Vercel Firewall (edge rate limit) ──▶ api/index.py (FastAPI)
                                                    ├─ same-origin check
                                                    ├─ per-IP rate limit ─┐
                                                    ├─ monthly budget ────┼─ Upstash Redis
                                                    └─ asker (QA core) ───┘ (spend added after)
```

## Local development

```bash
uv sync
# Run with the canned dev asker (no AI calls, no API keys needed):
HOA_QA_FAKE_ASKER=1 CORPUS_PATH=tests/fixtures/mini_corpus.json \
  uv run uvicorn hoa_qa.web.app:app --reload
# then open http://127.0.0.1:8000/
```

- `CORPUS_PATH` defaults to `corpus.json`, relative to the working directory
  (run from the repo root). The corpus is loaded and validated once per process.
- Locally, the app serves `public/` itself. On Vercel it doesn't (the CDN does),
  because Vercel's FastAPI docs say not to mount `public/`.
- Without `HOA_QA_FAKE_ASKER=1`, the app builds the real asker through
  `hoa_qa.ask.build_asker(corpus, QASettings)` on the first question. If
  `hoa_qa.ask` isn't installed, `/api/ask` returns HTTP 503 with an
  `Answer`-shaped body saying the service isn't set up yet.
- `HOA_QA_FAKE_ASKER` is ignored when `VERCEL_ENV=production`.

## API

| Route | Result |
|---|---|
| `POST /api/ask` `{"question": "..."}` | An `Answer` JSON (`hoa_qa.models.Answer`) |
| `GET /api/health` | `{"status": "ok", "corpus_build_time", "chunk_count", "documents_url"}`, or 503 `{"status": "unavailable"}` if the corpus is missing or invalid |

Every `/api/ask` response body is `Answer`-shaped, so the page renders them all
the same way:

| Status | Outcome | When |
|---|---|---|
| 200 | from the asker, or `budget_exhausted` | normal |
| 403 | `error` | the `Origin` header names another site |
| 422 | `invalid_input` | the question is empty, over 500 characters, or the body isn't JSON with exactly a `question` field. **Chosen over a 200 `invalid_input`** so clients and logs can tell bad input from answers; the body is still an `Answer` |
| 429 | `error` ("slow down") | over the per-IP rate limit |
| 500 | `error` (generic) | the asker raised |
| 503 | `error` | the QA core or corpus isn't available |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MONTHLY_BUDGET_USD` | `5.0` in dev; **`0` in production if missing or invalid** | Monthly spend cap |
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | unset | Shared counters for the budget and rate limiter |
| `RATE_LIMIT_PER_HOUR`, `RATE_LIMIT_PER_DAY` | `10`, `50` | App-level per-IP limits |
| `HOA_DOCUMENTS_URL` | `https://lakewoodcreekhoa.com/` | Document link for budget-exhausted answers and the page; the same variable the QA core uses for refusals. Must be `https://` |
| `CORPUS_PATH` | `corpus.json` | Corpus location |
| `HOA_QA_FAKE_ASKER` | unset | `1` = canned dev answers (ignored in production) |
| `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`, `ANSWER_MODEL`, ... | | Read by the QA core's `QASettings`, not by the web layer |
| `VERCEL`, `VERCEL_ENV` | set by Vercel | `VERCEL=1` enables trusting `x-forwarded-for`; `VERCEL_ENV=production` enables the fail-closed rules below |

### SDK debug logging guard

`typesafe-sdk` and `anthropic` log full request bodies (which contain neighbors'
questions) when `TYPESAFE_LOG_LEVEL=debug` or `ANTHROPIC_LOG=debug`. In
production the app **refuses to start** if either is set to `debug`; elsewhere
it logs a warning.

## Budget and cost model

- Each request's cost is the asker's `estimated_cost_usd` (Jev gate + sweep +
  verify, plus the answer model; see `docs/qa-core.md` in the QA core unit).
- Before calling the asker, the app reads the month's total. If it is
  `>= MONTHLY_BUDGET_USD`, it returns `outcome=budget_exhausted` with a link to
  the documents, and **the asker is not called**.
- After the call, the cost is added with `INCRBYFLOAT` on `budget:YYYY-MM`
  (UTC calendar month) with a 40-day TTL, in one Upstash `/multi-exec`
  transaction.
- The check-then-add isn't one atomic step, so concurrent requests near the
  limit can overshoot by roughly (concurrent requests × cost per request): a few
  cents. Set provider-side spend limits (Anthropic console, TypeSafe if offered)
  as the hard backstop.
- Rate-limited, invalid and budget-exhausted requests cost nothing.

### Fail-closed behavior

| Situation | Result |
|---|---|
| `VERCEL_ENV=production` and Upstash not configured | Every question gets `budget_exhausted`. Per-instance memory can't enforce a shared cap, so the app refuses instead of running uncapped |
| `VERCEL_ENV=production` and `MONTHLY_BUDGET_USD` missing/invalid | Budget is $0, so every question gets `budget_exhausted` |
| Upstash errors while reading the budget | That request gets `budget_exhausted` |
| Upstash errors while adding spend | The answer is still returned; the error type is logged |
| Upstash errors in the rate limiter | The request is allowed (fail open); the budget still caps spend |
| Preview deployments without Upstash | In-memory counters (per function instance) |

## Rate limiting

Two layers:

1. **Edge: Vercel Firewall rule** (below). Stops floods before they start a
   function.
2. **App: per-IP fixed windows** in the same counter store as the budget
   (Upstash in production, memory in dev): `RATE_LIMIT_PER_HOUR` (10) and
   `RATE_LIMIT_PER_DAY` (50). Keys hold a SHA-256 digest of the IP, not the IP.

**Client IP and spoofing.** The app trusts `x-forwarded-for` (first entry) and
`x-real-ip` **only when `VERCEL=1`**. Vercel's edge overwrites these headers with
the real client IP and doesn't forward client-supplied values
([request headers](https://vercel.com/docs/headers/request-headers)). Anywhere
else, a client could set them to any value and get a fresh limit per request,
so the app uses the socket peer address instead. If you put another proxy in
front of Vercel, all traffic will appear to come from that proxy's IPs.

### Vercel Firewall rule (edge layer)

Checked against the live Vercel docs
([WAF Rate Limiting](https://vercel.com/docs/vercel-firewall/vercel-waf/rate-limiting),
[vercel.json `routes.mitigate`](https://vercel.com/docs/project-configuration/vercel-json#routes),
2026-09-27):

- **Rate limits can't be set in `vercel.json`.** Its `mitigate` route option
  only supports `challenge` or `deny`, so the rule is configured in the
  dashboard (or the Vercel REST API).
- Hobby and Pro allow **fixed windows of 10 s to 10 min** keyed by IP (or JA4
  digest); Hobby gets one rate-limit rule per project. A 1-hour window is
  Enterprise-only, so the hourly and daily limits stay in the app. The edge rule
  is a burst limit.
- Counters are per region, so the effective limit can be a bit higher.

Steps:

1. Vercel dashboard → the project → **Firewall** → **Configure** → **+ New Rule**.
2. Name: `ask-burst-limit`.
3. **If**: `Request Path` `equals` `/api/ask` (add `Method` `equals` `POST` if
   you like).
4. **Then**: **Rate Limit** → **Fixed Window**, **Time Window** `600` s (10 min),
   **Request Limit** `5`, key **IP**, action **Default (429)**.
   Optional: start with action **Log** to watch the traffic first.
5. **Save Rule** → **Review Changes** → **Publish**.

## Security headers and CORS

`SecurityHeadersMiddleware` adds these to every app response, and
`vercel.json` repeats them for every path, because CDN-served `public/` files
never reach the app (a test keeps the two copies identical):

- `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'`.
  Citation links to the HOA's PDFs are plain navigations, which CSP doesn't
  govern, so no directive needs widening.
- `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`,
  a restrictive `Permissions-Policy`, `X-Frame-Options: DENY`,
  `Cross-Origin-Opener-Policy: same-origin`, and `Cache-Control: no-store` on `/api/*`.
- **CORS:** there is no CORS middleware, so no `Access-Control-Allow-*` header is
  ever sent and browsers block cross-origin reads and JSON preflights. The app
  also returns 403 for a POST whose `Origin` names another host, and FastAPI
  rejects non-JSON bodies (422), which blocks "simple" cross-site form posts.
  Non-browser clients can still call the API; the rate limits and budget cover
  them.
- OpenAPI docs (`/docs`, `/redoc`, `/openapi.json`) are disabled.

The page renders response data with `textContent`/`createElement` only, and uses
a citation URL as a link only if it starts with `https://` (a test checks that
`app.js` never uses `innerHTML`).

## Privacy

Logs contain the request ID, outcome, latency, estimated cost, and exception
*type*. They never contain question text, answer text, exception messages, or
IP addresses.

## Vercel project setup

Checked against the live Vercel docs
([Python runtime](https://vercel.com/docs/functions/runtimes/python),
[FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi),
[`vercel.json` `functions`](https://vercel.com/docs/project-configuration/vercel-json#functions),
2026-09-27):

- **Runtime:** Vercel detects FastAPI from `pyproject.toml` and installs
  dependencies from `pyproject.toml` + `uv.lock`. Python 3.12 is the default and
  matches `requires-python`.
- **Entrypoint:** `api/index.py` isn't one of Vercel's default entrypoint paths,
  so `pyproject.toml` sets `[tool.vercel] entrypoint = "api.index:app"`. With the
  FastAPI preset, that one function handles every request that isn't a
  `public/` file, including `/api/*`.
- **Static files:** `public/` is served from the CDN at `/`.
- **Corpus:** `vercel.json` → `functions["api/index.py"].includeFiles =
  "corpus.json"` bundles the CI-built corpus into the function. The file is
  gitignored and never committed.
- `maxDuration` is 60 s. `excludeFiles` keeps tests, docs and `public/` out of
  the function bundle.

One-time setup:

1. Create a Vercel project for this repo (`vercel link` locally, or the dashboard).
   Turn **off** automatic Git deployments: `deploy.yml` deploys, so the corpus is
   always bundled.
2. **Upstash:** add Upstash Redis from the Vercel Marketplace (Storage →
   Upstash → Redis) and connect it to the project, or create a database at
   upstash.com. The app reads `UPSTASH_REDIS_REST_URL` and
   `UPSTASH_REDIS_REST_TOKEN`; if the integration injects the REST credentials
   under other names (e.g. `KV_REST_API_URL`/`KV_REST_API_TOKEN`), add these two
   with the same values. Use the read-write token. Scope them to Production (and
   Preview, if you want shared counters there).
3. Project environment variables: `MONTHLY_BUDGET_USD`, `TYPESAFE_API_KEY`,
   `ANTHROPIC_API_KEY`, `ANSWER_MODEL`, and optionally `HOA_DOCUMENTS_URL` and
   the rate limits. Don't set `TYPESAFE_LOG_LEVEL`/`ANTHROPIC_LOG` to `debug`.
4. Create the Firewall rule above.
5. GitHub repository secrets: `VERCEL_TOKEN` (an account token), `VERCEL_ORG_ID`
   and `VERCEL_PROJECT_ID` (both from `.vercel/project.json` after `vercel link`).

## Deploy workflow (`.github/workflows/deploy.yml`)

- **Pull request:** preview deployment. **Push to `main`:** production.
- With no Vercel secrets (including fork PRs, which never get secrets), the job
  writes a "Deploy skipped" summary and succeeds.
- It downloads the `corpus` artifact (`corpus.json` + `corpus_manifest.json`)
  from the **latest successful `build-corpus.yml` run on `main`** and fails with a
  clear message if there's no such run, the artifact expired, or a file is
  missing. It then validates the corpus with `load_corpus`.
- Then `vercel pull` → `vercel build` → `vercel deploy --prebuilt` with the
  Vercel CLI pinned (`VERCEL_CLI_VERSION`).
- The workflow has `contents: read` only; the job adds `actions: read` to download
  the artifact. Actions are SHA-pinned and `persist-credentials: false`.

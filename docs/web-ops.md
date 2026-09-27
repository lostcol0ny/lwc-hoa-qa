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
  `hoa_qa.ask.build_asker(corpus, QASettings.from_env(env))` on the first
  question. If the API keys are missing (or any QA setting is invalid),
  `/api/ask` returns HTTP 503 with an `Answer`-shaped body saying the service
  isn't set up yet; only the exception *type* is logged.
- The web `Asker`/`AskResult` protocols (`src/hoa_qa/web/deps.py`) use
  read-only properties, and `build_asker` assigns the QA core's `QAAsker` to
  `Asker`, so pyright fails the build if the two units' contracts drift.
- On shutdown (FastAPI lifespan) the app awaits `aclose()` on the asker (which
  closes the Jev and Anthropic clients) and on the Upstash client.
- `HOA_QA_FAKE_ASKER` is ignored when `VERCEL_ENV=production`, and
  `build_asker` re-checks `settings.production`, so the fake can't be served
  in production even from a hand-built settings object.

## API

| Route | Result |
|---|---|
| `POST /api/ask` `{"question": "..."}` | An `Answer` JSON (`hoa_qa.models.Answer`) |
| `GET /api/health` | `{"status": "ok", "corpus_build_time", "chunk_count", "documents_url", "budget_config", "budget_config_missing"}`, or 503 `{"status": "unavailable"}` if the corpus is missing or invalid |
| `GET /documents` | 307 redirect to `HOA_DOCUMENTS_URL` (re-checked: https, no userinfo; else the default). The page's documents links point here so they work without JS |

`budget_config` reports the spend cap and Redis configuration health:
- In production, if Redis credentials are half-configured (an incomplete pair),
  it reports `"redis_config_incomplete"` and lists the missing variable(s) in
  `budget_config_missing`. If neither pair is configured, it reports
  `"redis_not_configured"` (with `budget_config_missing` empty).
  Health reports never leak secrets or URLs.
- When Redis is operational (or in dev where in-memory stores are allowed),
  it reports `"budget_below_reservation"` if `MONTHLY_BUDGET_USD < R` (meaning
  every question will get `budget_exhausted`), or `"ok"`.
It reveals neither amount. Until the first question builds the asker, R is the
env default `BUDGET_RESERVE_PER_REQUEST_USD`; afterwards it includes the
asker's `max_cost_usd`. The same condition is logged as a WARNING at startup
(env default R) and once more on the first question (real R).

Every `/api/ask` response body is `Answer`-shaped, so the page renders them all
the same way:

| Status | Outcome | When |
|---|---|---|
| 200 | from the asker, or `budget_exhausted` | normal |
| 403 | `error` | the `Origin` header isn't this site's exact origin, or is malformed |
| 422 | `invalid_input` | the question is empty, over 500 characters, or the body isn't JSON with exactly a `question` field. **Chosen over a 200 `invalid_input`** so clients and logs can tell bad input from answers; the body is still an `Answer` |
| 429 | `error` ("slow down") | over the per-IP rate limit |
| 500 | `error` (generic) | the asker raised |
| 503 | `error` | the QA core or corpus isn't available |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MONTHLY_BUDGET_USD` | `5.0` in dev; **`0` in production if missing or invalid** | Monthly spend cap |
| `BUDGET_RESERVE_PER_REQUEST_USD` | `0.05` | Minimum per-request reservation R (see below). Must be positive; an invalid value stops the app from starting |
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | unset | Shared counters; resolved as a pair, taking precedence over the `KV_*` fallback pair |
| `KV_REST_API_URL`, `KV_REST_API_TOKEN` | unset | Marketplace fallback pair; `KV_REST_API_READ_ONLY_TOKEN` is never used |
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

Each request's cost is the asker's `estimated_cost_usd` (Jev gate + sweep +
verify, plus the answer model; see `docs/qa-core.md`). The budget counter is
`budget_micros:YYYY-MM` (UTC calendar month) with a 40-day TTL.

**Units.** Budget counters hold **integer micro-dollars** (1 µ$ = $0.000001),
updated with `INCRBY`, never `INCRBYFLOAT`. Dollars are converted only at the
edges (`usd_to_micros`, `usd_limit_to_micros`, `micros_to_usd` in
`src/hoa_qa/budget.py`): charges and reservations round **up** to the next
µ$, the budget limit rounds **down**, so rounding can only refuse early.
Sums are exact, with no float drift and no epsilon.

**Old float keys.** Before this change the counter was `budget:YYYY-MM`, written
with `INCRBYFLOAT`. The new `budget_micros:` prefix means such a value can never
reach `INCRBY` (which would reject a non-integer and fail every reservation
closed). The app was never launched on the old keys, so there's no migration:
any leftover `budget:*` key is ignored and expires on its own 40-day TTL.

### Reserve, then reconcile

1. **Reserve.** Before calling the asker, the app atomically adds a worst-case
   amount **R** to the month's counter, but only if the new total stays within
   `MONTHLY_BUDGET_USD`:

   `R = max(BUDGET_RESERVE_PER_REQUEST_USD, asker.max_cost_usd)`

   `max_cost_usd` is optional on the asker (read with `getattr`); invalid values
   are ignored.
   - Upstash: one `EVAL` of a Lua script (`RESERVE_SCRIPT` in
     `src/hoa_qa/budget.py`) that does GET → compare → `INCRBY` → `EXPIRE`.
     Upstash runs a script as a single atomic step under a lock, so concurrent
     reservations can't interleave
     ([EVAL](https://upstash.com/docs/redis/commands/scripting/eval), which is
     also available over the REST API as a JSON-array POST).
   - In memory (dev/tests): the same check-and-add under an `asyncio.Lock`.
     The critical section awaits between the read and the write (as a network
     store would), so without the lock requests really interleave; a test
     swaps the lock for a no-op and asserts that reservations then overspend,
     which proves the concurrency tests can catch a missing lock.
   - Tests run `RESERVE_SCRIPT` itself under [lupa](https://pypi.org/project/lupa/)
     (a dev dependency), with `redis.call` bound to the fake Upstash, so the
     Lua is exercised, not a Python re-implementation of it.
2. **Refuse on any failure.** If the reservation doesn't fit, or the store
   errors (for example Redis reads work but writes fail), the request gets
   `outcome=budget_exhausted` and **the asker is never called**.
3. **Reconcile.** After the call, the app adds `actual − R` (usually negative)
   to the same month key the reservation used, so a request that straddles
   midnight UTC on the 1st settles against the month that admitted it.
   - It reconciles with whatever cost the asker reports, including on `error`
     outcomes. If the asker raises, it uses an `estimated_cost_usd` attribute
     on the exception if there is one; otherwise **the full reservation stays**.
   - A missing, negative or non-finite cost keeps the full reservation.
   - If the reconcile write fails, **the full reservation stays**. It's logged
     with the request ID only and not retried.
   - If `actual > R` (it shouldn't happen), the excess is added and a warning
     is logged.

### The guarantee

Admission is atomic, so the counter never passes the budget because of
concurrency: with budget B, at most `floor(B / R)` requests can be in flight
at once, however many IPs send them (a test fires 50 concurrent requests at
B = 3R and exactly 3 are admitted). The only way real spend can exceed B is a
request whose actual cost exceeds its reservation:

`worst-case overspend = Σ max(0, actual_i − R_i)`

That is **zero** when the asker's `max_cost_usd` is an honest upper bound (or R
is set at or above the true per-request maximum). The counter itself is
**exact to 1 µ$**: amounts are integers, and each conversion from dollars
rounds against spending (costs up, the limit down). Every accounting failure
errs the other way: it keeps reservations and over-counts, which can refuse
questions early but never spends past the cap.

Consequences:
- If `MONTHLY_BUDGET_USD < R`, nothing is admitted; `/api/health` reports
  `"budget_config": "budget_below_reservation"` only when there is no Redis
  config error, and a WARNING is logged.
- Near the cap, questions are refused while `spend + R > B`, even if the real
  cost would have fit. That margin (at most R) goes unused.
- Provider-side spend limits (Anthropic console, TypeSafe if offered) remain a
  good backstop against a dishonest or buggy `max_cost_usd`.
- Rate-limited, invalid and budget-exhausted requests cost nothing.

### Fail-closed behavior

| Situation | Result |
|---|---|
| `VERCEL_ENV=production` and Redis not configured or half-configured | Every question gets `budget_exhausted`. Per-instance memory can't enforce a shared cap, so the app refuses instead of running uncapped. Startup logs an error and `/api/health` reports the specific reason |
| `VERCEL_ENV=production` and `MONTHLY_BUDGET_USD` missing/invalid | Budget is $0, so every question gets `budget_exhausted` |
| Upstash errors while reserving (read or write) | That request gets `budget_exhausted`; the asker isn't called |
| Upstash errors while reconciling | The answer is returned and the full reservation stays counted; the error type and request ID are logged |
| Upstash errors in the rate limiter | The request is allowed (fail open); see below |
| Preview deployments without Upstash | In-memory counters (per function instance) |

## Rate limiting

Two layers:

1. **Edge: Vercel Firewall rule** (below). Stops floods before they start a
   function.
2. **App: per-IP fixed windows** in the same counter store as the budget
   (Upstash in production, memory in dev): `RATE_LIMIT_PER_HOUR` (10) and
   `RATE_LIMIT_PER_DAY` (50). Keys hold a SHA-256 digest of the IP, not the IP.

**Why the app limiter fails open.** The rate limiter protects fairness (one
neighbor can't use up the month), not money. Money is bounded by the atomic
budget reservation, which fails closed and applies across all IPs. If Upstash
writes fail, the reservation already refuses every question, so a fail-open
limiter can't let spend through. Failing closed on a limiter blip would only
lock out neighbors while the money bound stays the same.

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
  ever sent and browsers block cross-origin reads and JSON preflights. FastAPI
  also rejects non-JSON bodies (422), which blocks "simple" cross-site form
  posts.
- **Origin check** on `POST /api/ask`: the `Origin` header must equal this
  site's origin, compared as a normalized (scheme, hostname, effective port)
  tuple. Default ports are implicit, so `https://hoa.example` and
  `https://hoa.example:443` match, but `http://hoa.example` doesn't match an
  HTTPS site.
  - The expected origin comes from the `Host` header. On Vercel the scheme comes
    from `x-forwarded-proto` (set by Vercel's edge; `https` if absent), because
    the function itself sees an internal connection. Locally it comes from the
    request URL.
  - Malformed origins get 403, never a 500: `null`, other schemes, paths,
    userinfo, bad ports, whitespace. The default port is filled in only when
    no port is given, so an explicit `:0` is rejected rather than treated as
    443. Control characters and `?`/`#` are rejected **before** parsing,
    because `urlsplit` silently strips tabs/newlines and drops an empty
    query or fragment (`https://hoa.example?`).
  - **A missing `Origin` is allowed.** Browsers send `Origin` on every POST, so
    only non-browser clients omit it, and they could forge any value anyway.
    The rate limits and budget cover them.
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
   `vercel.json` sets [`git.deploymentEnabled: false`](https://vercel.com/docs/project-configuration/git-configuration#git.deploymentenabled)
   to disable automatic Git deployments on all branches. The only deploy path is
   `.github/workflows/deploy.yml`, which downloads and bundles the gitignored
   `corpus.json`; a Git deployment would lack that artifact.
2. **Upstash:** add Upstash Redis from the Vercel Marketplace (Storage →
   Upstash → Redis) and connect it to the project, or create a database at
   upstash.com. The [Marketplace integration](https://vercel.com/marketplace/upstash/upstash-kv)
   injects `KV_REST_API_URL` and `KV_REST_API_TOKEN`, also documented in
   [Upstash's integration example](https://upstash.com/docs/redis/tutorials/nextjs_with_redis).
   The app resolves credentials as **pairs**: it uses the UPSTASH pair
   (`UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`) if both are set,
   otherwise falling back to the KV pair (`KV_REST_API_URL` and `KV_REST_API_TOKEN`)
   if both are set. URLs and tokens from different families are never mixed.
   Any **incomplete** pair (exactly one of URL or token set in either family) is a
   configuration error, even if the other family is complete. In production,
   a configuration error or missing credentials fails closed (refusing all questions),
   logs a clear startup error, and reports a specific reason in `/api/health`
   (`"redis_config_incomplete: missing <VAR>"` or `"redis_not_configured"`) without
   leaking secrets or URLs. In development, an incomplete pair logs a warning and
   falls back to in-memory counters. `KV_REST_API_READ_ONLY_TOKEN` is never used:
   counters need writes. Both budget and rate limiting use this resolution.
   Scope credentials to Production (and Preview for shared counters there).
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

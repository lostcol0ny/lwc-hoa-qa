# QA core

`hoa_qa.ask.build_asker(corpus, settings)` returns an async `Asker`:
`await asker(question) -> AskResult`, with `.answer` (a `hoa_qa.models.Answer`)
and `.estimated_cost_usd`. The asker also exposes `max_cost_usd`, a worst-case
bound for one call that the web budget reserves up front. The CLI calls the
same pipeline:

```sh
uv run hoa-qa ask "How much is the fine for a second violation?" --corpus corpus.json
uv run hoa-qa ask "..." --corpus corpus.json --json   # full AskResult
```

## Pipeline (spec §4)

| Step | Module | What happens | Outcome if it stops here |
|---|---|---|---|
| 1. Validate | `ask.clean_question` | Drop control (`Cc`) and format (`Cf`, e.g. zero-width, bidi) characters, collapse whitespace, then require 1–500 characters. No model calls. | `invalid_input` |
| 2. Gate | `retrieval/gate.py` | One Jev `Noul`: "Is `message` a question about the Lakewood Creek HOA, its rules, governance, fees, amenities, or neighborhood?" The question is sent as state (data), never inside the instructions. | `refused_off_topic`, with a link to the documents; the answer model is never called |
| 3. Sweep | `retrieval/sweep.py` | One `Noul` per chunk ("Does this passage help answer the question?"), batched **one request per `doc_id`**. Documents over Jev's limits are split across requests, and a single chunk over the limits is split into sub-passages (its score is the best of its parts). Keep the top `SWEEP_TOP_K` at or above `SWEEP_THRESHOLD`; ties keep corpus order. | `not_found` ("couldn't find it; contact the Board") with no free-form answer |
| 4. Answer | `answer/prompt.py`, `answer/provider.py` | The answer model gets the §3.2 policy in the system prompt and only data in the user turn: `<passages>` (each with `chunk_id`, `citation_label`, `authority`, `effective_date`, and a superseded/informal note) and `<question>`. Our own tags inside the data are defanged. The output (JSON schema via `output_config.format`, validated by pydantic `AnswerDraft`) is a list of **claims**. Each claim is one short factual statement (`kind`: `answer` or `conflict`, plus an `essential` flag) with 1–3 citations `{chunk_id, quote}`. The output also carries `confidence` and `refer_to_board`. Caps: 8 claims, 3 citations per claim, 400 characters per statement. | `not_found` if the model returns no claims |
| 5a. Quote check | `verify/quotes.py` | Per citation: the quote must be a substring of its chunk's `text_clean` after normalizing whitespace, case, and curly quotes, and its `chunk_id` must be one of the passages shown. Failing citations are dropped. A claim with no surviving citation fails. | |
| 5b. Support check | `verify/support.py` | One Jev `Noul` **per claim**: "Do the passages in `claims[i].passages` support this specific claim, `claims[i].statement`?" Yes means *everything* the statement asserts is stated in the passages. Claims are batched under the Jev limits; a claim too large for any request fails closed. Below `SUPPORT_THRESHOLD`, the claim fails. | |
| 5c. Retry | `ask.py` | If any claim fails, or no answer claim survives, or the output was invalid, regenerate **once**, listing the failed claims. Then apply the drop rule below. | `not_found` |
| 6. Respond | `ask.py` | Compose the `Answer` from verified claims only (see below): `Citation.url = citation_url(chunk)`, the §5 disclaimer, and a uuid4 `request_id`. | `answered` |

### How the answer text is built, and the claim-drop rule

- `answer_text` is the verified `answer` claims' statements, in order. Code may
  append two fixed sentences: `OMITTED_NOTE` when a claim was dropped, and
  `BOARD_REFERRAL` when `refer_to_board` is set. The model never writes free
  text that reaches the user without verification.
- `conflicts_noted` is the verified `conflict` claims' statements.
- `citations` is the verified claims' quote-checked citations, deduplicated.
- **Drop rule** (after the one regenerate):
  - If any **essential** claim still fails, the result is `not_found`.
  - If no `answer` claim survives, the result is `not_found`.
  - Otherwise the failed non-essential claims are dropped, the survivors are
    returned as `answered`, and `OMITTED_NOTE` is appended.

  The `essential` flag can only make the outcome stricter. Unverified text is
  never shown whatever the flag says.
- The pipeline never returns `answered` with zero citations: every surviving
  claim has at least one quote-checked citation.

Any exception from a provider becomes `Outcome.error`. The log records only the
exception *type*, because SDK error messages can echo request bodies. When Jev
batches fail, every sibling batch still runs to completion. `PartialCostError`
carries the tokens the successful batches billed, and `ask()` adds them to
`AskResult.estimated_cost_usd` even on the `error` outcome, so the web budget
sees that spend.

## Configuration

`QASettings.from_env()` reads these variables. Empty values mean "use the default".

| Variable | Default | Meaning |
|---|---|---|
| `TYPESAFE_API_KEY` | (required) | Jev API key |
| `ANTHROPIC_API_KEY` | (required) | Answer-model API key |
| `ANSWER_MODEL` | `claude-haiku-4-5` | Anthropic model ID for step 4 |
| `JEV_MODEL` | `jev-latest` | Jev model; pin a version (e.g. `jev-1.13.0`) once thresholds are tuned against it |
| `GATE_THRESHOLD` | `0.5` | Minimum on-topic probability |
| `SWEEP_THRESHOLD` | `0.3` | Minimum relevance probability for a chunk |
| `SWEEP_TOP_K` | `8` | Maximum passages sent to the answer model |
| `JEV_CONCURRENCY` | `4` | Maximum concurrent Jev requests per `ask()` (sweep and support share one semaphore). `SWEEP_CONCURRENCY` is still read as a fallback. |
| `SUPPORT_THRESHOLD` | `0.5` | Minimum support probability for a claim |
| `HOA_DOCUMENTS_URL` | `https://lakewoodcreekhoa.com/` | Link used in refusal / not-found messages |

Keys are held as `SecretStr`, so they never appear in `repr()` or logs.

## Jev request sizing (guaranteed limits)

Jev 1.13 allows 64K tokens per request (state plus all questions), and 32K for
the state plus the longest question. The SDK has no tokenizer or
token-counting endpoint (it only calls `/v1/systemone` and `/v1/models`), so
`retrieval/jev.py` uses a deliberately conservative bound:

- `conservative_tokens(text)` = ceil(ASCII bytes / 2.5) + 1 per non-ASCII
  UTF-8 byte. English averages about 4–4.5 characters per token, so the ASCII
  term has roughly 1.6–1.8× headroom. The non-ASCII term is the byte-fallback
  worst case.
- Fixed overheads: `REQUEST_OVERHEAD_TOKENS` = 256 per request (the JSON
  envelope and model field), and `QUESTION_OVERHEAD_TOKENS` = 32 per question
  (name, type, criteria keys).
- `request_tokens` returns (whole request, state + longest question) under that
  bound. `pack` only emits groups within both limits and returns anything too
  large on its own separately. **No request known to exceed a limit is ever
  sent.**
  - In the sweep, an oversized chunk is split at whitespace into sub-passages
    that each fit. If even minimal pieces cannot fit, the chunk is skipped with
    a warning that logs only its `chunk_id`.
  - In the support check, an oversized claim fails closed (the warning logs
    only its index).

Residual risk: ASCII text that tokenizes worse than 2.5 bytes/token (long runs
of symbols or digits) could still exceed the estimate. The API would reject
that request, and the result would be an `error` outcome with its partial cost
counted.

## Cost model

`AskResult.estimated_cost_usd` = Jev input tokens × Jev rate + answer input
tokens × input rate + answer output tokens × output rate. The token counts come
from each SDK's usage fields (Jev's `usage.input_tokens`, falling back to our
bound if the API omits it). All rates live in `answer/pricing.py`, with their
sources and dates. An unlisted `ANSWER_MODEL` is costed at the most expensive
tier, so the budget errs toward stopping early.

| Component | Rate (2026-09-27) | Typical tokens | Typical cost |
|---|---|---|---|
| Jev gate | $0.042 / MTok input, output free | ~150 | ~$0.00001 |
| Jev sweep (full corpus, ~74K tokens + per-question overhead) | same | ~90–110K | ~$0.004 |
| Jev support check | same | ~2–4K | ~$0.0001 |
| `claude-haiku-4-5` answer | $1 / MTok in, $5 / MTok out | ~3–4K in, ~500 out | ~$0.006 |
| **Answered question** | | | **~$0.01** (up to ~2× if it regenerates) |

These are estimates. Integration (unit 5) replaces them with measured numbers.
A gate refusal costs only the gate. Invalid input costs nothing.

### Worst-case bound: `max_cost_usd`

Computed once per asker by `ask.max_cost_usd(corpus, settings, answer_model)`.
Every text is sized with `conservative_tokens` and the worst inputs the
pipeline allows:

```
Q  = a 500-character question of 4-byte characters
S  = a 400-character claim statement of 4-byte characters

jev_tokens =
    gate(Q)
  + Σ over plan_batches(Q, corpus) of request_tokens(batch)    # full sweep
  + 2 × MAX_CLAIMS × request_tokens(one claim S citing the 3 largest chunks)
                                                               # support, both attempts
answer_in  = tokens(system + user prompt with Q, the SWEEP_TOP_K largest chunks,
             and the retry list of MAX_CLAIMS failed statements S)
           + ANSWER_REQUEST_OVERHEAD_TOKENS (2048)
max_cost_usd = jev_tokens × Jev rate
             + 2 × (answer_in × input rate + MAX_OUTPUT_TOKENS (4096) × output rate)
```

A support request per claim is an upper bound, since packing claims together
only removes overhead. The retry prompt is the larger of the two answer prompts,
so it's used for both attempts. The bound comes from the same token estimate as
request sizing, so it carries the same residual risk.

For `tests/fixtures/mini_corpus.json` with default settings (`claude-haiku-4-5`):

| Term | Tokens | USD |
|---|---|---|
| Jev: gate 2,480 + sweep 17,277 (7 requests) + support 2 × 17,224 | 54,205 | $0.00228 |
| Answer: 2 × (19,034 in + 4,096 out) | | $0.07903 |
| **`max_cost_usd`** | | **$0.08130** |

The two full-length outputs dominate the bound. On the real corpus the sweep
term grows with the corpus: roughly its UTF-8 size ÷ 2.5, plus per-question overhead.

## Logging and privacy

`hoa_qa.ask` logs one structured line per request (also attached as
`record.hoa_qa`): `request_id`, `outcome`, `latency_ms`, `jev_input_tokens`,
`answer_model`, `answer_calls`, `answer_input_tokens`, `answer_output_tokens`,
`citations`, `estimated_cost_usd`, and `notes` (`invalid_draft`, `no_claims`,
`claims_failed=N`). It **never logs question, answer, or claim text**
(spec §6), and tests assert this.

Do **not** set `TYPESAFE_LOG_LEVEL=debug` or `ANTHROPIC_LOG=debug` in
production. At debug level both SDKs log request bodies, which contain the
question.

## Testing

`tests/qa/` uses fake Jev and answer providers (`tests/qa/qa_fakes.py`) against
`tests/fixtures/mini_corpus.json`. It needs no network and no keys. The fakes
bill the same conservative token counts the pipeline uses, and the fake Jev
asserts that every request it receives is within both limits. The real SDK
adapters are covered offline through `httpx2.MockTransport`.

`tests/qa/test_live.py` (`@pytest.mark.live`) runs only when both API keys are
set. It asks about a **second violation** (expects the 2023 Rules' $125) and a
**continuing violation** (expects $75 per day). The 2023 tiers are First
($75), Second ($125), and Continuing ($75 per day).

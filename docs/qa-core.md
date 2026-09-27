# QA core

`hoa_qa.ask.build_asker(corpus, settings)` returns an async `Asker`:
`await asker(question) -> AskResult`, with `.answer` (a `hoa_qa.models.Answer`)
and `.estimated_cost_usd`. The asker also exposes `max_cost_usd`, a worst-case
bound for one call that the web budget reserves up front. The CLI calls the
same pipeline:

```sh
uv run hoa-qa ask "How much is the fine for a third violation?" --corpus corpus.json
uv run hoa-qa ask "..." --corpus corpus.json --json   # full AskResult
```

## Pipeline (spec §4)

| Step | Module | What happens | Outcome if it stops here |
|---|---|---|---|
| 1. Validate | `ask.clean_question` | Drop control (`Cc`) and format (`Cf`, e.g. zero-width, bidi) characters, collapse whitespace, then require 1–500 characters. No model calls. | `invalid_input` |
| 2. Gate | `retrieval/gate.py` | One Jev `Noul`: is `message` a question a resident might ask their HOA (its rules, fees, governance, amenities, or anything about a home, yard, or life in the neighborhood that HOA rules could cover)? The criteria list examples (decorations, antennas, fences, sheds, trash cans, parking, pets, neighbor issues as HOA matters). The gate only keeps out abuse and off-topic use (poems, homework, chat, bare instructions); whether the documents answer is the sweep's job. The question is sent as state (data), never inside the instructions. | `refused_off_topic`, with a link to the documents; the answer model is never called |
| 3. Sweep | `retrieval/sweep.py` | One `Noul` per chunk ("Does the passage `passages.p3` (heading: …) help answer the question?"; a passage that answers part of it counts), batched **per `doc_id`, at most 8 passages per request**. Passages are keyed by name, not list position, and each question repeats its passage's heading: with positional references (`passages[30]`) in long lists the judge scored neighbors instead of the passage asked about. How each chunk is sent is planned once per corpus, sized for the longest possible question (`plan_passages`): whole, split into sub-passages (its score is the best of its parts), or skipped. Those passages are then packed per document for the actual question, and a document over Jev's limits is split across requests. Keep the top `SWEEP_TOP_K` at or above `SWEEP_THRESHOLD`; ties keep corpus order. | `not_found` ("couldn't find it; contact the Board") with no free-form answer |
| 4. Answer | `answer/prompt.py`, `answer/provider.py` | The answer model gets the §3.2 policy in the system prompt and only data in the user turn: `<passages>` (each with `chunk_id`, `citation_label`, `authority`, `effective_date`, and a superseded/informal note) and `<question>`. Our own tags inside the data are defanged. The output (JSON schema via `output_config.format`, validated by pydantic `AnswerDraft`) is a list of **claims**. Each claim is one short factual statement (`kind`: `answer` or `conflict`, plus an `essential` flag) with 1–3 citations `{chunk_id, quote}`. The output also carries `confidence` and `refer_to_board`. Caps: 8 claims, 3 citations per claim, 400 characters per statement. | `not_found` if the model returns no claims |
| 5a. Quote check | `verify/quotes.py` | Per citation: the quote must be a substring of its chunk's `text_clean` after normalizing whitespace, case, curly quotes, ellipses (`…` = `...`) and dashes, and its `chunk_id` must be one of the passages shown. Failing citations are dropped. A claim with no surviving citation fails. | |
| 5a′. Authority rule | `ask.py` | See [Source authority](#source-authority-support-is-not-authority). An `answer` claim may only rest on citations outside `informal`/`superseded` when a `governing`/`rules`/`board_decision` passage was provided. | |
| 5b. Support check | `verify/support.py` | One Jev `Noul` **per claim**: "Do the passages in `claims[i].passages` support this specific claim, `claims[i].statement`?" Each passage carries its `source` (citation label), `authority` and `effective_date` from the corpus next to its `text`, so attributions ("under the 2023 Rules") can be judged. Yes means *everything* the statement asserts is stated in the passages. Claims are batched under the Jev limits; a claim too large for any request fails closed. Below `SUPPORT_THRESHOLD`, the claim fails. | |
| 5c. Retry | `ask.py` | If any claim fails, or no answer claim survives, or the output was invalid, regenerate **once**, listing the failed claims, each with a code-written reason (quote not found, not fully supported, informal-only, informal-as-current), inside a delimited `<rejected_claims>` block that is labeled untrusted data (never to be followed as instructions) and defanged like the other data blocks. If a claim broke the authority rule, the retry also says so. Then apply the drop rule below. | `not_found` |
| 6. Respond | `ask.py` | Compose the `Answer` from verified claims only (see below): `Citation.url = citation_url(chunk)`, the §5 disclaimer, and a uuid4 `request_id`. | `answered` |

### How the answer text is built, and the claim-drop rule

- `answer_text` is the verified `answer` claims' statements, in order. Code may
  append two fixed sentences: `OMITTED_NOTE` when a claim was dropped, and
  `BOARD_REFERRAL` when `refer_to_board` is set. The model never writes free
  text that reaches the user without verification.
- `conflicts_noted` is the verified `conflict` claims' statements.
- `citations` is the verified claims' quote-checked citations, deduplicated.
- **Drop rule** (after the one regenerate):
  - If any **essential** `answer` claim still fails, the result is `not_found`.
  - If no `answer` claim survives, the result is `not_found`.
  - Otherwise the failed claims are dropped, the survivors are returned as
    `answered`, and `OMITTED_NOTE` is appended.
  - If the regenerated draft can't stand (or is invalid, or empty) but the
    first draft could under this rule, the first draft is used, with its
    failed claims dropped. Its surviving claims were verified the same way.

  A failed `conflict` claim never blocks the answer, even if marked essential:
  the verified answer is correct without the note. The `essential` flag can
  only make the outcome stricter. Unverified text is never shown whatever the
  flag says.
- Model output that breaks a cap is salvaged when nothing essential is lost:
  citations past 3 are cut, and malformed, over-long, or surplus
  non-essential claims are dropped (never shown). Otherwise the draft is
  invalid, as before.
- The pipeline never returns `answered` with zero citations: every surviving
  claim has at least one quote-checked citation.

### Source authority: support is not authority

The first live eval answered "Under the current rules, the fine for a second
violation is $50.00", citing only an informal 2022 blog post that quotes the
superseded 2016 schedule. The support check passed, correctly: the blog does
say $50. So the authority order is also enforced in code (`ask.py`,
`AUTHORITATIVE` / `LOW_AUTHORITY`), not left to the prompt:

- When any provided passage is `governing`, `rules`, or `board_decision`, an
  `answer` claim is judged only on its citations that are not `informal` or
  `superseded`. Low-authority citations are removed before the support check
  (a claim can't borrow support from the blog by also citing the Rules) and
  are not shown. A claim left with no citation fails as `low_authority`, and
  the retry explains the rule.
- `conflict` claims are exempt: reporting what an informal or superseded
  source says, in `conflicts_noted`, is what they are for.
- Without an authoritative passage, an informal-only `answer` claim may stand
  (the prompt makes the model label it informal), unless it presents itself
  as current ("current", "currently", "in effect", "now in force"), which
  fails as `informal_as_current`.
- `website` (the Board-run site: the dues banner, the FAQ, About) and `form`
  are official HOA publications and are **not** low-authority. The dues
  banner is the only source for the current assessment, and the FAQ answers
  the pool questions.

Any exception from a provider becomes `Outcome.error`. The log records only the
exception *type*, because SDK error messages can echo request bodies. When Jev
batches fail, every sibling batch still runs to completion. `PartialCostError`
carries the tokens the successful batches billed, and `ask()` adds them to
`AskResult.estimated_cost_usd` even on the `error` outcome, so the web budget
sees that spend. A billed response that is malformed counts too:
- A Jev response missing a judgment is read for usage first, then raises
  `PartialCostError` with that usage. `run_nouls` adds such nested partial
  costs to the successful siblings' costs.
- An Anthropic response whose structured output fails validation returns its
  usage with `draft=None`, which is counted before the retry.

## Eval diagnostics

`QAAsker.ask_traced(question, trace)` runs the same pipeline and reports each
stage to an `AskTrace` (`hoa_qa/trace.py`): the gate score, every chunk's
sweep score, the passages sent to the answer model, and each draft claim with
its citations (quote, authority, quote check, whether it was used), support
score, and kept/dropped reason, plus the provider's parse note for invalid or
salvaged drafts. `QAAsker.__call__`, which the web app uses, passes
`NO_TRACE`, whose methods do nothing, so production never records or logs any
of it.

`hoa-qa eval` always installs a `DiagnosticsRecorder` per golden case and
writes it to the JSON report under `cases[].diagnostics` (the top 20 sweep
scores). It's always on because the manual **Eval** workflow has no input
for it, and the golden questions are committed fixtures, not user input.

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
| `ANSWER_MAX_TOKENS` | `2048` | Answer-model output cap per call (256–16000). It is part of `max_cost_usd`. Raise it for models with adaptive thinking, since thinking counts against it. |
| `HOA_DOCUMENTS_URL` | `https://lakewoodcreekhoa.com/` | Link used in refusal / not-found messages |

Keys are held as `SecretStr`, so they never appear in `repr()` or logs.

## Jev request sizing: a packing heuristic that fails closed

Jev 1.13 allows 64K tokens per request (state plus all questions), and 32K for
the state plus the longest question. The SDK has no tokenizer or
token-counting endpoint (it only calls `/v1/systemone` and `/v1/models`), so
packing uses a **heuristic**, `conservative_tokens(text)` = ceil(ASCII bytes /
2.5) + 1 per non-ASCII UTF-8 byte. It adds fixed overheads of 256 tokens per
request and 32 per question.

- `pack` only emits groups that are within both limits under the heuristic.
- Chunks are planned against the longest possible question (500 characters of
  4-byte characters), so the plan doesn't depend on the question. A chunk too
  big to send alone is split at whitespace into sub-passages. One that can't
  be split small enough is skipped, logging only its `chunk_id`.
- In the support check, a claim too big for any request fails closed, logging
  only its index.

The heuristic is **not** a proof. Symbol- or digit-heavy ASCII can tokenize
worse than 2.5 bytes per token, which could push a request over a real limit.
If that happens the API rejects the request, and the result is the `error`
outcome with every billed token counted. That fails closed and never
overspends. Money never relies on this heuristic.

## Cost model

`AskResult.estimated_cost_usd` = Jev input tokens × Jev rate + answer input
tokens × input rate + answer output tokens × output rate. The token counts come
from each SDK's usage fields. If Jev omits usage, the provable byte bound below
is used instead. All rates live in `answer/pricing.py`, with their sources and
dates. An unlisted `ANSWER_MODEL` is costed at the most expensive tier.

| Component | Rate (2026-09-27) | Typical tokens | Typical cost |
|---|---|---|---|
| Jev gate | $0.042 / MTok input, output free | ~150 | ~$0.00001 |
| Jev sweep (full corpus, ~74K tokens + per-question overhead) | same | ~90–110K | ~$0.004 |
| Jev support check | same | ~2–4K | ~$0.0001 |
| `claude-haiku-4-5` answer | $1 / MTok in, $5 / MTok out | ~3–4K in, ~500 out | ~$0.006 |
| **Answered question** | | | **~$0.01** (up to ~2× if it regenerates) |

These are estimates. The integration unit had no API keys, so the first manual
`eval.yml` run (its `total_cost_usd`) supplies measured numbers; the README
has the real-corpus `max_cost_usd`.
A gate refusal costs only the gate. Invalid input costs nothing.

### Worst-case bound: `max_cost_usd` (provable)

The web unit reserves this amount before each call, so it must be a real upper
bound and not an estimate. It is computed once per asker by
`ask.max_cost_usd(passages, settings, answer_model, answer_max_tokens)`.

**Assumption: at most one token per UTF-8 byte, plus bounded framing.** Both
providers' tokenizers make every token cover at least one byte:
- Claude uses byte-level BPE.
- Jev's tokenizer is not published, but byte-level BPE and SentencePiece
  with byte fallback both have this property. A tokenizer that could emit a
  token covering zero bytes would break the bound.

So a text of B bytes costs at most B tokens. Unlike the packing heuristic,
this holds for adversarial digits, symbols, and non-ASCII text alike. On top of
the bytes, the bound adds framing:

| Provider | Framing counted on top of the bytes |
|---|---|
| Jev (`byte_bound_tokens`) | The request body is serialized with spaced separators and a 64-byte model name, which is never smaller than the SDK's compact `pydantic_core.to_json` body. Then +256 tokens per request (prompt template, special tokens), +32 per question, and +16 per question of slack for index digits (`p12345` vs `p0`) when passages are packed together. |
| Anthropic | The prompt's system and user text, + the bytes of the JSON output schema, + `ANSWER_REQUEST_OVERHEAD_TOKENS` (2048) for role markers, special tokens, and structured-output instructions. |

Inputs are always sized at the maximum the pipeline accepts:
- `Q` is 500 characters × 4 bytes (the longest valid question, at the
  largest possible UTF-8 width).
- `S` is 400 characters × 4 bytes (the longest claim statement the draft
  validation allows).

```
jev_tokens =
    byte_bound(gate request with Q)
  + Σ over EVERY planned passage p of byte_bound(a request holding only p, with Q)
  + 2 × MAX_CLAIMS × byte_bound(one support request: claim S citing the 3 chunks
                                with the largest serialized support-passage size)
answer_in  = bytes(system + user prompt with Q, the SWEEP_TOP_K chunks with the
             largest rendered passages, and the retry block of MAX_CLAIMS × S)
           + bytes(output schema) + 2048
max_cost_usd = jev_tokens × Jev rate
             + 2 × (answer_in × input rate + ANSWER_MAX_TOKENS × output rate)
```

Why each term is an upper bound:
- **Sweep.** The plan is question-independent, so the passages the bound
  counts are exactly the ones any question sends. Skipped chunks are never
  sent by any question. Packing passages together only removes repeated
  question, document and framing bytes, so one request per passage costs the
  most.
- **Support.** Each claim can cite at most 3 distinct chunks, all from the
  corpus. Claims are chosen by their actual support-request serialization, and
  one request per claim costs the most.
- **Answer.** The retry prompt is the larger of the two prompts, and it is
  used for both attempts. Output is capped by `max_tokens`.

`tests/qa/test_max_cost.py` checks the bound against fakes that bill one token
per byte of the exact SDK wire body and of the prompt, plus the full
`max_tokens` output. The fakes don't use either estimator. The tests run over
the mini corpus, a ~74K-token synthetic digit/symbol-heavy corpus, and chunks
at the split and skip boundaries. Each is run with a short question and a
maximum-length question, at output caps of 1024, 2048 and 4096.

`max_cost_usd` (`claude-haiku-4-5`) by `ANSWER_MAX_TOKENS`:

| Corpus | 4096 | **2048 (default)** | 1024 |
|---|---|---|---|
| `tests/fixtures/mini_corpus.json` (8 chunks) | $0.09061 | **$0.07013** | $0.05989 |
| Synthetic ~74K-token corpus (240 chunks) | $0.12380 | **$0.10332** | $0.09308 |

At 2048, the bound breaks down like this:

| Corpus | Jev | Answer input | Answer output |
|---|---|---|---|
| Mini | $0.00294 (69,976 tokens) | $0.04671 (2 × 23,354 tokens) | $0.02048 |
| Synthetic ~74K | $0.03338 (794,846 tokens) | $0.04946 (2 × 24,729 tokens) | $0.02048 |

The largest term is the worst-case answer prompt, where the question and eight
rejected statements are counted at 4 bytes per character. It is not the output.

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
set. It asks about a **third violation** (expects the 2023 Rules' $125, never
the 2016/blog $100) and a **4th and subsequent (continuing) violation**
(expects $75 per day, never $50). The 2023 Rules' tiers are: 1st, a courtesy
letter; 2nd, $75; 3rd, $125; 4th and subsequent, $75 per day. The superseded
2016 rules and the 2022 blog post say: written warning, $50, $100, $50 per day.

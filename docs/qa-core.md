# QA core

`hoa_qa.ask.build_asker(corpus, settings)` returns an async `Asker`:
`await asker(question) -> AskResult`, with `.answer` (a `hoa_qa.models.Answer`)
and `.estimated_cost_usd`. The CLI calls the same pipeline:

```sh
uv run hoa-qa ask "How much is the fine for a second violation?" --corpus corpus.json
uv run hoa-qa ask "..." --corpus corpus.json --json   # full AskResult
```

## Pipeline (spec §4)

| Step | Module | What happens | Outcome if it stops here |
|---|---|---|---|
| 1. Validate | `ask.clean_question` | Drop control (`Cc`) and format (`Cf`, e.g. zero-width, bidi) characters, collapse whitespace, then require 1–500 characters. No model calls. | `invalid_input` |
| 2. Gate | `retrieval/gate.py` | One Jev `Noul`: "Is `message` a question about the Lakewood Creek HOA, its rules, governance, fees, amenities, or neighborhood?" The question is sent as state (data), never inside the instructions. | `refused_off_topic`, with a link to the documents; the answer model is never called |
| 3. Sweep | `retrieval/sweep.py` | One `Noul` per chunk ("Does this passage help answer the question?"), batched **one request per `doc_id`** in one `system_one` call; documents that exceed Jev's limits are split. Batches run concurrently under a semaphore. Keep the top `SWEEP_TOP_K` at or above `SWEEP_THRESHOLD`; ties keep corpus order. | `not_found` ("couldn't find it; contact the Board") with no free-form answer |
| 4. Answer | `answer/prompt.py`, `answer/provider.py` | The answer model gets the §3.2 policy in the system prompt and only data in the user turn: `<passages>` (each with `chunk_id`, `citation_label`, `authority`, `effective_date`, and a superseded/informal note) and `<question>`. Our own tags inside the data are defanged so it cannot close its block. Output is constrained by a JSON schema (`output_config.format`) and validated with pydantic (`AnswerDraft`). | |
| 5a. Quote check | `verify/quotes.py` | Each quote must be a substring of its chunk's `text_clean` after normalizing whitespace, case, and curly quotes, and its `chunk_id` must be one of the passages shown. Empty quotes and duplicates are dropped. | |
| 5b. Support check | `verify/support.py` | One Jev `Noul` per surviving citation ("Does this passage support this claim?"), batched in one request. Drop below `SUPPORT_THRESHOLD`. | |
| 5c. Retry | `ask.py` | If no citation survives, or the output was invalid, regenerate **once** and tell the model its citations failed. If that also fails, stop. The pipeline never returns `answered` with zero citations. | `not_found` |
| 6. Respond | `ask.py` | `Answer` with `Citation.url = citation_url(chunk)`, the §5 disclaimer, and a uuid4 `request_id`. | `answered` |

Any exception from a provider becomes `Outcome.error`. The cost already spent
is still reported, and the log records only the exception *type*, because SDK
error messages can echo request bodies.

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
| `SWEEP_CONCURRENCY` | `4` | Maximum concurrent sweep requests |
| `SUPPORT_THRESHOLD` | `0.5` | Minimum support probability for a citation |
| `HOA_DOCUMENTS_URL` | `https://lakewoodcreekhoa.com/` | Link used in refusal / not-found messages |

Keys are held as `SecretStr`, so they never appear in `repr()` or logs.

## Jev request sizing

Jev 1.13 allows 64K tokens per request (state plus all questions), and 32K for
the state plus the longest question. `retrieval/jev.pack` greedily fills each
batch while both budgets hold, using a conservative estimate of about 4
characters per token. Because a document's passages all live in the state, the
32K state budget is usually the one that binds. A single passage that exceeds
the limits on its own still gets a batch of its own (logged), so the API
rejects it visibly instead of it being dropped in silence. Ingest should keep
chunks well below this size.

## Cost model

`AskResult.estimated_cost_usd` = Jev input tokens × Jev rate + answer input
tokens × input rate + answer output tokens × output rate. The token counts come
from each SDK's usage fields (Jev's `usage.input_tokens`, falling back to our
estimate if the API omits it). All rates live in `answer/pricing.py`, with
their sources and dates. An unlisted `ANSWER_MODEL` is costed at the most
expensive tier, so the budget errs toward stopping early.

| Component | Rate (2026-09-27) | Typical tokens | Typical cost |
|---|---|---|---|
| Jev gate | $0.042 / MTok input, output free | ~150 | ~$0.00001 |
| Jev sweep (full corpus, ~74K tokens + per-question overhead) | same | ~90–110K | ~$0.004 |
| Jev support check | same | ~2–4K | ~$0.0001 |
| `claude-haiku-4-5` answer | $1 / MTok in, $5 / MTok out | ~3–4K in, ~300 out | ~$0.005 |
| **Answered question** | | | **~$0.01** (up to ~2× if it regenerates) |

These are estimates. Integration (unit 5) replaces them with measured numbers.
A gate refusal costs only the gate. Invalid input costs nothing.

## Logging and privacy

`hoa_qa.ask` logs one structured line per request (also attached as
`record.hoa_qa`): `request_id`, `outcome`, `latency_ms`, `jev_input_tokens`,
`answer_model`, `answer_calls`, `answer_input_tokens`, `answer_output_tokens`,
`citations`, `estimated_cost_usd`, and `notes` (`invalid_draft`,
`citations_failed`). It **never logs question or answer text** (spec §6), and
a test asserts this.

Do **not** set `TYPESAFE_LOG_LEVEL=debug` or `ANTHROPIC_LOG=debug` in
production. At debug level both SDKs log request bodies, which contain the
question.

## Testing

`tests/qa/` uses fake Jev and answer providers (`tests/qa/qa_fakes.py`) against
`tests/fixtures/mini_corpus.json`. It needs no network and no keys. The real
SDK adapters are covered offline through `httpx2.MockTransport`.
`tests/qa/test_live.py` (`@pytest.mark.live`) runs only when both API keys are
set. It asks the third-offense question and expects the 2023 Rules' $125.

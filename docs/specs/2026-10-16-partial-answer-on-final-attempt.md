# Partial answer on the final attempt

Date: 2026-10-16. Status: built.

## Problem

The answer model marks each claim `essential` ("the answer would be wrong or
misleading without it"). Under the drop rule (docs/qa-core.md), one failed
essential `answer` claim means the draft cannot stand: it is regenerated
once, and if the second draft cannot stand either, the result is
`not_found` (`unverified`, "I found related passages but couldn't verify a
precise answer").

The model over-applies the flag. In the eval runs of 2026-10-15 (38822382735
and 38824221224, 58 questions), 5 drafts could not stand. In every one, at
least one other essential claim verified:

| Case | Failed essential claim | Reason | Essential claims verified |
|---|---|---|---|
| dues-history | an informal newsletter quote | low_authority | 3 |
| injection-dues-zero, draft 1 | "$99 per quarter from 2023" | unsupported 0.35 | 1 |
| injection-dues-zero, draft 2 | an informal newsletter quote | low_authority | 2 |
| statute-cicaa-applies | a definition paraphrase | unsupported 0.47 | 2 |
| upcoming-collection-policy, drafts 1 and 2 | the policy's checklist as one claim | not_yet_in_effect, then unsupported 0.17 | 1 |

Two cases ended `not_found` although the claims that verified in the second
draft were a correct answer on their own (for injection-dues-zero: the 2022
minutes record $99 a quarter for 2023; the website shows $113 a quarter for
2026). Retrieval was not the cause: the eval's `retrieval_hit` was true for
both.

## Decision

On the **final** attempt only, a draft that cannot stand under the drop rule
may still stand as a **partial answer** if at least one essential `answer`
claim verified. Its failed claims are dropped, as for any dropped claim, and
the answer says it may be incomplete.

Attempt 1 is unchanged: a draft that cannot stand is still regenerated, with
its failed claims listed. Skipping that retry (or skipping it only for
failure reasons a retry can't fix, such as `low_authority`) was considered
and deferred: the model's own reason for marking a claim essential is that
the answer would mislead without it, so the retry stays the first remedy, and
evals will show whether partial answers are good enough to use earlier.

Unverified text is never shown, as before. Only which verified claims may
stand changes.

Rejected alternatives:

- **Best of both drafts** (keep whichever draft verified more essential
  claims). The first draft's streamed sentences were already withdrawn
  (`reset`) when the retry began, so they would have to be streamed again,
  and in both `not_found` cases the second draft alone was enough.
- **Combine both drafts' verified claims.** The drafts word the same facts
  differently, so the result can repeat or contradict itself, and nothing
  checks the combination.

## Behavior

In `QAAsker._run` (`ask.py`), after verifying a draft:

1. Clean (nothing failed and an answer claim survived): answered, as now.
2. `_can_drop(verified)`: answered with `OMITTED_NOTE`, as now.
3. **New:** final attempt and `_can_stand_partially(verified)`: answered
   with `PARTIAL_NOTE` in place of `OMITTED_NOTE`.
4. Otherwise: regenerate (attempt 1) or `not_found(unverified)` (final), as
   now.

`_can_stand_partially(verified)` is true when at least one claim with
`kind == "answer"`, `essential`, and `ok` exists. A verified non-essential
claim alone isn't enough: when every essential claim failed, nothing the
model considered central was verified.

`PARTIAL_NOTE`: "Part of this answer could not be verified against the
documents and was left out, so it may be incomplete." It appears instead
of `OMITTED_NOTE`, never both. `_answered` gets the note to use, rather than
the boolean `dropped`.

The board referral, statute disclaimer, CICAA note, and conflict notes behave
as for any answered result.

**Streaming:** the draft-2 sentences already on screen are verified claims
of the draft being kept, so no `reset` is sent; the answer completes.

## Observability

- Trace tag `partial` (in `_tags`, beside `retried` and `claims-dropped`),
  set when `PARTIAL_NOTE` is in the answer. `claims-dropped` stays tied to
  `OMITTED_NOTE`, so the two never overlap.
- `_log_attempt` logs the attempt's terminal state as `partial` (the other
  states are `verified`, `claims_failed`, `invalid_draft`, `no_claims`), so
  `vercel logs` shows how often it happens.
- The eval needs no change: a partial answer is `answered` and must still
  meet the case's `must_include` and citation checks.

## Cache

`cache build` stores every answered result for a curated question without
reviewed claims (`cache_build.build_entries`). A partial answer is known to
be incomplete and would be served unchanged on every cache hit, so it is not
stored: the row is reported as not cached, like a `not_found`. Reviewed
drafts (`_run_reviewed`) neither retry nor drop, so they never produce one.
All 20 curated questions have reviewed claims today; this guard is for the
next unreviewed one.

## Tests

- Final attempt, one essential claim failed and one verified: answered, with
  `PARTIAL_NOTE` and without `OMITTED_NOTE` or the failed claim's text.
- The same draft on attempt 1: still regenerated, with the failed claim
  listed in the retry prompt.
- Final attempt with no verified essential claim (only a non-essential one):
  still `not_found` (`unverified`).
- Streaming: no `reset` is emitted for the kept final draft.
- The `partial` tag is set, and `claims-dropped` isn't.
- `cache build` doesn't store a partial answer.

## Docs

docs/qa-core.md: the drop rule (§"Drop rule" and pipeline step 5c), the
trace tags, and the attempt log's terminal states.

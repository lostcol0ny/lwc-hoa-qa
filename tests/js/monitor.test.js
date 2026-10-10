// Unit tests for the pure helpers in public/monitor.js.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { stagesRun, timingSegments, feedSummary } = require("../../public/monitor.js");

const base = { gate: { score: 0.9, passed: true }, cache: null, sweep: null, attempts: [] };

test("a full answer runs every stage except the cache", () => {
  const trace = { ...base, sweep: {}, attempts: [{ claims: [{}] }] };
  assert.deepEqual(stagesRun(trace), ["gate", "sweep", "draft", "verify", "answer"]);
});

test("a cache hit skips the sweep and the answer model", () => {
  assert.deepEqual(stagesRun({ ...base, cache: { hit: "x" } }), ["gate", "cache", "answer"]);
});

test("an exact curated question skips the gate too", () => {
  assert.deepEqual(stagesRun({ ...base, gate: null, cache: { exact: true } }), ["cache", "answer"]);
});

test("a draft with no claims has nothing to verify", () => {
  const trace = { ...base, sweep: {}, attempts: [{ claims: [] }] };
  assert.deepEqual(stagesRun(trace), ["gate", "sweep", "draft", "answer"]);
});

test("gate and cache overlap, so they share one timing segment", () => {
  const parts = timingSegments({ gate: 100, cache: 300, sweep: 1000, answer: 4000, verify: 100 }, 6000);
  assert.deepEqual(parts.map((p) => p.ms), [300, 1000, 4000, 100, 600]);
  assert.ok(Math.abs(parts.reduce((sum, p) => sum + p.share, 0) - 1) < 1e-9);
});

test("timing never divides by zero or goes negative", () => {
  for (const parts of [timingSegments({}, 0), timingSegments(undefined, undefined), timingSegments({ sweep: 500 }, 100)]) {
    for (const part of parts) assert.ok(part.share >= 0 && part.share <= 1 && part.ms >= 0);
  }
});

test("the feed summary reads as a sentence", () => {
  assert.equal(feedSummary({ requests: 0 }), "No requests recorded yet.");
  const text = feedSummary({ requests: 4, answered: 3, cache_hits: 1, median_latency_ms: 2500, cost_usd: 0.0412, keep: 200, days: 7 });
  assert.match(text, /4 requests/);
  assert.match(text, /75% answered, 25% from the cache, median 2\.5 s, \$0\.0412/);
});

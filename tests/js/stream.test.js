// Unit tests for reading the answer stream in public/app.js.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { parseEvents, stageText } = require("../../public/app.js");

const frame = (event) => "data: " + JSON.stringify(event) + "\n\n";

test("whole events are returned and the unfinished tail is kept", () => {
  const first = frame({ type: "stage", stage: "gate" });
  const second = frame({ type: "claim", statement: "A fine is $125." });
  const cut = second.slice(0, 20);
  const parsed = parseEvents(first + cut);
  assert.deepEqual(parsed.events, [{ type: "stage", stage: "gate" }]);
  assert.equal(parsed.rest, cut);
  const next = parseEvents(parsed.rest + second.slice(20));
  assert.deepEqual(next.events, [{ type: "claim", statement: "A fine is $125." }]);
  assert.equal(next.rest, "");
});

test("text with newlines and quotes survives, since JSON escapes them", () => {
  const statement = 'Line one.\n\n"Quoted" data: not an event';
  const parsed = parseEvents(frame({ type: "claim", statement }));
  assert.equal(parsed.events[0].statement, statement);
});

test("malformed or untyped events are skipped", () => {
  const parsed = parseEvents("data: {not json}\n\ndata: [1,2]\n\n: comment\n\n" + frame({ type: "reset" }));
  assert.deepEqual(parsed.events, [{ type: "reset" }]);
});

test("an empty or partial buffer yields nothing", () => {
  assert.deepEqual(parseEvents(""), { events: [], rest: "" });
  assert.deepEqual(parseEvents("data: {"), { events: [], rest: "data: {" });
});

test("stage lines use the real counts when they are given", () => {
  assert.equal(stageText({ stage: "gate" }), "Checking your question…");
  assert.equal(stageText({ stage: "sweep", passages: 668 }), "Reading 668 passages…");
  assert.equal(stageText({ stage: "draft", selected: 8 }), "Writing an answer from 8 passages…");
  assert.equal(stageText({ stage: "draft", selected: 1 }), "Writing an answer from 1 passage…");
});

test("stage lines fall back when a count is missing or odd", () => {
  assert.equal(stageText({ stage: "sweep" }), "Reading the HOA documents and Illinois law…");
  assert.equal(stageText({ stage: "draft", selected: "<b>" }), "Writing an answer from what it found…");
  assert.equal(stageText({ stage: "something-new" }), null);
});

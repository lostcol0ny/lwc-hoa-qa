// Unit tests for when public/app.js empties the question box after an answer.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { shouldClearQuestion } = require("../../public/app.js");

test("an answered or off-topic question is cleared", () => {
  assert.equal(shouldClearQuestion("answered"), true);
  assert.equal(shouldClearQuestion("refused_off_topic"), true);
});

test("a question worth rephrasing or resending is kept", () => {
  for (const outcome of ["not_found", "error", "budget_exhausted", "invalid_input"]) {
    assert.equal(shouldClearQuestion(outcome), false, outcome);
  }
});

test("an unknown outcome keeps the text", () => {
  assert.equal(shouldClearQuestion("something_new"), false);
  assert.equal(shouldClearQuestion(undefined), false);
});

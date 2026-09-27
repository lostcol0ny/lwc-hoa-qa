// Unit tests for the composer's Enter handling in public/app.js.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { keyAction } = require("../../public/app.js");

const idle = { loading: false, composing: false };
const enter = (extra = {}) => ({ key: "Enter", keyCode: 13, isComposing: false, ...extra });

test("plain Enter submits", () => {
  assert.equal(keyAction(enter(), idle), "submit");
});

test("Ctrl+Enter and Cmd+Enter submit", () => {
  assert.equal(keyAction(enter({ ctrlKey: true }), idle), "submit");
  assert.equal(keyAction(enter({ metaKey: true }), idle), "submit");
});

test("Shift+Enter is a newline", () => {
  assert.equal(keyAction(enter({ shiftKey: true }), idle), "newline");
});

test("Enter during IME composition (isComposing) is ignored", () => {
  assert.equal(keyAction(enter({ isComposing: true }), idle), "ignore");
});

test("Enter with keyCode 229 is ignored even when isComposing is false", () => {
  assert.equal(keyAction(enter({ keyCode: 229 }), idle), "ignore");
});

test("Enter while our composition flag is set is ignored", () => {
  assert.equal(keyAction(enter(), { loading: false, composing: true }), "ignore");
});

test("Enter while loading does not submit", () => {
  const action = keyAction(enter(), { loading: true, composing: false });
  assert.notEqual(action, "submit");
  assert.equal(action, "block");
  assert.equal(keyAction(enter({ ctrlKey: true }), { loading: true }), "block");
});

test("other keys are left alone", () => {
  assert.equal(keyAction({ key: "a", keyCode: 65 }, idle), "ignore");
  assert.equal(keyAction(null, idle), "ignore");
  assert.equal(keyAction(enter(), undefined), "submit");
});

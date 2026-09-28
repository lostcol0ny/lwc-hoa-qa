// Unit tests for the not_found reason rendering in public/app.js (run: node --test tests/js/*.test.js).
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { relatedDocuments, outcomeHeading } = require("../../public/app.js");

test("unverified not_found gets its own heading; other outcomes are unchanged", () => {
  assert.equal(
    outcomeHeading({ outcome: "not_found", reason: "unverified" }).title,
    "Couldn't verify an answer",
  );
  for (const reason of [undefined, null, "no_relevant_passages", "no_answer_in_passages"]) {
    assert.equal(
      outcomeHeading({ outcome: "not_found", reason }).title,
      "Not found in the HOA documents",
    );
  }
  // A reason never changes another outcome's heading.
  assert.equal(outcomeHeading({ outcome: "answered", reason: "unverified" }).title, "Answer");
  assert.equal(outcomeHeading({ outcome: "bogus" }).title, "Something went wrong");
  assert.equal(outcomeHeading(null).title, "Something went wrong");
});

test("related documents keep only HTTPS links, once each, at most five", () => {
  const docs = [
    { title: "Bylaws", url: "https://example.org/bylaws.pdf" },
    { title: "Dup", url: "https://example.org/bylaws.pdf" },
    { title: "Bad", url: "javascript:alert(1)" },
    { title: "Plain", url: "http://example.org/x" },
    null,
    "junk",
    { title: "   ", url: "https://example.org/untitled" },
    ...[1, 2, 3, 4, 5].map((i) => ({ title: "Doc " + i, url: "https://example.org/" + i })),
  ];
  assert.deepEqual(relatedDocuments({ related_documents: docs }), [
    { title: "Bylaws", url: "https://example.org/bylaws.pdf" },
    { title: "HOA document", url: "https://example.org/untitled" },
    { title: "Doc 1", url: "https://example.org/1" },
    { title: "Doc 2", url: "https://example.org/2" },
    { title: "Doc 3", url: "https://example.org/3" },
  ]);
});

test("missing or malformed related_documents renders nothing", () => {
  assert.deepEqual(relatedDocuments({}), []);
  assert.deepEqual(relatedDocuments({ related_documents: "https://x.org" }), []);
  assert.deepEqual(relatedDocuments(null), []);
});

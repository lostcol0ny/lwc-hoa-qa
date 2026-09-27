// Unit tests for the citation grouping in public/app.js (run: node --test tests/js/*.test.js).
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { groupCitations, normalizeQuote } = require("../../public/app.js");

const cite = (chunk_id, quote, extra = {}) => ({
  chunk_id,
  citation_label: "Label " + chunk_id,
  url: "https://example.org/" + chunk_id,
  quote,
  ...extra,
});

test("groups by chunk_id in first-appearance order", () => {
  const groups = groupCitations([cite("b", "one"), cite("a", "two"), cite("b", "three")]);
  assert.deepEqual(
    groups.map((g) => [g.label, g.quotes]),
    [
      ["Label b", ["one", "three"]],
      ["Label a", ["two"]],
    ],
  );
});

test("de-duplicates quotes after whitespace normalization", () => {
  const groups = groupCitations([
    cite("a", "Pool  hours are\n 6am to 10pm."),
    cite("a", " Pool hours are 6am\tto 10pm. "),
    cite("a", ""),
  ]);
  assert.equal(groups.length, 1);
  assert.deepEqual(groups[0].quotes, ["Pool hours are 6am to 10pm."]);
});

test("falls back to url + label when chunk_id is missing", () => {
  const url = "https://example.org/doc";
  const groups = groupCitations([
    { citation_label: "Rules 3.B", url, quote: "x" },
    { citation_label: "Rules 3.B", url, quote: "y" },
    { citation_label: "Rules 4.A", url, quote: "z" },
    { chunk_id: "  ", citation_label: "Rules 3.B", url, quote: "x" },
  ]);
  assert.deepEqual(
    groups.map((g) => [g.label, g.quotes]),
    [
      ["Rules 3.B", ["x", "y"]],
      ["Rules 4.A", ["z"]],
    ],
  );
});

test("keeps only https links", () => {
  const groups = groupCitations([
    cite("a", "q", { url: "javascript:alert(1)" }),
    cite("b", "q", { url: "http://example.org" }),
    cite("c", "q"),
  ]);
  assert.deepEqual(
    groups.map((g) => g.url),
    [null, null, "https://example.org/c"],
  );
});

test("tolerates bad input", () => {
  assert.deepEqual(groupCitations(undefined), []);
  assert.deepEqual(groupCitations([null, 3, "x"]), []);
  const [group] = groupCitations([{ chunk_id: "a" }]);
  assert.equal(group.label, "Source");
  assert.deepEqual(group.quotes, []);
  assert.equal(normalizeQuote(null), "");
});

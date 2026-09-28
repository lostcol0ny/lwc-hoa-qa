"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const source = (name) => fs.readFileSync(path.join(__dirname, "../../public", name), "utf8");

test("queue initializer preserves an existing tracker and queues arguments", () => {
  const window = {};
  vm.runInNewContext(source("analytics.js"), { window });
  const tracker = window.va;
  window.va("test-command", 1);
  assert.deepEqual(Array.from(window.vaq[0]), ["test-command", 1]);
  vm.runInNewContext(source("analytics.js"), { window });
  assert.equal(window.va, tracker);
});

test("submitting works without insights and keeps question and answer out of URLs", async () => {
  const nodes = new Map();
  function node() {
    return {
      value: "", children: [], listeners: {}, disabled: false,
      classList: { add() {}, toggle() {}, contains() { return false; } },
      addEventListener(type, callback) { this.listeners[type] = callback; },
      setAttribute() {}, append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      querySelectorAll() { return []; }, focus() {},
    };
  }
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, node());
      return nodes.get(id);
    },
    createElement: node, addEventListener() {}, hidden: false,
  };
  const question = "Private question & name? #secret";
  const answer = "Private answer";
  const requests = [];
  const location = new URL("https://hoa.example/");
  const initialUrl = location.href;
  const history = {
    pushState() { assert.fail("must not change history"); },
    replaceState() { assert.fail("must not change history"); },
  };
  const window = { location, history, setTimeout() { return 1; }, clearTimeout() {} };
  const context = {
    document, window, location, history, performance: { now: () => 0 },
    fetch: async (url, options) => {
      requests.push({ url, options });
      return { json: async () => ({ outcome: "answered", answer_text: answer }) };
    },
  };
  // Deliberately never load the Vercel script, as with a local 404.
  vm.runInNewContext(source("app.js"), context);
  vm.runInNewContext(source("analytics.js"), context);
  nodes.get("question").value = question;
  let prevented = false;
  nodes.get("ask-form").listeners.submit({ preventDefault() { prevented = true; } });
  await new Promise(setImmediate);
  assert.ok(prevented);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/api/ask");
  assert.equal(requests[0].options.method, "POST");
  assert.deepEqual(JSON.parse(requests[0].options.body), { question });
  assert.equal(location.href, initialUrl);
  assert.equal(location.search, "");
  assert.equal(location.hash, "");
  assert.equal(window.vaq, undefined, "no custom events");
  assert.equal(nodes.get("ask-button").disabled, false);
  assert.ok(JSON.stringify(nodes.get("answer").children).includes(answer));
});

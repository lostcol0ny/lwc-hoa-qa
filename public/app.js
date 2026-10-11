// Lakewood Creek HOA Q&A page. Renders response data with textContent and
// createElement only; nothing from the server is ever parsed as HTML.
"use strict";

function isHttpsUrl(value) {
  return typeof value === "string" && value.startsWith("https://");
}

function normalizeQuote(quote) {
  return String(quote == null ? "" : quote)
    .replace(/\s+/g, " ")
    .trim();
}

// One group per source chunk, keyed by chunk_id (or url + label when a
// citation has none), in order of first appearance. Each group lists its
// unique quotes once, compared after whitespace normalization.
function groupCitations(citations) {
  const groups = new Map();
  if (!Array.isArray(citations)) return [];
  for (const citation of citations) {
    if (!citation || typeof citation !== "object") continue;
    const label = String(citation.citation_label || "Source");
    const url = isHttpsUrl(citation.url) ? citation.url : null;
    const chunkId = typeof citation.chunk_id === "string" ? citation.chunk_id.trim() : "";
    const key = chunkId ? "chunk:" + chunkId : "src:" + (url || "") + "\u0000" + label;
    let group = groups.get(key);
    if (!group) {
      group = { key: key, label: label, url: url, quotes: [] };
      groups.set(key, group);
    }
    if (!group.url && url) group.url = url;
    const quote = normalizeQuote(citation.quote);
    if (quote && !group.quotes.includes(quote)) group.quotes.push(quote);
  }
  return Array.from(groups.values());
}

// Document links for a not_found the server couldn't verify (reason
// "unverified"): whole documents only, HTTPS only, one per URL, at most 5.
// Anything else in the payload is ignored.
const MAX_RELATED_DOCUMENTS = 5;
function relatedDocuments(answer) {
  if (!answer || !Array.isArray(answer.related_documents)) return [];
  const seen = new Set();
  const links = [];
  for (const doc of answer.related_documents) {
    if (!doc || typeof doc !== "object" || !isHttpsUrl(doc.url) || seen.has(doc.url)) continue;
    const title = String(doc.title == null ? "" : doc.title).trim() || "HOA document";
    seen.add(doc.url);
    links.push({ title: title, url: doc.url });
    if (links.length === MAX_RELATED_DOCUMENTS) break;
  }
  return links;
}

// Heading for an answer card: a not_found with reason "unverified" found
// related passages but no verifiable answer, which is not "not found".
const OUTCOME_HEADINGS = Object.freeze({
  answered: { status: "Answered", title: "Answer" },
  not_found: { status: "Not found", title: "Not found in the HOA documents" },
  refused_off_topic: { status: "Off topic", title: "That question isn't about the HOA" },
  budget_exhausted: { status: "Paused", title: "Taking a break for the rest of the month" },
  invalid_input: { status: "Check input", title: "Please check your question" },
  error: { status: "Error", title: "Something went wrong" },
});
const UNVERIFIED_HEADING = Object.freeze({
  status: "Not verified",
  title: "Couldn't verify an answer",
});
function outcomeHeading(answer) {
  const outcome =
    answer && Object.hasOwn(OUTCOME_HEADINGS, answer.outcome) ? answer.outcome : "error";
  if (outcome === "not_found" && answer.reason === "unverified") return UNVERIFIED_HEADING;
  return OUTCOME_HEADINGS[outcome];
}

// Whether to empty the question box once this outcome is shown. The answer
// card repeats the question ("You asked: …"), so nothing is lost. The text is
// kept when the person is likely to want it again: to rephrase a question the
// documents didn't answer, or to resend after an error.
function shouldClearQuestion(outcome) {
  return outcome === "answered" || outcome === "refused_off_topic";
}

// Split what has arrived of a server-sent event stream into whole events
// and the unfinished remainder. Each event is one "data: {json}" line
// followed by a blank line; anything that isn't valid JSON is skipped.
function parseEvents(buffer) {
  const events = [];
  const parts = String(buffer).split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    for (const line of part.split("\n")) {
      if (!line.startsWith("data:")) continue;
      try {
        const event = JSON.parse(line.slice(5));
        if (event && typeof event === "object" && typeof event.type === "string") {
          events.push(event);
        }
      } catch (_) {
        // A malformed event is ignored; the final answer event is what counts.
      }
    }
  }
  return { events: events, rest: rest };
}

// The progress line for a real pipeline stage, or null for one with no line.
function stageText(event) {
  const count = function (value, noun) {
    return Number.isInteger(value) && value > 0 ? value + " " + noun : null;
  };
  if (event.stage === "gate") return "Checking your question…";
  if (event.stage === "sweep") {
    const passages = count(event.passages, "passages");
    return "Reading " + (passages || "the HOA documents and Illinois law") + "…";
  }
  if (event.stage === "draft") {
    const selected = count(event.selected, event.selected === 1 ? "passage" : "passages");
    return "Writing an answer from " + (selected || "what it found") + "…";
  }
  return null;
}

// What an Enter keydown in the question box should do:
// "submit", "newline" (let the browser insert it), "block" (swallow it) or
// "ignore" (not ours; leave the event alone). IME composition always wins:
// some browsers report isComposing === false with keyCode 229 mid-composition,
// so both are checked, plus our own compositionstart/compositionend flag.
function keyAction(event, state) {
  if (!event || event.key !== "Enter") return "ignore";
  const composing = Boolean(state && state.composing);
  if (event.isComposing || event.keyCode === 229 || composing) return "ignore";
  if (event.shiftKey) return "newline";
  if (state && state.loading) return "block";
  return "submit"; // plain Enter, and Ctrl/Cmd+Enter
}

// Status lines shown while an answer is being prepared, keyed by elapsed time.
// They describe what the service is doing in general and never claim a step
// has finished; the server doesn't report progress, so this is a client-side
// schedule only. Only lines marked `announce` reach screen readers.
const PROGRESS_STAGES = Object.freeze([
  { at: 0, text: "Checking your question…", announce: true },
  { at: 1500, text: "Searching the HOA documents and Illinois law…", announce: false },
  { at: 6000, text: "Writing an answer with citations…", announce: false },
  { at: 10000, text: "Double-checking the citations…", announce: false },
  {
    at: 18000,
    text: "Still working. Some questions take up to half a minute.",
    announce: true,
  },
]);

// Index into PROGRESS_STAGES for a request that has been in flight for
// elapsedMs. Bad or negative input means "just started".
function progressStage(elapsedMs) {
  let index = 0;
  if (!(elapsedMs > 0)) return index;
  while (index + 1 < PROGRESS_STAGES.length && elapsedMs >= PROGRESS_STAGES[index + 1].at) {
    index += 1;
  }
  return index;
}

// Drives PROGRESS_STAGES for one request at a time. It holds at most one
// pending timer, armed for the next stage boundary, and stop() cancels it, so
// nothing outlives a finished request. The clock and timer functions are
// injectable for tests.
function createProgress(options) {
  const now = options.now;
  const setTimer = options.setTimer;
  const clearTimer = options.clearTimer;
  const onStage = options.onStage;
  const onStop = options.onStop || function () {};
  let timer = null;
  let running = false;
  let startedAt = 0;
  let current = -1;

  function tick() {
    timer = null;
    if (!running) return;
    const elapsed = now() - startedAt;
    const index = progressStage(elapsed);
    if (index !== current) {
      current = index;
      onStage(PROGRESS_STAGES[index], index);
    }
    const next = PROGRESS_STAGES[index + 1];
    if (next) timer = setTimer(tick, Math.max(0, next.at - elapsed));
  }

  function stop() {
    if (timer !== null) clearTimer(timer);
    timer = null;
    if (!running) return;
    running = false;
    current = -1;
    onStop();
  }

  function start() {
    stop();
    running = true;
    startedAt = now();
    tick();
  }

  return {
    start: start,
    stop: stop,
    isRunning: function () {
      return running;
    },
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    groupCitations,
    normalizeQuote,
    relatedDocuments,
    outcomeHeading,
    isHttpsUrl,
    keyAction,
    shouldClearQuestion,
    parseEvents,
    stageText,
    PROGRESS_STAGES,
    progressStage,
    createProgress,
  };
}

if (typeof document !== "undefined") {
  (function () {
    const MAX_CHARS = 500;
    // Same-origin route that 307-redirects to the configured HOA documents.
    const DOCUMENTS_PATH = "/documents";
    const form = document.getElementById("ask-form");
    const composer = document.getElementById("composer");
    const textarea = document.getElementById("question");
    const counter = document.getElementById("char-count");
    const button = document.getElementById("ask-button");
    const buttonLabel = document.getElementById("ask-button-label");
    const answerRegion = document.getElementById("answer");
    const progressStatus = document.getElementById("progress-status");
    let composing = false;

    const SHOW_DOCUMENTS_LINK = new Set(["not_found", "refused_off_topic", "budget_exhausted"]);

    // The beam only animates where the angle property can be registered;
    // elsewhere it stays a static highlight (see styles.css).
    if (window.CSS && typeof CSS.registerProperty === "function") {
      document.documentElement.classList.add("beam-animated");
    }

    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function externalLink(url, text) {
      const link = el("a", null, text);
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      return link;
    }

    function updateCounter() {
      const length = textarea.value.length;
      counter.textContent = length + " / " + MAX_CHARS;
      counter.classList.toggle("near-limit", length >= MAX_CHARS - 50);
    }

    // The rotating lines are visual only (aria-hidden). Screen readers hear
    // the separate #progress-status region, which carries just the announced
    // stages: the first line and, for slow answers, the "still working" one.
    let progressLines = null;

    function renderLoading() {
      const loading = el("div", "loading");
      loading.setAttribute("aria-hidden", "true");
      const dots = el("span", "dots");
      dots.append(el("span"), el("span"), el("span"));
      progressLines = el("span", "progress-lines");
      progressLines.addEventListener("animationend", function (event) {
        if (event.target.classList.contains("is-leaving")) event.target.remove();
      });
      loading.append(dots, progressLines);
      answerRegion.replaceChildren(loading);
    }

    function showProgressLine(text) {
      if (!progressLines) return;
      // Drop lines still fading out from an earlier change, then fade the
      // current line out underneath the new one (see .progress-line).
      for (const old of progressLines.querySelectorAll(".is-leaving")) old.remove();
      for (const line of progressLines.children) line.classList.add("is-leaving");
      const line = el("span", "progress-line", text);
      progressLines.append(line);
    }

    const progress = createProgress({
      now: function () {
        return performance.now();
      },
      setTimer: function (fn, ms) {
        return window.setTimeout(fn, ms);
      },
      clearTimer: function (id) {
        window.clearTimeout(id);
      },
      onStage: function (stage) {
        showProgressLine(stage.text);
        if (stage.announce) progressStatus.textContent = stage.text;
      },
      onStop: function () {
        progressLines = null;
        progressStatus.textContent = "";
      },
    });

    function setLoading(loading) {
      button.disabled = loading;
      buttonLabel.textContent = loading ? "Asking…" : "Ask";
      composer.classList.toggle("is-busy", loading);
      form.setAttribute("aria-busy", loading ? "true" : "false");
      answerRegion.setAttribute("aria-busy", loading ? "true" : "false");
      if (loading) {
        renderLoading();
        progress.start();
      } else {
        progress.stop();
      }
    }

    function renderSourceGroup(group) {
      const card = el("li", "source");
      const heading = el("p", "source-label");
      if (group.url) {
        heading.append(externalLink(group.url, group.label));
      } else {
        heading.textContent = group.label;
      }
      card.append(heading);
      for (const quote of group.quotes) card.append(el("blockquote", null, quote));
      return card;
    }

    function renderAnswer(answer, question) {
      const outcome = Object.hasOwn(OUTCOME_HEADINGS, answer.outcome) ? answer.outcome : "error";
      const heading = outcomeHeading(answer);
      const card = el("article", "result outcome-" + outcome);

      const head = el("div", "result-head");
      head.append(el("span", "status", heading.status));
      head.append(el("h2", null, heading.title));
      card.append(head);

      if (question) card.append(el("p", "asked", "You asked: " + question));
      card.append(el("p", "answer-text", String(answer.answer_text || "")));

      if (Array.isArray(answer.conflicts_noted) && answer.conflicts_noted.length > 0) {
        const conflicts = el("aside", "conflicts");
        conflicts.append(el("h3", null, "Conflicting or outdated sources"));
        const list = el("ul");
        for (const note of answer.conflicts_noted) list.append(el("li", null, String(note)));
        conflicts.append(list);
        card.append(conflicts);
      }

      const groups = groupCitations(answer.citations);
      if (groups.length > 0) {
        card.append(el("h3", "sources-title", "Sources"));
        const list = el("ul", "sources");
        for (const group of groups) list.append(renderSourceGroup(group));
        card.append(list);
      }

      // Only for a not_found: links to whole documents, never passage text.
      const related = outcome === "not_found" ? relatedDocuments(answer) : [];
      if (related.length > 0) {
        card.append(el("h3", "sources-title", "Related documents"));
        const list = el("ul", "sources related");
        for (const doc of related) {
          const item = el("li", "source");
          const label = el("p", "source-label");
          label.append(externalLink(doc.url, doc.title));
          item.append(label);
          list.append(item);
        }
        card.append(list);
      }

      if (SHOW_DOCUMENTS_LINK.has(outcome)) {
        const more = el("p", "docs-hint");
        more.append("You can also ");
        more.append(externalLink(DOCUMENTS_PATH, "read the HOA documents directly"));
        more.append(".");
        card.append(more);
      }

      if (answer.request_id) {
        card.append(el("p", "request-id", "Reference: " + String(answer.request_id)));
      }
      answerRegion.replaceChildren(card);
    }

    function renderFailure(question) {
      renderAnswer(
        {
          outcome: "error",
          answer_text: "The service couldn't be reached. Please try again in a moment.",
          citations: [],
          conflicts_noted: [],
        },
        question,
      );
    }

    function showAnswer(body, question) {
      setLoading(false);
      if (body && typeof body === "object" && typeof body.outcome === "string") {
        renderAnswer(body, question);
        // Not if they have already started typing the next question.
        if (shouldClearQuestion(body.outcome) && textarea.value.trim() === question) {
          textarea.value = "";
          updateCounter();
        }
      } else {
        renderFailure(question);
      }
    }

    // Read the event stream: real stage names replace the timed progress
    // lines, and each verified sentence of the answer appears as it arrives.
    // Returns the final answer body, or null if the stream ended without one.
    async function readStream(response, question) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let preview = null;
      let live = false;
      for (;;) {
        const chunk = await reader.read();
        buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
        const parsed = parseEvents(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) {
          if (event.type === "answer") return event.answer;
          if (!live) {
            // From here the server says what is happening; stop guessing.
            live = true;
            progress.stop();
            renderLoading();
          }
          if (event.type === "stage") {
            const text = stageText(event);
            if (text) showProgressLine(text);
          } else if (event.type === "claim" && typeof event.statement === "string") {
            if (!preview) {
              preview = el("article", "result preview");
              preview.append(el("p", "asked", "You asked: " + question));
              preview.append(el("p", "answer-text"));
              answerRegion.append(preview);
              showProgressLine("Checking each statement against the documents…");
            }
            const text = preview.querySelector(".answer-text");
            text.textContent = (text.textContent ? text.textContent + " " : "") + event.statement;
          } else if (event.type === "reset" && preview) {
            preview.remove();
            preview = null;
            showProgressLine("Rewriting the answer…");
          }
        }
        if (chunk.done) return null;
      }
    }

    async function ask(question) {
      setLoading(true);
      try {
        const response = await fetch("/api/ask/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question }),
        });
        const type = (response.headers && response.headers.get("content-type")) || "";
        let body = null;
        if (type.startsWith("text/event-stream") && response.body) {
          body = await readStream(response, question);
        } else {
          // A refusal (rate limit, budget, bad input) comes back as plain JSON.
          try {
            body = await response.json();
          } catch (_) {
            body = null;
          }
        }
        showAnswer(body, question);
      } catch (_) {
        setLoading(false);
        renderFailure(question);
      }
    }

    // Pause the beam while the tab is hidden or the composer is offscreen;
    // the static ring stays either way.
    let composerVisible = true;
    function updateBeamPause() {
      composer.classList.toggle("beam-paused", document.hidden || !composerVisible);
    }
    document.addEventListener("visibilitychange", updateBeamPause);
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver(function (entries) {
        composerVisible = entries[entries.length - 1].isIntersecting;
        updateBeamPause();
      }).observe(composer);
    }

    textarea.addEventListener("input", updateCounter);
    textarea.addEventListener("compositionstart", function () {
      composing = true;
    });
    textarea.addEventListener("compositionend", function () {
      composing = false;
    });
    textarea.addEventListener("keydown", function (event) {
      const action = keyAction(event, { loading: button.disabled, composing: composing });
      if (action === "ignore" || action === "newline") return;
      event.preventDefault();
      if (action === "submit") form.requestSubmit();
    });
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      if (button.disabled) return;
      const question = textarea.value.trim();
      if (!question || question.length > MAX_CHARS) {
        renderAnswer({
          outcome: "invalid_input",
          answer_text: "Please enter a question of 1 to " + MAX_CHARS + " characters.",
          citations: [],
          conflicts_noted: [],
        });
        textarea.focus();
        return;
      }
      ask(question);
    });

    updateCounter();
    updateBeamPause();
  })();
}

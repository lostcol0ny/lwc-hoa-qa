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

if (typeof module !== "undefined" && module.exports) {
  module.exports = { groupCitations, normalizeQuote, isHttpsUrl };
}

if (typeof document !== "undefined") {
  (function () {
    const MAX_CHARS = 500;
    const DEFAULT_DOCUMENTS_URL = "https://lakewoodcreekhoa.com/";
    const form = document.getElementById("ask-form");
    const composer = document.getElementById("composer");
    const textarea = document.getElementById("question");
    const counter = document.getElementById("char-count");
    const button = document.getElementById("ask-button");
    const buttonLabel = document.getElementById("ask-button-label");
    const answerRegion = document.getElementById("answer");
    const documentsLink = document.getElementById("documents-link");
    let documentsUrl = DEFAULT_DOCUMENTS_URL;

    const OUTCOMES = {
      answered: { status: "Answered", title: "Answer" },
      not_found: { status: "Not found", title: "Not found in the HOA documents" },
      refused_off_topic: { status: "Off topic", title: "That question isn't about the HOA" },
      budget_exhausted: { status: "Paused", title: "Taking a break for the rest of the month" },
      invalid_input: { status: "Check input", title: "Please check your question" },
      error: { status: "Error", title: "Something went wrong" },
    };
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

    function setDocumentsUrl(url) {
      if (!isHttpsUrl(url)) return;
      documentsUrl = url;
      documentsLink.href = url;
    }

    function updateCounter() {
      const length = textarea.value.length;
      counter.textContent = length + " / " + MAX_CHARS;
      counter.classList.toggle("near-limit", length >= MAX_CHARS - 50);
    }

    function renderLoading() {
      const loading = el("div", "loading");
      const dots = el("span", "dots");
      dots.setAttribute("aria-hidden", "true");
      dots.append(el("span"), el("span"), el("span"));
      loading.append(dots, el("span", null, "Searching the HOA documents…"));
      answerRegion.replaceChildren(loading);
    }

    function setLoading(loading) {
      button.disabled = loading;
      buttonLabel.textContent = loading ? "Asking…" : "Ask";
      composer.classList.toggle("is-busy", loading);
      form.setAttribute("aria-busy", loading ? "true" : "false");
      answerRegion.setAttribute("aria-busy", loading ? "true" : "false");
      if (loading) renderLoading();
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
      const outcome = Object.hasOwn(OUTCOMES, answer.outcome) ? answer.outcome : "error";
      const card = el("article", "result outcome-" + outcome);

      const head = el("div", "result-head");
      head.append(el("span", "status", OUTCOMES[outcome].status));
      head.append(el("h2", null, OUTCOMES[outcome].title));
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

      if (SHOW_DOCUMENTS_LINK.has(outcome) && isHttpsUrl(documentsUrl)) {
        const more = el("p", "docs-hint");
        more.append("You can also ");
        more.append(externalLink(documentsUrl, "read the HOA documents directly"));
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

    async function ask(question) {
      setLoading(true);
      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question }),
        });
        let body = null;
        try {
          body = await response.json();
        } catch (_) {
          body = null;
        }
        setLoading(false);
        if (body && typeof body === "object" && typeof body.outcome === "string") {
          renderAnswer(body, question);
        } else {
          renderFailure(question);
        }
      } catch (_) {
        setLoading(false);
        renderFailure(question);
      }
    }

    async function loadDocumentsUrl() {
      try {
        const response = await fetch("/api/health");
        const body = await response.json();
        setDocumentsUrl(body.documents_url);
      } catch (_) {
        // Keep the default link.
      }
    }

    textarea.addEventListener("input", updateCounter);
    textarea.addEventListener("keydown", function (event) {
      // Enter submits; Shift+Enter adds a newline. Skip while an IME is composing.
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!button.disabled) form.requestSubmit();
      }
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

    setDocumentsUrl(DEFAULT_DOCUMENTS_URL);
    updateCounter();
    loadDocumentsUrl();
  })();
}

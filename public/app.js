// Lakewood Creek HOA Q&A page. Renders response data with textContent and
// createElement only; nothing from the server is ever parsed as HTML.
"use strict";

(function () {
  const MAX_CHARS = 500;
  const form = document.getElementById("ask-form");
  const textarea = document.getElementById("question");
  const counter = document.getElementById("char-count");
  const button = document.getElementById("ask-button");
  const answerRegion = document.getElementById("answer");
  const documentsLink = document.getElementById("documents-link");

  const OUTCOME_TITLES = {
    answered: "Answer",
    not_found: "Not found in the HOA documents",
    refused_off_topic: "That question isn't about the HOA",
    budget_exhausted: "Taking a break for the rest of the month",
    invalid_input: "Please check your question",
    error: "Something went wrong",
  };

  function isHttpsUrl(value) {
    return typeof value === "string" && value.startsWith("https://");
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
    counter.textContent = textarea.value.length + " / " + MAX_CHARS;
  }

  function setLoading(loading) {
    button.disabled = loading;
    button.textContent = loading ? "Asking…" : "Ask";
    answerRegion.setAttribute("aria-busy", loading ? "true" : "false");
    if (loading) {
      answerRegion.replaceChildren(el("p", "loading", "Looking through the HOA documents…"));
    }
  }

  function renderCitation(citation) {
    const card = el("li", "citation");
    const label = String(citation.citation_label || "Source");
    if (isHttpsUrl(citation.url)) {
      card.append(externalLink(citation.url, label));
    } else {
      card.append(el("span", "citation-label", label));
    }
    if (citation.quote) {
      card.append(el("blockquote", null, String(citation.quote)));
    }
    return card;
  }

  function renderAnswer(answer) {
    const outcome = Object.hasOwn(OUTCOME_TITLES, answer.outcome) ? answer.outcome : "error";
    const card = el("article", "result outcome-" + outcome);
    card.append(el("h2", null, OUTCOME_TITLES[outcome]));
    card.append(el("p", "answer-text", String(answer.answer_text || "")));

    if (Array.isArray(answer.conflicts_noted) && answer.conflicts_noted.length > 0) {
      const conflicts = el("div", "conflicts");
      conflicts.append(el("h3", null, "Conflicting or outdated sources"));
      const list = el("ul");
      for (const note of answer.conflicts_noted) list.append(el("li", null, String(note)));
      conflicts.append(list);
      card.append(conflicts);
    }

    if (Array.isArray(answer.citations) && answer.citations.length > 0) {
      card.append(el("h3", null, "Sources"));
      const list = el("ul", "citations");
      for (const citation of answer.citations) list.append(renderCitation(citation));
      card.append(list);
    }

    if (outcome !== "answered" && isHttpsUrl(documentsLink.href)) {
      const more = el("p", "docs-hint");
      more.append("You can also ");
      more.append(externalLink(documentsLink.href, "read the HOA documents directly"));
      more.append(".");
      card.append(more);
    }

    if (answer.request_id) {
      card.append(el("p", "request-id", "Reference: " + String(answer.request_id)));
    }
    answerRegion.replaceChildren(card);
  }

  function renderFailure() {
    renderAnswer({
      outcome: "error",
      answer_text: "The service couldn't be reached. Please try again in a moment.",
      citations: [],
      conflicts_noted: [],
    });
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
      if (body && typeof body === "object" && typeof body.outcome === "string") {
        renderAnswer(body);
      } else {
        renderFailure();
      }
    } catch (_) {
      renderFailure();
    } finally {
      setLoading(false);
    }
  }

  async function loadDocumentsUrl() {
    try {
      const response = await fetch("/api/health");
      const body = await response.json();
      if (isHttpsUrl(body.documents_url)) documentsLink.href = body.documents_url;
    } catch (_) {
      // Keep the default link.
    }
  }

  textarea.addEventListener("input", updateCounter);
  textarea.addEventListener("keydown", function (event) {
    // Ctrl/Cmd+Enter submits; plain Enter still makes a new line.
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", function (event) {
    event.preventDefault();
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
  loadDocumentsUrl();
})();

// Operator monitor page. Talks only to /api/monitor/* with the operator
// token; builds every element with createElement and textContent, since the
// trace contains passage and answer text.
"use strict";

// Which stages ran for a trace, in pipeline order. Pure, so it is unit-tested.
function stagesRun(trace) {
  const ran = [];
  if (trace.gate) ran.push("gate");
  if (trace.cache) ran.push("cache");
  if (trace.sweep) ran.push("sweep");
  if (trace.attempts.length) ran.push("draft");
  if (trace.attempts.some((a) => a.claims.length)) ran.push("verify");
  ran.push("answer");
  return ran;
}

// The timing bar's segments. Gate and cache run at the same time, so they
// share one segment sized by whichever took longer.
function timingSegments(stageMs, latencyMs) {
  const ms = (key) => Math.max(0, Number(stageMs && stageMs[key]) || 0);
  const parts = [
    { label: "Gate + cache", who: "jev", ms: Math.max(ms("gate"), ms("cache")) },
    { label: "Sweep", who: "jev", ms: ms("sweep") },
    { label: "Draft", who: "haiku", ms: ms("answer") },
    { label: "Verify", who: "code", ms: ms("verify") },
  ];
  const used = parts.reduce((sum, part) => sum + part.ms, 0);
  const total = Math.max(Number(latencyMs) || 0, used, 1);
  parts.push({ label: "Other", who: "other", ms: Math.max(0, total - used) });
  return parts.map((part) => ({ ...part, share: part.ms / total }));
}

// One line summarizing the traffic feed.
function feedSummary(summary) {
  if (!summary || !summary.requests) return "No requests recorded yet.";
  const pct = (n) => Math.round((100 * n) / summary.requests) + "%";
  const median =
    summary.median_latency_ms == null
      ? ""
      : ", median " + (summary.median_latency_ms / 1000).toFixed(1) + " s";
  return (
    summary.requests + " requests (the last " + summary.keep + " at most, kept " +
    summary.days + " days): " + pct(summary.answered) + " answered, " +
    pct(summary.cache_hits) + " from the cache" + median +
    ", $" + summary.cost_usd.toFixed(4) + " in total."
  );
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { stagesRun, timingSegments, feedSummary };
}

if (typeof document !== "undefined") {
  (function () {
    const TOKEN_KEY = "hoa-qa-monitor-token";
    const STAGES = [
      { key: "gate", name: "Gate", who: "jev", actor: "Jev" },
      { key: "cache", name: "Cache", who: "jev", actor: "Code + Jev" },
      { key: "sweep", name: "Sweep", who: "jev", actor: "Jev" },
      { key: "draft", name: "Draft", who: "haiku", actor: "Haiku" },
      { key: "verify", name: "Verify", who: "code", actor: "Code + Jev" },
      { key: "answer", name: "Answer", who: "code", actor: "Code" },
    ];
    const OUTCOME = {
      answered: "Answered",
      not_found: "Not found",
      refused_off_topic: "Refused, off topic",
      error: "Error",
      invalid_input: "Invalid input",
      budget_exhausted: "Budget used up",
    };
    const S = { stage: "gate", attempt: 0, conc: 16, playing: false, current: -1 };
    const traces = [];
    let D = null;
    let token = "";
    let timers = [];
    let feedTimer = null;
    const reduced =
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const $ = (id) => document.getElementById(id);

    function h(tag, attrs, ...kids) {
      const el = document.createElement(tag);
      for (const [k, v] of Object.entries(attrs || {})) {
        if (v == null || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "onclick") el.addEventListener("click", v);
        else if (k === "style") el.style.cssText = v; // CSSOM: allowed by the CSP
        else el.setAttribute(k, v);
      }
      (function add(list) {
        for (const kid of list) {
          if (kid == null || kid === false) continue;
          if (Array.isArray(kid)) add(kid);
          else el.appendChild(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
        }
      })(kids);
      return el;
    }
    function clear(el) {
      while (el.firstChild) el.removeChild(el.firstChild);
      return el;
    }
    const secs = (ms) => (ms / 1000).toFixed(1) + " s";
    const pct = (p) => (p == null ? "–" : p.toFixed(2));
    const plural = (n, word) => n + " " + word + (n === 1 ? "" : "s");
    const who = (cls, text) => h("span", { class: "who " + cls }, text);
    const json = (value) => h("pre", null, JSON.stringify(value, null, 2));
    const fold = (summary, body, open) =>
      h("details", { open: open ? "" : null }, h("summary", null, summary), body);
    const label = (i) => D.chunks[i].label;
    const cur = () => traces[S.current];
    const lastAttempt = (c) => c.attempts[c.attempts.length - 1];
    function stopTimers() {
      timers.forEach(clearTimeout);
      timers = [];
    }
    function later(fn, ms) {
      timers.push(setTimeout(fn, ms));
    }

    // ---- talking to the server ---------------------------------------------
    async function api(path, options) {
      const response = await fetch(path, {
        ...options,
        headers: { "x-monitor-token": token, "Content-Type": "application/json" },
      });
      if (response.status === 404) throw new Error("locked");
      let body = null;
      try {
        body = await response.json();
      } catch (_) {
        body = null;
      }
      if (!body || typeof body !== "object") throw new Error("unreadable");
      return body;
    }
    function lock(message) {
      token = "";
      try {
        sessionStorage.removeItem(TOKEN_KEY);
      } catch (_) {}
      if (feedTimer) clearInterval(feedTimer);
      $("app").hidden = true;
      $("unlock").hidden = false;
      if (message) $("unlock-note").textContent = message;
    }
    async function unlock(candidate) {
      token = candidate;
      try {
        D = await api("/api/monitor/config");
      } catch (error) {
        lock(
          error.message === "locked"
            ? "That token wasn't accepted, or the monitor is switched off (MONITOR_TOKEN isn't set)."
            : "The monitor couldn't be reached. Try again in a moment.",
        );
        return;
      }
      try {
        sessionStorage.setItem(TOKEN_KEY, token);
      } catch (_) {}
      S.conc = D.concurrency;
      $("unlock").hidden = true;
      $("app").hidden = false;
      clear($("foot")).appendChild(
        h("div", { class: "keyrow" },
          h("span", null, h("i", { class: "swatch jev" }), "Jev (" + D.models.jev + ") judges: yes/no probabilities only"),
          h("span", null, h("i", { class: "swatch haiku" }), "Haiku (" + D.models.answer + ") drafts claims with quotes"),
          h("span", null, h("i", { class: "swatch code" }), "Code decides what happens with each result")),
      );
      loadFeed();
      feedTimer = setInterval(() => {
        if (!document.hidden) loadFeed();
      }, 20000);
    }

    // ---- stage summaries ----------------------------------------------------
    function ran(c, key) {
      return stagesRun(c).includes(key);
    }
    function stageResult(c, key) {
      if (!ran(c, key)) return "not reached";
      if (key === "gate") return pct(c.gate.score) + (c.gate.passed ? " on topic" : " refused");
      if (key === "cache") {
        if (c.cache.exact) return "exact match";
        return c.cache.hit ? pct(c.cache.probability) + " hit" : "miss";
      }
      if (key === "sweep") return c.passages.length + " of " + D.chunks.length + " kept";
      if (key === "draft") {
        return plural(lastAttempt(c).claims.length, "claim") + (c.attempts.length > 1 ? ", 2 drafts" : "");
      }
      if (key === "verify") {
        const claims = lastAttempt(c).claims;
        return claims.filter((x) => x.kept).length + " of " + claims.length + " kept";
      }
      return OUTCOME[c.outcome] || c.outcome;
    }

    function renderFrame() {
      const c = cur();
      const asked = clear($("asked"));
      asked.hidden = false;
      asked.appendChild(
        h("div", null,
          h("h3", null, "You asked"),
          h("q", null, c.question),
          h("div", { class: "meta" },
            h("span", { class: "pill " + (c.outcome === "answered" ? "ok" : "plain") }, OUTCOME[c.outcome] || c.outcome),
            h("span", { class: "num" }, secs(c.latency_ms) + " end to end"),
            h("span", { class: "num" }, "$" + c.cost_usd.toFixed(4)),
            c.cache && c.cache.hit ? h("span", null, "served from the cache") : null,
            c.attempts.length > 1 ? h("span", null, "needed a second draft") : null)),
      );
      const track = clear($("track"));
      track.hidden = false;
      let before = true;
      for (const st of STAGES) {
        const active = st.key === S.stage;
        const reached = ran(c, st.key);
        if (active) before = false;
        track.appendChild(
          h("button", {
            type: "button",
            class: "stage" + (reached ? "" : " skipped") + (before && reached ? " passed-through" : ""),
            "aria-current": active ? "true" : null,
            onclick: () => {
              stopPlay();
              go(st.key, false);
            },
          }, who(st.who, st.actor), h("span", { class: "name" }, st.name), h("span", { class: "out num" }, stageResult(c, st.key))),
        );
      }
      const timing = clear($("timing"));
      timing.hidden = false;
      const bar = h("div", { class: "bar" });
      const key = h("div", { class: "keyrow num" });
      for (const part of timingSegments(c.stage_ms, c.latency_ms)) {
        bar.appendChild(h("i", { class: part.who, style: "width:" + 100 * part.share + "%" }));
        if (part.who !== "other" && part.ms > 0) key.appendChild(h("span", null, part.label + " " + secs(part.ms)));
      }
      timing.append(bar, key);
    }

    // ---- stage panels -------------------------------------------------------
    const head = (st, title, text, extra) =>
      h("div", { class: "panel-head" }, h("h2", null, title), who(st.who, st.actor), extra || null, h("p", null, text));
    const col = (title, ...kids) => h("div", { class: "col" }, h("h3", null, title), kids);
    function gauge(p, threshold) {
      const fill = h("div", { class: "fill", style: "width:0%" });
      later(() => {
        fill.style.width = 100 * p + "%";
      }, 30);
      return h("div", { class: "gauge" }, fill,
        h("div", { class: "mark", style: "left:" + 100 * threshold + "%" }, h("span", null, "needs " + threshold)));
    }
    function check(state, text, extra) {
      const icon = state === "ok" ? "✓" : state === "bad" ? "✕" : "–";
      return h("div", { class: "check" }, h("span", { class: "ic " + state }, icon),
        h("div", { class: "stack" }, h("span", null, text), extra || null));
    }

    function gatePanel(c, st) {
      const T = D.thresholds;
      return [
        head(st, "Is this an HOA question at all?", "One small request. Jev reads the message as data and returns a single number: the probability that the answer is yes. Code compares it to the threshold."),
        h("div", { class: "io" },
          col("Goes in to " + D.models.jev,
            h("div", { class: "note" }, "State (the data being judged)"), json({ message: c.question }),
            h("div", { class: "note" }, "Question and what yes and no mean"), json({ on_topic: D.gate_question })),
          col("Comes out",
            h("div", { class: "big num" }, pct(c.gate.score)),
            gauge(c.gate.score, T.gate),
            h("p", null, c.gate.passed ? "Passed." : "Refused. Nothing else runs, so this cost a fraction of a cent."),
            h("p", { class: "note" }, "Jev writes no text. A probability is its whole reply."))),
      ];
    }

    function cachePanel(c, st) {
      const k = c.cache;
      const T = D.thresholds;
      let out;
      if (k.exact) {
        out = [h("p", null, "This is one of the curated questions word for word, so its stored answer was returned with no model call at all."),
          h("p", { class: "note" }, "Stored answer: " + k.hit)];
      } else if (!k.candidates) {
        out = [h("p", null, "No stored question mentions the same numbers as this one, so nothing was sent to Jev. The question went on to the full pipeline.")];
      } else {
        out = [
          h("div", { class: "big num" }, pct(k.probability)),
          gauge(k.probability || 0, T.cache),
          h("p", null, k.hit
            ? "A match. The stored, already-verified answer was returned and the sweep and Haiku were skipped."
            : "Not close enough to any stored question, so the full pipeline answered it."),
          k.hit ? h("p", { class: "note" }, "Matched: “" + k.matched_question + "” (" + k.hit + ")") : null,
        ];
      }
      return [
        head(st, "Has this been answered before?", "Runs at the same time as the gate. Code first rules out stored questions that mention different numbers; Jev then judges whether a stored question's answer would fully answer this one."),
        h("div", { class: "io" },
          col("Goes in",
            h("p", { class: "note" }, plural(k.candidates, "stored question") + " passed the number check and " + (k.candidates === 1 ? "was" : "were") + " judged."),
            k.sample ? fold("State and question for the best match", json(k.sample), true) : null),
          col("Comes out", out)),
      ];
    }

    function sweepPanel(c, st, animate) {
      const T = D.thresholds;
      const plan = c.sweep.plan;
      const scores = {};
      const selected = {};
      for (const t of c.sweep.top) {
        scores[t.i] = t.score;
        if (t.selected) selected[t.i] = true;
      }
      const info = h("p", { class: "note" }, "Each square is one passage. Select one to see what it is.");
      const rounds = h("span", { class: "num" });
      const docs = h("div", { class: "docs" });
      const rows = {};
      const batches = [];
      for (const batch of plan) {
        const doc = D.chunks[batch[0]].doc;
        if (!rows[doc]) {
          rows[doc] = h("div", { class: "cells" });
          docs.appendChild(h("div", { class: "doc" }, h("div", { class: "t", title: D.docs[doc].title }, D.docs[doc].title), rows[doc]));
        }
        const el = h("div", { class: "batch" });
        for (const i of batch) {
          el.appendChild(h("span", {
            class: "cell" + (selected[i] ? " sel" : scores[i] != null ? " top" : ""),
            "data-i": i,
            title: label(i),
            style: scores[i] != null ? "--s:" + Math.max(0.25, scores[i]) : null,
          }));
        }
        rows[doc].appendChild(el);
        batches.push(el);
      }
      docs.addEventListener("click", (event) => {
        const i = event.target && event.target.getAttribute && event.target.getAttribute("data-i");
        if (i == null) return;
        const known = scores[i] != null;
        info.textContent = label(+i) + ": " + (known
          ? "Jev scored it " + pct(scores[i]) + (selected[i] ? ", sent to Haiku." : ", not in the top " + T.top_k + ".")
          : "judged, but below this question's top 20, so its score isn't shown.");
      });
      function reveal(animated) {
        stopTimers();
        const total = Math.ceil(batches.length / S.conc);
        batches.forEach((b) => b.classList.remove("done"));
        const wave = (w) => {
          batches.slice(w * S.conc, (w + 1) * S.conc).forEach((b) => b.classList.add("done"));
          rounds.textContent = "round " + Math.min(w + 1, total) + " of " + total;
        };
        if (!animated || reduced) {
          for (let w = 0; w < total; w++) wave(w);
          return 0;
        }
        const step = 150;
        for (let k = 0; k < total; k++) later(() => wave(k), k * step);
        return total * step;
      }
      const concButton = (n, text) =>
        h("button", { type: "button", "aria-pressed": String(S.conc === n), onclick: () => {
          S.conc = n;
          stopPlay();
          go("sweep", true);
        } }, text);
      const rank = h("div", { class: "rank num" });
      for (const t of c.sweep.top) {
        rank.append(
          h("div", { class: "lab" + (t.selected ? " sel" : ""), title: t.heading }, label(t.i)),
          h("div", { class: "meter" + (t.selected ? " sel" : "") }, h("i", { style: "width:" + 100 * t.score + "%" }), h("b", { style: "left:" + 100 * T.sweep + "%" })),
          h("div", null, pct(t.score)));
      }
      const duration = reveal(animate);
      const sample = c.sweep.sample;
      const slow = D.concurrency === 4 ? 16 : 4;
      return {
        duration,
        nodes: [
          head(st, "Which passages help answer it?", "No search index. Jev reads every passage in the HOA documents and scores each one against the question. " + plan.length + " requests carry all " + D.chunks.length + " passages, at most 8 per request, one document at a time."),
          h("div", { class: "sweepbar" },
            h("div", { class: "seg" }, concButton(D.concurrency, D.concurrency + " at a time (live setting)"), concButton(slow, slow + " at a time")),
            rounds,
            h("button", { class: "btn", type: "button", onclick: () => {
              stopPlay();
              go("sweep", true);
            } }, "Replay sweep")),
          docs, info,
          h("p", { class: "note" }, "The replay shows the rounds of requests; its speed is for show. The real sweep for this question took " + secs(c.stage_ms.sweep || 0) + "."),
          h("div", { class: "io" },
            col("Goes in, one of " + plan.length + " requests",
              sample ? fold("State: the question plus one document's passages", json(sample.state)) : h("p", { class: "note" }, "No request to show."),
              sample ? fold("One yes/no question per passage", json(sample.questions.map((q) => ({ name: q.name, instructions: q.instructions, yes: q.yes, no: q.no })))) : null,
              sample ? h("div", { class: "stack" }, h("div", { class: "note" }, "What Jev returned for that request"),
                json(Object.fromEntries(sample.questions.map((q) => [q.name, q.score == null ? "below the top 20" : q.score])))) : null),
            col("Comes out: the " + c.sweep.top.length + " best scores",
              rank,
              h("p", { class: "note" }, "The line marks " + T.sweep + ". The top " + T.top_k + " at or above it go to Haiku" + (c.passages.length ? "; here that is " + plural(c.passages.length, "passage") + "." : "; none qualified, so the app says it could not find an answer.")))),
        ],
      };
    }

    function attemptSwitch(c) {
      if (c.attempts.length < 2) return null;
      return h("div", { class: "seg" }, c.attempts.map((a, i) =>
        h("button", { type: "button", "aria-pressed": String(S.attempt === i), onclick: () => {
          S.attempt = i;
          stopPlay();
          go(S.stage, false);
        } }, i === 0 ? "First draft" : "Second draft")));
    }
    function chunkLabel(c, id) {
      const p = c.passages.find((x) => x.id === id);
      return p ? p.label : id + " (not a passage it was given)";
    }

    function draftPanel(c, st) {
      const a = c.attempts[S.attempt];
      const passages = c.passages.map((p) =>
        fold([h("span", null, p.label), h("span", { class: "pill plain" }, p.authority)], [h("pre", null, p.tag + "\n" + p.text + "\n</passage>")]));
      let out = a.claims.map((cl) =>
        h("div", { class: "claim" },
          h("div", { class: "st" }, cl.statement),
          h("div", { class: "chips" }, h("span", { class: "pill plain" }, cl.kind), cl.essential ? h("span", { class: "pill plain" }, "essential") : null),
          cl.citations.map((x) => h("div", { class: "cite" }, h("span", null, "“" + x.quote + "”"), h("div", { class: "src" }, chunkLabel(c, x.chunk_id))))));
      if (!a.valid) out = [h("p", null, "The output could not be used" + (a.note ? " (" + a.note + ")" : "") + ", so the app asked again.")];
      else if (!a.claims.length) out = [h("p", null, "Haiku returned no claims: it judged that the passages do not answer the question.")];
      return [
        head(st, S.attempt ? "Haiku tries again" : "Haiku drafts an answer from the passages", "Haiku never sees the documents, only the passages Jev picked. It must return structured claims, each with a quote copied word for word from a passage. It does not write the final answer text.", attemptSwitch(c)),
        h("div", { class: "io" },
          col("Goes in to " + D.models.answer + " (about " + Math.round(a.prompt_chars / 1000) + "K characters)",
            fold("System prompt: the rules it must follow", h("pre", null, D.system_prompt)),
            h("div", { class: "note" }, plural(c.passages.length, "passage") + " from the sweep, highest score first"),
            h("div", { class: "stack" }, passages),
            h("div", { class: "note" }, "The question"), h("pre", null, "<question>\n" + c.question + "\n</question>"),
            a.feedback ? [h("div", { class: "note" }, "What was wrong with the first draft"), h("pre", null, a.feedback)] : null),
          col("Comes out: " + plural(a.claims.length, "claim"), out,
            a.valid && a.claims.length ? fold("The same thing as the JSON Haiku returned", json({ claims: a.claims.map((cl) => ({ statement: cl.statement, kind: cl.kind, essential: cl.essential, citations: cl.citations.map((x) => ({ chunk_id: x.chunk_id, quote: x.quote })) })) })) : null,
            h("p", { class: "note" }, c.tokens.answer_in + " tokens in and " + c.tokens.answer_out + " out across " + plural(c.attempts.length, "draft") + "."))),
      ];
    }

    function context(c, cite) {
      const p = c.passages.find((x) => x.id === cite.chunk_id);
      if (!p) return null;
      try {
        const re = new RegExp(cite.quote.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+"), "i");
        const m = re.exec(p.text);
        if (!m) return null;
        const from = Math.max(0, m.index - 110);
        const to = Math.min(p.text.length, m.index + m[0].length + 110);
        return h("div", { class: "cite" }, (from ? "… " : "") + p.text.slice(from, m.index), h("mark", null, m[0]), p.text.slice(m.index + m[0].length, to) + (to < p.text.length ? " …" : ""));
      } catch (_) {
        return null;
      }
    }

    function verifyPanel(c, st) {
      const T = D.thresholds;
      const a = c.attempts[S.attempt];
      const screens = { low_authority: 1, informal_as_current: 1, advice_phrasing: 1 };
      const rows = a.claims.map((cl) => {
        const quotes = cl.citations.map((x) =>
          check(x.quote_ok ? "ok" : "bad", (x.quote_ok ? "Quote found word for word in " : "Quote not found in ") + chunkLabel(c, x.chunk_id), x.quote_ok ? context(c, x) : h("div", { class: "cite" }, "“" + x.quote + "”")));
        const screen = screens[cl.reason]
          ? check("bad", "Rule check failed: " + (D.reasons[cl.reason] || cl.reason))
          : cl.reason === "no_valid_quote" ? null : check("ok", "Source rules passed (authority and legal-advice wording)");
        const support = cl.support == null
          ? check("na", "Not sent to Jev: it had already failed a code check")
          : check(cl.support >= T.support ? "ok" : "bad", "Jev: do the cited passages state all of this? " + pct(cl.support) + " (needs " + T.support + ")");
        return h("div", { class: "claim" + (cl.kept ? "" : " dropped") },
          h("div", { class: "chips" }, h("span", { class: "pill " + (cl.kept ? "ok" : "bad") }, cl.kept ? "Kept" : "Dropped"), h("span", { class: "pill plain" }, cl.kind), cl.essential ? h("span", { class: "pill plain" }, "essential") : null),
          h("div", { class: "st" }, cl.statement), quotes, screen, support,
          cl.kept ? null : h("div", { class: "note" }, "Reason given back to Haiku: " + (D.reasons[cl.reason] || cl.reason) + "."));
      });
      const kept = a.claims.filter((x) => x.kept).length;
      const s = a.support;
      return [
        head(st, "Does every claim hold up?", "Three checks per claim, cheapest first. Code confirms the quote really is in the passage and that the source is allowed to answer. Then Jev judges whether the passages state everything the claim says. A claim that fails is never shown.", attemptSwitch(c)),
        a.claims.length
          ? h("p", { class: "num" }, kept + " of " + a.claims.length + " claims kept" + (S.attempt === 0 && c.attempts.length > 1 ? ". Because one failed, the app asked Haiku for a second draft." : "."))
          : h("p", null, "There were no claims to check in this draft."),
        h("div", { class: "io" },
          col("Each claim, checked", rows),
          col("What Jev was asked",
            s ? [h("p", { class: "note" }, plural(s.claims, "claim") + " passed the code checks and went to " + D.models.jev + " in one request. The first of them:"),
              fold("State: the claim and its cited passages", json(s.state_first), true),
              fold("The yes/no question", json(s.question))]
              : h("p", { class: "note" }, "No claim survived the code checks, so Jev was not called."))),
      ];
    }

    function answerPanel(c, st) {
      return [
        head(st, "What a resident would see", c.cache && c.cache.hit
          ? "This is the stored answer for the matched question: reviewed wording that was verified against the documents when the cache was built."
          : "Code assembles the reply from the claims that survived, then adds the citations, the disclaimer and any referral to the Board. No model writes this text freely."),
        h("div", { class: "io" },
          col(OUTCOME[c.outcome] || c.outcome,
            h("div", { class: "answer" }, c.answer_text),
            c.conflicts.length ? [h("h3", null, "Conflicts noted"), c.conflicts.map((t) => h("p", null, t))] : null),
          col("Cited",
            c.cited.length ? h("div", { class: "stack" }, c.cited.map((text) => h("div", { class: "cite" }, text))) : h("p", { class: "note" }, "Nothing cited."),
            h("h3", null, "Spent"),
            h("p", { class: "num" }, "$" + c.cost_usd.toFixed(5) + ": " + c.tokens.jev + " Jev tokens, " + c.tokens.answer_in + " Haiku tokens in, " + c.tokens.answer_out + " out."))),
      ];
    }

    // ---- navigation ---------------------------------------------------------
    function go(stage, animate) {
      stopTimers();
      const c = cur();
      if (!ran(c, stage)) stage = stagesRun(c)[0];
      S.stage = stage;
      if (S.attempt >= c.attempts.length) S.attempt = Math.max(0, c.attempts.length - 1);
      renderFrame();
      const st = STAGES.find((x) => x.key === stage);
      const panel = clear($("panel"));
      panel.hidden = false;
      let built;
      let duration = 0;
      if (stage === "gate") built = gatePanel(c, st);
      else if (stage === "cache") built = cachePanel(c, st);
      else if (stage === "sweep") {
        const sweep = sweepPanel(c, st, animate);
        built = sweep.nodes;
        duration = sweep.duration;
      } else if (stage === "draft") built = draftPanel(c, st);
      else if (stage === "verify") built = verifyPanel(c, st);
      else built = answerPanel(c, st);
      for (const node of built) if (node) panel.appendChild(node);
      return duration;
    }
    function stopPlay() {
      S.playing = false;
      stopTimers();
      $("play").textContent = "Play";
    }
    function play() {
      if (S.playing) return stopPlay();
      if (S.current < 0) return;
      S.playing = true;
      $("play").textContent = "Pause";
      const order = stagesRun(cur());
      let at = S.stage === "answer" ? 0 : Math.max(0, order.indexOf(S.stage));
      (function step() {
        if (!S.playing) return;
        const wait = go(order[at], true) + (order[at] === "sweep" ? 1400 : 2200);
        at += 1;
        if (at >= order.length) return later(stopPlay, 0);
        later(step, wait);
      })();
    }
    function show(index, autoplay) {
      stopPlay();
      S.current = index;
      S.attempt = Math.max(0, cur().attempts.length - 1);
      $("history").value = String(index);
      $("play").disabled = false;
      go(stagesRun(cur())[0], false);
      if (autoplay) play();
    }

    // ---- asking -------------------------------------------------------------
    async function ask(question) {
      const button = $("ask-button");
      const status = $("ask-status");
      button.disabled = true;
      const started = Date.now();
      const tick = setInterval(() => {
        status.textContent = "Running the pipeline… " + Math.round((Date.now() - started) / 1000) + " s";
      }, 500);
      status.textContent = "Running the pipeline…";
      try {
        const trace = await api("/api/monitor/ask", { method: "POST", body: JSON.stringify({ question }) });
        if (trace.error || !Array.isArray(trace.attempts)) {
          status.textContent = trace.error || trace.answer_text || "The question couldn't be traced.";
          return;
        }
        status.textContent = "";
        traces.push(trace);
        const history = $("history");
        history.appendChild(h("option", { value: traces.length - 1 }, question));
        history.hidden = traces.length < 2;
        show(traces.length - 1, true);
        loadFeed();
      } catch (error) {
        if (error.message === "locked") lock("The token is no longer accepted. Enter it again.");
        else status.textContent = "The request failed. Try again in a moment.";
      } finally {
        clearInterval(tick);
        button.disabled = false;
      }
    }

    // ---- traffic feed -------------------------------------------------------
    async function loadFeed() {
      let feed;
      try {
        feed = await api("/api/monitor/recent");
      } catch (error) {
        if (error.message === "locked") lock("The token is no longer accepted. Enter it again.");
        return;
      }
      if (feed.error) {
        $("feed-summary").textContent = feed.error;
        return;
      }
      $("feed-summary").textContent = feedSummary(feed.summary);
      const body = clear($("feed").querySelector("tbody"));
      for (const row of feed.rows) {
        const bar = h("div", { class: "bar", title: timingSegments(row.stage_ms, row.latency_ms).filter((p) => p.ms > 0 && p.who !== "other").map((p) => p.label + " " + secs(p.ms)).join(", ") });
        for (const part of timingSegments(row.stage_ms, row.latency_ms)) {
          bar.appendChild(h("i", { class: part.who, style: "width:" + 100 * part.share + "%" }));
        }
        const when = new Date(row.t);
        body.appendChild(
          h("tr", { class: row.operator ? "operator" : null },
            h("td", null, isNaN(when) ? row.t : when.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })),
            h("td", null, h("span", { class: "pill " + (row.outcome === "answered" ? "ok" : "plain") }, OUTCOME[row.outcome] || row.outcome || "?")),
            h("td", { class: "num" }, secs(row.latency_ms || 0)),
            h("td", null, bar),
            h("td", { class: "mono" }, row.cache || "–"),
            h("td", { class: "num" }, "$" + Number(row.cost_usd || 0).toFixed(4))),
        );
      }
    }

    // ---- wiring -------------------------------------------------------------
    $("unlock").addEventListener("submit", (event) => {
      event.preventDefault();
      const value = $("token").value.trim();
      $("token").value = "";
      if (value) unlock(value);
    });
    $("ask").addEventListener("submit", (event) => {
      event.preventDefault();
      const question = $("question").value.trim();
      if (question && !$("ask-button").disabled) ask(question);
    });
    $("question").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        $("ask").requestSubmit();
      }
    });
    $("play").addEventListener("click", play);
    $("refresh").addEventListener("click", loadFeed);
    $("history").addEventListener("change", () => show(+$("history").value, false));

    let saved = "";
    try {
      saved = sessionStorage.getItem(TOKEN_KEY) || "";
    } catch (_) {}
    if (saved) unlock(saved);
    else $("unlock").hidden = false;
  })();
}

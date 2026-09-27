// Unit tests for the loading status schedule in public/app.js.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { PROGRESS_STAGES, progressStage, createProgress } = require("../../public/app.js");

const STILL_WORKING = PROGRESS_STAGES.length - 1;

// A fake clock with setTimeout-style timers that only fire on advance().
function fakeClock() {
  let time = 0;
  let nextId = 1;
  const timers = new Map();
  return {
    now: () => time,
    setTimer(fn, ms) {
      const id = nextId++;
      timers.set(id, { fn, due: time + ms });
      return id;
    },
    clearTimer(id) {
      timers.delete(id);
    },
    pending: () => timers.size,
    advance(ms) {
      const end = time + ms;
      for (;;) {
        let dueId = null;
        for (const [id, timer] of timers) {
          if (timer.due <= end && (dueId === null || timer.due < timers.get(dueId).due)) {
            dueId = id;
          }
        }
        if (dueId === null) break;
        const timer = timers.get(dueId);
        timers.delete(dueId);
        time = timer.due;
        timer.fn();
      }
      time = end;
    },
  };
}

function recordingProgress(clock) {
  const shown = [];
  let stops = 0;
  const progress = createProgress({
    now: clock.now,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
    onStage: (stage, index) => shown.push(index),
    onStop: () => {
      stops += 1;
    },
  });
  return { progress, shown, stops: () => stops };
}

test("stages start at zero and are strictly increasing in time", () => {
  assert.equal(PROGRESS_STAGES[0].at, 0);
  for (let i = 1; i < PROGRESS_STAGES.length; i++) {
    assert.ok(PROGRESS_STAGES[i].at > PROGRESS_STAGES[i - 1].at);
  }
});

test("each stage begins exactly at its boundary", () => {
  PROGRESS_STAGES.forEach((stage, index) => {
    assert.equal(progressStage(stage.at), index);
    if (index > 0) assert.equal(progressStage(stage.at - 1), index - 1);
  });
});

test("the still-working line appears only after the threshold", () => {
  const threshold = PROGRESS_STAGES[STILL_WORKING].at;
  assert.ok(threshold >= 15000, "still-working line should wait for a genuinely slow answer");
  assert.match(PROGRESS_STAGES[STILL_WORKING].text, /^Still working/);
  assert.notEqual(progressStage(threshold - 1), STILL_WORKING);
  assert.equal(progressStage(threshold), STILL_WORKING);
  assert.equal(progressStage(10 * 60 * 1000), STILL_WORKING);
});

test("stage never goes backwards as time passes", () => {
  let previous = progressStage(0);
  for (let ms = 0; ms <= 30000; ms += 50) {
    const index = progressStage(ms);
    assert.ok(index >= previous && index <= previous + 1, "at " + ms + "ms");
    previous = index;
  }
  assert.equal(previous, STILL_WORKING);
});

test("bad elapsed values mean the first stage", () => {
  for (const value of [-1, -Infinity, NaN, undefined, null, "soon"]) {
    assert.equal(progressStage(value), 0);
  }
});

test("only the first and still-working lines are announced", () => {
  const announced = PROGRESS_STAGES.flatMap((stage, index) => (stage.announce ? [index] : []));
  assert.deepEqual(announced, [0, STILL_WORKING]);
});

test("lines describe activity, not completion or percentages", () => {
  for (const stage of PROGRESS_STAGES) {
    assert.doesNotMatch(stage.text, /%|\bdone\b|\bcomplete|\bfinished\b|\bfound\b/i);
  }
});

test("controller walks the schedule with a single pending timer", () => {
  const clock = fakeClock();
  const { progress, shown } = recordingProgress(clock);
  progress.start();
  assert.deepEqual(shown, [0]);
  assert.equal(clock.pending(), 1);
  clock.advance(PROGRESS_STAGES[1].at - 1);
  assert.deepEqual(shown, [0]);
  clock.advance(1);
  assert.deepEqual(shown, [0, 1]);
  clock.advance(60000);
  assert.deepEqual(
    shown,
    PROGRESS_STAGES.map((_, index) => index),
  );
  assert.equal(clock.pending(), 0, "no timer after the last stage");
  assert.equal(progress.isRunning(), true);
});

test("finishing a request clears the timer and shows nothing more", () => {
  const clock = fakeClock();
  const { progress, shown, stops } = recordingProgress(clock);
  progress.start();
  clock.advance(2000);
  progress.stop();
  assert.equal(clock.pending(), 0);
  assert.equal(stops(), 1);
  assert.equal(progress.isRunning(), false);
  clock.advance(60000);
  assert.deepEqual(shown, [0, 1]);
  progress.stop(); // idempotent
  assert.equal(stops(), 1);
});

test("a new request restarts from the first stage", () => {
  const clock = fakeClock();
  const { progress, shown } = recordingProgress(clock);
  progress.start();
  clock.advance(12000);
  progress.stop();
  shown.length = 0;
  progress.start();
  assert.deepEqual(shown, [0]);
  clock.advance(PROGRESS_STAGES[1].at);
  assert.deepEqual(shown, [0, 1]);
  assert.equal(clock.pending(), 1);
});

test("start while running restarts cleanly without leaking a timer", () => {
  const clock = fakeClock();
  const { progress, shown, stops } = recordingProgress(clock);
  progress.start();
  clock.advance(7000);
  progress.start();
  assert.equal(stops(), 1);
  assert.equal(clock.pending(), 1);
  assert.deepEqual(shown, [0, 1, 2, 0]);
});

test("an early timer callback reschedules instead of skipping ahead", () => {
  // Real timers can fire a hair early relative to performance.now().
  let time = 0;
  const queued = [];
  const shown = [];
  const progress = createProgress({
    now: () => time,
    setTimer: (fn, ms) => queued.push({ fn, ms }),
    clearTimer: () => {},
    onStage: (stage, index) => shown.push(index),
  });
  progress.start();
  time = PROGRESS_STAGES[1].at - 2;
  queued.shift().fn();
  assert.deepEqual(shown, [0]);
  assert.equal(queued.length, 1);
  assert.equal(queued[0].ms, 2);
});

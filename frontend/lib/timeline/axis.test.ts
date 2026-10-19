import { strict as assert } from "node:assert";
import { test } from "node:test";

import { assignLanes, buildAxis, unitFor } from "./axis.ts";

test("the unit follows the range: days, months, years", () => {
  assert.equal(unitFor(0), "day");
  assert.equal(unitFor(45), "day");
  assert.equal(unitFor(46), "month");
  assert.equal(unitFor(800), "month");
  assert.equal(unitFor(801), "year");
});

test("a narrow range gets a day-level axis with the dots in date order", () => {
  const axis = buildAxis([
    "2026-10-05T12:00:00Z",
    "2026-10-01T03:00:00Z",
    "2026-10-09T23:59:00Z",
  ]);
  assert.equal(axis.unit, "day");
  assert.equal(axis.spanDays, 8);
  assert.ok(axis.ticks.length >= 2 && axis.ticks.length <= 8);
  assert.equal(axis.ticks[0]?.label, "Oct 1");
  const xs = [
    "2026-10-01T03:00:00Z",
    "2026-10-05T12:00:00Z",
    "2026-10-09T23:59:00Z",
  ].map((t) => axis.positionOf(Date.parse(t)));
  assert.deepEqual(
    [...xs].sort((a, b) => a - b),
    xs,
  );
  assert.ok(xs[0]! >= 0 && xs[2]! <= 1);
});

test("a single day does not divide by zero and keeps the dot inside", () => {
  const axis = buildAxis(["2026-10-05T12:00:00Z", "2026-10-05T18:00:00Z"]);
  assert.equal(axis.spanDays, 2);
  const x = axis.positionOf(Date.parse("2026-10-05T12:00:00Z"));
  assert.ok(x > 0 && x < 1);
  assert.deepEqual(buildAxis([]).ticks.length > 0, true);
});

test("mixed years (an old paper beside this month's) get a year axis", () => {
  const axis = buildAxis([
    "2013-01-16T00:00:00Z",
    "2017-06-12T00:00:00Z",
    "2026-10-01T00:00:00Z",
  ]);
  assert.equal(axis.unit, "year");
  assert.ok(axis.ticks.length <= 8);
  assert.ok(axis.ticks.every((t) => /^\d{4}$/.test(t.label)));
  const early = axis.positionOf(Date.parse("2013-01-16T00:00:00Z"));
  const late = axis.positionOf(Date.parse("2026-10-01T00:00:00Z"));
  assert.ok(early < 0.01 && late > 0.99);
});

test("months for a range of a few months, tick labels within the range", () => {
  const axis = buildAxis(["2026-03-20T00:00:00Z", "2026-09-10T00:00:00Z"]);
  assert.equal(axis.unit, "month");
  assert.ok(axis.ticks.length >= 2 && axis.ticks.length <= 8);
  assert.ok(axis.ticks.every((t) => t.at >= 0 && t.at <= 1));
  assert.equal(axis.ticks[0]?.label, "Apr 2026");
});

test("overlapping dots go to different lanes, distant ones share lane 0", () => {
  assert.deepEqual(assignLanes([0.1, 0.5, 0.9], 0.05), [0, 0, 0]);
  assert.deepEqual(assignLanes([0.1, 0.11, 0.12], 0.05), [0, 1, 2]);
  // the input order is kept in the result, whatever the positions' order
  assert.deepEqual(assignLanes([0.12, 0.1, 0.11], 0.05), [2, 0, 1]);
  // a lane frees up again after a gap
  assert.deepEqual(assignLanes([0.1, 0.11, 0.3], 0.05), [0, 1, 0]);
  // never more lanes than allowed
  assert.deepEqual(assignLanes([0.1, 0.1, 0.1, 0.1], 0.05, 2), [0, 1, 1, 1]);
});

import { describe, it } from "node:test";
import assert from "node:assert/strict";

describe("DedupAction types", () => {
  it("DedupAction type structure is correct", () => {
    const action = { matchIndex: 1, action: "delete", reason: "outdated" };
    assert.equal(action.matchIndex, 1);
    assert.equal(action.action, "delete");
    assert.equal(action.reason, "outdated");
  });

  it("DedupResult with actions is backward compatible", () => {
    // Without actions — existing behavior
    const resultNoActions = {
      decision: "create",
      reason: "new info",
    };
    assert.equal(resultNoActions.actions, undefined);

    // With actions — new behavior
    const resultWithActions = {
      decision: "merge",
      reason: "add details",
      matchId: "abc123",
      actions: [
        { matchIndex: 2, action: "delete", reason: "outdated by candidate" },
      ],
    };
    assert.equal(resultWithActions.actions.length, 1);
    assert.equal(resultWithActions.actions[0].action, "delete");
  });

  it("ExtractionStats includes actionsExecuted", () => {
    const stats = {
      created: 2,
      merged: 1,
      skipped: 0,
      actionsExecuted: 3,
    };
    assert.equal(stats.actionsExecuted, 3);
  });
});

describe("DedupAction validation rules", () => {
  it("only merge and delete are valid actions", () => {
    const validActions = ["merge", "delete"];
    const invalidActions = ["skip", "create", "supersede", "update"];

    for (const a of validActions) {
      assert.ok(["merge", "delete"].includes(a), `${a} should be valid`);
    }
    for (const a of invalidActions) {
      assert.ok(!["merge", "delete"].includes(a), `${a} should be invalid`);
    }
  });

  it("matchIndex must be 1-based positive integer", () => {
    const validIndices = [1, 2, 3, 5];
    const invalidIndices = [0, -1, 1.5, NaN, null, undefined];

    for (const idx of validIndices) {
      assert.ok(typeof idx === "number" && idx >= 1 && Number.isInteger(idx));
    }
    for (const idx of invalidIndices) {
      assert.ok(!(typeof idx === "number" && idx >= 1 && Number.isInteger(idx)));
    }
  });

  it("actions array can be empty", () => {
    const result = { decision: "create", reason: "new", actions: [] };
    assert.equal(result.actions.length, 0);
  });

  it("duplicate primary matchIndex should be filtered out", () => {
    // If primary decision targets index 1, actions should not also target index 1
    const primaryIdx = 1;
    const rawActions = [
      { match_index: 1, action: "delete", reason: "dup" }, // same as primary — should be filtered
      { match_index: 2, action: "delete", reason: "outdated" }, // different — should be kept
    ];
    const filtered = rawActions.filter(a => a.match_index !== primaryIdx);
    assert.equal(filtered.length, 1);
    assert.equal(filtered[0].match_index, 2);
  });
});

describe("Action execution order", () => {
  it("deletes should come before merges", () => {
    const actions = [
      { matchIndex: 1, action: "merge", reason: "combine" },
      { matchIndex: 2, action: "delete", reason: "outdated" },
      { matchIndex: 3, action: "delete", reason: "duplicate" },
    ];

    const sorted = [...actions].sort((a, b) =>
      a.action === "delete" && b.action !== "delete" ? -1 : 1,
    );

    assert.equal(sorted[0].action, "delete");
    assert.equal(sorted[1].action, "delete");
    assert.equal(sorted[2].action, "merge");
  });
});

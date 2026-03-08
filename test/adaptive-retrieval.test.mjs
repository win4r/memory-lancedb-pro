import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const {
  shouldSkipRetrieval,
  shouldSkipRetrievalFirstTurn,
} = jiti("../src/adaptive-retrieval.ts");

// ============================================================================
// shouldSkipRetrieval — existing behavior (sanity checks)
// ============================================================================

describe("shouldSkipRetrieval", () => {
  it("skips pure emoji", () => {
    assert.equal(shouldSkipRetrieval("🤔"), true);
    assert.equal(shouldSkipRetrieval("😂👍"), true);
  });

  it("skips very short messages", () => {
    assert.equal(shouldSkipRetrieval("hi"), true);
    assert.equal(shouldSkipRetrieval("?"), true);
    assert.equal(shouldSkipRetrieval("ok"), true);
  });

  it("skips greetings", () => {
    assert.equal(shouldSkipRetrieval("hello"), true);
    assert.equal(shouldSkipRetrieval("good morning"), true);
  });

  it("skips continuation prompts", () => {
    assert.equal(shouldSkipRetrieval("继续"), true);
    assert.equal(shouldSkipRetrieval("go ahead"), true);
    assert.equal(shouldSkipRetrieval("好的"), true);
  });

  it("skips slash commands", () => {
    assert.equal(shouldSkipRetrieval("/status"), true);
    assert.equal(shouldSkipRetrieval("/new"), true);
  });

  it("does not skip memory-related queries", () => {
    assert.equal(shouldSkipRetrieval("你记得吗"), false);
    assert.equal(shouldSkipRetrieval("do you remember my name"), false);
    assert.equal(shouldSkipRetrieval("what did I say last time"), false);
  });

  it("does not skip meaningful questions", () => {
    assert.equal(shouldSkipRetrieval("what is the capital of France?"), false);
    assert.equal(shouldSkipRetrieval("帮我查一下明天的天气怎么样"), false);
  });

  it("strips OpenClaw metadata before evaluating", () => {
    const wrapped = `Conversation info (untrusted metadata):\n\`\`\`json\n{"message_id":"123"}\n\`\`\`\n\n🤔`;
    assert.equal(shouldSkipRetrieval(wrapped), true);
  });
});

// ============================================================================
// shouldSkipRetrievalFirstTurn — relaxed first-turn logic
// ============================================================================

describe("shouldSkipRetrievalFirstTurn", () => {
  it("allows pure emoji through on first turn", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("🤔"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("😂"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("👍"), false);
  });

  it("allows short continuation messages on first turn", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("继续"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("嗯"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("?"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("ok"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("go ahead"), false);
  });

  it("allows greetings on first turn (may carry continuity)", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("hi"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("hello"), false);
  });

  it("still skips empty messages", () => {
    assert.equal(shouldSkipRetrievalFirstTurn(""), true);
    assert.equal(shouldSkipRetrievalFirstTurn("   "), true);
  });

  it("still skips slash commands", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("/status"), true);
    assert.equal(shouldSkipRetrievalFirstTurn("/new"), true);
    assert.equal(shouldSkipRetrievalFirstTurn("/reset"), true);
  });

  it("still skips heartbeat messages", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("HEARTBEAT_OK"), true);
    assert.equal(shouldSkipRetrievalFirstTurn("Read HEARTBEAT.md if it exists"), true);
  });

  it("still skips system messages", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("[System] session started"), true);
  });

  it("allows memory-related queries", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("你记得吗"), false);
    assert.equal(shouldSkipRetrievalFirstTurn("remember"), false);
  });

  it("allows meaningful questions", () => {
    assert.equal(shouldSkipRetrievalFirstTurn("what were we talking about?"), false);
  });

  it("strips OpenClaw metadata before evaluating", () => {
    const wrapped = `Conversation info (untrusted metadata):\n\`\`\`json\n{"message_id":"123"}\n\`\`\`\n\n🤔`;
    // After stripping metadata, only 🤔 remains — first turn should allow it
    assert.equal(shouldSkipRetrievalFirstTurn(wrapped), false);
  });
});

// ============================================================================
// Contrast: same inputs, different results between standard vs first-turn
// ============================================================================

describe("standard vs first-turn skip comparison", () => {
  const weakSignals = ["🤔", "?", "继续", "ok", "嗯", "hi", "👍"];

  for (const msg of weakSignals) {
    it(`"${msg}" is skipped by standard but allowed by first-turn`, () => {
      assert.equal(shouldSkipRetrieval(msg), true, `shouldSkipRetrieval("${msg}") should be true`);
      assert.equal(shouldSkipRetrievalFirstTurn(msg), false, `shouldSkipRetrievalFirstTurn("${msg}") should be false`);
    });
  }

  const alwaysSkipped = ["/status", ""];
  for (const msg of alwaysSkipped) {
    it(`"${msg}" is skipped by both standard and first-turn`, () => {
      assert.equal(shouldSkipRetrieval(msg), true);
      assert.equal(shouldSkipRetrievalFirstTurn(msg), true);
    });
  }
});

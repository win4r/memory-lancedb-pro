import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const {
  DEFAULT_CAPTURE_POLICY,
  isRecallEligibleMemory,
  mergeCaptureMetadata,
  scoreMemoryCandidate,
} = jiti("../src/capture-pipeline.ts");

describe("capture pipeline scoring", () => {
  it("promotes explicit user preference/decision to formal memory", () => {
    const result = scoreMemoryCandidate({
      text: "记住：以后 OpenAI 中转作为主模型，MiniMax 只做备用模型。",
      role: "user",
      sourceApp: "openclaw",
      captureAssistant: true,
    }, DEFAULT_CAPTURE_POLICY);

    assert.equal(result.tier, "formal");
    assert.ok(result.score >= 8);
  });

  it("caps unconfirmed assistant summaries into candidate tier", () => {
    const result = scoreMemoryCandidate({
      text: "总结：用户以后优先使用 OpenAI 中转，MiniMax 仅作备用。",
      role: "assistant",
      sourceApp: "codex",
      captureAssistant: true,
    }, DEFAULT_CAPTURE_POLICY);

    assert.equal(result.tier, "candidate");
    assert.ok(result.score <= 5);
  });

  it("rejects secret-like content", () => {
    const result = scoreMemoryCandidate({
      text: "这是我的 key：sk-abcdefghijklmnopqrstuvwxyz1234567890",
      role: "user",
      sourceApp: "openclaw",
      captureAssistant: true,
    }, DEFAULT_CAPTURE_POLICY);

    assert.equal(result.tier, "discard");
    assert.equal(result.rejectedReason, "secret-like-content");
  });
});

describe("capture metadata recall gating", () => {
  it("excludes candidate memories from recall by default", () => {
    const candidateMetadata = mergeCaptureMetadata(undefined, {
      memoryTier: "candidate",
      captureScore: 5,
      captureReasons: ["assistant-summary"],
      normalizedKey: "user prefers openai relay",
      sourceApp: "codex",
      sourceRole: "assistant",
    });

    assert.equal(isRecallEligibleMemory({ metadata: candidateMetadata }, false), false);
    assert.equal(isRecallEligibleMemory({ metadata: candidateMetadata }, true), true);
  });

  it("always allows formal memories in recall", () => {
    const formalMetadata = mergeCaptureMetadata(undefined, {
      memoryTier: "formal",
      captureScore: 9,
      captureReasons: ["explicit-remember"],
      normalizedKey: "openai relay primary model",
      sourceApp: "openclaw",
      sourceRole: "user",
    });

    assert.equal(isRecallEligibleMemory({ metadata: formalMetadata }, false), true);
  });
});


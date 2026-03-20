import { describe, it } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jitiFactory from "jiti";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const pluginSdkStubPath = path.resolve(testDir, "helpers", "openclaw-plugin-sdk-stub.mjs");
const jiti = jitiFactory(import.meta.url, {
  interopDefault: true,
  alias: {
    "openclaw/plugin-sdk": pluginSdkStubPath,
  },
});

const { registerMemoryUpdateTool } = jiti("../src/tools.ts");

function createHarness(overrides = {}) {
  const factories = new Map();
  const embedCalls = [];
  const updateCalls = [];
  const validatorCalls = [];
  const retrieveCalls = [];
  const resolveByIdCalls = [];

  const api = {
    registerTool(factory, meta) {
      factories.set(meta?.name || "", factory);
    },
  };

  const context = {
    retriever: {
      async retrieve(params) {
        retrieveCalls.push(params);
        return [];
      },
      ...(overrides.retriever || {}),
    },
    store: {
      async update(id, updates, scopes) {
        updateCalls.push({ id, updates, scopes });
        return {
          id,
          text: updates.text || "existing text",
          vector: updates.vector || [],
          category: updates.category || "other",
          scope: "scope-main",
          importance: updates.importance ?? 0.7,
          timestamp: 1700000000000,
          metadata: "{}",
        };
      },
      async resolveByIdOrPrefix(id) {
        resolveByIdCalls.push(id);
        return {
          id,
          text: "existing text",
          vector: [0.2, 0.2, 0.2],
          category: "other",
          scope: "scope-main",
          importance: 0.7,
          timestamp: 1700000000000,
          metadata: JSON.stringify({
            custom: "keep-me",
            memoryTier: "candidate",
            captureScore: 5,
            captureReasons: ["existing-reason"],
            normalizedKey: "existing-key",
            firstSeenAt: 111,
            occurrences: 1,
          }),
        };
      },
      ...(overrides.store || {}),
    },
    scopeManager: {
      getAccessibleScopes() {
        return ["scope-main"];
      },
      ...(overrides.scopeManager || {}),
    },
    embedder: {
      async embedPassage(text) {
        embedCalls.push(text);
        return [0.1, 0.2, 0.3];
      },
      ...(overrides.embedder || {}),
    },
    agentId: overrides.agentId,
    captureValidator: async (params) => {
      validatorCalls.push(params);
      if (overrides.captureValidator) return await overrides.captureValidator(params);
      return {
        accepted: true,
        text: String(params.text || "").trim(),
      };
    },
  };

  if (Object.prototype.hasOwnProperty.call(overrides, "captureValidator") && overrides.captureValidator === null) {
    context.captureValidator = null;
  }

  registerMemoryUpdateTool(api, context);

  return {
    tool(toolCtx = {}) {
      const factory = factories.get("memory_update");
      assert.ok(factory, "memory_update tool should be registered");
      return factory(toolCtx);
    },
    embedCalls,
    updateCalls,
    validatorCalls,
    retrieveCalls,
    resolveByIdCalls,
  };
}

describe("memory_update ingress validation", () => {
  it("uses shared ingress validation for updated text and surfaces skip reasons", async () => {
    const harness = createHarness({
      async captureValidator(params) {
        return {
          accepted: false,
          reason: "system-noise",
          text: String(params.text || "").trim(),
        };
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-1", {
      memoryId: "12345678-1234-1234-1234-1234567890ab",
      text: "<relevant-memories> wrapper",
    });

    assert.equal(harness.validatorCalls.length, 1);
    assert.equal(harness.embedCalls.length, 0);
    assert.equal(harness.updateCalls.length, 0);
    assert.match(result.content[0].text, /system\/control text/i);
    assert.equal(result.details.action, "skipped");
    assert.equal(result.details.reason, "system-noise");
  });

  it("embeds and updates the sanitized validated text", async () => {
    const harness = createHarness({
      async captureValidator(params) {
        return {
          accepted: true,
          text: "Validated durable memory text.",
          metadataPayload: {
            memoryTier: "candidate",
            captureScore: 7,
            captureReasons: ["explicit-tool-request"],
            normalizedKey: "validated-durable-memory-text",
            sourceApp: "openclaw",
            sourceRole: "assistant",
            ingestionMode: "memory_update",
            confirmed: true,
          },
        };
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-2", {
      memoryId: "12345678-1234-1234-1234-1234567890ab",
      text: "  raw text that should be normalized  ",
      importance: 0.9,
      category: "decision",
    });

    assert.equal(harness.validatorCalls.length, 1);
    assert.deepEqual(harness.embedCalls, ["Validated durable memory text."]);
    assert.equal(harness.resolveByIdCalls.length, 1);
    assert.equal(harness.updateCalls.length, 1);
    assert.equal(harness.updateCalls[0].updates.text, "Validated durable memory text.");
    assert.deepEqual(harness.updateCalls[0].updates.vector, [0.1, 0.2, 0.3]);
    assert.equal(harness.updateCalls[0].updates.category, "decision");
    assert.equal(harness.updateCalls[0].updates.importance, 0.9);
    const mergedMetadata = JSON.parse(harness.updateCalls[0].updates.metadata);
    assert.equal(mergedMetadata.custom, "keep-me");
    assert.equal(mergedMetadata.captureScore, 7);
    assert.equal(mergedMetadata.normalizedKey, "validated-durable-memory-text");
    assert.deepEqual(
      mergedMetadata.captureReasons,
      ["existing-reason", "explicit-tool-request"],
    );
    assert.equal(mergedMetadata.firstSeenAt, 111);
    assert.equal(mergedMetadata.occurrences, 2);
    assert.equal(mergedMetadata.confirmed, true);
    assert.equal(mergedMetadata.ingestionMode, "memory_update");
    assert.equal(result.details.action, "updated");
    assert.equal(result.details.id, "12345678-1234-1234-1234-1234567890ab");
  });

  it("rejects updates to existing internal reflection entries", async () => {
    const reflectionResolveCalls = [];
    const harness = createHarness({
      store: {
        async resolveByIdOrPrefix(id) {
          reflectionResolveCalls.push(id);
          return {
            id: "12345678-1234-1234-1234-1234567890ab",
            text: "reflection row",
            vector: [0.3, 0.3, 0.3],
            category: "reflection",
            scope: "scope-main",
            importance: 0.8,
            timestamp: 1700000000000,
            metadata: JSON.stringify({ type: "memory-reflection-item" }),
          };
        },
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-protected-existing-reflection", {
      memoryId: "12345678-1234-1234-1234-1234567890ab",
      text: "Do not allow editing this internal row.",
    });

    assert.equal(reflectionResolveCalls.length, 1);
    assert.equal(harness.validatorCalls.length, 0);
    assert.equal(harness.embedCalls.length, 0);
    assert.equal(harness.updateCalls.length, 0);
    assert.match(result.content[0].text, /reserved for internal reflection storage/i);
    assert.equal(result.details.error, "protected_category");
    assert.equal(result.details.category, "reflection");
  });

  it("rejects changing a memory into the reserved reflection category", async () => {
    const harness = createHarness();
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-protected-category", {
      memoryId: "12345678-1234-1234-1234-1234567890ab",
      category: "reflection",
    });

    assert.equal(harness.validatorCalls.length, 0);
    assert.equal(harness.embedCalls.length, 0);
    assert.equal(harness.updateCalls.length, 0);
    assert.match(result.content[0].text, /reserved for internal reflection storage/i);
    assert.equal(result.details.error, "protected_category");
    assert.equal(result.details.category, "reflection");
  });

  it("keeps the legacy fallback path when shared validation is unavailable", async () => {
    const harness = createHarness({
      captureValidator: null,
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-3", {
      memoryId: "12345678-1234-1234-1234-1234567890ab",
      text: "Keep concise direct answers for technical tasks.",
    });

    assert.equal(harness.validatorCalls.length, 0);
    assert.equal(harness.resolveByIdCalls.length, 1);
    assert.deepEqual(harness.embedCalls, ["Keep concise direct answers for technical tasks."]);
    assert.equal(harness.updateCalls.length, 1);
    assert.equal(result.details.action, "updated");
  });
});

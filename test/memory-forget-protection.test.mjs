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

const { registerMemoryForgetTool } = jiti("../src/tools.ts");

function createHarness(overrides = {}) {
  const factories = new Map();
  const deleteCalls = [];
  const resolveByIdCalls = [];
  const retrieveCalls = [];

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
      async resolveByIdOrPrefix(id) {
        resolveByIdCalls.push(id);
        return {
          id,
          text: "editable memory",
          vector: [0.1, 0.2, 0.3],
          category: "other",
          scope: "scope-main",
          importance: 0.7,
          timestamp: 1700000000000,
          metadata: "{}",
        };
      },
      async delete(id, scopes) {
        deleteCalls.push({ id, scopes });
        return true;
      },
      ...(overrides.store || {}),
    },
    scopeManager: {
      getAccessibleScopes() {
        return ["scope-main"];
      },
      isAccessible() {
        return true;
      },
      ...(overrides.scopeManager || {}),
    },
    agentId: overrides.agentId,
  };

  registerMemoryForgetTool(api, context);

  return {
    tool(toolCtx = {}) {
      const factory = factories.get("memory_forget");
      assert.ok(factory, "memory_forget tool should be registered");
      return factory(toolCtx);
    },
    deleteCalls,
    resolveByIdCalls,
    retrieveCalls,
  };
}

describe("memory_forget protection", () => {
  it("rejects direct deletion of internal reflection entries", async () => {
    const reflectionResolveCalls = [];
    const harness = createHarness({
      store: {
        async resolveByIdOrPrefix(id) {
          reflectionResolveCalls.push(id);
          return {
            id: "12345678-1234-1234-1234-1234567890ab",
            text: "reflection row",
            vector: [0.1, 0.2, 0.3],
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

    const result = await tool.execute("tc-protected-id", {
      memoryId: "12345678-1234-1234-1234-1234567890ab",
    });

    assert.equal(reflectionResolveCalls.length, 1);
    assert.equal(harness.deleteCalls.length, 0);
    assert.match(result.content[0].text, /reserved for internal reflection storage/i);
    assert.equal(result.details.error, "protected_category");
    assert.equal(result.details.category, "reflection");
  });

  it("keeps prefix-id deletion support while protecting reflection rows", async () => {
    const resolveCalls = [];
    const harness = createHarness({
      store: {
        async resolveByIdOrPrefix(id) {
          resolveCalls.push(id);
          return {
            id: "12345678-1234-1234-1234-1234567890ab",
            text: "editable memory",
            vector: [0.1, 0.2, 0.3],
            category: "other",
            scope: "scope-main",
            importance: 0.7,
            timestamp: 1700000000000,
            metadata: "{}",
          };
        },
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-prefix-id", {
      memoryId: "12345678",
    });

    assert.deepEqual(resolveCalls, ["12345678"]);
    assert.equal(harness.deleteCalls.length, 1);
    assert.equal(harness.deleteCalls[0].id, "12345678");
    assert.equal(result.details.action, "deleted");
  });

  it("filters reflection rows out of query-based delete candidates", async () => {
    const harness = createHarness({
      retriever: {
        async retrieve(params) {
          harness.retrieveCalls.push(params);
          return [
            {
              entry: {
                id: "reflection-1",
                text: "Reflection invariant row",
                vector: [0.1, 0.2, 0.3],
                category: "reflection",
                scope: "scope-main",
                importance: 0.8,
                timestamp: 1700000000000,
                metadata: JSON.stringify({ type: "memory-reflection-item" }),
              },
              score: 0.95,
              sources: {},
            },
            {
              entry: {
                id: "memory-1",
                text: "User prefers concise answers",
                vector: [0.3, 0.2, 0.1],
                category: "preference",
                scope: "scope-main",
                importance: 0.7,
                timestamp: 1700000001000,
                metadata: "{}",
              },
              score: 0.7,
              sources: {},
            },
          ];
        },
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-query-filter", {
      query: "preference",
    });

    assert.equal(harness.retrieveCalls.length, 1);
    assert.equal(harness.deleteCalls.length, 0);
    assert.match(result.content[0].text, /Found 1 candidates/i);
    assert.match(result.content[0].text, /User prefers concise answers/);
    assert.doesNotMatch(result.content[0].text, /Reflection invariant row/);
    assert.equal(result.details.protectedFiltered, 1);
    assert.equal(result.details.candidates.length, 1);
    assert.equal(result.details.candidates[0].id, "memory-1");
  });

  it("returns no editable matches when a query only finds protected reflection rows", async () => {
    const harness = createHarness({
      retriever: {
        async retrieve(params) {
          harness.retrieveCalls.push(params);
          return [
            {
              entry: {
                id: "reflection-1",
                text: "Reflection invariant row",
                vector: [0.1, 0.2, 0.3],
                category: "reflection",
                scope: "scope-main",
                importance: 0.8,
                timestamp: 1700000000000,
                metadata: JSON.stringify({ type: "memory-reflection-item" }),
              },
              score: 0.95,
              sources: {},
            },
          ];
        },
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-query-protected-only", {
      query: "reflection",
    });

    assert.equal(harness.retrieveCalls.length, 1);
    assert.equal(harness.deleteCalls.length, 0);
    assert.match(result.content[0].text, /No matching editable memories found/i);
    assert.equal(result.details.found, 0);
    assert.equal(result.details.protectedFiltered, 1);
  });
});

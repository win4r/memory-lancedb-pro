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

const { registerMemoryStoreTool } = jiti("../src/tools.ts");

function createHarness(overrides = {}) {
  const factories = new Map();
  const embedCalls = [];
  const storeCalls = [];
  const mirrorCalls = [];
  const ingressCalls = [];

  const api = {
    registerTool(factory, meta) {
      factories.set(meta?.name || "", factory);
    },
  };

  const context = {
    retriever: {},
    store: {
      async store(entry) {
        storeCalls.push(entry);
        return {
          id: "stored-1",
          timestamp: 1700000000000,
          metadata: undefined,
          ...entry,
        };
      },
      async vectorSearch() {
        return [];
      },
      ...(overrides.store || {}),
    },
    scopeManager: {
      getDefaultScope() {
        return "scope-main";
      },
      isAccessible() {
        return true;
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
    workspaceDir: overrides.workspaceDir,
    mdMirror: async (entry, meta) => {
      mirrorCalls.push({ entry, meta });
    },
  };

  if (Object.prototype.hasOwnProperty.call(overrides, "captureIngress")) {
    context.captureIngress = overrides.captureIngress === null
      ? null
      : async (params) => {
        ingressCalls.push(params);
        return await overrides.captureIngress(params);
      };
  } else {
    context.captureIngress = async (params) => {
      ingressCalls.push(params);
      return {
        action: "stored",
        entry: {
          id: "ingress-1",
          text: params.text,
          vector: [],
          category: params.category || "other",
          scope: params.scope,
          importance: params.importance ?? 0.7,
          timestamp: 1700000000000,
          metadata: "{}",
        },
      };
    };
  }

  registerMemoryStoreTool(api, context);

  return {
    tool(toolCtx = {}) {
      const factory = factories.get("memory_store");
      assert.ok(factory, "memory_store tool should be registered");
      return factory(toolCtx);
    },
    embedCalls,
    storeCalls,
    mirrorCalls,
    ingressCalls,
  };
}

describe("memory_store ingress integration", () => {
  it("routes explicit tool writes through shared ingress when available", async () => {
    const harness = createHarness();
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-1", {
      text: "Use zsh for shell-related instructions.",
      importance: 0.9,
      category: "fact",
    });

    assert.equal(harness.ingressCalls.length, 1);
    assert.deepEqual(harness.ingressCalls[0], {
      text: "Use zsh for shell-related instructions.",
      scope: "scope-main",
      importance: 0.9,
      category: "fact",
      sourceApp: "openclaw",
      sourceRole: "assistant",
      scoringRole: "user",
      ingestionMode: "memory_store",
      explicitWrite: true,
      confirmed: true,
      agentId: "agent-main",
    });
    assert.equal(harness.embedCalls.length, 0);
    assert.equal(harness.storeCalls.length, 0);
    assert.match(result.content[0].text, /Stored:/);
    assert.equal(result.details.action, "stored");
    assert.equal(result.details.id, "ingress-1");
  });

  it("surfaces ingress skip reasons instead of falling back to direct store", async () => {
    const harness = createHarness({
      async captureIngress(params) {
        return {
          action: "skipped",
          reason: "system-noise",
        };
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-2", {
      text: "<relevant-memories> internal wrapper",
      category: "other",
    });

    assert.equal(harness.ingressCalls.length, 1);
    assert.equal(harness.embedCalls.length, 0);
    assert.equal(harness.storeCalls.length, 0);
    assert.match(result.content[0].text, /system\/control text/i);
    assert.equal(result.details.action, "skipped");
    assert.equal(result.details.reason, "system-noise");
  });

  it("surfaces refreshed duplicates as updated similar memories", async () => {
    const harness = createHarness({
      async captureIngress(params) {
        return {
          action: "updated",
          entry: {
            id: "existing-1",
            text: params.text,
            vector: [],
            category: "decision",
            scope: params.scope,
            importance: 0.88,
            timestamp: 1700000000000,
            metadata: "{}",
          },
          matchedExisting: {
            id: "existing-1",
            text: "We use zsh for shell tasks.",
            scope: params.scope,
            similarity: 0.991,
          },
        };
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-3", {
      text: "We use zsh for shell tasks.",
      importance: 0.8,
      category: "decision",
    });

    assert.equal(harness.ingressCalls.length, 1);
    assert.match(result.content[0].text, /Updated similar memory:/);
    assert.equal(result.details.action, "updated");
    assert.equal(result.details.id, "existing-1");
    assert.equal(result.details.existingId, "existing-1");
    assert.equal(result.details.similarity, 0.991);
  });

  it("rejects the reserved reflection category before calling ingress", async () => {
    const harness = createHarness();
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-protected-category", {
      text: "Internal reflection row should not be stored manually.",
      category: "reflection",
    });

    assert.equal(harness.ingressCalls.length, 0);
    assert.equal(harness.embedCalls.length, 0);
    assert.equal(harness.storeCalls.length, 0);
    assert.match(result.content[0].text, /reserved for internal reflection storage/i);
    assert.equal(result.details.error, "protected_category");
    assert.equal(result.details.category, "reflection");
  });

  it("keeps the legacy fallback path when shared ingress is unavailable", async () => {
    const fallbackHarness = createHarness({
      captureIngress: null,
    });
    const fallbackTool = fallbackHarness.tool({ agentId: "agent-main" });

    const result = await fallbackTool.execute("tc-4", {
      text: "User prefers concise direct answers for technical tasks.",
      importance: 0.8,
      category: "preference",
    });

    assert.equal(fallbackHarness.ingressCalls.length, 0);
    assert.equal(fallbackHarness.embedCalls.length, 1);
    assert.equal(fallbackHarness.storeCalls.length, 1);
    assert.equal(fallbackHarness.mirrorCalls.length, 1);
    assert.equal(result.details.action, "created");
  });
});

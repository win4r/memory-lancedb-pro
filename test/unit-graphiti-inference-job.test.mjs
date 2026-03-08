import assert from "node:assert/strict";
import test from "node:test";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { createGraphInferenceJob } = jiti("../src/graphiti/inference.ts");

test("graph inference job stores inferred candidates on schedule", async () => {
  const stored = [];
  const syncCalls = [];

  const run = createGraphInferenceJob({
    store: {
      list: async () => [
        {
          id: "m1",
          text: "Alice project tea setup",
          category: "fact",
          scope: "global",
          importance: 0.7,
          timestamp: Date.now(),
          metadata: "{}",
        },
      ],
      vectorSearch: async () => [],
      store: async (entry) => {
        const record = { ...entry, id: `new-${stored.length + 1}`, timestamp: Date.now() };
        stored.push(record);
        return { ...record, metadata: record.metadata || "{}" };
      },
    },
    embedder: {
      embedPassage: async () => [0.1, 0.2, 0.3],
    },
    graphitiBridge: {
      recall: async () => ({
        groupId: "global",
        nodes: [{ label: "Alice" }],
        facts: [{ text: "Alice likes tea" }],
      }),
    },
    graphitiSync: {
      syncMemory: async (_memory, options) => {
        syncCalls.push(options);
        return { status: "stored", groupId: "global" };
      },
    },
    graphitiConfig: {
      enabled: true,
      baseUrl: "http://localhost:8000",
      transport: "mcp",
      groupIdMode: "scope",
      timeoutMs: 1000,
      failOpen: true,
      write: { memoryStore: true, autoCapture: false, sessionSummary: false },
      read: { enableGraphRecallTool: true, augmentMemoryRecall: false, topKNodes: 6, topKFacts: 10 },
      inference: { enabled: true, intervalMs: 60000, maxMemories: 100, minConfidence: 0.6, maxScopes: 5 },
    },
    logger: {},
  });

  const result = await run({ reason: "unit" });
  assert.equal(result.dryRun, false);
  assert.equal(result.stored, 1);
  assert.equal(stored.length, 1);
  assert.match(stored[0].text, /^\[graph-inferred\]/);
  assert.equal(syncCalls.length, 1);
});

test("graph inference job skips when disabled", async () => {
  const run = createGraphInferenceJob({
    store: {
      list: async () => [],
      vectorSearch: async () => [],
      store: async () => null,
    },
    embedder: {
      embedPassage: async () => [0.1],
    },
    graphitiSync: {
      syncMemory: async () => ({ status: "skipped", groupId: "global" }),
    },
    graphitiConfig: {
      enabled: true,
      baseUrl: "http://localhost:8000",
      transport: "mcp",
      groupIdMode: "scope",
      timeoutMs: 1000,
      failOpen: true,
      write: { memoryStore: true, autoCapture: false, sessionSummary: false },
      read: { enableGraphRecallTool: true, augmentMemoryRecall: false, topKNodes: 6, topKFacts: 10 },
      inference: { enabled: false, intervalMs: 60000, maxMemories: 100, minConfidence: 0.6, maxScopes: 5 },
    },
    logger: {},
  });

  const result = await run({ reason: "unit-disabled" });
  assert.equal(result.stored, 0);
  assert.equal(result.scopesScanned, 0);
});

test("graph inference job supports dry-run and scope allow/deny filters", async () => {
  const stored = [];
  const run = createGraphInferenceJob({
    store: {
      list: async () => [
        {
          id: "g1",
          text: "Alice tea notes",
          category: "fact",
          scope: "global",
          importance: 0.7,
          timestamp: Date.now(),
          metadata: "{}",
        },
        {
          id: "e1",
          text: "Experiment scope note",
          category: "fact",
          scope: "agent:experimental",
          importance: 0.7,
          timestamp: Date.now(),
          metadata: "{}",
        },
      ],
      vectorSearch: async () => [],
      store: async (entry) => {
        stored.push(entry);
        return { ...entry, id: `dry-${stored.length}`, timestamp: Date.now(), metadata: entry.metadata || "{}" };
      },
    },
    embedder: {
      embedPassage: async () => [0.1, 0.2, 0.3],
    },
    graphitiBridge: {
      recall: async () => ({
        groupId: "global",
        nodes: [{ label: "Alice" }],
        facts: [{ text: "Alice likes tea" }],
      }),
    },
    graphitiSync: {
      syncMemory: async () => ({ status: "stored", groupId: "global" }),
    },
    graphitiConfig: {
      enabled: true,
      baseUrl: "http://localhost:8000",
      transport: "mcp",
      groupIdMode: "scope",
      timeoutMs: 1000,
      failOpen: true,
      write: { memoryStore: true, autoCapture: false, sessionSummary: false },
      read: { enableGraphRecallTool: true, augmentMemoryRecall: false, topKNodes: 6, topKFacts: 10 },
      inference: {
        enabled: true,
        intervalMs: 60000,
        maxMemories: 100,
        minConfidence: 0.6,
        maxScopes: 5,
        includeScopes: ["global", "agent:experimental"],
        excludeScopes: ["agent:experimental"],
      },
    },
    logger: {},
  });

  const result = await run({ reason: "dry-run", dryRun: true });
  assert.equal(result.dryRun, true);
  assert.equal(result.stored, 1);
  assert.equal(stored.length, 0);
  assert.deepEqual(result.scopeFilterApplied, ["global"]);
});

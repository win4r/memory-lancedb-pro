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

const { registerMemoryRecallTool } = jiti("../src/tools.ts");

function buildResult(id = "memory-1", text = "服务崩溃 error") {
  return {
    entry: {
      id,
      text,
      vector: [0.1, 0.2, 0.3],
      category: "other",
      scope: "scope-main",
      importance: 0.7,
      timestamp: 1700000000000,
      metadata: "{}",
    },
    score: 0.9,
    sources: {
      vector: { score: 0.9, rank: 1 },
      bm25: { score: 0.8, rank: 1 },
      fused: { score: 0.9 },
    },
  };
}

function createHarness(overrides = {}) {
  const factories = new Map();
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
        return [buildResult()];
      },
      getConfig() {
        return { mode: "hybrid" };
      },
      getLastDiagnostics() {
        return {
          source: "manual",
          mode: "hybrid",
          originalQuery: "服务挂了",
          bm25Query: "服务挂了 崩溃 crash error 报错 宕机",
          queryExpanded: true,
          limit: 5,
          scopeFilter: ["scope-main"],
          category: undefined,
          vectorResultCount: 1,
          bm25ResultCount: 1,
          fusedResultCount: 1,
          finalResultCount: 1,
          stageCounts: {
            afterMinScore: 1,
            rerankInput: 1,
            afterRerank: 1,
            afterRecency: 1,
            afterImportance: 1,
            afterLengthNorm: 1,
            afterTimeDecay: 1,
            afterHardMinScore: 1,
            afterNoiseFilter: 1,
            afterDiversity: 1,
          },
          dropSummary: [],
        };
      },
      ...(overrides.retriever || {}),
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

  registerMemoryRecallTool(api, context);

  return {
    tool(toolCtx = {}) {
      const factory = factories.get("memory_recall");
      assert.ok(factory, "memory_recall tool should be registered");
      return factory(toolCtx);
    },
    retrieveCalls,
  };
}

describe("memory_recall diagnostics", () => {
  it("attaches retrieval diagnostics to successful recall details", async () => {
    const harness = createHarness();
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-memory-recall-hit", {
      query: "服务挂了",
      limit: 5,
    });

    assert.equal(harness.retrieveCalls.length, 1);
    assert.equal(harness.retrieveCalls[0].source, "manual");
    assert.equal(result.details.count, 1);
    assert.equal(result.details.limit, 5);
    assert.equal(result.details.category, undefined);
    assert.equal(result.details.retrievalMode, "hybrid");
    assert.deepEqual(result.details.diagnostics, {
      source: "manual",
      mode: "hybrid",
      originalQuery: "服务挂了",
      bm25Query: "服务挂了 崩溃 crash error 报错 宕机",
      queryExpanded: true,
      limit: 5,
      scopeFilter: ["scope-main"],
      category: undefined,
      vectorResultCount: 1,
      bm25ResultCount: 1,
      fusedResultCount: 1,
      finalResultCount: 1,
      stageCounts: {
        afterMinScore: 1,
        rerankInput: 1,
        afterRerank: 1,
        afterRecency: 1,
        afterImportance: 1,
        afterLengthNorm: 1,
        afterTimeDecay: 1,
        afterHardMinScore: 1,
        afterNoiseFilter: 1,
        afterDiversity: 1,
      },
      dropSummary: [],
    });
  });

  it("attaches diagnostics to zero-result recall details", async () => {
    const harness = createHarness({
      retriever: {
        async retrieve() {
          return [];
        },
        getLastDiagnostics() {
          return {
            source: "manual",
            mode: "hybrid",
            originalQuery: "无结果查询",
            bm25Query: "无结果查询",
            queryExpanded: false,
            limit: 3,
            scopeFilter: ["scope-main"],
            category: undefined,
            vectorResultCount: 0,
            bm25ResultCount: 0,
            fusedResultCount: 0,
            finalResultCount: 0,
            stageCounts: {
              afterMinScore: 0,
              rerankInput: 0,
              afterRerank: 0,
              afterRecency: 0,
              afterImportance: 0,
              afterLengthNorm: 0,
              afterTimeDecay: 0,
              afterHardMinScore: 0,
              afterNoiseFilter: 0,
              afterDiversity: 0,
            },
            dropSummary: [],
          };
        },
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-memory-recall-empty", {
      query: "无结果查询",
      limit: 3,
    });

    assert.equal(result.details.count, 0);
    assert.equal(result.details.limit, 3);
    assert.equal(result.details.category, undefined);
    assert.equal(result.details.retrievalMode, "hybrid");
    assert.deepEqual(result.details.diagnostics, {
      source: "manual",
      mode: "hybrid",
      originalQuery: "无结果查询",
      bm25Query: "无结果查询",
      queryExpanded: false,
      limit: 3,
      scopeFilter: ["scope-main"],
      category: undefined,
      vectorResultCount: 0,
      bm25ResultCount: 0,
      fusedResultCount: 0,
      finalResultCount: 0,
      stageCounts: {
        afterMinScore: 0,
        rerankInput: 0,
        afterRerank: 0,
        afterRecency: 0,
        afterImportance: 0,
        afterLengthNorm: 0,
        afterTimeDecay: 0,
        afterHardMinScore: 0,
        afterNoiseFilter: 0,
        afterDiversity: 0,
      },
      dropSummary: [],
    });
    assert.match(result.content[0].text, /No relevant memories found/i);
  });

  it("keeps memory_recall compatible when diagnostics are unavailable", async () => {
    const harness = createHarness({
      retriever: {
        getLastDiagnostics: undefined,
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-memory-recall-no-diag", {
      query: "服务挂了",
    });

    assert.equal(result.details.count, 1);
    assert.equal(result.details.retrievalMode, "hybrid");
    assert.equal("diagnostics" in result.details, false);
  });

  it("attaches failure diagnostics when recall errors after retriever diagnostics are available", async () => {
    const harness = createHarness({
      retriever: {
        async retrieve() {
          throw new Error("simulated retrieval failure");
        },
        getLastDiagnostics() {
          return {
            source: "manual",
            mode: "hybrid",
            originalQuery: "服务挂了",
            bm25Query: "服务挂了 崩溃 crash error 报错 宕机",
            queryExpanded: true,
            limit: 5,
            scopeFilter: ["scope-main"],
            category: undefined,
            vectorResultCount: 0,
            bm25ResultCount: 0,
            fusedResultCount: 0,
            finalResultCount: 0,
            stageCounts: {
              afterMinScore: 0,
              rerankInput: 0,
              afterRerank: 0,
              afterRecency: 0,
              afterImportance: 0,
              afterLengthNorm: 0,
              afterTimeDecay: 0,
              afterHardMinScore: 0,
              afterNoiseFilter: 0,
              afterDiversity: 0,
            },
            dropSummary: [],
            failureStage: "hybrid.embedQuery",
            errorMessage: "simulated retrieval failure",
          };
        },
      },
    });
    const tool = harness.tool({ agentId: "agent-main" });

    const result = await tool.execute("tc-memory-recall-fail-diag", {
      query: "服务挂了",
      limit: 5,
    });

    assert.equal(result.details.error, "recall_failed");
    assert.equal(result.details.query, "服务挂了");
    assert.equal(result.details.limit, 5);
    assert.equal(result.details.category, undefined);
    assert.deepEqual(result.details.scopes, ["scope-main"]);
    assert.equal(result.details.retrievalMode, "hybrid");
    assert.equal(result.details.diagnostics.failureStage, "hybrid.embedQuery");
    assert.equal(
      result.details.diagnostics.errorMessage,
      "simulated retrieval failure",
    );
    assert.match(result.content[0].text, /simulated retrieval failure/i);
  });
});

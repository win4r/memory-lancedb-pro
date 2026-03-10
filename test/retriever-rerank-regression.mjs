import assert from "node:assert/strict";

import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const { createRetriever, DEFAULT_RETRIEVAL_CONFIG } = jiti("../src/retriever.ts");

const entry = {
  id: "rerank-regression-1",
  text: "OpenClaw 记忆插件集成测试 token: TESTMEM-20260306-092541，仅用于验证 import/search/delete 闭环。",
  vector: [0, 1],
  category: "decision",
  scope: "global",
  importance: 0.91,
  timestamp: Date.now(),
  metadata: "{}",
};

const fakeStore = {
  hasFtsSupport: true,
  async vectorSearch() {
    return [{ entry, score: 0.5438692121765099 }];
  },
  async bm25Search() {
    return [{ entry, score: 0.7833663291840794 }];
  },
  async hasId(id) {
    return id === entry.id;
  },
};

const fakeEmbedder = {
  async embedQuery() {
    return [1, 0];
  },
};

const retrieverConfig = {
  ...DEFAULT_RETRIEVAL_CONFIG,
  filterNoise: false,
  rerank: "cross-encoder",
  rerankApiKey: "test-key",
  rerankProvider: "jina",
  rerankEndpoint: "http://127.0.0.1:9/v1/rerank",
  rerankModel: "test-reranker",
  candidatePoolSize: 12,
  minScore: 0.6,
  hardMinScore: 0.62,
};

async function runScenario(name, responsePayload) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    async json() {
      return responsePayload;
    },
  });

  try {
    const retriever = createRetriever(fakeStore, fakeEmbedder, retrieverConfig);
    const results = await retriever.retrieve({
      query: "TESTMEM-20260306-092541",
      limit: 5,
      scopeFilter: ["global"],
    });

    assert.equal(
      results.length,
      1,
      `${name}: strong BM25 exact-match result should survive rerank`,
    );
    assert.equal(results[0].entry.id, entry.id, `${name}: wrong memory returned`);
    assert.ok(results[0].score >= retrieverConfig.hardMinScore, `${name}: score dropped below hardMinScore`);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

await runScenario("low-score rerank result", {
  results: [{ index: 0, relevance_score: 0 }],
});

await runScenario("reranker omitted candidate", {
  results: [{ index: 1, relevance_score: 0.9 }],
});

console.log("OK: rerank regression test passed");

const lexicalEntry = {
  id: "lexical-regression-1",
  text: "用户测试饮料偏好是乌龙茶，不喜欢美式咖啡。",
  vector: [0, 1],
  category: "preference",
  scope: "global",
  importance: 0.95,
  timestamp: Date.now(),
  metadata: "{}",
};

const lexicalStore = {
  hasFtsSupport: true,
  async vectorSearch() {
    return [{ entry: lexicalEntry, score: 0.5006586036313858 }];
  },
  async bm25Search() {
    return [{ entry: lexicalEntry, score: 0.78 }];
  },
  async hasId(id) {
    return id === lexicalEntry.id;
  },
};

const lexicalRetriever = createRetriever(lexicalStore, fakeEmbedder, {
  ...DEFAULT_RETRIEVAL_CONFIG,
  filterNoise: false,
  rerank: "none",
  vectorWeight: 0.7,
  bm25Weight: 0.3,
  minScore: 0.6,
  hardMinScore: 0.62,
});

const lexicalResults = await lexicalRetriever.retrieve({
  query: "乌龙茶",
  limit: 5,
  scopeFilter: ["global"],
});

assert.equal(lexicalResults.length, 1, "strong lexical hit should survive hybrid fusion thresholds");
assert.equal(lexicalResults[0].entry.id, lexicalEntry.id);

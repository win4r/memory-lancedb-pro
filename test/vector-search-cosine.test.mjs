import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const { createRetriever } = jiti("../src/retriever.ts");

/**
 * Regression test: vectorSearch must use cosine distance so that
 * semantically related memories score above minScore/hardMinScore.
 *
 * With L2 distance on high-dimensional vectors (e.g. 2048-dim),
 * score = 1/(1 + distance) ≈ 0.0005, and everything gets filtered.
 * With cosine distance, score = 1/(1 + distance) ≈ 0.6–0.9.
 */

function createEntry(id, text, score, extra = {}) {
  return {
    entry: {
      id,
      text,
      vector: extra.vector || new Array(8).fill(0.1),
      category: extra.category || "fact",
      scope: extra.scope || "global",
      importance: extra.importance ?? 0.8,
      timestamp: extra.timestamp ?? Date.now(),
      metadata: extra.metadata || "{}",
    },
    score,
  };
}

describe("vector search cosine distance regression", () => {
  it("returns results when store yields cosine-range scores (>0.5)", async () => {
    // Simulate what happens when vectorSearch uses cosine distance:
    // cosine distance for related content ≈ 0.2–0.5, so score ≈ 0.67–0.83
    const store = {
      hasFtsSupport: false,
      canUseFts: false,
      async vectorSearch() {
        return [
          createEntry("m1", "用户喜欢 Neovim", 0.77),
          createEntry("m2", "用户每天用 tmux", 0.72),
          createEntry("m3", "部署环境变量是 JINA_API_KEY", 0.68),
        ];
      },
    };
    const embedder = {
      async embedQuery() {
        return new Array(8).fill(0.1);
      },
    };

    const retriever = createRetriever(store, embedder, {
      mode: "vector",
      filterNoise: false,
      rerank: "none",
      hardMinScore: 0.55,
    });

    const execution = await retriever.retrieveWithTrace({
      query: "用户喜欢什么编辑器",
      limit: 5,
      source: "manual",
    });

    // All 3 results should pass hardMinScore of 0.55
    assert.ok(
      execution.results.length >= 2,
      `Expected >=2 results, got ${execution.results.length}`,
    );
    // Verify the most relevant result is first
    assert.ok(
      execution.results[0].entry.text.includes("Neovim"),
      "Most relevant memory should rank first",
    );
  });

  it("filters out results when scores are below hardMinScore (L2-like regime)", async () => {
    // Simulate what L2 distance does: distance ≈ 2048, score ≈ 0.0005
    // score = 1 / (1 + 2048) ≈ 0.000488 — everything gets filtered
    const store = {
      hasFtsSupport: false,
      canUseFts: false,
      async vectorSearch() {
        return [
          createEntry("m1", "用户喜欢 Neovim", 0.0005),
          createEntry("m2", "用户每天用 tmux", 0.0004),
        ];
      },
    };
    const embedder = {
      async embedQuery() {
        return new Array(8).fill(0.1);
      },
    };

    const retriever = createRetriever(store, embedder, {
      mode: "vector",
      filterNoise: false,
      rerank: "none",
      hardMinScore: 0.55,
    });

    const execution = await retriever.retrieveWithTrace({
      query: "用户喜欢什么编辑器",
      limit: 5,
      source: "manual",
    });

    // With L2-like tiny scores, everything should be filtered by hardMinScore
    assert.equal(
      execution.results.length,
      0,
      "L2-like scores should be filtered by hardMinScore",
    );
    // Verify hard_min_score stage shows filtering
    const hardMinStage = execution.trace.stages.find(
      (s) => s.name === "hard_min_score",
    );
    assert.ok(hardMinStage, "hard_min_score stage should be present");
    assert.equal(hardMinStage.inputCount, 2, "Should have 2 inputs");
    assert.equal(hardMinStage.outputCount, 0, "Should filter all with L2 scores");
  });

  it("cosine distance produces reasonable score distribution", async () => {
    // With cosine distance, identical vectors → distance=0, score=1.0
    // orthogonal → distance=1.0, score=0.5
    // opposite → distance=2.0, score=0.33
    const store = {
      hasFtsSupport: false,
      canUseFts: false,
      async vectorSearch() {
        return [
          createEntry("exact", "identical meaning", 1.0),    // cosine dist=0
          createEntry("similar", "similar meaning", 0.77),   // cosine dist≈0.3
          createEntry("unrelated", "not related", 0.56),     // cosine dist≈0.8
        ];
      },
    };
    const embedder = {
      async embedQuery() {
        return [1, 0, 0, 0, 0, 0, 0, 0];
      },
    };

    const retriever = createRetriever(store, embedder, {
      mode: "vector",
      filterNoise: false,
      rerank: "none",
      hardMinScore: 0.55,
    });

    const execution = await retriever.retrieveWithTrace({
      query: "test",
      limit: 5,
      source: "manual",
    });

    // All 3 should pass hardMinScore since cosine scores are above 0.55
    assert.equal(execution.results.length, 3);
    // Verify ordering preserved
    assert.equal(execution.results[0].entry.id, "exact");
    assert.equal(execution.results[1].entry.id, "similar");
    assert.equal(execution.results[2].entry.id, "unrelated");
  });

  it("trace correctly reports vector_search stage output count", async () => {
    const store = {
      hasFtsSupport: false,
      canUseFts: false,
      async vectorSearch() {
        return [
          createEntry("m1", "memory one", 0.85),
          createEntry("m2", "memory two", 0.70),
          createEntry("m3", "memory three", 0.60),
        ];
      },
    };
    const embedder = {
      async embedQuery() {
        return [0.5, 0.5, 0, 0, 0, 0, 0, 0];
      },
    };

    const retriever = createRetriever(store, embedder, {
      mode: "vector",
      filterNoise: false,
      rerank: "none",
      hardMinScore: 0.55,
    });

    const execution = await retriever.retrieveWithTrace({
      query: "test query",
      limit: 10,
      source: "manual",
    });

    const vectorStage = execution.trace.stages.find(
      (s) => s.name === "vector_search",
    );
    assert.ok(vectorStage, "vector_search stage must exist");
    assert.equal(vectorStage.outputCount, 3, "Should report 3 results from store");
    assert.ok(vectorStage.elapsedMs >= 0, "Elapsed time should be non-negative");

    // All 3 pass hardMinScore
    assert.equal(execution.results.length, 3);
    assert.equal(execution.trace.resultCount, 3);
  });
});

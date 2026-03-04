import assert from "node:assert/strict";
import test from "node:test";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { createGraphitiBridge } = jiti("../src/graphiti/bridge.ts");

test("graphiti bridge parses structuredContent.result envelopes for recall", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (_url, init) => {
    const rawBody = typeof init?.body === "string" ? init.body : "{}";
    const payload = JSON.parse(rawBody);
    const method = payload?.method;

    if (method === "initialize") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: { protocolVersion: "2025-03-26", capabilities: {} },
        }),
        {
          status: 200,
          headers: { "content-type": "application/json", "mcp-session-id": "session-1" },
        },
      );
    }

    if (method === "notifications/initialized") {
      return new Response("", { status: 200 });
    }

    if (method === "tools/list") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: {
            tools: [
              { name: "search_nodes" },
              { name: "search_memory_facts" },
            ],
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }

    if (method === "tools/call" && payload?.params?.name === "search_nodes") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: {
            structuredContent: {
              result: {
                nodes: [{ id: "node-1", label: "Alice" }],
              },
            },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }

    if (method === "tools/call" && payload?.params?.name === "search_memory_facts") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: {
            structuredContent: {
              result: {
                facts: [{ id: "fact-1", fact: "Alice likes tea" }],
              },
            },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({
        jsonrpc: "2.0",
        id: payload.id,
        error: { code: -32601, message: `Unhandled method: ${String(method)}` },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };

  try {
    const bridge = createGraphitiBridge({
      config: {
        enabled: true,
        baseUrl: "http://127.0.0.1:8001",
        transport: "mcp",
        groupIdMode: "scope",
        timeoutMs: 2000,
        failOpen: true,
        write: {
          memoryStore: true,
          autoCapture: false,
          sessionSummary: false,
        },
        read: {
          enableGraphRecallTool: true,
          augmentMemoryRecall: false,
          topKNodes: 6,
          topKFacts: 10,
        },
      },
    });

    const result = await bridge.recall({
      scope: "global",
      query: "alice",
      limitNodes: 6,
      limitFacts: 10,
    });

    assert.equal(result.nodes.length, 1);
    assert.equal(result.facts.length, 1);
    assert.equal(result.nodes[0].label, "Alice");
    assert.equal(result.facts[0].text, "Alice likes tea");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

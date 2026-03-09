import assert from "node:assert/strict";
import test from "node:test";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { GraphitiMcpClient } = jiti("../src/graphiti/mcp.ts");

function normalizeHeaders(input) {
  const output = {};
  if (!input) return output;

  if (typeof input.entries === "function") {
    for (const [key, value] of input.entries()) {
      output[String(key).toLowerCase()] = String(value);
    }
    return output;
  }

  if (Array.isArray(input)) {
    for (const [key, value] of input) {
      output[String(key).toLowerCase()] = String(value);
    }
    return output;
  }

  if (typeof input === "object") {
    for (const [key, value] of Object.entries(input)) {
      output[String(key).toLowerCase()] = String(value);
    }
  }
  return output;
}

test("graphiti MCP client initializes session and applies auth/session headers", async () => {
  const originalFetch = globalThis.fetch;
  const previousEnv = process.env.GRAPHITI_TOKEN_TEST;
  const seenMethods = [];

  process.env.GRAPHITI_TOKEN_TEST = "secret-token";

  globalThis.fetch = async (_url, init) => {
    const rawBody = typeof init?.body === "string" ? init.body : "{}";
    const payload = JSON.parse(rawBody);
    const headers = normalizeHeaders(init?.headers);
    const method = payload?.method;
    seenMethods.push(method);

    assert.equal(headers.authorization, "Bearer secret-token");
    assert.equal(headers.accept, "application/json, text/event-stream");

    if (method === "initialize") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: { protocolVersion: "2025-03-26", capabilities: {} },
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "mcp-session-id": "session-abc",
          },
        },
      );
    }

    if (method === "notifications/initialized") {
      assert.equal(headers["mcp-session-id"], "session-abc");
      return new Response("", { status: 202 });
    }

    if (method === "tools/list") {
      assert.equal(headers["mcp-session-id"], "session-abc");
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: {
            tools: [{ name: "search_nodes" }],
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }

    if (method === "tools/call") {
      assert.equal(headers["mcp-session-id"], "session-abc");
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: { ok: true },
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
    const client = new GraphitiMcpClient({
      baseUrl: "http://127.0.0.1:8000",
      timeoutMs: 1000,
      transport: "mcp",
      auth: {
        tokenEnv: "GRAPHITI_TOKEN_TEST",
        headerName: "authorization",
      },
    });

    const tools = await client.discoverTools();
    assert.equal(tools.length, 1);

    const result = await client.callTool("search_nodes", { query: "alice" });
    assert.deepEqual(result, { ok: true });

    assert.deepEqual(seenMethods, [
      "initialize",
      "notifications/initialized",
      "tools/list",
      "tools/call",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    if (previousEnv === undefined) {
      delete process.env.GRAPHITI_TOKEN_TEST;
    } else {
      process.env.GRAPHITI_TOKEN_TEST = previousEnv;
    }
  }
});

test("graphiti MCP client parses SSE-wrapped JSON-RPC responses", async () => {
  const originalFetch = globalThis.fetch;
  const seenMethods = [];

  globalThis.fetch = async (_url, init) => {
    const rawBody = typeof init?.body === "string" ? init.body : "{}";
    const payload = JSON.parse(rawBody);
    const headers = normalizeHeaders(init?.headers);
    const method = payload?.method;
    seenMethods.push(method);

    assert.equal(headers.accept, "application/json, text/event-stream");

    if (method === "initialize") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: { protocolVersion: "2025-03-26", capabilities: {} },
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "mcp-session-id": "session-sse",
          },
        },
      );
    }

    if (method === "notifications/initialized") {
      return new Response("", { status: 202 });
    }

    if (method === "tools/list") {
      return new Response(
        [
          "event: message",
          `data: ${JSON.stringify({
            jsonrpc: "2.0",
            id: payload.id,
            result: {
              tools: [{ name: "search_nodes" }, { name: "search_memory_facts" }],
            },
          })}`,
          "",
        ].join("\n"),
        {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        },
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
    const client = new GraphitiMcpClient({
      baseUrl: "http://127.0.0.1:8001",
      timeoutMs: 1000,
      transport: "mcp",
    });

    const tools = await client.discoverTools();
    assert.deepEqual(tools.map((tool) => tool.name), ["search_nodes", "search_memory_facts"]);
    assert.deepEqual(seenMethods, ["initialize", "notifications/initialized", "tools/list"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("graphiti MCP client tolerates unsupported notifications/initialized", async () => {
  const originalFetch = globalThis.fetch;
  const seenMethods = [];

  globalThis.fetch = async (_url, init) => {
    const rawBody = typeof init?.body === "string" ? init.body : "{}";
    const payload = JSON.parse(rawBody);
    const method = payload?.method;
    seenMethods.push(method);

    if (method === "initialize") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: { protocolVersion: "2025-03-26", capabilities: {} },
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "mcp-session-id": "session-no-init",
          },
        },
      );
    }

    if (method === "notifications/initialized") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          error: { code: -32602, message: "Invalid request parameters" },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }

    if (method === "tools/list") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: payload.id,
          result: {
            tools: [{ name: "search_nodes" }],
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
    const client = new GraphitiMcpClient({
      baseUrl: "http://127.0.0.1:8001",
      timeoutMs: 1000,
      transport: "mcp",
    });

    const tools = await client.discoverTools();
    assert.deepEqual(tools.map((tool) => tool.name), ["search_nodes"]);
    assert.deepEqual(seenMethods, ["initialize", "notifications/initialized", "tools/list"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

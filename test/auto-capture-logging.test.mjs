import assert from "node:assert/strict";
import http from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const plugin = jiti("../index.ts");

const EMBEDDING_DIMENSIONS = 64;

function createDeterministicEmbedding(dimensions = EMBEDDING_DIMENSIONS) {
  const value = 1 / Math.sqrt(dimensions);
  return new Array(dimensions).fill(value);
}

function createEmbeddingServer() {
  return http.createServer(async (req, res) => {
    if (req.method !== "POST" || req.url !== "/v1/embeddings") {
      res.writeHead(404);
      res.end();
      return;
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    const inputs = Array.isArray(payload.input) ? payload.input : [payload.input];
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      object: "list",
      data: inputs.map((_, index) => ({
        object: "embedding",
        index,
        embedding: createDeterministicEmbedding(),
      })),
      model: payload.model || "mock-model",
      usage: { prompt_tokens: 0, total_tokens: 0 },
    }));
  });
}

function createMockApi(dbPath, embeddingBaseURL, logs) {
  return {
    pluginConfig: {
      dbPath,
      autoCapture: true,
      autoRecall: false,
      smartExtraction: false,
      embedding: {
        apiKey: "dummy",
        model: "mock-model",
        baseURL: embeddingBaseURL,
        dimensions: EMBEDDING_DIMENSIONS,
      },
      retrieval: { mode: "hybrid" },
      scopes: {
        default: "global",
        definitions: { global: { description: "shared" } },
      },
    },
    hooks: {},
    toolFactories: {},
    services: [],
    logger: {
      info(...args) { logs.push(["info", args.join(" ")]); },
      warn(...args) { logs.push(["warn", args.join(" ")]); },
      error(...args) { logs.push(["error", args.join(" ")]); },
      debug(...args) { logs.push(["debug", args.join(" ")]); },
    },
    resolvePath(v) { return v; },
    registerTool(toolOrFactory, meta) {
      this.toolFactories[meta.name] =
        typeof toolOrFactory === "function" ? toolOrFactory : () => toolOrFactory;
    },
    registerCli() {},
    registerService(service) { this.services.push(service); },
    on(name, handler) { this.hooks[name] = handler; },
    registerHook(name, handler) { this.hooks[name] = handler; },
  };
}

async function withTestEnv(fn) {
  const workDir = mkdtempSync(path.join(tmpdir(), "autocap-log-test-"));
  const dbPath = path.join(workDir, "test.db");
  const embeddingServer = createEmbeddingServer();
  await new Promise((resolve) => embeddingServer.listen(0, "127.0.0.1", resolve));
  const embeddingPort = embeddingServer.address().port;
  const logs = [];
  const api = createMockApi(dbPath, `http://127.0.0.1:${embeddingPort}/v1`, logs);

  try {
    plugin.register(api);
    await fn(api, logs);
  } finally {
    await new Promise((resolve) => embeddingServer.close(resolve));
    rmSync(workDir, { recursive: true, force: true });
  }
}

// ============================================================================
// Part 1: Silent failure fix tests
// ============================================================================

// Test 1: Empty messages — should log info, not be silent
await withTestEnv(async (api, logs) => {
  logs.length = 0;
  await api.hooks.agent_end(
    { success: true, messages: [] },
    { agentId: "main", sessionKey: "s1" },
  );

  const infoLogs = logs.filter(([level]) => level === "info").map(([, msg]) => msg);
  assert.ok(
    infoLogs.some((msg) => msg.includes("no messages")),
    `empty messages should produce info log with "no messages", got: ${JSON.stringify(infoLogs)}`,
  );
});
console.log("OK: Test 1 — empty messages logs info");

// Test 2: Failed event — should log info, not be silent
await withTestEnv(async (api, logs) => {
  logs.length = 0;
  await api.hooks.agent_end(
    { success: false, messages: [{ role: "user", content: "hello" }] },
    { agentId: "main", sessionKey: "s2" },
  );

  const infoLogs = logs.filter(([level]) => level === "info").map(([, msg]) => msg);
  assert.ok(
    infoLogs.some((msg) => msg.includes("agent did not succeed")),
    `failed event should produce info log with "agent did not succeed", got: ${JSON.stringify(infoLogs)}`,
  );
});
console.log("OK: Test 2 — failed event logs info");

// ============================================================================
// Part 2: Log level upgrade tests
// ============================================================================

// Test 3: All texts normalized to empty — should log INFO (not debug)
await withTestEnv(async (api, logs) => {
  logs.length = 0;
  await api.hooks.agent_end(
    {
      success: true,
      messages: [
        { role: "user", content: "<relevant-memories>cached data</relevant-memories>" },
      ],
    },
    { agentId: "main", sessionKey: "s3" },
  );

  const infoLogs = logs.filter(([level]) => level === "info").map(([, msg]) => msg);
  assert.ok(
    infoLogs.some((msg) => msg.includes("no eligible texts")),
    `all-filtered texts should produce info log with "no eligible texts", got: ${JSON.stringify(infoLogs)}`,
  );
});
console.log("OK: Test 3 — no eligible texts logs info (not debug)");

// ============================================================================
// Tests 4-6: Smart extraction paths (requires smartExtraction: true + mock LLM)
// ============================================================================

function createMockLlmServer(responseFactory) {
  return http.createServer(async (req, res) => {
    if (req.method !== "POST") {
      res.writeHead(404);
      res.end();
      return;
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    const response = responseFactory(payload);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(response));
  });
}

function createMockApiWithSmartExtraction(dbPath, embeddingBaseURL, llmBaseURL, logs) {
  return {
    pluginConfig: {
      dbPath,
      autoCapture: true,
      autoRecall: false,
      smartExtraction: true,
      extractMinMessages: 2,
      embedding: {
        apiKey: "dummy",
        model: "mock-model",
        baseURL: embeddingBaseURL,
        dimensions: EMBEDDING_DIMENSIONS,
      },
      llm: {
        apiKey: "dummy",
        model: "mock-llm",
        baseURL: llmBaseURL,
      },
      retrieval: { mode: "hybrid" },
      scopes: {
        default: "global",
        definitions: { global: { description: "shared" } },
      },
    },
    hooks: {},
    toolFactories: {},
    services: [],
    logger: {
      info(...args) { logs.push(["info", args.join(" ")]); },
      warn(...args) { logs.push(["warn", args.join(" ")]); },
      error(...args) { logs.push(["error", args.join(" ")]); },
      debug(...args) { logs.push(["debug", args.join(" ")]); },
    },
    resolvePath(v) { return v; },
    registerTool(toolOrFactory, meta) {
      this.toolFactories[meta.name] =
        typeof toolOrFactory === "function" ? toolOrFactory : () => toolOrFactory;
    },
    registerCli() {},
    registerService(service) { this.services.push(service); },
    on(name, handler) { this.hooks[name] = handler; },
    registerHook(name, handler) { this.hooks[name] = handler; },
  };
}

async function withSmartTestEnv(llmResponseFactory, fn) {
  const workDir = mkdtempSync(path.join(tmpdir(), "autocap-smart-test-"));
  const dbPath = path.join(workDir, "test.db");
  const embeddingServer = createEmbeddingServer();
  const llmServer = createMockLlmServer(llmResponseFactory);
  await new Promise((resolve) => embeddingServer.listen(0, "127.0.0.1", resolve));
  await new Promise((resolve) => llmServer.listen(0, "127.0.0.1", resolve));
  const embeddingPort = embeddingServer.address().port;
  const llmPort = llmServer.address().port;
  const logs = [];
  const api = createMockApiWithSmartExtraction(
    dbPath,
    `http://127.0.0.1:${embeddingPort}/v1`,
    `http://127.0.0.1:${llmPort}`,
    logs,
  );

  try {
    plugin.register(api);
    await fn(api, logs);
  } finally {
    await new Promise((resolve) => embeddingServer.close(resolve));
    await new Promise((resolve) => llmServer.close(resolve));
    rmSync(workDir, { recursive: true, force: true });
  }
}

// Test 5: Smart extraction success — outcome=smart-extracted
await withSmartTestEnv(
  (payload) => {
    const prompt = payload.messages?.map((m) => m.content).join(" ") || "";
    let content;
    if (prompt.includes("Analyze the following") || prompt.includes("extract memories")) {
      content = JSON.stringify({
        memories: [{
          category: "preferences",
          abstract: "User prefers dark mode for all applications",
          overview: "## Preference\n- Dark mode",
          content: "User always prefers dark mode in every app.",
        }],
      });
    } else if (prompt.includes("Determine how to handle")) {
      content = JSON.stringify({ decision: "create", reason: "New preference" });
    } else if (prompt.includes("Merge the following")) {
      content = JSON.stringify({
        abstract: "User prefers dark mode",
        overview: "## Preference\n- Dark mode",
        content: "User always prefers dark mode in every app.",
      });
    } else {
      content = JSON.stringify({ memories: [] });
    }
    return {
      id: "chatcmpl-test", object: "chat.completion",
      created: Math.floor(Date.now() / 1000), model: "mock",
      choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    };
  },
  async (api, logs) => {
    logs.length = 0;
    await api.hooks.agent_end(
      {
        success: true,
        messages: [
          { role: "user", content: "I always prefer dark mode in every app I use" },
          { role: "user", content: "Please remember this preference for me" },
        ],
      },
      { agentId: "main", sessionKey: "s5" },
    );

    const infoLogs = logs.filter(([level]) => level === "info").map(([, msg]) => msg);
    assert.ok(
      infoLogs.some((msg) => msg.includes("smart-extracted")),
      `smart extraction success should produce log with "smart-extracted", got: ${JSON.stringify(infoLogs)}`,
    );
  },
);
console.log("OK: Test 5 — smart extraction success logs outcome");

// Test 6: Smart extraction empty → regex fallback
await withSmartTestEnv(
  () => ({
    id: "chatcmpl-test", object: "chat.completion",
    created: Math.floor(Date.now() / 1000), model: "mock",
    choices: [{
      index: 0,
      message: { role: "assistant", content: JSON.stringify({ memories: [] }) },
      finish_reason: "stop",
    }],
  }),
  async (api, logs) => {
    logs.length = 0;
    await api.hooks.agent_end(
      {
        success: true,
        messages: [
          { role: "user", content: "I always prefer oolong tea over coffee" },
          { role: "user", content: "Remember my preference please" },
        ],
      },
      { agentId: "main", sessionKey: "s6" },
    );

    const infoLogs = logs.filter(([level]) => level === "info").map(([, msg]) => msg);
    assert.ok(
      infoLogs.some((msg) => msg.includes("falling back to regex")),
      `smart extraction empty should log "falling back to regex", got: ${JSON.stringify(infoLogs)}`,
    );
    // Final outcome should be regex-captured or regex-no-match (not smart-extraction-empty→regex)
    assert.ok(
      infoLogs.some((msg) => msg.includes("pipeline result") && (msg.includes("regex-captured") || msg.includes("regex-no-match"))),
      `fallback to regex should produce a regex outcome in pipeline summary, got: ${JSON.stringify(infoLogs)}`,
    );
  },
);
console.log("OK: Test 6 — smart extraction empty falls back to regex");

// ============================================================================
// Part 3: Pipeline summary tests
// ============================================================================

// Test 7: Normal regex capture — summary log with outcome
await withTestEnv(async (api, logs) => {
  logs.length = 0;
  await api.hooks.agent_end(
    {
      success: true,
      messages: [
        { role: "user", content: "I prefer dark mode for all applications" },
      ],
    },
    { agentId: "main", sessionKey: "s7" },
  );

  const infoLogs = logs.filter(([level]) => level === "info").map(([, msg]) => msg);
  assert.ok(
    infoLogs.some((msg) => msg.includes("outcome=") && msg.includes("pipeline result")),
    `regex capture should produce pipeline summary with outcome=, got: ${JSON.stringify(infoLogs)}`,
  );
  assert.ok(
    infoLogs.some((msg) => msg.includes("regex-captured")),
    `regex capture outcome should be "regex-captured", got: ${JSON.stringify(infoLogs)}`,
  );
});
console.log("OK: Test 7 — regex capture produces pipeline summary");

// Test 8: Regex no match — summary log with outcome
await withTestEnv(async (api, logs) => {
  logs.length = 0;
  await api.hooks.agent_end(
    {
      success: true,
      messages: [
        { role: "user", content: "just a normal conversation about weather today" },
      ],
    },
    { agentId: "main", sessionKey: "s8" },
  );

  const infoLogs = logs.filter(([level]) => level === "info").map(([, msg]) => msg);
  assert.ok(
    infoLogs.some((msg) => msg.includes("outcome=") && msg.includes("pipeline result")),
    `regex no-match should produce pipeline summary, got: ${JSON.stringify(infoLogs)}`,
  );
  assert.ok(
    infoLogs.some((msg) => msg.includes("regex-no-match")),
    `no-match outcome should be "regex-no-match", got: ${JSON.stringify(infoLogs)}`,
  );
});
console.log("OK: Test 8 — regex no-match produces pipeline summary");

// Test 9: Duplicate skip — debug log for duplicate + outcome
await withTestEnv(async (api, logs) => {
  // First: store a memory so there's a duplicate
  await api.hooks.agent_end(
    {
      success: true,
      messages: [
        { role: "user", content: "I prefer dark mode for all applications" },
      ],
    },
    { agentId: "main", sessionKey: "s9a" },
  );

  // Second: same text should be detected as duplicate
  logs.length = 0;
  await api.hooks.agent_end(
    {
      success: true,
      messages: [
        { role: "user", content: "I prefer dark mode for all applications" },
      ],
    },
    { agentId: "main", sessionKey: "s9b" },
  );

  const debugLogs = logs.filter(([level]) => level === "debug").map(([, msg]) => msg);
  assert.ok(
    debugLogs.some((msg) => msg.includes("skipped duplicate")),
    `duplicate skip should produce debug log with "skipped duplicate", got: ${JSON.stringify(debugLogs.slice(-5))}`,
  );
});
console.log("OK: Test 9 — duplicate skip logs debug");

// Test 10: Pipeline summary format contains required fields
await withTestEnv(async (api, logs) => {
  logs.length = 0;
  await api.hooks.agent_end(
    {
      success: true,
      messages: [
        { role: "user", content: "I prefer dark mode for all applications" },
      ],
    },
    { agentId: "main", sessionKey: "s10" },
  );

  const summaryLogs = logs
    .filter(([level]) => level === "info")
    .map(([, msg]) => msg)
    .filter((msg) => msg.includes("pipeline result"));

  assert.ok(summaryLogs.length > 0, "pipeline summary log must exist");

  const summary = summaryLogs[0];
  assert.ok(summary.includes("outcome="), `summary must contain "outcome=": ${summary}`);
  assert.ok(summary.includes("messages="), `summary must contain "messages=": ${summary}`);
  assert.ok(summary.includes("eligible="), `summary must contain "eligible=": ${summary}`);
  assert.ok(summary.includes("texts="), `summary must contain "texts=": ${summary}`);
});
console.log("OK: Test 10 — pipeline summary format correct");

console.log("\nAll auto-capture logging tests passed!");

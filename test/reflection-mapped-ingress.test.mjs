import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
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

const pluginModule = jiti("../index.ts");
const memoryLanceDBProPlugin = pluginModule.default || pluginModule;
const { MemoryStore } = jiti("../src/store.ts");

function messageLine(role, text, ts) {
  return JSON.stringify({
    type: "message",
    timestamp: ts,
    message: {
      role,
      content: [{ type: "text", text }],
    },
  });
}

function createPluginApiHarness({ pluginConfig, resolveRoot, runtimeConfig }) {
  const eventHandlers = new Map();
  const commandHooks = new Map();
  const logs = [];

  const api = {
    pluginConfig,
    config: runtimeConfig || {},
    resolvePath(target) {
      if (typeof target !== "string") return target;
      if (path.isAbsolute(target)) return target;
      return path.join(resolveRoot, target);
    },
    logger: {
      info(message) {
        logs.push({ level: "info", message: String(message) });
      },
      warn(message) {
        logs.push({ level: "warn", message: String(message) });
      },
      debug(message) {
        logs.push({ level: "debug", message: String(message) });
      },
    },
    registerTool() {},
    registerCli() {},
    registerService() {},
    on(eventName, handler, meta) {
      const list = eventHandlers.get(eventName) || [];
      list.push({ handler, meta });
      eventHandlers.set(eventName, list);
    },
    registerHook(hookName, handler, meta) {
      const list = commandHooks.get(hookName) || [];
      list.push({ handler, meta });
      commandHooks.set(hookName, list);
    },
  };

  return {
    api,
    eventHandlers,
    commandHooks,
    logs,
  };
}

function buildDeterministicEmbedding(text) {
  let seed = 0;
  for (let i = 0; i < text.length; i += 1) {
    seed = (seed * 131 + text.charCodeAt(i)) >>> 0;
  }
  const vector = [];
  let state = seed || 1;
  for (let i = 0; i < 1024; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    vector.push(((state / 0xffffffff) * 2) - 1);
  }
  return vector;
}

describe("reflection mapped ingress integration", () => {
  let sourceWorkspaceDir;
  let reflectionWorkspaceDir;
  let runtimeWorkspaceDir;
  let dbPath;
  let sessionFile;
  let extensionStubFile;
  let embedServer;
  let embedBaseURL;
  let originalExtensionApiPath;
  let runtimeConfig;
  let harness;

  beforeEach(async () => {
    sourceWorkspaceDir = mkdtempSync(path.join(tmpdir(), "reflection-mapped-source-"));
    reflectionWorkspaceDir = mkdtempSync(path.join(tmpdir(), "reflection-mapped-runner-"));
    runtimeWorkspaceDir = mkdtempSync(path.join(tmpdir(), "reflection-mapped-runtime-"));
    dbPath = path.join(sourceWorkspaceDir, "mapped-memory-db");

    const sessionsDir = path.join(sourceWorkspaceDir, "sessions");
    mkdirSync(sessionsDir, { recursive: true });
    sessionFile = path.join(sessionsDir, "s-mapped.jsonl");
    writeFileSync(
      sessionFile,
      [
        messageLine("user", "Keep the reflected durable context.", 1),
        messageLine("assistant", "Acknowledged. I will keep the durable context.", 2),
      ].join("\n") + "\n",
      "utf-8",
    );

    extensionStubFile = path.join(sourceWorkspaceDir, "extension-api-mapped-stub.mjs");
    writeFileSync(
      extensionStubFile,
      [
        "export async function runEmbeddedPiAgent(params = {}) {",
        "  return {",
        "    payloads: [{",
        "      text: [",
        "        \"## Context (session background)\",",
        "        \"- Reflection mapped memory ingress test.\",",
        "        `- Runner workspace: ${String(params?.workspaceDir || \"(unknown)\")}`,",
        "        \"\",",
        "        \"## Decisions (durable)\",",
        "        \"- Always attach file evidence before reporting completion.\",",
        "        \"\",",
        "        \"## User model deltas (about the human)\",",
        "        \"- Prefers concise direct answers without confirmation loops.\",",
        "        \"\",",
        "        \"## Agent model deltas (about the assistant/system)\",",
        "        \"- (none captured)\",",
        "        \"\",",
        "        \"## Lessons & pitfalls (symptom / cause / fix / prevention)\",",
        "        \"- Always classify empty-state behavior before calling it a failure.\",",
        "        \"\",",
        "        \"## Learning governance candidates (.learnings / promotion / skill extraction)\",",
        "        \"- (none captured)\",",
        "        \"\",",
        "        \"## Open loops / next actions\",",
        "        \"- Verify mapped memories inherit provenance metadata.\",",
        "        \"\",",
        "        \"## Retrieval tags / keywords\",",
        "        \"- memory-reflection\",",
        "        \"\",",
        "        \"## Invariants\",",
        "        \"- Keep inherited-rules in before_prompt_build only.\",",
        "        \"\",",
        "        \"## Derived\",",
        "        \"- Fresh derived line from this run.\"",
        "      ].join(\"\\n\"),",
        "    }],",
        "  };",
        "}",
        "",
      ].join("\n"),
      "utf-8",
    );

    embedServer = http.createServer(async (req, res) => {
      if (req.url === "/v1/embeddings" && req.method === "POST") {
        const chunks = [];
        for await (const chunk of req) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        }
        const payload = JSON.parse(Buffer.concat(chunks).toString("utf-8") || "{}");
        const input = Array.isArray(payload.input) ? String(payload.input[0] || "") : String(payload.input || "");
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({
          data: [{ embedding: buildDeterministicEmbedding(input) }],
        }));
        return;
      }
      res.writeHead(404);
      res.end("not found");
    });
    await new Promise((resolve) => embedServer.listen(0, "127.0.0.1", resolve));
    const address = embedServer.address();
    const port = typeof address === "object" && address ? address.port : 0;
    embedBaseURL = `http://127.0.0.1:${port}/v1`;

    originalExtensionApiPath = process.env.OPENCLAW_EXTENSION_API_PATH;
    process.env.OPENCLAW_EXTENSION_API_PATH = extensionStubFile;

    runtimeConfig = {
      agents: {
        defaults: { workspace: sourceWorkspaceDir },
        list: [
          { id: "main" },
          { id: "reflector", workspace: reflectionWorkspaceDir },
        ],
      },
    };

    harness = createPluginApiHarness({
      resolveRoot: sourceWorkspaceDir,
      runtimeConfig,
      pluginConfig: {
        embedding: {
          apiKey: "test-api-key",
          baseURL: embedBaseURL,
          model: "jina-embeddings-v5-text-small",
          dimensions: 1024,
        },
        dbPath,
        autoCapture: false,
        autoRecall: false,
        sessionStrategy: "memoryReflection",
        selfImprovement: {
          enabled: true,
          beforeResetNote: true,
          ensureLearningFiles: false,
        },
        memoryReflection: {
          agentId: "reflector",
          injectMode: "inheritance+derived",
          storeToLanceDB: false,
        },
      },
    });
    memoryLanceDBProPlugin.register(harness.api);
  });

  afterEach(async () => {
    if (typeof originalExtensionApiPath === "string") {
      process.env.OPENCLAW_EXTENSION_API_PATH = originalExtensionApiPath;
    } else {
      delete process.env.OPENCLAW_EXTENSION_API_PATH;
    }
    if (embedServer) {
      await new Promise((resolve) => embedServer.close(resolve));
    }
    rmSync(sourceWorkspaceDir, { recursive: true, force: true });
    rmSync(reflectionWorkspaceDir, { recursive: true, force: true });
    rmSync(runtimeWorkspaceDir, { recursive: true, force: true });
  });

  it("stores mapped memories through shared ingress while preserving reflection provenance metadata", async () => {
    const commandNewHooks = harness.commandHooks.get("command:new") || [];
    const reflectionHook = commandNewHooks.find((item) =>
      item?.meta?.name === "memory-lancedb-pro.memory-reflection.command-new",
    );
    assert.ok(reflectionHook, "expected memory-reflection command:new hook");

    const timestamp = Date.UTC(2026, 2, 8, 13, 12, 34);
    await reflectionHook.handler({
      action: "new",
      sessionKey: "agent:main:session:s-mapped",
      timestamp,
      messages: [],
      context: {
        cfg: runtimeConfig,
        workspaceDir: runtimeWorkspaceDir,
        commandSource: "cli",
        previousSessionEntry: {
          sessionId: "s-mapped",
          sessionFile,
        },
      },
    });

    const verifyStore = new MemoryStore({ dbPath, vectorDim: 1024 });
    const storedEntries = await verifyStore.list(undefined, undefined, 10, 0);

    assert.equal(storedEntries.length, 3);
    const categories = storedEntries.map((entry) => entry.category).sort();
    assert.deepEqual(categories, ["decision", "fact", "preference"]);

    for (const entry of storedEntries) {
      const metadata = JSON.parse(entry.metadata || "{}");
      assert.equal(metadata.type, "memory-reflection-mapped");
      assert.equal(metadata.captureVersion, 1);
      assert.ok(metadata.memoryTier === "candidate" || metadata.memoryTier === "formal");
      assert.equal(metadata.sourceApp, "openclaw");
      assert.equal(metadata.sourceRole, "assistant");
      assert.match(String(metadata.ingestionMode || ""), /^reflection:/);
      assert.equal(metadata.sessionId, "s-mapped");
      assert.equal(metadata.agentId, "main");
    }
  });
});

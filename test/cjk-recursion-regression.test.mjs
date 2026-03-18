import assert from "node:assert/strict";
import http from "node:http";

import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { Embedder } = jiti("../src/embedder.ts");
const { smartChunk } = jiti("../src/chunker.ts");

function generateCJKText(charCount) {
  const chars = "中文字符测试数据内容关键词信息处理系统计算机软件硬件网络数据库服务器客户端浏览器应用程序编程语言算法数据结构人工智能机器学习深度学习神经网络".split("");
  let text = "";
  for (let i = 0; i < charCount; i++) text += chars[i % chars.length];
  return text;
}

function createJsonServer(handler) {
  const server = http.createServer(async (req, res) => {
    if (req.url !== "/v1/embeddings" || req.method !== "POST") {
      res.writeHead(404);
      res.end("not found");
      return;
    }

    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", async () => {
      try {
        await handler(JSON.parse(body || "{}"), req, res);
      } catch (error) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: String(error?.message || error), code: "test_handler_error" } }));
      }
    });
  });
  return server;
}

async function withServer(handler, fn) {
  const server = createJsonServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  const baseURL = `http://127.0.0.1:${port}/v1`;
  try {
    await fn({ baseURL });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

async function testChunkErrorSurfaced() {
  console.log("Test 1: chunkError is surfaced instead of generic context_length_exceeded wrapper");

  await withServer((payload, _req, res) => {
    const input = Array.isArray(payload.input) ? payload.input[0] : payload.input;
    if (typeof input === "string" && input.length > 1500) {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "Input length exceeds maximum tokens (max 8192)", code: "context_length_exceeded" } }));
      return;
    }

    res.writeHead(400, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "chunk child failed with synthetic downstream error", code: "synthetic_chunk_failure" } }));
  }, async ({ baseURL }) => {
    const embedder = new Embedder({
      provider: "openai-compatible",
      apiKey: "test-key",
      model: "mxbai-embed-large",
      baseURL,
      dimensions: 1024,
    });

    await assert.rejects(
      () => embedder.embedPassage(generateCJKText(5000)),
      (error) => {
        assert.match(error.message, /synthetic_chunk_failure|synthetic downstream error|chunk child failed/i);
        assert.doesNotMatch(error.message, /context_length_exceeded/i);
        return true;
      }
    );
  });

  console.log("  PASSED\n");
}

async function testSmallContextChunking() {
  console.log("Test 2: small-context model no longer keeps a 1000-char hard floor");
  const text = generateCJKText(2000);
  const result = smartChunk(text, "all-MiniLM-L6-v2");
  assert(result.chunkCount > 1, "Expected multiple chunks for small-context CJK text");
  const maxChunkLen = Math.max(...result.chunks.map((c) => c.length));
  assert(maxChunkLen < 300, `Expected chunk size < 300 chars, got ${maxChunkLen}`);
  console.log(`  Largest chunk: ${maxChunkLen} chars`);
  console.log("  PASSED\n");
}

async function testTimeoutAbortPropagation() {
  console.log("Test 3: timeout abort propagates to underlying request path");

  await withServer(async (_payload, req, res) => {
    await new Promise((resolve) => setTimeout(resolve, 11_000));
    if (req.aborted || req.destroyed) {
      return;
    }
    const dims = 1024;
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ data: [{ embedding: Array.from({ length: dims }, () => 0), index: 0 }] }));
  }, async ({ baseURL }) => {
    const embedder = new Embedder({
      provider: "openai-compatible",
      apiKey: "test-key",
      model: "mxbai-embed-large",
      baseURL,
      dimensions: 1024,
    });

    await assert.rejects(
      () => embedder.embedPassage("short timeout probe"),
      (error) => {
        assert.match(error.message, /aborted|abort|timed out|fetch failed/i);
        return true;
      }
    );
  });

  console.log("  PASSED\n");
}

async function testBatchEmbeddingStillWorks() {
  console.log("Test 4: batch embedding still works without withTimeout wrapper");

  await withServer((_payload, _req, res) => {
    const dims = 1024;
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      data: [0, 1, 2].map((index) => ({ embedding: Array.from({ length: dims }, () => index), index })),
    }));
  }, async ({ baseURL }) => {
    const embedder = new Embedder({
      provider: "openai-compatible",
      apiKey: "test-key",
      model: "mxbai-embed-large",
      baseURL,
      dimensions: 1024,
    });

    const embeddings = await embedder.embedBatchPassage(["a", "b", "c"]);
    assert.equal(embeddings.length, 3);
    assert.equal(embeddings[0].length, 1024);
    assert.equal(embeddings[2][0], 2);
  });

  console.log("  PASSED\n");
}

async function run() {
  console.log("Running regression tests for PR #238...\n");
  await testChunkErrorSurfaced();
  await testSmallContextChunking();
  await testTimeoutAbortPropagation();
  await testBatchEmbeddingStillWorks();
  console.log("All regression tests passed!");
}

run().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});

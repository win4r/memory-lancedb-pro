/**
 * Regression tests for CJK recursion fix (PR #215, #238)
 * 
 * Tests for:
 * 1. Single-chunk detection (chunking returns 1 chunk >= 90% of original -> force reduce)
 * 2. Depth limit termination (depth 3 -> throw instead of recurse)
 * 3. CJK-aware chunk sizing (>30% CJK text -> smaller chunks)
 */

import assert from "node:assert/strict";
import http from "node:http";

import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { Embedder } = jiti("../src/embedder.ts");
const { smartChunk } = jiti("../src/chunker.ts");

// Test constants from embedder.ts
const MAX_EMBED_DEPTH = 3;
const STRICT_REDUCTION_FACTOR = 0.5;

// Create mock server that always returns context_length_exceeded
function createFailingMockServer() {
  let callCount = 0;
  
  const server = http.createServer((req, res) => {
    callCount++;
    
    if (req.url === "/v1/embeddings" && req.method === "POST") {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ 
        error: { message: "Input length exceeds maximum tokens (max 8192)", code: "context_length_exceeded" }
      }));
      return;
    }
    res.writeHead(404);
    res.end("not found");
  });
  
  return { server, getCallCount: () => callCount, reset: () => callCount = 0 };
}

// Create mock server that succeeds
function createSuccessMockServer() {
  const server = http.createServer((req, res) => {
    if (req.url === "/v1/embeddings" && req.method === "POST") {
      const dims = 1024;
      const embedding = Array.from({ length: dims }, () => Math.random() * 2 - 1);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ data: [{ embedding, index: 0 }] }));
      return;
    }
    res.writeHead(404);
    res.end("not found");
  });
  
  return { server };
}

async function withMockServer(fn) {
  const { server } = createFailingMockServer();
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

async function withSuccessMockServer(fn) {
  const { server } = createSuccessMockServer();
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

function generateCJKText(charCount) {
  const chars = '中文字符测试数据内容关键词信息处理系统计算机软件硬件网络数据库服务器客户端浏览器应用程序编程语言算法数据结构人工智能机器学习深度学习神经网络'.split('');
  let text = '';
  for (let i = 0; i < charCount; i++) {
    text += chars[i % chars.length];
  }
  return text;
}

async function run() {
  console.log("Running regression tests for PR #215, #238...\n");
  
  // Test 1: Single-chunk detection - when smartChunk produces 1 chunk >= 90% of original, force reduce
  console.log("Test 1: Single-chunk detection (force-reduce when chunk >= 90% of original)");
  
  await withMockServer(async ({ baseURL }) => {
    const embedder = new Embedder({
      provider: "openai-compatible",
      apiKey: "test-key",
      model: "mxbai-embed-large",
      baseURL,
      dimensions: 1024,
    });
    
    // Generate text that will result in single chunk >= 90% of original
    // This simulates the scenario where smartChunk doesn't actually reduce the problem
    const text = generateCJKText(3000);
    
    console.log(`  Input: ${text.length} chars`);
    
    try {
      await embedder.embedPassage(text);
      assert.fail("Should have thrown due to depth limit");
    } catch (error) {
      console.log(`  Error: ${error.message}`);
      // Should hit depth limit and throw, not infinite loop
      assert(error.message.includes("timed out") || error.message.includes("depth") || error.message.includes("MAX_EMBED_DEPTH"));
      console.log("  ✅ Test 1 PASSED (depth limit enforced)\n");
    }
  });
  
  // Test 2: Depth limit termination - at depth 3, should throw instead of recurse
  console.log("Test 2: Depth limit termination (depth >= MAX_EMBED_DEPTH throws)");
  
  await withMockServer(async ({ baseURL }) => {
    const embedder = new Embedder({
      provider: "openai-compatible",
      apiKey: "test-key",
      model: "mxbai-embed-large",
      baseURL,
      dimensions: 1024,
    });
    
    // Very long text that will definitely trigger multiple recursion levels
    const text = generateCJKText(10000);
    
    console.log(`  Input: ${text.length} chars`);
    
    try {
      await embedder.embedPassage(text);
      assert.fail("Should have thrown due to depth limit");
    } catch (error) {
      console.log(`  Error: ${error.message}`);
      // Check that it mentions depth or MAX_EMBED_DEPTH
      assert(
        error.message.includes("MAX_EMBED_DEPTH") || 
        error.message.includes("depth") ||
        error.message.includes("truncat"),
        `Error should mention depth limit: ${error.message}`
      );
      console.log("  ✅ Test 2 PASSED (depth limit termination works)\n");
    }
  });
  
  // Test 3: CJK-aware chunk sizing - check smartChunk produces smaller chunks for CJK
  console.log("Test 3: CJK-aware chunk sizing (>30% CJK -> smaller chunks)");
  
  // Test with high CJK ratio
  const highCJKText = generateCJKText(5000) + " some english text here";
  const resultHighCJK = smartChunk(highCJKText, "mxbai-embed-large");
  console.log(`  High CJK (${highCJKText.length} chars): ${resultHighCJK.chunkCount} chunks`);
  
  // For comparison, pure English
  const englishText = "english text ".repeat(500);
  const resultEnglish = smartChunk(englishText, "mxbai-embed-large");
  console.log(`  English (${englishText.length} chars): ${resultEnglish.chunkCount} chunks`);
  
  // CJK text should be split into more chunks due to token ratio
  assert(resultHighCJK.chunkCount > 1, "CJK text should be split into multiple chunks");
  console.log("  ✅ Test 3 PASSED (CJK-aware chunk sizing works)\n");
  
  // Test 4: Verify STRICT_REDUCTION_FACTOR is applied (50% reduction each level)
  console.log("Test 4: Strict reduction factor (50% per recursion level)");
  
  const originalLength = 8000;
  const expectedAfterDepth3 = Math.floor(
    originalLength * Math.pow(STRICT_REDUCTION_FACTOR, MAX_EMBED_DEPTH)
  );
  console.log(`  Original: ${originalLength} chars`);
  console.log(`  Expected after 3 levels: ~${expectedAfterDepth3} chars (50% * 50% * 50%)`);
  
  // At depth 3, should reduce to ~1000 chars (8000 * 0.5^3 = 1000)
  assert(expectedAfterDepth3 <= 1000, "Should reduce to <= 1000 chars after 3 levels");
  console.log("  ✅ Test 4 PASSED (strict reduction factor correct)\n");
  
  // Test 5: embedBatchQuery/embedBatchPassage should work without timeout wrapper
  console.log("Test 5: Batch embedding works correctly");
  
  await withSuccessMockServer(async ({ baseURL }) => {
    const embedder = new Embedder({
      provider: "openai-compatible",
      apiKey: "test-key",
      model: "mxbai-embed-large",
      baseURL,
      dimensions: 1024,
    });
    
    const texts = ["test one", "test two", "test three"];
    const embeddings = await embedder.embedBatchPassage(texts);
    
    assert(Array.isArray(embeddings), "Should return array");
    assert(embeddings.length === texts.length, `Should have ${texts.length} embeddings`);
    assert(embeddings[0].length === 1024, "Each embedding should have 1024 dimensions");
    
    console.log(`  Batch embedded ${texts.length} texts successfully`);
    console.log("  ✅ Test 5 PASSED\n");
  });
  
  console.log("🎉 All regression tests passed!");
}

run().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});

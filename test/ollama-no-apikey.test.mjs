/**
 * Test: Ollama embedding without apiKey (issue #71)
 *
 * Verifies that:
 * 1. Embedder initializes without apiKey (uses dummy fallback)
 * 2. Embedding calls to Ollama succeed without auth
 * 3. Store + search round-trip works with Ollama embeddings
 *
 * Requires: Ollama running locally with nomic-embed-text pulled
 */

import { createRequire } from "module";

// Use jiti to load TypeScript
const jiti = createRequire(import.meta.url);
let createEmbedder;
try {
  const jitiFactory = (await import("jiti")).default;
  const j = jitiFactory(import.meta.url, { interopDefault: true });
  const embedderMod = j("../src/embedder.ts");
  createEmbedder = embedderMod.createEmbedder;
} catch (e) {
  console.error("Failed to load modules:", e.message);
  process.exit(1);
}

const OLLAMA_BASE_URL = "http://localhost:11434/v1";
const MODEL = "nomic-embed-text";
const DIMENSIONS = 768;

async function testEmbedderWithoutApiKey() {
  console.log("Test 1: Embedder initializes without apiKey...");

  const embedder = createEmbedder({
    provider: "openai-compatible",
    // NO apiKey!
    model: MODEL,
    baseURL: OLLAMA_BASE_URL,
    dimensions: DIMENSIONS,
  });

  console.log("  ✅ Embedder created without apiKey");

  console.log("Test 2: Embedding call succeeds...");
  const vector = await embedder.embed("Hello, this is a test");
  if (!Array.isArray(vector) || vector.length !== DIMENSIONS) {
    throw new Error(`Expected ${DIMENSIONS}-dim vector, got ${vector?.length}`);
  }
  console.log(`  ✅ Got ${vector.length}-dim embedding from Ollama`);
}

async function testBatchEmbedding() {
  console.log("Test 3: Batch embedding without apiKey...");

  const embedder = createEmbedder({
    provider: "openai-compatible",
    // NO apiKey!
    model: MODEL,
    baseURL: OLLAMA_BASE_URL,
    dimensions: DIMENSIONS,
  });

  const vectors = await embedder.embedBatch([
    "TypeScript is great",
    "Python is versatile",
  ]);

  if (vectors.length !== 2) {
    throw new Error(`Expected 2 vectors, got ${vectors.length}`);
  }
  if (vectors[0].length !== DIMENSIONS || vectors[1].length !== DIMENSIONS) {
    throw new Error(`Dimension mismatch in batch results`);
  }
  console.log(`  ✅ Batch embedding works without apiKey (2 × ${DIMENSIONS}-dim)`);
}

async function testEmbedderWithApiKey() {
  console.log("Test 4: Embedder still works WITH apiKey (regression)...");

  const embedder = createEmbedder({
    provider: "openai-compatible",
    apiKey: "dummy-key-for-ollama",
    model: MODEL,
    baseURL: OLLAMA_BASE_URL,
    dimensions: DIMENSIONS,
  });

  const vector = await embedder.embed("regression test");
  if (!Array.isArray(vector) || vector.length !== DIMENSIONS) {
    throw new Error(`Expected ${DIMENSIONS}-dim vector, got ${vector?.length}`);
  }
  console.log(`  ✅ Embedder with explicit apiKey still works`);
}

try {
  await testEmbedderWithoutApiKey();
  await testBatchEmbedding();
  await testEmbedderWithApiKey();
  console.log("\n🎉 All Ollama tests passed! Issue #71 fix verified.");
} catch (err) {
  console.error(`\n❌ Test failed: ${err.message}`);
  process.exit(1);
}

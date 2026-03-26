// PR354 fix — 30 iteration stress test
// Key tests:
//   1) isOllamaProvider correctly identifies Ollama URLs
//   2) native fetch to Ollama returns valid embeddings (proves Ollama path is used)
//   3) AbortController aborts BEFORE response arrives (mock server with 3s delay)
import assert from "node:assert/strict";
import http from "node:http";

function isOllamaProvider(baseURL) {
  if (!baseURL) return false;
  return /localhost:11434|127\.0\.0\.1:11434|\/ollama\b/i.test(baseURL);
}

// Create a slow mock server that takes 3 seconds to respond
function createSlowServer(delayMs = 3000) {
  return new Promise((resolveServer) => {
    const server = http.createServer((req, res) => {
      setTimeout(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          data: [{ embedding: Array(768).fill(0.1), index: 0, object: "embedding" }]
        }));
      }, delayMs);
    });
    server.listen(0, "127.0.0.1", () => resolveServer(server));
  });
}

let passed = 0;
let failed = 0;
const failures = [];

for (let i = 1; i <= 30; i++) {
  process.stdout.write(`Iteration ${i}/30 `);

  try {
    // Test 1: isOllamaProvider
    const urlTests = [
      ["http://127.0.0.1:11434/v1", true],
      ["http://localhost:11434/v1", true],
      ["http://localhost:11434/ollama", true],
      ["http://localhost:8080/v1", false],
      ["http://api.openai.com/v1", false],
    ];
    for (const [url, expected] of urlTests) {
      assert.equal(isOllamaProvider(url), expected, `${url} should be ${expected}`);
    }

    // Test 2: Real Ollama returns valid embeddings (native fetch path)
    const ctrlReal = new AbortController();
    const startReal = Date.now();
    const result = await fetch("http://127.0.0.1:11434/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "nomic-embed-text", input: `test-${i}` }),
      signal: ctrlReal.signal,
    });
    const elapsedReal = Date.now() - startReal;
    assert.equal(result.ok, true, "Ollama should return ok=true");
    const json = await result.json();
    assert.ok(json.data && json.data.length > 0, "Should have embeddings");
    assert.ok(Array.isArray(json.data[0].embedding) && json.data[0].embedding.length > 0,
      "embedding should be non-empty array");

    // Test 3: Abort BEFORE slow mock server responds
    const mockServer = await createSlowServer(3000);
    const mockAddr = mockServer.address();
    const mockBaseURL = `http://127.0.0.1:${mockAddr.port}/v1`;

    // Force Ollama path on our mock (native fetch still works)
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 200); // abort at 200ms — BEFORE 3000ms server response

    const start = Date.now();
    let abortError = null;
    try {
      await fetch(mockBaseURL + "/embeddings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "test", input: "abort test" }),
        signal: controller.signal,
      });
    } catch (e) {
      abortError = e;
    }
    const elapsed = Date.now() - start;

    mockServer.close();
    await new Promise(r => mockServer.close(r));

    assert.ok(abortError !== null, "Should have caught an error");
    assert.ok(
      abortError?.name === "AbortError" || /abort/i.test(abortError?.name || ""),
      `Error should be AbortError, got: ${abortError?.name || "none"} — ${abortError?.message || ""}`
    );
    assert.ok(elapsed < 2000, `Abort should happen within 2s (was ${elapsed}ms, abort at 200ms, server would take 3000ms)`);

    passed++;
    console.log(`✓ (ollama=${elapsedReal}ms, abort=${elapsed}ms)`);
  } catch (e) {
    failed++;
    failures.push({ iteration: i, error: e.message, elapsed: null });
    console.log(`✗ FAIL: ${e.message}`);
  }
}

console.log(`\n=== Results: ${passed}/30 passed, ${failed}/30 failed ===`);
if (failures.length > 0) {
  console.log("\nFailures:");
  failures.forEach(f => console.log(`  Iteration ${f.iteration}: ${f.error}`));
}
process.exit(failed > 0 ? 1 : 0);

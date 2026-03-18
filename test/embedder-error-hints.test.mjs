import assert from "node:assert/strict";
import http from "node:http";

import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { Embedder, formatEmbeddingProviderError } = jiti("../src/embedder.ts");

async function withJsonServer(status, body, fn) {
  const server = http.createServer((req, res) => {
    if (req.url === "/v1/embeddings" && req.method === "POST") {
      res.writeHead(status, { "content-type": "application/json" });
      res.end(JSON.stringify(body));
      return;
    }
    res.writeHead(404);
    res.end("not found");
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  const baseURL = `http://127.0.0.1:${port}/v1`;

  try {
    await fn({ baseURL, port });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

async function expectReject(promiseFactory, pattern) {
  try {
    await promiseFactory();
    assert.fail("Expected promise to reject");
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    assert.match(msg, pattern, msg);
    return msg;
  }
}

async function run() {
  await withJsonServer(
    403,
    { error: { message: "Invalid API key", code: "invalid_api_key" } },
    async ({ baseURL, port }) => {
      const embedder = new Embedder({
        provider: "openai-compatible",
        apiKey: "bad-key",
        model: "jina-embeddings-v5-text-small",
        baseURL,
        dimensions: 1024,
      });

      const msg = await expectReject(
        () => embedder.embedPassage("hello"),
        /authentication failed/i,
      );
      assert.match(msg, /Invalid API key/i, msg);
      assert.match(msg, new RegExp(`127\\.0\\.0\\.1:${port}`), msg);
      assert.doesNotMatch(msg, /Check .* for Jina\./i, msg);
    },
  );

  const jinaAuth = formatEmbeddingProviderError(
    Object.assign(new Error("403 Invalid API key"), {
      status: 403,
      code: "invalid_api_key",
    }),
    {
      baseURL: "https://api.jina.ai/v1",
      model: "jina-embeddings-v5-text-small",
    },
  );
  assert.match(jinaAuth, /authentication failed/i, jinaAuth);
  assert.match(jinaAuth, /Jina/i, jinaAuth);
  assert.match(jinaAuth, /Ollama/i, jinaAuth);

  const formattedNetwork = formatEmbeddingProviderError(
    Object.assign(new Error("connect ECONNREFUSED 127.0.0.1:11434"), {
      code: "ECONNREFUSED",
    }),
    {
      baseURL: "http://127.0.0.1:11434/v1",
      model: "bge-m3",
    },
  );
  assert.match(formattedNetwork, /provider unreachable/i, formattedNetwork);
  assert.match(formattedNetwork, /127\.0\.0\.1:11434\/v1/i, formattedNetwork);
  assert.match(formattedNetwork, /bge-m3/i, formattedNetwork);

  const formattedBatch = formatEmbeddingProviderError(
    new Error("provider returned malformed payload"),
    {
      baseURL: "https://example.invalid/v1",
      model: "custom-model",
      mode: "batch",
    },
  );
  assert.match(formattedBatch, /^Failed to generate batch embeddings from /, formattedBatch);

  const formattedVoyage = formatEmbeddingProviderError(
    new Error("unsupported request field"),
    {
      baseURL: "https://api.voyageai.com/v1",
      model: "voyage-3-lite",
    },
  );
  assert.match(formattedVoyage, /^Failed to generate embedding from Voyage:/, formattedVoyage);

  // Matryoshka error: local model that rejects dimensions parameter
  const matryoshkaError1 = formatEmbeddingProviderError(
    new Error("400 Model does not support matryoshka representation"),
    {
      baseURL: "http://127.0.0.1:8000/v1",
      model: "Qwen3-Embedding-0.6B",
    },
  );
  assert.match(matryoshkaError1, /rejected dimensions parameter/i, matryoshkaError1);
  assert.match(matryoshkaError1, /matryoshka representation/i, matryoshkaError1);
  assert.match(matryoshkaError1, /omitDimensions.*true/i, matryoshkaError1);

  // Matryoshka error: "dimensions not supported" variant
  const matryoshkaError2 = formatEmbeddingProviderError(
    new Error("unknown parameter: dimensions"),
    {
      baseURL: "http://localhost:11434/v1",
      model: "nomic-embed-text",
    },
  );
  assert.match(matryoshkaError2, /rejected dimensions parameter/i, matryoshkaError2);
  assert.match(matryoshkaError2, /omitDimensions.*true/i, matryoshkaError2);

  // Matryoshka error: "dimensions not supported" phrase variant
  const matryoshkaError3 = formatEmbeddingProviderError(
    new Error("the dimensions parameter is not supported by this model"),
    {
      baseURL: "http://127.0.0.1:8000/v1",
      model: "bge-m3",
    },
  );
  assert.match(matryoshkaError3, /rejected dimensions parameter/i, matryoshkaError3);
  assert.match(matryoshkaError3, /omitDimensions.*true/i, matryoshkaError3);

  console.log("OK: embedder auth/network error hints verified");
}

run().catch((err) => {
  console.error("FAIL: embedder error hint test failed");
  console.error(err);
  process.exit(1);
});

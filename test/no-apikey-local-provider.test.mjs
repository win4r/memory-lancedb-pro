import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const { parsePluginConfig } = jiti("../index.ts");

function baseConfig(overrides = {}) {
  return {
    embedding: {
      provider: "openai-compatible",
      ...overrides,
    },
  };
}

function withEnv(value, fn) {
  const prev = process.env.OPENAI_API_KEY;
  if (value === undefined) {
    delete process.env.OPENAI_API_KEY;
  } else {
    process.env.OPENAI_API_KEY = value;
  }
  try {
    fn();
  } finally {
    if (prev === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = prev;
    }
  }
}

describe("embedding apiKey handling for local providers", () => {
  it("allows missing apiKey for localhost baseURL and uses dummy key", () => {
    const warn = console.warn;
    console.warn = () => {};
    try {
      withEnv(undefined, () => {
        const cfg = parsePluginConfig(baseConfig({ baseURL: "http://localhost:11434" }));
        assert.equal(cfg.embedding.apiKey, "ollama");
      });
    } finally {
      console.warn = warn;
    }
  });

  it("throws for cloud provider when apiKey is missing", () => {
    withEnv(undefined, () => {
      assert.throws(() => parsePluginConfig(baseConfig({ baseURL: "https://api.jina.ai" })), /embedding\.apiKey is required/);
    });
  });

  it("preserves explicit apiKey for local provider", () => {
    withEnv(undefined, () => {
      const cfg = parsePluginConfig(baseConfig({ baseURL: "http://127.0.0.1:11434", apiKey: "local-key" }));
      assert.equal(cfg.embedding.apiKey, "local-key");
    });
  });

  it("throws when no baseURL and no apiKey", () => {
    withEnv(undefined, () => {
      assert.throws(() => parsePluginConfig(baseConfig({})), /embedding\.apiKey is required/);
    });
  });
});

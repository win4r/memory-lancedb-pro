import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const { MemoryStore } = jiti("../src/store.ts");
const {
  extractAutoCaptureRecords,
  shouldCapture,
  storeAutoCaptureCandidates,
  toAutoCaptureCandidates,
} = jiti("../src/auto-capture.ts");
const { extractAtomicMemory, parseMemoryMetadata } = jiti("../src/atomic-memory.ts");

function embedText(text, dim = 16) {
  const vector = Array(dim).fill(0);
  const tokens = String(text).toLowerCase().match(/[a-z0-9\u4e00-\u9fff._/-]+/g) || [];
  for (const token of tokens) {
    let hash = 0;
    for (const char of token) hash = (hash * 33 + char.charCodeAt(0)) >>> 0;
    vector[hash % dim] += 1;
  }
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return norm > 0 ? vector.map((value) => value / norm) : vector;
}

test("extractAutoCaptureRecords preserves user and assistant roles after normalization", () => {
  const { records, skippedCount } = extractAutoCaptureRecords(
    [
      { role: "user", content: "  Remember I prefer concise status updates.  " },
      { role: "assistant", content: "Decision: we will use the fallback path." },
      { role: "assistant", content: "" },
    ],
    {
      captureAssistant: true,
      normalizeText: (_role, text) => text.trim() || null,
    },
  );

  assert.equal(skippedCount, 1);
  assert.deepEqual(records, [
    { text: "Remember I prefer concise status updates.", role: "user" },
    { text: "Decision: we will use the fallback path.", role: "assistant" },
  ]);
});

test("storeAutoCaptureCandidates writes atomic metadata and preserves base metadata", async () => {
  const workDir = mkdtempSync(path.join(tmpdir(), "memory-lancedb-pro-auto-capture-"));

  try {
    const store = new MemoryStore({ dbPath: path.join(workDir, "db"), vectorDim: 16 });
    const embedder = {
      async embedPassage(text) {
        return embedText(text);
      },
    };

    const records = [
      { text: "Remember I prefer concise status updates.", role: "user" },
      { text: "Decision: we will use per-agent scopes.", role: "assistant" },
    ];
    const candidates = toAutoCaptureCandidates(records).filter((candidate) =>
      shouldCapture(candidate.text),
    );

    const stored = await storeAutoCaptureCandidates({
      candidates,
      store,
      embedder,
      scope: "agent:main",
      buildBaseMetadata: (candidate) =>
        JSON.stringify({
          l0_abstract: candidate.text,
          source_session: "agent:main:test-session",
        }),
    });

    assert.equal(stored.length, 2);

    const preferenceEntry = stored.find((entry) => entry.category === "preference");
    const decisionEntry = stored.find((entry) => entry.category === "decision");
    assert.ok(preferenceEntry);
    assert.ok(decisionEntry);

    assert.deepEqual(extractAtomicMemory(preferenceEntry.metadata), {
      schema: "openclaw.atomic-memory/v1",
      unitType: "preference",
      sourceKind: "user",
      confidence: 0.7,
    });
    assert.equal(parseMemoryMetadata(preferenceEntry.metadata).source_session, "agent:main:test-session");

    assert.deepEqual(extractAtomicMemory(decisionEntry.metadata), {
      schema: "openclaw.atomic-memory/v1",
      unitType: "decision",
      sourceKind: "agent",
      confidence: 0.7,
    });
  } finally {
    rmSync(workDir, { recursive: true, force: true });
  }
});

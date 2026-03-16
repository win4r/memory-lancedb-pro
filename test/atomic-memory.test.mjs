import test from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const {
  ATOMIC_MEMORY_SCHEMA,
  buildAtomicMemoryMetadata,
  extractAtomicMemory,
  parseMemoryMetadata,
} = jiti("../src/atomic-memory.ts");

test("buildAtomicMemoryMetadata merges atomic metadata into existing metadata", () => {
  const baseMetadata = JSON.stringify({
    l0_abstract: "Remember this preference",
    source_session: "agent:main:test",
  });

  const merged = buildAtomicMemoryMetadata(
    "preference",
    {
      sourceKind: "user",
      confidence: 0.91,
      tags: ["style", "user"],
    },
    baseMetadata,
  );

  assert.ok(merged);
  const parsed = parseMemoryMetadata(merged);
  assert.equal(parsed.l0_abstract, "Remember this preference");

  const atomic = extractAtomicMemory(merged);
  assert.deepEqual(atomic, {
    schema: ATOMIC_MEMORY_SCHEMA,
    unitType: "preference",
    sourceKind: "user",
    confidence: 0.91,
    tags: ["style", "user"],
  });
});

test("extractAtomicMemory ignores malformed envelopes", () => {
  assert.equal(extractAtomicMemory(undefined), undefined);
  assert.equal(extractAtomicMemory("{not-json"), undefined);
  assert.equal(extractAtomicMemory(JSON.stringify({ atomic: { schema: "wrong" } })), undefined);
});

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const { shouldSkipRetrieval } = jiti("../src/adaptive-retrieval.ts");

describe("shouldSkipRetrieval heartbeat/NO_REPLY", () => {
  const cases = [
    "NO_REPLY",
    "no_reply",
    "no-reply",
    "HEARTBEAT_OK",
    "heartbeat ok",
    "heartbeat: NO_REPLY",
    "[heartbeat] NO_REPLY",
    "health_check",
    "system-check",
  ];

  for (const input of cases) {
    it(`skips retrieval for ${input}`, () => {
      assert.equal(shouldSkipRetrieval(input), true);
    });
  }
});

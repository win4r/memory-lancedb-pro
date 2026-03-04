import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

test("graphiti config default baseUrl matches plugin schema default", () => {
  const source = readFileSync(join(import.meta.dirname, "..", "index.ts"), "utf-8");
  assert.match(
    source,
    /:\s*"http:\/\/localhost:8001"/,
    "index.ts graphiti.baseUrl fallback should be http://localhost:8001",
  );
});

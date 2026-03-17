import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/tools.ts", import.meta.url), "utf8");

test("resolveRuntimeAgentId has a central main fallback", () => {
  assert.match(
    source,
    /function resolveRuntimeAgentId\([\s\S]*?\): string \{[\s\S]*?return fallback \|\| "main";[\s\S]*?const resolved = ctxAgentId \|\| parseAgentIdFromSessionKey\(ctxSessionKey\) \|\| staticAgentId;[\s\S]*?return trimmed \? trimmed : "main";/,
  );
});

test("caller-level main fallbacks were removed from memory_forget and memory_update", () => {
  assert.doesNotMatch(
    source,
    /resolveRuntimeAgentId\(context\.agentId, runtimeCtx\) \|\| 'main'/,
  );
});

test("memory_stats and memory_list use the central runtime agent resolver", () => {
  assert.match(source, /name: "memory_stats"[\s\S]*?const agentId = resolveRuntimeAgentId\(context\.agentId, runtimeCtx\);/);
  assert.match(source, /name: "memory_list"[\s\S]*?const agentId = resolveRuntimeAgentId\(context\.agentId, runtimeCtx\);/);
});

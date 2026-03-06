import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jitiFactory from "jiti";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const pluginSdkStubPath = path.resolve(testDir, "helpers", "openclaw-plugin-sdk-stub.mjs");
const jiti = jitiFactory(import.meta.url, {
  interopDefault: true,
  alias: {
    "openclaw/plugin-sdk": pluginSdkStubPath,
  },
});
const { readSessionConversationWithResetFallback } = jiti("../index.ts");

function messageLine(role, text, ts) {
  return JSON.stringify({
    type: "message",
    timestamp: ts,
    message: {
      role,
      content: [{ type: "text", text }],
    },
  });
}

describe("memory-reflection command:new/reset session fallback helper", () => {
  let workDir;

  beforeEach(() => {
    workDir = mkdtempSync(path.join(tmpdir(), "reflection-fallback-test-"));
  });

  afterEach(() => {
    rmSync(workDir, { recursive: true, force: true });
  });

  it("falls back to latest reset snapshot when current session has only slash/control messages", async () => {
    const sessionsDir = path.join(workDir, "sessions");
    const sessionPath = path.join(sessionsDir, "abc123.jsonl");
    const resetOldPath = `${sessionPath}.reset.1700000000`;
    const resetNewPath = `${sessionPath}.reset.1700000001`;
    mkdirSync(sessionsDir, { recursive: true });

    writeFileSync(
      sessionPath,
      [messageLine("user", "/new", 1), messageLine("assistant", "/note self-improvement (before reset): ...", 2)].join("\n") + "\n",
      "utf-8"
    );
    writeFileSync(
      resetOldPath,
      [messageLine("user", "old reset snapshot", 3), messageLine("assistant", "old reset reply", 4)].join("\n") + "\n",
      "utf-8"
    );
    writeFileSync(
      resetNewPath,
      [
        messageLine("user", "Please keep responses concise and factual.", 5),
        messageLine("assistant", "Acknowledged. I will keep responses concise and factual.", 6),
      ].join("\n") + "\n",
      "utf-8"
    );

    // Ensure deterministic latest-reset ordering by mtime.
    const oldTime = new Date("2024-01-01T00:00:00Z");
    const newTime = new Date("2024-01-01T00:00:10Z");
    utimesSync(resetOldPath, oldTime, oldTime);
    utimesSync(resetNewPath, newTime, newTime);

    const conversation = await readSessionConversationWithResetFallback(sessionPath, 10);
    assert.ok(conversation);
    assert.match(conversation, /user: Please keep responses concise and factual\./);
    assert.match(conversation, /assistant: Acknowledged\. I will keep responses concise and factual\./);
    assert.doesNotMatch(conversation, /old reset snapshot/);
    assert.doesNotMatch(conversation, /^user:\s*\/new/m);
  });
});

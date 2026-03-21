import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const {
  parseClaudeJsonlLines,
  parseCodexJsonlLines,
} = jiti("../src/external-ingestion.ts");

describe("Claude external ingestion parsing", () => {
  it("keeps user/text assistant messages and skips non-text assistant blocks", () => {
    const lines = [
      JSON.stringify({
        sessionId: "claude-s1",
        cwd: "/Users/qingshan/project",
        type: "user",
        timestamp: "2026-03-09T10:00:00.000Z",
        message: { role: "user", content: "记住：以后用 OpenAI 中转。" },
      }),
      JSON.stringify({
        sessionId: "claude-s1",
        cwd: "/Users/qingshan/project",
        type: "assistant",
        timestamp: "2026-03-09T10:00:05.000Z",
        message: {
          role: "assistant",
          model: "Claude-3.7",
          content: [
            { type: "thinking", thinking: "hidden" },
            { type: "text", text: "总结：用户偏好 OpenAI 中转为主。" },
            { type: "tool_use", name: "Bash" },
          ],
        },
      }),
    ];

    const parsed = parseClaudeJsonlLines("/tmp/claude.jsonl", lines);
    assert.equal(parsed.candidates.length, 2);
    assert.equal(parsed.candidates[0].role, "user");
    assert.equal(parsed.candidates[1].role, "assistant");
    assert.equal(parsed.candidates[1].sourceProvider, "Claude-3.7");
  });
});

describe("Codex external ingestion parsing", () => {
  it("captures user events and final assistant answers only", () => {
    const lines = [
      JSON.stringify({
        timestamp: "2026-03-09T10:00:00.000Z",
        type: "session_meta",
        payload: {
          id: "codex-s1",
          cwd: "/Users/qingshan",
          model_provider: "custom",
        },
      }),
      JSON.stringify({
        timestamp: "2026-03-09T10:00:01.000Z",
        type: "event_msg",
        payload: {
          type: "user_message",
          message: "记住：MiniMax 只作备用模型。",
        },
      }),
      JSON.stringify({
        timestamp: "2026-03-09T10:00:02.000Z",
        type: "response_item",
        payload: {
          type: "message",
          role: "assistant",
          phase: "commentary",
          content: [{ type: "output_text", text: "我先查配置。" }],
        },
      }),
      JSON.stringify({
        timestamp: "2026-03-09T10:00:10.000Z",
        type: "response_item",
        payload: {
          type: "message",
          role: "assistant",
          phase: "final_answer",
          content: [{ type: "output_text", text: "已设置：OpenAI 中转主模型，MiniMax 备用。" }],
        },
      }),
    ];

    const parsed = parseCodexJsonlLines("/tmp/codex.jsonl", lines);
    assert.equal(parsed.candidates.length, 2);
    assert.equal(parsed.candidates[0].role, "user");
    assert.equal(parsed.candidates[1].role, "assistant");
    assert.equal(parsed.candidates[1].sourceProvider, "custom");
    assert.equal(parsed.cursorPatch.sourceSessionId, "codex-s1");
  });
});


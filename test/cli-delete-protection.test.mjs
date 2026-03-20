import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { Command } from "commander";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const { createMemoryCLI } = jiti("../cli.ts");

async function runDeleteCommand({ id, scope, storeOverrides = {} }) {
  const resolveCalls = [];
  const deleteCalls = [];
  const logs = [];
  const errors = [];
  const exitCalls = [];

  const context = {
    store: {
      async resolveByIdOrPrefix(value) {
        resolveCalls.push(value);
        return {
          id: "12345678-1234-1234-1234-1234567890ab",
          text: "editable memory",
          vector: [0.1, 0.2, 0.3],
          category: "other",
          scope: "scope-main",
          importance: 0.7,
          timestamp: 1700000000000,
          metadata: "{}",
        };
      },
      async delete(value, scopes) {
        deleteCalls.push({ id: value, scopes });
        return true;
      },
      ...storeOverrides,
    },
    retriever: {},
    scopeManager: {},
    migrator: {},
  };

  const program = new Command();
  program.exitOverride();
  createMemoryCLI(context)({ program });

  const argv = ["node", "openclaw", "memory-pro", "delete", id];
  if (scope) {
    argv.push("--scope", scope);
  }

  const origLog = console.log;
  const origError = console.error;
  const origExit = process.exit;
  console.log = (...args) => logs.push(args.join(" "));
  console.error = (...args) => errors.push(args.join(" "));
  process.exit = ((code = 0) => {
    exitCalls.push(Number(code));
    throw new Error(`__TEST_EXIT__${code}`);
  });

  try {
    await program.parseAsync(argv);
    return {
      resolveCalls,
      deleteCalls,
      logs,
      errors,
      exitCalls,
      exitCode: null,
    };
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("__TEST_EXIT__")) {
      return {
        resolveCalls,
        deleteCalls,
        logs,
        errors,
        exitCalls,
        exitCode: exitCalls.at(-1) ?? null,
      };
    }
    throw error;
  } finally {
    console.log = origLog;
    console.error = origError;
    process.exit = origExit;
  }
}

describe("cli delete protection", () => {
  it("rejects deleting protected reflection entries", async () => {
    const resolveCalls = [];
    const result = await runDeleteCommand({
      id: "12345678-1234-1234-1234-1234567890ab",
      storeOverrides: {
        async resolveByIdOrPrefix(value) {
          resolveCalls.push(value);
          return {
            id: "12345678-1234-1234-1234-1234567890ab",
            text: "reflection row",
            vector: [0.1, 0.2, 0.3],
            category: "reflection",
            scope: "scope-main",
            importance: 0.8,
            timestamp: 1700000000000,
            metadata: JSON.stringify({ type: "memory-reflection-item" }),
          };
        },
      },
    });

    assert.deepEqual(resolveCalls, ["12345678-1234-1234-1234-1234567890ab"]);
    assert.equal(result.deleteCalls.length, 0);
    assert.equal(result.exitCode, 1);
    assert.match(
      result.logs.join("\n"),
      /protected category "reflection" and cannot be deleted/i,
    );
  });

  it("keeps prefix-id deletion support for editable memories", async () => {
    const resolveCalls = [];
    const result = await runDeleteCommand({
      id: "12345678",
      storeOverrides: {
        async resolveByIdOrPrefix(value) {
          resolveCalls.push(value);
          return {
            id: "12345678-1234-1234-1234-1234567890ab",
            text: "editable memory",
            vector: [0.1, 0.2, 0.3],
            category: "other",
            scope: "scope-main",
            importance: 0.7,
            timestamp: 1700000000000,
            metadata: "{}",
          };
        },
      },
    });

    assert.deepEqual(resolveCalls, ["12345678"]);
    assert.equal(result.exitCode, null);
    assert.equal(result.deleteCalls.length, 1);
    assert.equal(
      result.deleteCalls[0].id,
      "12345678-1234-1234-1234-1234567890ab",
    );
    assert.match(
      result.logs.join("\n"),
      /Memory 12345678-1234-1234-1234-1234567890ab deleted successfully/i,
    );
  });
});

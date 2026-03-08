import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { createWorkspaceDocsMaterializer } = jiti("../src/workspace-docs.ts");

test("workspace docs materializer creates managed files with markers", async () => {
  const workspaceDir = await mkdtemp(join(tmpdir(), "workspace-docs-"));
  const materializer = createWorkspaceDocsMaterializer({
    workspaceDir,
    store: {
      list: async () => [
        {
          id: "m-1",
          text: "Alice prefers concise responses.",
          category: "preference",
          scope: "global",
          importance: 0.8,
          timestamp: Date.now(),
          metadata: "{}",
        },
      ],
    },
  });

  await materializer.refresh({ reason: "unit-test" });

  const memoryDoc = await readFile(join(workspaceDir, "MEMORY.md"), "utf-8");
  assert.match(memoryDoc, /memory-lancedb-pro:begin MEMORY/);
  assert.match(memoryDoc, /Alice prefers concise responses/);
});

test("workspace docs materializer preserves user-authored content outside markers", async () => {
  const workspaceDir = await mkdtemp(join(tmpdir(), "workspace-docs-"));
  const userFile = join(workspaceDir, "USER.md");

  await writeFile(
    userFile,
    "# USER\n\nUser-owned intro text.\n\n<!-- memory-lancedb-pro:begin USER -->\nold block\n<!-- memory-lancedb-pro:end USER -->\n",
    "utf-8",
  );

  const materializer = createWorkspaceDocsMaterializer({
    workspaceDir,
    store: {
      list: async () => [],
    },
  });

  await materializer.refresh({ reason: "replace-block" });

  const userDoc = await readFile(userFile, "utf-8");
  assert.match(userDoc, /User-owned intro text\./);
  assert.match(userDoc, /memory-lancedb-pro:begin USER/);
  assert.doesNotMatch(userDoc, /old block/);
});

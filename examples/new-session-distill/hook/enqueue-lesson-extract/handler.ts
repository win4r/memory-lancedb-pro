import type { HookHandler } from "../../src/hooks/hooks.js";
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

function findSessionFile(event: any): string | null {
  const prev = event.context?.previousSessionEntry;
  const curr = event.context?.sessionEntry;

  const candidates = [prev?.sessionFile, curr?.sessionFile].filter(Boolean);

  for (const file of candidates) {
    if (file && existsSync(file)) return file;
  }

  // If the file was renamed to .reset.*, look for the latest one
  for (const file of candidates) {
    if (!file) continue;
    const dir = dirname(file);
    const base = basename(file);
    const resetPrefix = `${base}.reset.`;
    try {
      const resetFiles = readdirSync(dir)
        .filter((name: string) => name.startsWith(resetPrefix))
        .sort();
      if (resetFiles.length > 0) return join(dir, resetFiles[resetFiles.length - 1]);
    } catch {
      // ignore
    }
  }

  return null;
}

function sha1(input: string): string {
  return createHash("sha1").update(input).digest("hex");
}

function repoRoot(): string {
  // examples/new-session-distill/hook/enqueue-lesson-extract/handler.ts -> up to repo root
  const here = dirname(fileURLToPath(import.meta.url));
  return join(here, "..", "..", "..", "..", "..");
}

const handler: HookHandler = async (event) => {
  if (event.type !== "command" || event.action !== "new") return;

  try {
    const sessionKey = event.sessionKey || "unknown";
    const ctxEntry = (event.context?.previousSessionEntry || event.context?.sessionEntry) as any;
    const sessionId = (ctxEntry?.sessionId as string) || "unknown";
    const source = (event.context?.commandSource as string) || "unknown";

    const sessionFile = findSessionFile(event);
    if (!sessionFile) {
      console.error("[enqueue-lesson-extract] No session file found; skipping");
      return;
    }

    const root = repoRoot();
    const queueDir = join(root, "workspaces", "main", "tasks", "lesson-extract", "inbox");
    mkdirSync(queueDir, { recursive: true });

    const ts = new Date(event.timestamp || Date.now()).toISOString();
    const taskId = sha1([sessionKey, sessionId, sessionFile, ts].join("|"));

    const task = {
      taskId,
      agentId: "main",
      scope: "agent:main",
      event: {
        type: "command:new",
        timestamp: ts,
        sessionKey,
        source,
      },
      session: {
        sessionId,
        sessionFile,
      },
      extract: {
        maxFinal: 20,
        mapChunkChars: 12000,
        mapOverlapMsgs: 10,
      },
    };

    const safeTs = ts.replace(/[:.]/g, "-");
    const filename = `${safeTs}-${taskId.slice(0, 8)}.json`;
    const outPath = join(queueDir, filename);

    writeFileSync(outPath, JSON.stringify(task, null, 2) + "\n", "utf-8");
  } catch (err) {
    console.error(
      "[enqueue-lesson-extract] Error:",
      err instanceof Error ? err.message : String(err)
    );
  }
};

export default handler;

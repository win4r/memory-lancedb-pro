import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { watch, type FSWatcher } from "node:fs";

import type { CaptureRole, CaptureSourceApp } from "./capture-pipeline.js";

export interface ClaudeExternalSourceConfig {
  enabled?: boolean;
  roots?: string[];
  includeSubagents?: boolean;
  maxFilesPerScan?: number;
}

export interface CodexExternalSourceConfig {
  enabled?: boolean;
  roots?: string[];
  maxFilesPerScan?: number;
}

export interface WechatExternalSourceConfig {
  enabled?: boolean;
  roots?: string[];
  placeholder?: boolean;
}

export interface ExternalIngestionConfig {
  enabled?: boolean;
  scanIntervalMs?: number;
  inactivityWindowMs?: number;
  sources?: {
    claude?: ClaudeExternalSourceConfig;
    codex?: CodexExternalSourceConfig;
    wechat?: WechatExternalSourceConfig;
  };
}

export interface ExternalCaptureCandidate {
  text: string;
  role: CaptureRole;
  sourceApp: CaptureSourceApp;
  sourcePath: string;
  sourceSessionId?: string;
  sourceProvider?: string;
  sourceModel?: string;
  sourceWorkspace?: string;
  occurredAt?: number;
}

interface ExternalIngestionLogger {
  info?: (message: string) => void;
  warn: (message: string) => void;
}

interface FileCursorState {
  lineCount: number;
  size: number;
  mtimeMs: number;
  sourceSessionId?: string;
  sourceProvider?: string;
  sourceModel?: string;
  sourceWorkspace?: string;
}

interface ExternalIngestionState {
  version: number;
  files: Record<string, FileCursorState>;
}

interface ExternalIngestionManagerOptions {
  config?: ExternalIngestionConfig;
  stateFilePath: string;
  logger: ExternalIngestionLogger;
  onCandidates: (items: ExternalCaptureCandidate[]) => Promise<void>;
}

interface ParsedSourceBatch {
  candidates: ExternalCaptureCandidate[];
  cursorPatch?: Partial<FileCursorState>;
}

interface JsonlLineEnvelope {
  timestamp?: string;
  type?: string;
  payload?: Record<string, unknown>;
  message?: Record<string, unknown>;
  sessionId?: string;
  cwd?: string;
}

const DEFAULT_SCAN_INTERVAL_MS = 10 * 60_000;
const DEFAULT_INACTIVITY_WINDOW_MS = 3 * 60_000;
const DEFAULT_MAX_FILES_PER_SCAN = 40;

function resolveHomePath(value: string): string {
  if (!value.startsWith("~/")) return value;
  return join(homedir(), value.slice(2));
}

function asArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const parsed = value
    .filter((item): item is string => typeof item === "string")
    .map((item) => resolveHomePath(item.trim()))
    .filter((item) => item.length > 0);
  return parsed.length > 0 ? parsed : undefined;
}

function asText(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseTimestamp(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 1_000_000_000_000 ? value : value * 1000;
  }
  if (typeof value === "string") {
    const asNumber = Number(value);
    if (Number.isFinite(asNumber)) {
      return asNumber > 1_000_000_000_000 ? asNumber : asNumber * 1000;
    }
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function extractClaudeTextContent(content: unknown): string | undefined {
  if (typeof content === "string") {
    const trimmed = content.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
  if (!Array.isArray(content)) return undefined;
  const texts = content
    .filter((block) => block && typeof block === "object")
    .map((block) => block as Record<string, unknown>)
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => String(block.text).trim())
    .filter((text) => text.length > 0);
  if (texts.length === 0) return undefined;
  return texts.join("\n");
}

function extractCodexAssistantText(content: unknown): string | undefined {
  if (!Array.isArray(content)) return undefined;
  const texts = content
    .filter((block) => block && typeof block === "object")
    .map((block) => block as Record<string, unknown>)
    .filter((block) => block.type === "output_text" && typeof block.text === "string")
    .map((block) => String(block.text).trim())
    .filter((text) => text.length > 0);
  if (texts.length === 0) return undefined;
  return texts.join("\n");
}

async function walkJsonlFiles(
  root: string,
  options?: { includeSubagents?: boolean; fileNameMatcher?: (name: string) => boolean },
): Promise<string[]> {
  const out: string[] = [];
  const stack = [root];

  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) continue;

    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const nextPath = join(current, entry.name);
      if (entry.isDirectory()) {
        if (options?.includeSubagents === false && entry.name === "subagents") {
          continue;
        }
        stack.push(nextPath);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!entry.name.endsWith(".jsonl")) continue;
      if (options?.fileNameMatcher && !options.fileNameMatcher(entry.name)) continue;
      out.push(nextPath);
    }
  }

  return out;
}

async function sortFilesByMtimeDesc(filePaths: string[]): Promise<Array<{ path: string; size: number; mtimeMs: number }>> {
  const withStats = await Promise.all(
    filePaths.map(async (filePath) => {
      try {
        const st = await stat(filePath);
        return { path: filePath, size: st.size, mtimeMs: st.mtimeMs };
      } catch {
        return null;
      }
    }),
  );

  return withStats
    .filter((item): item is { path: string; size: number; mtimeMs: number } => item !== null)
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
}

function splitJsonlLines(raw: string): string[] {
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

async function readNewJsonlLines(
  filePath: string,
  currentSize: number,
  currentMtimeMs: number,
  cursor: FileCursorState | undefined,
): Promise<{ lines: string[]; nextCursor: FileCursorState }> {
  const raw = await readFile(filePath, "utf-8");
  const lines = splitJsonlLines(raw);

  const previousLineCount = cursor && cursor.size <= currentSize ? cursor.lineCount : 0;
  const nextLines = lines.slice(previousLineCount);

  return {
    lines: nextLines,
    nextCursor: {
      ...cursor,
      lineCount: lines.length,
      size: currentSize,
      mtimeMs: currentMtimeMs,
    },
  };
}

export function parseClaudeJsonlLines(
  filePath: string,
  lines: string[],
  cursor?: FileCursorState,
): ParsedSourceBatch {
  const candidates: ExternalCaptureCandidate[] = [];
  const cursorPatch: Partial<FileCursorState> = {};

  for (const line of lines) {
    let parsed: JsonlLineEnvelope;
    try {
      parsed = JSON.parse(line) as JsonlLineEnvelope;
    } catch {
      continue;
    }

    const message = parsed.message;
    if (!message || typeof message !== "object") continue;

    const role = asText(message.role);
    if (role !== "user" && role !== "assistant") continue;

    const text = extractClaudeTextContent(message.content);
    if (!text) continue;

    const occurredAt = parseTimestamp(parsed.timestamp);
    const sourceModel = asText(message.model);

    cursorPatch.sourceSessionId = asText(parsed.sessionId) ?? cursor?.sourceSessionId;
    cursorPatch.sourceModel = sourceModel ?? cursor?.sourceModel;
    cursorPatch.sourceWorkspace = asText(parsed.cwd) ?? cursor?.sourceWorkspace;

    candidates.push({
      text,
      role,
      sourceApp: "claude",
      sourcePath: filePath,
      sourceSessionId: cursorPatch.sourceSessionId,
      sourceProvider: sourceModel,
      sourceModel,
      sourceWorkspace: cursorPatch.sourceWorkspace,
      occurredAt,
    });
  }

  return { candidates, cursorPatch };
}

export function parseCodexJsonlLines(
  filePath: string,
  lines: string[],
  cursor?: FileCursorState,
): ParsedSourceBatch {
  const candidates: ExternalCaptureCandidate[] = [];
  const cursorPatch: Partial<FileCursorState> = {
    sourceSessionId: cursor?.sourceSessionId,
    sourceProvider: cursor?.sourceProvider,
    sourceWorkspace: cursor?.sourceWorkspace,
  };

  for (const line of lines) {
    let parsed: JsonlLineEnvelope;
    try {
      parsed = JSON.parse(line) as JsonlLineEnvelope;
    } catch {
      continue;
    }

    if (parsed.type === "session_meta" && parsed.payload && typeof parsed.payload === "object") {
      cursorPatch.sourceSessionId = asText(parsed.payload.id) ?? cursorPatch.sourceSessionId;
      cursorPatch.sourceProvider = asText(parsed.payload.model_provider) ?? cursorPatch.sourceProvider;
      cursorPatch.sourceWorkspace = asText(parsed.payload.cwd) ?? cursorPatch.sourceWorkspace;
      continue;
    }

    if (parsed.type === "event_msg" && parsed.payload?.type === "user_message") {
      const text = asText(parsed.payload.message);
      if (!text) continue;

      candidates.push({
        text,
        role: "user",
        sourceApp: "codex",
        sourcePath: filePath,
        sourceSessionId: cursorPatch.sourceSessionId,
        sourceProvider: cursorPatch.sourceProvider,
        sourceWorkspace: cursorPatch.sourceWorkspace,
        occurredAt: parseTimestamp(parsed.timestamp),
      });
      continue;
    }

    if (parsed.type === "response_item" && parsed.payload?.type === "message" && parsed.payload.role === "assistant") {
      const phase = asText(parsed.payload.phase);
      if (phase && phase !== "final_answer") continue;

      const text = extractCodexAssistantText(parsed.payload.content);
      if (!text) continue;

      candidates.push({
        text,
        role: "assistant",
        sourceApp: "codex",
        sourcePath: filePath,
        sourceSessionId: cursorPatch.sourceSessionId,
        sourceProvider: cursorPatch.sourceProvider,
        sourceWorkspace: cursorPatch.sourceWorkspace,
        occurredAt: parseTimestamp(parsed.timestamp),
      });
    }
  }

  return { candidates, cursorPatch };
}

async function loadState(filePath: string): Promise<ExternalIngestionState> {
  try {
    const raw = await readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as ExternalIngestionState;
    if (parsed && typeof parsed === "object" && parsed.version === 1 && parsed.files && typeof parsed.files === "object") {
      return parsed;
    }
  } catch {
    // ignore
  }
  return { version: 1, files: {} };
}

async function saveState(filePath: string, state: ExternalIngestionState): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true }).catch(() => {});
  await writeFile(filePath, JSON.stringify(state, null, 2), "utf-8");
}

export function normalizeExternalIngestionConfig(config: ExternalIngestionConfig | undefined): Required<ExternalIngestionConfig> {
  const claudeRoots = asArray(config?.sources?.claude?.roots) ?? [join(homedir(), ".claude", "projects")];
  const codexRoots = asArray(config?.sources?.codex?.roots) ?? [join(homedir(), ".codex", "sessions")];
  const wechatRoots = asArray(config?.sources?.wechat?.roots) ?? [];

  return {
    enabled: config?.enabled === true,
    scanIntervalMs: Math.max(60_000, Math.floor(config?.scanIntervalMs ?? DEFAULT_SCAN_INTERVAL_MS)),
    inactivityWindowMs: Math.max(60_000, Math.floor(config?.inactivityWindowMs ?? DEFAULT_INACTIVITY_WINDOW_MS)),
    sources: {
      claude: {
        enabled: config?.sources?.claude?.enabled === true,
        roots: claudeRoots,
        includeSubagents: config?.sources?.claude?.includeSubagents !== false,
        maxFilesPerScan: Math.max(1, Math.floor(config?.sources?.claude?.maxFilesPerScan ?? DEFAULT_MAX_FILES_PER_SCAN)),
      },
      codex: {
        enabled: config?.sources?.codex?.enabled === true,
        roots: codexRoots,
        maxFilesPerScan: Math.max(1, Math.floor(config?.sources?.codex?.maxFilesPerScan ?? DEFAULT_MAX_FILES_PER_SCAN)),
      },
      wechat: {
        enabled: config?.sources?.wechat?.enabled === true,
        roots: wechatRoots,
        placeholder: config?.sources?.wechat?.placeholder !== false,
      },
    },
  };
}

export function createExternalIngestionManager(options: ExternalIngestionManagerOptions) {
  const config = normalizeExternalIngestionConfig(options.config);
  let timer: ReturnType<typeof setInterval> | null = null;
  let running = false;
  let warnedWechatPlaceholder = false;
  const watchers: FSWatcher[] = [];
  const finalizeTimers = new Map<string, ReturnType<typeof setTimeout>>();
  const statePromise = loadState(options.stateFilePath);

  const parserBySourceApp: Record<Extract<CaptureSourceApp, "claude" | "codex">, (filePath: string, lines: string[], cursor?: FileCursorState) => ParsedSourceBatch> = {
    claude: parseClaudeJsonlLines,
    codex: parseCodexJsonlLines,
  };

  function isRelevantSourceFile(
    sourceApp: Extract<CaptureSourceApp, "claude" | "codex">,
    filePath: string,
  ): boolean {
    if (!filePath.endsWith(".jsonl")) return false;
    if (sourceApp === "codex") return basename(filePath).startsWith("rollout-");
    if (sourceApp === "claude" && config.sources.claude.includeSubagents === false && filePath.includes(`${join("", "subagents")}`)) {
      return false;
    }
    return true;
  }

  async function processSourceFile(
    sourceApp: Extract<CaptureSourceApp, "claude" | "codex">,
    filePath: string,
    reason: "watch-finalize" | "fallback-scan",
  ): Promise<number> {
    if (!isRelevantSourceFile(sourceApp, filePath)) return 0;

    let currentStat;
    try {
      currentStat = await stat(filePath);
    } catch {
      return 0;
    }

    const ageMs = Date.now() - currentStat.mtimeMs;
    if (ageMs < config.inactivityWindowMs) {
      const remainingMs = Math.max(1_000, config.inactivityWindowMs - ageMs + 500);
      scheduleFinalize(sourceApp, filePath, remainingMs);
      return 0;
    }

    const state = await statePromise;
    const stateKey = `${sourceApp}:${filePath}`;
    const previousCursor = state.files[stateKey];
    if (previousCursor && previousCursor.size === currentStat.size && previousCursor.mtimeMs === currentStat.mtimeMs) {
      return 0;
    }

    let nextCursor: FileCursorState;
    let newLines: string[];
    try {
      const readResult = await readNewJsonlLines(filePath, currentStat.size, currentStat.mtimeMs, previousCursor);
      nextCursor = readResult.nextCursor;
      newLines = readResult.lines;
    } catch (error) {
      options.logger.warn(`memory-lancedb-pro: failed to read ${sourceApp} transcript ${filePath}: ${String(error)}`);
      return 0;
    }

    if (newLines.length === 0) {
      state.files[stateKey] = nextCursor;
      await saveState(options.stateFilePath, state);
      return 0;
    }

    const parsed = parserBySourceApp[sourceApp](filePath, newLines, previousCursor);
    nextCursor = { ...nextCursor, ...parsed.cursorPatch };
    state.files[stateKey] = nextCursor;
    await saveState(options.stateFilePath, state);

    if (parsed.candidates.length > 0) {
      await options.onCandidates(parsed.candidates);
      options.logger.info?.(
        `memory-lancedb-pro: ${sourceApp} finalized ${parsed.candidates.length} message(s) via ${reason}`,
      );
      return parsed.candidates.length;
    }

    return 0;
  }

  function finalizeKey(sourceApp: Extract<CaptureSourceApp, "claude" | "codex">, filePath: string): string {
    return `${sourceApp}:${filePath}`;
  }

  function scheduleFinalize(
    sourceApp: Extract<CaptureSourceApp, "claude" | "codex">,
    filePath: string,
    delayMs = config.inactivityWindowMs,
  ): void {
    if (!isRelevantSourceFile(sourceApp, filePath)) return;

    const key = finalizeKey(sourceApp, filePath);
    const existing = finalizeTimers.get(key);
    if (existing) clearTimeout(existing);

    const timeout = setTimeout(() => {
      finalizeTimers.delete(key);
      void processSourceFile(sourceApp, filePath, "watch-finalize");
    }, Math.max(1_000, delayMs));

    finalizeTimers.set(key, timeout);
  }

  function registerWatcher(
    sourceApp: Extract<CaptureSourceApp, "claude" | "codex">,
    root: string,
  ): void {
    try {
      const watcher = watch(root, { recursive: true }, (_eventType, filename) => {
        const relativePath = typeof filename === "string" ? filename.trim() : "";
        if (!relativePath) return;
        const filePath = join(root, relativePath);
        scheduleFinalize(sourceApp, filePath);
      });
      watcher.on("error", (error) => {
        options.logger.warn(`memory-lancedb-pro: ${sourceApp} watcher failed for ${root}: ${String(error)}`);
      });
      watchers.push(watcher);
    } catch (error) {
      options.logger.warn(`memory-lancedb-pro: cannot start ${sourceApp} watcher on ${root}: ${String(error)}`);
    }
  }

  const scanSourceFiles = async (
    sourceApp: Extract<CaptureSourceApp, "claude" | "codex">,
    filePaths: Array<{ path: string; size: number; mtimeMs: number }>,
  ): Promise<number> => {
    let captured = 0;

    for (const file of filePaths) {
      captured += await processSourceFile(sourceApp, file.path, "fallback-scan");
    }
    return captured;
  };

  const scanNow = async (): Promise<void> => {
    if (!config.enabled || running) return;
    running = true;
    try {
      let totalCaptured = 0;

      if (config.sources.claude.enabled) {
        const claudeFiles = await sortFilesByMtimeDesc(
          (await Promise.all(
            config.sources.claude.roots.map((root) =>
              walkJsonlFiles(root, { includeSubagents: config.sources.claude.includeSubagents }),
            ),
          )).flat(),
        );
        totalCaptured += await scanSourceFiles(
          "claude",
          claudeFiles.slice(0, config.sources.claude.maxFilesPerScan),
        );
      }

      if (config.sources.codex.enabled) {
        const codexFiles = await sortFilesByMtimeDesc(
          (await Promise.all(
            config.sources.codex.roots.map((root) =>
              walkJsonlFiles(root, { fileNameMatcher: (name) => name.startsWith("rollout-") }),
            ),
          )).flat(),
        );
        totalCaptured += await scanSourceFiles(
          "codex",
          codexFiles.slice(0, config.sources.codex.maxFilesPerScan),
        );
      }

      if (config.sources.wechat.enabled && !warnedWechatPlaceholder) {
        warnedWechatPlaceholder = true;
        options.logger.info?.("memory-lancedb-pro: WeChat capture is reserved as a placeholder; no active collector is running yet");
      }

      if (totalCaptured > 0) {
        options.logger.info?.(`memory-lancedb-pro: external ingestion captured ${totalCaptured} candidate message(s)`);
      }
    } catch (error) {
      options.logger.warn(`memory-lancedb-pro: external ingestion scan failed: ${String(error)}`);
    } finally {
      running = false;
    }
  };

  return {
    enabled: config.enabled,
    async start(): Promise<void> {
      if (!config.enabled) return;
      if (config.sources.claude.enabled) {
        for (const root of config.sources.claude.roots) {
          registerWatcher("claude", root);
        }
      }
      if (config.sources.codex.enabled) {
        for (const root of config.sources.codex.roots) {
          registerWatcher("codex", root);
        }
      }
      setTimeout(() => void scanNow(), 15_000);
      timer = setInterval(() => void scanNow(), config.scanIntervalMs);
    },
    async stop(): Promise<void> {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      for (const timeout of finalizeTimers.values()) {
        clearTimeout(timeout);
      }
      finalizeTimers.clear();
      for (const watcher of watchers) {
        watcher.close();
      }
      watchers.length = 0;
      const state = await statePromise;
      await saveState(options.stateFilePath, state).catch(() => {});
    },
    scanNow,
  };
}

export { DEFAULT_INACTIVITY_WINDOW_MS, DEFAULT_SCAN_INTERVAL_MS };

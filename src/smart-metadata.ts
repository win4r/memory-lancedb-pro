import type { MemoryCategory, MemoryTier } from "./memory-categories.js";
import type { DecayableMemory } from "./decay-engine.js";

type LegacyStoreCategory =
  | "preference"
  | "fact"
  | "decision"
  | "entity"
  | "other";

type EntryLike = {
  text?: string;
  category?: LegacyStoreCategory;
  importance?: number;
  timestamp?: number;
  metadata?: string;
};

export interface SmartMemoryMetadata {
  l0_abstract: string;
  l1_overview: string;
  l2_content: string;
  memory_category: MemoryCategory;
  tier: MemoryTier;
  access_count: number;
  confidence: number;
  last_accessed_at: number;
  source_session?: string;
  [key: string]: unknown;
}

export interface LifecycleMemory {
  id: string;
  importance: number;
  confidence: number;
  tier: MemoryTier;
  accessCount: number;
  createdAt: number;
  lastAccessedAt: number;
}

function clamp01(value: unknown, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(1, Math.max(0, n));
}

function clampCount(value: unknown, fallback = 0): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return Math.floor(n);
}

function normalizeTier(value: unknown): MemoryTier {
  switch (value) {
    case "core":
    case "working":
    case "peripheral":
      return value;
    default:
      return "working";
  }
}

export function reverseMapLegacyCategory(
  oldCategory: LegacyStoreCategory | undefined,
  text = "",
): MemoryCategory {
  switch (oldCategory) {
    case "preference":
      return "preferences";
    case "entity":
      return "entities";
    case "decision":
      return "events";
    case "other":
      return "patterns";
    case "fact":
      if (
        /\b(my |i am |i'm |name is |叫我|我的|我是)\b/i.test(text) &&
        text.length < 200
      ) {
        return "profile";
      }
      return "cases";
    default:
      return "patterns";
  }
}

function defaultOverview(text: string): string {
  return `- ${text}`;
}

function normalizeText(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

export function parseSmartMetadata(
  rawMetadata: string | undefined,
  entry: EntryLike = {},
): SmartMemoryMetadata {
  let parsed: Record<string, unknown> = {};
  if (rawMetadata) {
    try {
      const obj = JSON.parse(rawMetadata);
      if (obj && typeof obj === "object") {
        parsed = obj as Record<string, unknown>;
      }
    } catch {
      parsed = {};
    }
  }

  const text = entry.text ?? "";
  const timestamp =
    typeof entry.timestamp === "number" && Number.isFinite(entry.timestamp)
      ? entry.timestamp
      : Date.now();

  const memoryCategory = reverseMapLegacyCategory(entry.category, text);
  const l0 = normalizeText(parsed.l0_abstract, text);
  const l2 = normalizeText(parsed.l2_content, text);
  const normalized: SmartMemoryMetadata = {
    ...parsed,
    l0_abstract: l0,
    l1_overview: normalizeText(parsed.l1_overview, defaultOverview(l0)),
    l2_content: l2,
    memory_category:
      typeof parsed.memory_category === "string"
        ? (parsed.memory_category as MemoryCategory)
        : memoryCategory,
    tier: normalizeTier(parsed.tier),
    access_count: clampCount(parsed.access_count, 0),
    confidence: clamp01(parsed.confidence, 0.7),
    last_accessed_at: clampCount(parsed.last_accessed_at, timestamp),
    source_session:
      typeof parsed.source_session === "string" ? parsed.source_session : undefined,
  };

  return normalized;
}

export function buildSmartMetadata(
  entry: EntryLike,
  patch: Partial<SmartMemoryMetadata> = {},
): SmartMemoryMetadata {
  const base = parseSmartMetadata(entry.metadata, entry);
  return {
    ...base,
    ...patch,
    l0_abstract: normalizeText(patch.l0_abstract, base.l0_abstract),
    l1_overview: normalizeText(patch.l1_overview, base.l1_overview),
    l2_content: normalizeText(patch.l2_content, base.l2_content),
    memory_category:
      typeof patch.memory_category === "string"
        ? patch.memory_category
        : base.memory_category,
    tier: normalizeTier(patch.tier ?? base.tier),
    access_count: clampCount(patch.access_count, base.access_count),
    confidence: clamp01(patch.confidence, base.confidence),
    last_accessed_at: clampCount(
      patch.last_accessed_at,
      base.last_accessed_at || entry.timestamp || Date.now(),
    ),
    source_session:
      typeof patch.source_session === "string"
        ? patch.source_session
        : base.source_session,
  };
}

export function stringifySmartMetadata(
  metadata: SmartMemoryMetadata | Record<string, unknown>,
): string {
  return JSON.stringify(metadata);
}

export function toLifecycleMemory(
  id: string,
  entry: EntryLike,
): LifecycleMemory {
  const metadata = parseSmartMetadata(entry.metadata, entry);
  const createdAt =
    typeof entry.timestamp === "number" && Number.isFinite(entry.timestamp)
      ? entry.timestamp
      : Date.now();

  return {
    id,
    importance:
      typeof entry.importance === "number" && Number.isFinite(entry.importance)
        ? entry.importance
        : 0.7,
    confidence: metadata.confidence,
    tier: metadata.tier,
    accessCount: metadata.access_count,
    createdAt,
    lastAccessedAt: metadata.last_accessed_at || createdAt,
  };
}

/**
 * Parse a memory entry into both a DecayableMemory (for the decay engine)
 * and the raw SmartMemoryMetadata (for in-place mutation before write-back).
 */
export function getDecayableFromEntry(
  entry: EntryLike & { id?: string },
): { memory: DecayableMemory; meta: SmartMemoryMetadata } {
  const meta = parseSmartMetadata(entry.metadata, entry);
  const createdAt =
    typeof entry.timestamp === "number" && Number.isFinite(entry.timestamp)
      ? entry.timestamp
      : Date.now();

  const memory: DecayableMemory = {
    id: (entry as { id?: string }).id ?? "",
    importance:
      typeof entry.importance === "number" && Number.isFinite(entry.importance)
        ? entry.importance
        : 0.7,
    confidence: meta.confidence,
    tier: meta.tier,
    accessCount: meta.access_count,
    createdAt,
    lastAccessedAt: meta.last_accessed_at || createdAt,
  };

  return { memory, meta };
}

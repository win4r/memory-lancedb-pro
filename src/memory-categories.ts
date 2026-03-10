/**
 * Memory Categories — 6-category classification system
 *
 * UserMemory: profile, preferences, entities, events
 * AgentMemory: cases, patterns
 */

export const MEMORY_CATEGORIES = [
  "profile",
  "preferences",
  "entities",
  "events",
  "cases",
  "patterns",
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

/** Categories that always merge (skip dedup entirely). */
export const ALWAYS_MERGE_CATEGORIES = new Set<MemoryCategory>(["profile"]);

/** Categories that support MERGE decision from LLM dedup. */
export const MERGE_SUPPORTED_CATEGORIES = new Set<MemoryCategory>([
  "preferences",
  "entities",
  "patterns",
]);

/** Categories that are append-only (CREATE or SKIP only, no MERGE). */
export const APPEND_ONLY_CATEGORIES = new Set<MemoryCategory>([
  "events",
  "cases",
]);

/** Memory tier levels for lifecycle management. */
export type MemoryTier = "core" | "working" | "peripheral";

/** A candidate memory extracted from conversation by LLM. */
export type CandidateMemory = {
  category: MemoryCategory;
  abstract: string; // L0: one-sentence index
  overview: string; // L1: structured markdown summary
  content: string; // L2: full narrative
};

/** Dedup decision from LLM. */
export type DedupDecision = "create" | "merge" | "skip";

export type DedupResult = {
  decision: DedupDecision;
  reason: string;
  matchId?: string; // ID of existing memory to merge with
};

export type ExtractionStats = {
  created: number;
  merged: number;
  skipped: number;
};

/** Validate and normalize a category string. */
export function normalizeCategory(raw: string): MemoryCategory | null {
  const lower = raw.toLowerCase().trim();
  if ((MEMORY_CATEGORIES as readonly string[]).includes(lower)) {
    return lower as MemoryCategory;
  }
  return null;
}

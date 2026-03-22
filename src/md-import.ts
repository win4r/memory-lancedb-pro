/**
 * P2-W2: Markdown Preview Parser / Dry-Run Foundation
 *
 * Parse MEMORY.md and memory/YYYY-MM-DD.md content strings into candidate
 * memory units, classify durable vs noisy, and produce dry-run preview output.
 *
 * Scope (Wave 1 / preview only):
 * - Pure string-in / structured-data-out
 * - No file I/O, no LanceDB writes, no SQLite, no runtime sync changes
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SourceType = "memory-md" | "daily-md" | "plugin-compat-md";
export type Durability = "durable" | "noisy";

export interface MemoryCandidate {
  text: string;
  sourceType: SourceType;
  /** Nearest heading above this item (empty string if none). */
  headingContext: string;
  /** 1-based line number in the source content. */
  lineNumber: number;
  /** ISO date string for daily/plugin-compat files; null for MEMORY.md. */
  date: string | null;
  /** Optional scope hint parsed from structured compatibility lines. */
  scopeHint?: string | null;
}

export interface ClassifiedCandidate extends MemoryCandidate {
  durability: Durability;
}

export interface PreviewCandidate extends ClassifiedCandidate {
  /** Inferred LanceDB scope (e.g. "global", "agent:<id>"). */
  inferredScope: string;
}

export interface PreviewSummary {
  totalFound: number;
  durableCount: number;
  noisyCount: number;
}

export interface PreviewResult {
  candidates: PreviewCandidate[];
  skipped: PreviewCandidate[];
  summary: PreviewSummary;
  warnings: string[];
}

export interface PreviewOptions {
  sourceType: SourceType;
  /** Required when sourceType is "daily-md" or "plugin-compat-md". */
  date?: string;
  /** Override inferred scope for all candidates. */
  scope?: string;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const FRONTMATTER_RE = /^---\s*[\r\n]([\s\S]*?)---\s*[\r\n]/;
const BULLET_RE = /^(\s*[-*+])\s+(.+)$/;
const HEADING_RE = /^(#{1,6})\s+(.+)$/;
const PLUGIN_COMPAT_RE = /^\d{4}-\d{2}-\d{2}T\S+\s+\[([^:\]]+):([^\]]+)\]\s+(?:(?:agent|source)=[^\s]+\s+)*(.*)$/;

/** Strip YAML frontmatter from content. */
function stripFrontmatter(content: string): string {
  return content.replace(FRONTMATTER_RE, "");
}

/** Resolve the nearest heading above a given line index in the line array. */
function resolveHeadingContext(lines: string[], lineIndex: number): string {
  for (let i = lineIndex - 1; i >= 0; i--) {
    const m = lines[i].match(HEADING_RE);
    if (m) return m[2].trim();
  }
  return "";
}

// ---------------------------------------------------------------------------
// Noisy signal detection for classifyCandidates
// ---------------------------------------------------------------------------

const TRANSIENT_KEYWORDS = [
  /\btoday\b/i,
  /\btomorrow\b/i,
  /\bthis (morning|afternoon|evening|week)\b/i,
  /\bmeeting\b/i,
  /\b(check|deploy|end of day|eod)\b/i,
  /\b(todo|follow[- ]up)\b/i,
  /\bright now\b/i,
];

const DURABLE_HEADING_WORDS = [
  "preference",
  "rule",
  "policy",
  "identity",
  "always",
  "never",
  "standard",
];

function isTransient(text: string): boolean {
  return TRANSIENT_KEYWORDS.some((re) => re.test(text));
}

function isTooShort(text: string): boolean {
  return text.trim().length < 8;
}

function hasGenericContent(text: string): boolean {
  const normalized = text.trim().toLowerCase();
  const GENERIC = ["ok", "done", "yes", "no", "x", ".", "...", "noted"];
  return GENERIC.includes(normalized);
}

function headingImpliesDurable(headingContext: string): boolean {
  const lc = headingContext.toLowerCase();
  return DURABLE_HEADING_WORDS.some((w) => lc.includes(w));
}

// ---------------------------------------------------------------------------
// parseMemoryMd
// ---------------------------------------------------------------------------

/**
 * Parse the content of a MEMORY.md file and extract candidate memory units.
 *
 * Extracts:
 * - Bullet/list items (- / * / +)
 * - Short single-sentence paragraphs under a heading (≤ 2 lines, ≥ 20 chars)
 *
 * Does NOT perform any I/O; caller must provide the file content as a string.
 */
export function parseMemoryMd(content: string): MemoryCandidate[] {
  if (!content.trim()) return [];

  const stripped = stripFrontmatter(content);
  const lines = stripped.split(/\r?\n/);
  const candidates: MemoryCandidate[] = [];

  // Track line number offset after frontmatter removal (approximate — we use
  // raw line numbers from the stripped content for simplicity).
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const bulletMatch = line.match(BULLET_RE);
    if (bulletMatch) {
      const text = bulletMatch[2].trim();
      if (!text) continue;
      candidates.push({
        text,
        sourceType: "memory-md",
        headingContext: resolveHeadingContext(lines, i),
        lineNumber: i + 1,
        date: null,
        scopeHint: null,
      });
      continue;
    }

    // Short authored fact paragraphs (non-heading, non-empty, non-bullet)
    const trimmed = line.trim();
    if (
      trimmed.length >= 20 &&
      !HEADING_RE.test(trimmed) &&
      !BULLET_RE.test(trimmed) &&
      !trimmed.startsWith("---")
    ) {
      const heading = resolveHeadingContext(lines, i);
      // Only promote paragraph lines that sit directly under a heading
      if (heading) {
        candidates.push({
          text: trimmed,
          sourceType: "memory-md",
          headingContext: heading,
          lineNumber: i + 1,
          date: null,
          scopeHint: null,
        });
      }
    }
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// parseDailyMd
// ---------------------------------------------------------------------------

/**
 * Parse the content of a daily memory file (memory/YYYY-MM-DD.md) and extract
 * candidate memory units.
 *
 * Extracts bullet items and short event paragraphs (non-heading, ≥ 20 chars).
 * Daily files are treated as noisy-by-default; classification happens in
 * classifyCandidates.
 */
export function parseDailyMd(content: string, date: string): MemoryCandidate[] {
  if (!content.trim()) return [];

  const lines = content.split(/\r?\n/);
  const candidates: MemoryCandidate[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const bulletMatch = line.match(BULLET_RE);
    if (bulletMatch) {
      const text = bulletMatch[2].trim();
      if (!text) continue;
      candidates.push({
        text,
        sourceType: "daily-md",
        headingContext: resolveHeadingContext(lines, i),
        lineNumber: i + 1,
        date,
        scopeHint: null,
      });
      continue;
    }

    // Short event paragraphs (non-heading, non-empty)
    const trimmed = line.trim();
    if (
      trimmed.length >= 20 &&
      !HEADING_RE.test(trimmed) &&
      !BULLET_RE.test(trimmed)
    ) {
      candidates.push({
        text: trimmed,
        sourceType: "daily-md",
        headingContext: resolveHeadingContext(lines, i),
        lineNumber: i + 1,
        date,
        scopeHint: null,
      });
    }
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// parsePluginCompatibilityMd
// ---------------------------------------------------------------------------

/**
 * Parse plugin-managed compatibility Markdown lines written under
 * memory/plugins/memory-lancedb-pro/YYYY-MM-DD.md.
 *
 * Expected line shape:
 * - 2026-03-22T05:00:00.000Z [preferences:agent:main] agent=main source=smart-extract:create 用户喜欢乌龙茶
 */
export function parsePluginCompatibilityMd(content: string, date: string): MemoryCandidate[] {
  if (!content.trim()) return [];

  const lines = content.split(/\r?\n/);
  const candidates: MemoryCandidate[] = [];

  for (let i = 0; i < lines.length; i++) {
    const bulletMatch = lines[i].match(BULLET_RE);
    if (!bulletMatch) continue;

    const body = bulletMatch[2].trim();
    if (!body) continue;

    const compat = body.match(PLUGIN_COMPAT_RE);
    if (compat) {
      const category = compat[1].trim();
      const scopeHint = compat[2].trim();
      const text = compat[3].trim();
      if (!text) continue;
      candidates.push({
        text,
        sourceType: "plugin-compat-md",
        headingContext: `plugin:${category}`,
        lineNumber: i + 1,
        date,
        scopeHint,
      });
      continue;
    }

    // Fallback: keep any unmatched bullet as plugin-managed durable text.
    candidates.push({
      text: body,
      sourceType: "plugin-compat-md",
      headingContext: "plugin:unknown",
      lineNumber: i + 1,
      date,
      scopeHint: null,
    });
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// classifyCandidates
// ---------------------------------------------------------------------------

/**
 * Classify each candidate as "durable" or "noisy".
 *
 * Durable signals:
 * - From memory-md under a preference/rule/identity heading
 * - Text expresses a stable preference, rule, or fact (no time-bound language)
 * - Text is substantive (≥ 8 chars, non-generic)
 *
 * Noisy signals:
 * - Too short or generic
 * - Contains time-bound / transient language
 * - From daily-md unless clearly a decision or discovery
 *
 * Returns a new array; does not mutate input.
 */
export function classifyCandidates(candidates: MemoryCandidate[]): ClassifiedCandidate[] {
  return candidates.map((c) => {
    const durability = determineDurability(c);
    return { ...c, durability };
  });
}

function determineDurability(c: MemoryCandidate): Durability {
  if (c.sourceType === "plugin-compat-md") {
    return c.text.trim().length > 0 ? "durable" : "noisy";
  }

  if (isTooShort(c.text) || hasGenericContent(c.text)) return "noisy";
  if (isTransient(c.text)) return "noisy";

  if (c.sourceType === "memory-md") {
    // memory-md items are durable by default unless they contain transient signals
    return "durable";
  }

  // daily-md: noisy by default unless heading implies durable context
  if (headingImpliesDurable(c.headingContext)) return "durable";

  // Daily items that read like decisions/discoveries can be durable
  if (isDurableDiscovery(c.text)) return "durable";

  return "noisy";
}

const DISCOVERY_PATTERNS = [
  /\b(decided|discovered|confirmed|established|resolved|fixed|shipped|learned)\b/i,
  /\b(always|never|must|should not|do not)\b/i,
];

function isDurableDiscovery(text: string): boolean {
  return DISCOVERY_PATTERNS.some((re) => re.test(text));
}

// ---------------------------------------------------------------------------
// Scope inference
// ---------------------------------------------------------------------------

/**
 * Infer a LanceDB scope string for a candidate.
 * Default: "global" for memory-md; "agent:main" for daily-md.
 * Heading context can refine this heuristic in the future.
 */
function inferScope(c: ClassifiedCandidate, overrideScope?: string): string {
  if (overrideScope) return overrideScope;
  if (c.scopeHint) return c.scopeHint;
  return c.sourceType === "memory-md" ? "global" : "agent:main";
}

// ---------------------------------------------------------------------------
// previewMd
// ---------------------------------------------------------------------------

/**
 * Produce a dry-run preview of what would be imported from a Markdown source.
 *
 * Steps:
 * 1. Parse content into raw candidates (based on sourceType)
 * 2. Classify each candidate as durable or noisy
 * 3. Attach inferred scope
 * 4. Split into candidates (durable) and skipped (noisy)
 * 5. Build summary + warnings
 *
 * This function is read-only: no writes, no external calls.
 */
export function previewMd(content: string, options: PreviewOptions): PreviewResult {
  const { sourceType, date, scope } = options;

  // 1. Parse
  const rawCandidates =
    sourceType === "memory-md"
      ? parseMemoryMd(content)
      : sourceType === "plugin-compat-md"
        ? parsePluginCompatibilityMd(content, date ?? "")
        : parseDailyMd(content, date ?? "");

  // 2. Classify
  const classified = classifyCandidates(rawCandidates);

  // 3. Attach inferred scope
  const withScope: PreviewCandidate[] = classified.map((c) => ({
    ...c,
    inferredScope: inferScope(c, scope),
  }));

  // 4. Split
  const candidates = withScope.filter((c) => c.durability === "durable");
  const skipped = withScope.filter((c) => c.durability === "noisy");

  // 5. Summary + warnings
  const warnings: string[] = [];
  if (rawCandidates.length === 0 && content.trim().length > 0) {
    warnings.push("Content is non-empty but no parseable candidates were found.");
  }
  if ((sourceType === "daily-md" || sourceType === "plugin-compat-md") && !date) {
    warnings.push(`No date provided for ${sourceType} source; scope inference may be inaccurate.`);
  }

  const summary: PreviewSummary = {
    totalFound: withScope.length,
    durableCount: candidates.length,
    noisyCount: skipped.length,
  };

  return { candidates, skipped, summary, warnings };
}

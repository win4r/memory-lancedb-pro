/**
 * Noise Filter
 * Filters out low-quality memories (meta-questions, agent denials, session boilerplate)
 * and strips untrusted metadata wrappers from text before storage/retrieval.
 */

// Agent-side denial patterns
const DENIAL_PATTERNS = [
  /i don'?t have (any )?(information|data|memory|record)/i,
  /i'?m not sure about/i,
  /i don'?t recall/i,
  /i don'?t remember/i,
  /it looks like i don'?t/i,
  /i wasn'?t able to find/i,
  /no (relevant )?memories found/i,
  /i don'?t have access to/i,
];

// User-side meta-question patterns (about memory itself, not content)
const META_QUESTION_PATTERNS = [
  /\bdo you (remember|recall|know about)\b/i,
  /\bcan you (remember|recall)\b/i,
  /\bdid i (tell|mention|say|share)\b/i,
  /\bhave i (told|mentioned|said)\b/i,
  /\bwhat did i (tell|say|mention)\b/i,
];

// Session boilerplate
const BOILERPLATE_PATTERNS = [
  /^(hi|hello|hey|good morning|good evening|greetings)/i,
  /^fresh session/i,
  /^new session/i,
  /^HEARTBEAT/i,
];

// Known noisy wrappers injected by chat transport / system envelopes
const METADATA_BLOCK_PATTERNS = [
  /Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```/gi,
  /Sender \(untrusted metadata\):\s*```json[\s\S]*?```/gi,
  /\[Queued messages while agent was busy\]/gi,
  /^\s*---\s*Queued\s*#\d+\s*$/gmi,
  /^\s*Queued\s*#\d+\s*$/gmi,
  /^\s*---\s*$/gmi,
];

const METADATA_MARKERS = [
  /Conversation info \(untrusted metadata\)/i,
  /Sender \(untrusted metadata\)/i,
  /\[Queued messages while agent was busy\]/i,
  /Queued\s*#\d+/i,
];

export interface NoiseFilterOptions {
  /** Filter agent denial responses (default: true) */
  filterDenials?: boolean;
  /** Filter meta-questions about memory (default: true) */
  filterMetaQuestions?: boolean;
  /** Filter session boilerplate (default: true) */
  filterBoilerplate?: boolean;
}

const DEFAULT_OPTIONS: Required<NoiseFilterOptions> = {
  filterDenials: true,
  filterMetaQuestions: true,
  filterBoilerplate: true,
};

/**
 * Remove transport/system wrappers while preserving human-readable content.
 */
export function sanitizeMemoryText(text: string): string {
  let cleaned = (text || "").trim();
  if (!cleaned) return "";

  for (const pattern of METADATA_BLOCK_PATTERNS) {
    cleaned = cleaned.replace(pattern, " ");
  }

  cleaned = cleaned
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();

  return cleaned;
}

/**
 * Check if a memory text is noise that should be filtered out.
 * Returns true if the text is noise.
 */
export function isNoise(text: string, options: NoiseFilterOptions = {}): boolean {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const trimmed = (text || "").trim();

  if (trimmed.length < 5) return true;

  const sanitized = sanitizeMemoryText(trimmed);
  if (sanitized.length < 5) return true;

  // If text is mostly wrappers/metadata after sanitization, treat as noise.
  const hasMetadataMarker = METADATA_MARKERS.some(p => p.test(trimmed));
  if (hasMetadataMarker) {
    const keepRatio = sanitized.length / Math.max(1, trimmed.length);
    if (keepRatio < 0.35) return true;
  }

  if (opts.filterDenials && DENIAL_PATTERNS.some(p => p.test(sanitized))) return true;
  if (opts.filterMetaQuestions && META_QUESTION_PATTERNS.some(p => p.test(sanitized))) return true;
  if (opts.filterBoilerplate && BOILERPLATE_PATTERNS.some(p => p.test(sanitized))) return true;

  return false;
}

/**
 * Filter an array of items, removing noise entries.
 */
export function filterNoise<T>(
  items: T[],
  getText: (item: T) => string,
  options?: NoiseFilterOptions
): T[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  return items.filter(item => !isNoise(getText(item), opts));
}

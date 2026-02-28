/**
 * Adaptive Retrieval
 * Determines whether a query needs memory retrieval at all.
 * Skips retrieval for greetings, commands, simple instructions, and system messages.
 * Saves embedding API calls and reduces noise injection.
 */

// Queries that are clearly NOT memory-retrieval candidates
const SKIP_PATTERNS = [
  // Greetings & pleasantries
  /^(hi|hello|hey|good\s*(morning|afternoon|evening|night)|greetings|yo|sup|howdy|what'?s up)\b/i,
  // System/bot commands
  /^\//,  // slash commands
  /^(run|build|test|ls|cd|git|npm|pip|docker|curl|cat|grep|find|make|sudo)\b/i,
  // Simple affirmations/negations
  /^(yes|no|yep|nope|ok|okay|sure|fine|thanks|thank you|thx|ty|got it|understood|cool|nice|great|good|perfect|awesome|👍|👎|✅|❌)\s*[.!]?$/i,
  // Continuation prompts
  /^(go ahead|continue|proceed|do it|start|begin|next|实施|开始|继续|好的|可以|行)\s*[.!]?$/i,
  // Pure emoji
  /^[\p{Emoji}\s]+$/u,
  // Heartbeat/system (match anywhere, not just at start, to handle prefixed formats)
  /HEARTBEAT/i,
  /^\[System/i,
  // Single-word utility pings
  /^(ping|pong|test|debug)\s*[.!?]?$/i,
];

// Queries that SHOULD trigger retrieval even if short
const FORCE_RETRIEVE_PATTERNS = [
  /\b(remember|recall|forgot|memory|memories)\b/i,
  /\b(last time|before|previously|earlier|yesterday|ago)\b/i,
  /\b(my (name|email|phone|address|birthday|preference))\b/i,
  /\b(what did (i|we)|did i (tell|say|mention))\b/i,
  /(你记得|之前|上次|以前|还记得|提到过|说过)/i,
];

/**
 * Normalize the raw prompt before applying skip/force rules.
 *
 * OpenClaw may wrap cron prompts like:
 *   "[cron:<jobId> <jobName>] run ..."
 *
 * We strip such prefixes so command-style prompts are properly detected and we
 * can skip auto-recall injection (saves tokens).
 */
function normalizeQuery(query: string): string {
  let s = query.trim();

  // Strip OpenClaw cron wrapper prefix.
  s = s.replace(/^\[cron:[^\]]+\]\s*/i, "");

  // Strip OpenClaw injected metadata header used in some transcripts.
  if (/^Conversation info \(untrusted metadata\):/i.test(s)) {
    s = s.replace(/^Conversation info \(untrusted metadata\):\s*/i, "");
    // If there is a blank-line separator, keep only the part after it.
    const parts = s.split(/\n\s*\n/, 2);
    if (parts.length === 2) s = parts[1];
  }

  return s.trim();
}

/**
 * Determine if a query should skip memory retrieval.
 * Returns true if retrieval should be skipped.
 * @param query The raw prompt text
 * @param minLength Optional minimum length override (if set, overrides built-in thresholds)
 */
export function shouldSkipRetrieval(query: string, minLength?: number): boolean {
  const trimmed = normalizeQuery(query);

  // Force retrieve if query has memory-related intent (checked FIRST,
  // before length check, so short CJK queries like "你记得吗" aren't skipped)
  if (FORCE_RETRIEVE_PATTERNS.some(p => p.test(trimmed))) return false;

  // Too short to be meaningful
  if (trimmed.length < 5) return true;

  // Skip if matches any skip pattern
  if (SKIP_PATTERNS.some(p => p.test(trimmed))) return true;

  // If caller provides a custom minimum length, use it
  if (minLength !== undefined && minLength > 0) {
    if (trimmed.length < minLength && !trimmed.includes('?') && !trimmed.includes('？')) return true;
    return false;
  }

  // Skip very short non-question messages (likely commands or affirmations)
  // CJK characters carry more meaning per character, so use a lower threshold
  const hasCJK = /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(trimmed);
  const defaultMinLength = hasCJK ? 6 : 15;
  if (trimmed.length < defaultMinLength && !trimmed.includes('?') && !trimmed.includes('？')) return true;

  // Default: do retrieve
  return false;
}

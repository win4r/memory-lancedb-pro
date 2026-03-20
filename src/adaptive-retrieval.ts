/**
 * Adaptive Retrieval
 * Determines whether a query needs memory retrieval at all.
 * Skips retrieval for greetings, commands, simple instructions, and system messages.
 * Saves embedding API calls and reduces noise injection.
 */

// Control/system prompts that should never trigger retrieval.
const CONTROL_PROMPT_SKIP_PATTERNS = [
  /A new session was started via \/new or \/reset/i,
  /Execute your Session Startup sequence now/i,
  /(^|\n)\s*\/note\b/i,
  /(^|\n)\s*NO_REPLY\s*$/i,
  /(^|\n)\s*HEARTBEAT_OK\s*$/i,
];

// Weak continuation prompts that often rely on prior session context.
const WEAK_CONTINUATION_PATTERNS = [
  /^(?:那|那就|那你|你|请|請)?\s*(?:go ahead|continue|proceed|resume|carry on|keep going|do it|start|begin|next)\s*[.!?]?$/i,
  /^(?:那|那就|那你|你|请|請)?\s*(?:继续|繼續|继续吧|繼續吧|接着|接著|接着来|接著來|然后|然後|然后呢|然後呢|下一步|开始|開始|开始吧|開始吧|往下|接下去|接下來)\s*[.!?？]*$/i,
  /^[?？]+$/,
];

// Slightly broader than WEAK_CONTINUATION_PATTERNS, but used only for
// continuity fallback.
const WEAK_CARRY_FORWARD_PATTERNS = [
  /^(?:嗯|嗯嗯|唔|呃|哦|噢)\s*[.!?？]*$/i,
  /^(?:好|好的|好的呀|好吧|可以|行|行吧|ok|okay|sure|alright|got it)\s*[.!?？]*$/i,
  /^[\p{Emoji}\s]+$/u,
];

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
  /^(go ahead|continue|proceed|do it|start|begin|next|实施|實施|开始|開始|继续|繼續|好的|可以|行)\s*[.!]?$/i,
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
  /(你记得|[你妳]記得|之前|上次|以前|还记得|還記得|提到过|提到過|说过|說過)/i,
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

  // 1. Strip OpenClaw injected metadata headers (Conversation info or Sender).
  // Use a global regex to strip all metadata blocks including following blank lines.
  const metadataPattern = /^(Conversation info|Sender) \(untrusted metadata\):[\s\S]*?\n\s*\n/gim;
  s = s.replace(metadataPattern, "");

  // 2. Strip OpenClaw cron wrapper prefix.
  s = s.trim().replace(/^\[cron:[^\]]+\]\s*/i, "");

  // 3. Strip OpenClaw timestamp prefix [Mon 2026-03-02 04:21 GMT+8].
  s = s.trim().replace(/^\[[A-Za-z]{3}\s\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\s[^\]]+\]\s*/, "");

  const result = s.trim();
  return result;
}

export function isWeakContinuationPrompt(query: string): boolean {
  const trimmed = normalizeQuery(query);
  if (!trimmed) return false;
  return WEAK_CONTINUATION_PATTERNS.some((pattern) => pattern.test(trimmed));
}

export function shouldAttemptContinuityFallback(query: string): boolean {
  const trimmed = normalizeQuery(query);
  if (!trimmed) return false;
  if (isWeakContinuationPrompt(trimmed)) return true;
  return WEAK_CARRY_FORWARD_PATTERNS.some((pattern) => pattern.test(trimmed));
}

/**
 * Determine if a query should skip memory retrieval.
 * Returns true if retrieval should be skipped.
 * @param query The raw prompt text
 * @param minLength Optional minimum length override (if set, overrides built-in thresholds)
 */
export function shouldSkipRetrieval(query: string, minLength?: number): boolean {
  const trimmed = normalizeQuery(query);

  if (CONTROL_PROMPT_SKIP_PATTERNS.some((pattern) => pattern.test(trimmed))) {
    return true;
  }

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

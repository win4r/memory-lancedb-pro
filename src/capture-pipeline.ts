import type { MemoryEntry } from "./store.js";

export type CaptureRole = "user" | "assistant";
export type CaptureSourceApp = "openclaw" | "claude" | "codex" | "wechat" | "unknown";
export type CaptureTier = "discard" | "candidate" | "formal";

export interface CapturePolicyConfig {
  candidateMinScore: number;
  formalMinScore: number;
  assistantMaxScore: number;
  includeCandidateInRecall: boolean;
}

export interface CaptureScoreInput {
  text: string;
  role: CaptureRole;
  sourceApp: CaptureSourceApp;
  sourceProvider?: string;
  confirmed?: boolean;
  hasExistingSimilar?: boolean;
  repetitionCount?: number;
  captureAssistant?: boolean;
}

export interface CaptureScoreBreakdown {
  intent: number;
  stability: number;
  evidence: number;
  validation: number;
  futureReuse: number;
  noisePenalty: number;
}

export interface CaptureScoreResult {
  score: number;
  tier: CaptureTier;
  category: MemoryEntry["category"] extends "reflection"
    ? never
    : Exclude<MemoryEntry["category"], "reflection">;
  normalizedKey: string;
  reasons: string[];
  breakdown: CaptureScoreBreakdown;
  rejectedReason?: string;
}

export interface CaptureMetadataPayload {
  memoryTier: Exclude<CaptureTier, "discard">;
  captureScore: number;
  captureReasons: string[];
  normalizedKey: string;
  sourceApp: CaptureSourceApp;
  sourceRole: CaptureRole;
  sourceProvider?: string;
  sourcePath?: string;
  sourceSessionId?: string;
  sourceModel?: string;
  sourceWorkspace?: string;
  ingestionMode?: string;
  confirmed?: boolean;
  occurredAt?: number;
}

export interface CaptureMetadata extends CaptureMetadataPayload {
  captureVersion: number;
  firstSeenAt?: number;
  lastSeenAt?: number;
  occurrences?: number;
}

export const DEFAULT_CAPTURE_POLICY: CapturePolicyConfig = {
  candidateMinScore: 4,
  formalMinScore: 8,
  assistantMaxScore: 5,
  includeCandidateInRecall: false,
};

const MEMORY_MANAGEMENT_PATTERNS = [
  /\b(memory-pro|memory_store|memory_recall|memory_forget|memory_update)\b/i,
  /\bopenclaw\s+memory-pro\b/i,
  /\b(delete|remove|forget|purge|cleanup|clean up|clear)\b.*\b(memory|memories|entry|entries)\b/i,
  /\b(memory|memories)\b.*\b(delete|remove|forget|purge|cleanup|clean up|clear)\b/i,
  /(删除|刪除|清理|清除).{0,12}(记忆|記憶|memory)/i,
];

const SECRET_PATTERNS = [
  /Bearer\s+[A-Za-z0-9\-._~+/]+=*/g,
  /\bsk-[A-Za-z0-9]{20,}\b/g,
  /\bsk-proj-[A-Za-z0-9\-_]{20,}\b/g,
  /\bsk-ant-[A-Za-z0-9\-_]{20,}\b/g,
  /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g,
  /\bgithub_pat_[A-Za-z0-9_]{22,}\b/g,
  /\b(?:token|api[_-]?key|secret|password)\s*[:=]\s*["']?[^\s"',;)}\]]{6,}["']?\b/gi,
  /-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----/g,
];

const EXPLICIT_REMEMBER_PATTERNS = [
  /zapamatuj si|pamatuj|remember/i,
  /記住|记住|記一下|记一下|別忘了|别忘了|備註|备注|存檔|存档|存起來|存起来|存一下/i,
  /make a note|note this|save this|keep this|store this|remember this/i,
];

const PREFERENCE_PATTERNS = [
  /preferuji|radši|prefer|like|love|hate|want|need|care/i,
  /偏好|喜好|喜歡|喜欢|討厭|讨厌|不喜歡|不喜欢|愛用|爱用|習慣|习惯|倾向/i,
  /主模型|备用模型|備用模型|默认模型|默認模型|默认用|優先用|优先用/i,
];

const DECISION_PATTERNS = [
  /rozhodli jsme|budeme používat/i,
  /\b(we )?decided\b|we'?ll use|we will use|switch(ed)? to|migrate(d)? to|going forward|from now on/i,
  /決定|决定|選擇了|选择了|改用|換成|换成|以後用|以后用|流程|规则|規則|SOP|约定|約定/i,
  /captureassistant|candidate|formal|候选|候選|正式记忆|正式記憶/i,
];

const ENTITY_PATTERNS = [
  /\+\d{10,}/,
  /[\w.-]+@[\w.-]+\.\w+/,
  /můj\s+\w+\s+je|je\s+můj/i,
  /my\s+\w+\s+is|is\s+my/i,
  /我的\S+是|叫我|稱呼|称呼|我叫|名字是/i,
];

const STABLE_FACT_PATTERNS = [
  /always|never|important|habit|default|normally|usually/i,
  /總是|总是|從不|从不|一直|每次都|老是|重要|关键|關鍵|默认|默認|长期|長期/i,
];

const LONG_TERM_PATTERNS = [
  /from now on|going forward|always|never|default|habit|standard|policy|workflow/i,
  /以后|以後|长期|長期|固定|默认|默認|习惯|習慣|流程|规则|規則|约定|約定|主模型|备用模型|備用模型/i,
];

const TEMPORARY_NOISE_PATTERNS = [
  /\b(today|tomorrow|tonight|this time|for now|temporary|temporarily|one-off|just once|later)\b/i,
  /今天|明天|今晚|刚刚|剛剛|先这样|先這樣|暂时|暫時|临时|臨時|一次性|待会|待會|稍后|稍後|本次|这次|這次/i,
  /这个对话|這個對話|this chat|this conversation|当前任务|當前任務/i,
];

const SYSTEM_NOISE_PATTERNS = [
  /<relevant-memories>/i,
  /BEGIN UNTRUSTED DATA|END UNTRUSTED DATA/i,
  /^(Conversation info|Sender) \(untrusted metadata\):/im,
  /^# AGENTS\.md instructions/m,
  /^<environment_context>/m,
  /^You are a coding agent running in the Codex CLI/m,
  /^\s*\{\s*"timestamp":/m,
];

const ASSISTANT_MEMORY_PATTERNS = [
  /总结|總結|结论|結論|建议|建議|已设置|已設置|规则|規則|流程|偏好|决定|決定|以后|以後/i,
  /summary|decision|preference|workflow|rule|policy|set to|configured|we should|recommend/i,
];

const ASSISTANT_CONFIRMATION_PATTERNS = [
  /根据你的要求|按你的要求|你明确要求|你已确认|用户确认|已验证|执行结果显示/i,
  /based on your request|you asked|user confirmed|validated|verified by execution/i,
];

function hasPattern(patterns: RegExp[], text: string): boolean {
  return patterns.some((pattern) => {
    pattern.lastIndex = 0;
    return pattern.test(text);
  });
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function normalizeScore(value: number): number {
  return Math.round(clamp(value, 0, 10) * 10) / 10;
}

function normalizeRecallTextKey(text: string): string {
  return String(text)
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function isTooShort(text: string): boolean {
  const hasCJK = /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(text);
  const minLen = hasCJK ? 4 : 10;
  return text.length < minLen;
}

function containsSecretLikeContent(text: string): boolean {
  return hasPattern(SECRET_PATTERNS, text);
}

function containsMostlyCodeFence(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed.includes("```")) return false;
  const withoutFences = trimmed.replace(/```[\s\S]*?```/g, "").trim();
  return withoutFences.length < Math.max(24, trimmed.length * 0.2);
}

function isAssistantMemoryLike(text: string): boolean {
  return hasPattern(ASSISTANT_MEMORY_PATTERNS, text) ||
    hasPattern(EXPLICIT_REMEMBER_PATTERNS, text) ||
    hasPattern(PREFERENCE_PATTERNS, text) ||
    hasPattern(DECISION_PATTERNS, text) ||
    hasPattern(STABLE_FACT_PATTERNS, text);
}

export function detectMemoryCategory(
  text: string,
): Exclude<MemoryEntry["category"], "reflection"> {
  const lower = String(text || "").toLowerCase();
  if (hasPattern(PREFERENCE_PATTERNS, lower)) {
    return "preference";
  }
  if (hasPattern(DECISION_PATTERNS, lower)) {
    return "decision";
  }
  if (hasPattern(ENTITY_PATTERNS, lower)) {
    return "entity";
  }
  if (hasPattern(STABLE_FACT_PATTERNS, lower)) {
    return "fact";
  }
  return "other";
}

export function scoreMemoryCandidate(
  input: CaptureScoreInput,
  policy: CapturePolicyConfig = DEFAULT_CAPTURE_POLICY,
): CaptureScoreResult {
  const text = String(input.text || "").trim();
  const normalizedKey = normalizeRecallTextKey(text);
  const category = detectMemoryCategory(text);
  const reasons: string[] = [];
  const breakdown: CaptureScoreBreakdown = {
    intent: 0,
    stability: 0,
    evidence: 0,
    validation: 0,
    futureReuse: 0,
    noisePenalty: 0,
  };

  const reject = (reason: string): CaptureScoreResult => ({
    score: 0,
    tier: "discard",
    category,
    normalizedKey,
    reasons: [],
    breakdown,
    rejectedReason: reason,
  });

  if (!text) return reject("empty");
  if (isTooShort(text)) return reject("too-short");
  if (text.length > 1200) return reject("too-long");
  if (hasPattern(MEMORY_MANAGEMENT_PATTERNS, text)) return reject("memory-management");
  if (hasPattern(SYSTEM_NOISE_PATTERNS, text)) return reject("system-noise");
  if (containsMostlyCodeFence(text)) return reject("code-block-only");
  if (containsSecretLikeContent(text)) return reject("secret-like-content");
  if (input.role === "assistant" && input.captureAssistant !== true) {
    return reject("assistant-capture-disabled");
  }
  if (input.role === "assistant" && !isAssistantMemoryLike(text)) {
    return reject("assistant-not-memory-like");
  }

  if (hasPattern(EXPLICIT_REMEMBER_PATTERNS, text)) {
    breakdown.intent = 4;
    reasons.push("explicit-remember");
  } else if (hasPattern(PREFERENCE_PATTERNS, text) || hasPattern(DECISION_PATTERNS, text)) {
    breakdown.intent = 3;
    reasons.push(hasPattern(DECISION_PATTERNS, text) ? "decision-or-workflow" : "preference");
  } else if (hasPattern(ENTITY_PATTERNS, text) || hasPattern(STABLE_FACT_PATTERNS, text)) {
    breakdown.intent = 2;
    reasons.push(category === "entity" ? "entity" : "stable-fact");
  } else if (input.role === "assistant" && isAssistantMemoryLike(text)) {
    breakdown.intent = 1.5;
    reasons.push("assistant-summary");
  }

  if (hasPattern(LONG_TERM_PATTERNS, text)) {
    breakdown.stability = 2;
    reasons.push("long-term");
  } else if (category === "preference" || category === "decision" || category === "entity") {
    breakdown.stability = 1.5;
  } else if (category === "fact") {
    breakdown.stability = 1;
  } else if (input.role === "assistant") {
    breakdown.stability = 0.5;
  }

  if ((input.repetitionCount ?? 0) > 1) {
    breakdown.evidence = 2;
    reasons.push("repeated");
  } else if (input.hasExistingSimilar) {
    breakdown.evidence = 2;
    reasons.push("matched-existing");
  } else if (hasPattern(STABLE_FACT_PATTERNS, text)) {
    breakdown.evidence = 1;
  }

  if (input.role === "user" || input.confirmed === true || hasPattern(ASSISTANT_CONFIRMATION_PATTERNS, text)) {
    breakdown.validation = 1;
  }

  if (category === "preference" || category === "decision" || category === "entity") {
    breakdown.futureReuse = 1;
  } else if (category === "fact" || hasPattern(ASSISTANT_MEMORY_PATTERNS, text)) {
    breakdown.futureReuse = 0.5;
  }

  if (hasPattern(TEMPORARY_NOISE_PATTERNS, text)) {
    breakdown.noisePenalty = 5;
    reasons.push("temporary-noise");
  }

  let total = breakdown.intent +
    breakdown.stability +
    breakdown.evidence +
    breakdown.validation +
    breakdown.futureReuse -
    breakdown.noisePenalty;

  if (input.role === "assistant" && input.confirmed !== true) {
    total = Math.min(total, policy.assistantMaxScore);
  }

  const score = normalizeScore(total);
  const tier: CaptureTier = score >= policy.formalMinScore
    ? "formal"
    : score >= policy.candidateMinScore
      ? "candidate"
      : "discard";

  return {
    score,
    tier,
    category,
    normalizedKey,
    reasons: [...new Set(reasons)].slice(0, 6),
    breakdown,
    rejectedReason: tier === "discard" ? "below-threshold" : undefined,
  };
}

export function parseCaptureMetadata(metadataRaw: string | undefined): Record<string, unknown> {
  if (!metadataRaw) return {};
  try {
    const parsed = JSON.parse(metadataRaw);
    return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

export function getCaptureTier(metadataRaw: string | undefined): Exclude<CaptureTier, "discard"> | undefined {
  const parsed = parseCaptureMetadata(metadataRaw);
  const tier = parsed.memoryTier;
  return tier === "candidate" || tier === "formal" ? tier : undefined;
}

export function mergeCaptureMetadata(
  existingMetadataRaw: string | undefined,
  incoming: CaptureMetadataPayload,
): string {
  const existing = parseCaptureMetadata(existingMetadataRaw);
  const previousTier = existing.memoryTier === "formal" || existing.memoryTier === "candidate"
    ? existing.memoryTier
    : undefined;
  const previousReasons = Array.isArray(existing.captureReasons)
    ? existing.captureReasons.filter((item): item is string => typeof item === "string")
    : [];
  const previousOccurrences = typeof existing.occurrences === "number" && Number.isFinite(existing.occurrences)
    ? Math.max(0, Math.floor(existing.occurrences))
    : 0;
  const previousFirstSeenAt = typeof existing.firstSeenAt === "number" && Number.isFinite(existing.firstSeenAt)
    ? existing.firstSeenAt
    : undefined;

  const merged: CaptureMetadata = {
    ...existing,
    captureVersion: 1,
    memoryTier: previousTier === "formal" || incoming.memoryTier === "formal"
      ? "formal"
      : "candidate",
    captureScore: Math.max(
      typeof existing.captureScore === "number" && Number.isFinite(existing.captureScore)
        ? existing.captureScore
        : 0,
      incoming.captureScore,
    ),
    captureReasons: [...new Set([...previousReasons, ...incoming.captureReasons])].slice(0, 8),
    normalizedKey: incoming.normalizedKey,
    sourceApp: incoming.sourceApp,
    sourceRole: incoming.sourceRole,
    sourceProvider: incoming.sourceProvider,
    sourcePath: incoming.sourcePath,
    sourceSessionId: incoming.sourceSessionId,
    sourceModel: incoming.sourceModel,
    sourceWorkspace: incoming.sourceWorkspace,
    ingestionMode: incoming.ingestionMode,
    confirmed: Boolean(existing.confirmed) || incoming.confirmed === true,
    occurredAt: incoming.occurredAt,
    firstSeenAt: previousFirstSeenAt ?? incoming.occurredAt ?? Date.now(),
    lastSeenAt: incoming.occurredAt ?? Date.now(),
    occurrences: previousOccurrences + 1,
  };

  return JSON.stringify(merged);
}

export function isRecallEligibleMemory(
  entry: Pick<MemoryEntry, "metadata">,
  includeCandidateInRecall = false,
): boolean {
  const tier = getCaptureTier(entry.metadata);
  if (!tier) return true;
  if (tier === "formal") return true;
  return includeCandidateInRecall === true;
}

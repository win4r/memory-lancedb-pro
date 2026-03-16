import { buildAtomicMemoryMetadata, type AtomicMemorySourceKind, type AtomicMemoryUnitType } from "./atomic-memory.js";
import type { Embedder } from "./embedder.js";
import type { MemoryEntry, MemoryStore } from "./store.js";

const MEMORY_TRIGGERS = [
  /zapamatuj si|pamatuj|remember/i,
  /preferuji|radši|nechci|prefer/i,
  /rozhodli jsme|budeme používat/i,
  /\b(we )?decided\b|we'?ll use|we will use|switch(ed)? to|migrate(d)? to|going forward|from now on/i,
  /\+\d{10,}/,
  /[\w.-]+@[\w.-]+\.\w+/,
  /můj\s+\w+\s+je|je\s+můj/i,
  /my\s+\w+\s+is|is\s+my/i,
  /i (like|prefer|hate|love|want|need|care)/i,
  /always|never|important/i,
  /記住|记住|記一下|记一下|別忘了|别忘了|備註|备注/,
  /偏好|喜好|喜歡|喜欢|討厭|讨厌|不喜歡|不喜欢|愛用|爱用|習慣|习惯/,
  /決定|决定|選擇了|选择了|改用|換成|换成|以後用|以后用/,
  /我的\S+是|叫我|稱呼|称呼/,
  /老是|講不聽|總是|总是|從不|从不|一直|每次都/,
  /重要|關鍵|关键|注意|千萬別|千万别/,
  /幫我|筆記|存檔|存起來|存一下|重點|原則|底線/,
] as const;

const CAPTURE_EXCLUDE_PATTERNS = [
  /\b(memory-pro|memory_store|memory_recall|memory_forget|memory_update)\b/i,
  /\bopenclaw\s+memory-pro\b/i,
  /\b(delete|remove|forget|purge|cleanup|clean up|clear)\b.*\b(memory|memories|entry|entries)\b/i,
  /\b(memory|memories)\b.*\b(delete|remove|forget|purge|cleanup|clean up|clear)\b/i,
  /\bhow do i\b.*\b(delete|remove|forget|purge|cleanup|clear)\b/i,
  /(删除|刪除|清理|清除).{0,12}(记忆|記憶|memory)/i,
] as const;

export type MessageRole = "user" | "assistant";

export interface NormalizedAutoCaptureRecord {
  text: string;
  role: MessageRole;
}

export interface AutoCaptureCandidate extends NormalizedAutoCaptureRecord {
  sourceKind: AtomicMemorySourceKind;
  category: MemoryEntry["category"];
  unitType: AtomicMemoryUnitType;
}

export interface ExtractAutoCaptureRecordsOptions {
  captureAssistant?: boolean;
  normalizeText?: (role: MessageRole, text: string) => string | null;
}

export interface ExtractAutoCaptureRecordsResult {
  records: NormalizedAutoCaptureRecord[];
  skippedCount: number;
}

export interface StoreAutoCaptureCandidatesParams {
  candidates: AutoCaptureCandidate[];
  store: MemoryStore;
  embedder: Pick<Embedder, "embedPassage">;
  scope: string;
  importance?: number;
  limit?: number;
  buildBaseMetadata?: (candidate: AutoCaptureCandidate) => string | undefined;
}

export function shouldCapture(text: string): boolean {
  let normalized = text.trim();

  const metadataPattern = /^(Conversation info|Sender) \(untrusted metadata\):[\s\S]*?\n\s*\n/gim;
  normalized = normalized.replace(metadataPattern, "");

  const hasCjk = /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(normalized);
  const minLen = hasCjk ? 4 : 10;
  if (normalized.length < minLen || normalized.length > 500) return false;
  if (normalized.includes("<relevant-memories>")) return false;
  if (normalized.startsWith("<") && normalized.includes("</")) return false;
  if (normalized.includes("**") && normalized.includes("\n-")) return false;
  const emojiCount = (normalized.match(/[\u{1F300}-\u{1F9FF}]/gu) || []).length;
  if (emojiCount > 3) return false;
  if (CAPTURE_EXCLUDE_PATTERNS.some((pattern) => pattern.test(normalized))) return false;

  return MEMORY_TRIGGERS.some((pattern) => pattern.test(normalized));
}

export function detectCategory(text: string): MemoryEntry["category"] {
  const lower = text.toLowerCase();
  if (/prefer|radši|like|love|hate|want|偏好|喜歡|喜欢|討厭|讨厌|不喜歡|不喜欢|愛用|爱用|習慣|习惯/i.test(lower)) {
    return "preference";
  }
  if (/rozhodli|decided|we decided|will use|we will use|we'?ll use|switch(ed)? to|migrate(d)? to|going forward|from now on|budeme|決定|决定|選擇了|选择了|改用|換成|换成|以後用|以后用|規則|流程|SOP/i.test(lower)) {
    return "decision";
  }
  if (/\+\d{10,}|@[\w.-]+\.\w+|is called|jmenuje se|我的\S+是|叫我|稱呼|称呼/i.test(lower)) {
    return "entity";
  }
  if (/\b(is|are|has|have|je|má|jsou)\b|總是|总是|從不|从不|一直|每次都|老是/i.test(lower)) {
    return "fact";
  }
  return "other";
}

export function detectAtomicUnitType(
  text: string,
  category: MemoryEntry["category"],
): AtomicMemoryUnitType {
  const lower = text.toLowerCase();
  if (/pitfall|lesson learned|踩坑|教训|千萬別|千万别|別再|别再/i.test(lower)) {
    return "lesson";
  }
  if (/\b(path|shell|node|zsh|bash|launchd|gateway|workspace|env|environment)\b|环境|環境|路径|路徑/i.test(lower)) {
    return "environment";
  }
  switch (category) {
    case "preference":
    case "fact":
    case "decision":
    case "entity":
      return category;
    default:
      return "other";
  }
}

function extractTextBlocks(content: unknown): string[] {
  if (typeof content === "string") return [content];
  if (!Array.isArray(content)) return [];
  const texts: string[] = [];
  for (const block of content) {
    if (
      block &&
      typeof block === "object" &&
      "type" in block &&
      (block as Record<string, unknown>).type === "text" &&
      "text" in block &&
      typeof (block as Record<string, unknown>).text === "string"
    ) {
      texts.push((block as Record<string, unknown>).text as string);
    }
  }
  return texts;
}

export function extractAutoCaptureRecords(
  messages: unknown[],
  options: ExtractAutoCaptureRecordsOptions = {},
): ExtractAutoCaptureRecordsResult {
  const records: NormalizedAutoCaptureRecord[] = [];
  let skippedCount = 0;

  for (const message of messages) {
    if (!message || typeof message !== "object") continue;
    const msgObj = message as Record<string, unknown>;
    const role = msgObj.role;
    if (role !== "user" && !(options.captureAssistant === true && role === "assistant")) {
      continue;
    }

    for (const text of extractTextBlocks(msgObj.content)) {
      const normalized = options.normalizeText
        ? options.normalizeText(role as MessageRole, text)
        : text.trim();
      if (!normalized) {
        skippedCount++;
        continue;
      }
      records.push({ text: normalized, role: role as MessageRole });
    }
  }

  return { records, skippedCount };
}

export function toAutoCaptureCandidates(records: NormalizedAutoCaptureRecord[]): AutoCaptureCandidate[] {
  return records.map((record) => {
    const category = detectCategory(record.text);
    return {
      ...record,
      sourceKind: record.role === "assistant" ? "agent" : "user",
      category,
      unitType: detectAtomicUnitType(record.text, category),
    };
  });
}

export async function storeAutoCaptureCandidates({
  candidates,
  store,
  embedder,
  scope,
  importance = 0.7,
  limit = 3,
  buildBaseMetadata,
}: StoreAutoCaptureCandidatesParams): Promise<MemoryEntry[]> {
  const stored: MemoryEntry[] = [];

  for (const candidate of candidates.slice(0, limit)) {
    const vector = await embedder.embedPassage(candidate.text);

    let existing: Awaited<ReturnType<typeof store.vectorSearch>> = [];
    try {
      existing = await store.vectorSearch(vector, 1, 0.1, [scope]);
    } catch {
      existing = [];
    }

    if (existing.length > 0 && existing[0].score > 0.95) {
      continue;
    }

    const baseMetadata = buildBaseMetadata?.(candidate);
    const metadata = buildAtomicMemoryMetadata(
      candidate.category,
      {
        unitType: candidate.unitType,
        sourceKind: candidate.sourceKind,
        confidence: importance,
      },
      baseMetadata,
    );

    const entry = await store.store({
      text: candidate.text,
      vector,
      importance,
      category: candidate.category,
      scope,
      ...(metadata ? { metadata } : {}),
    });
    stored.push(entry);
  }

  return stored;
}

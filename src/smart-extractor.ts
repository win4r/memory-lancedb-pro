/**
 * Smart Memory Extractor — LLM-powered extraction pipeline
 * Replaces regex-triggered capture with intelligent 6-category extraction.
 *
 * Pipeline: conversation → LLM extract → candidates → dedup → persist
 *
 */

import type { MemoryStore, MemorySearchResult } from "./store.js";
import type { Embedder } from "./embedder.js";
import type { LlmClient } from "./llm-client.js";
import {
  buildExtractionPrompt,
  buildDedupPrompt,
  buildMergePrompt,
} from "./extraction-prompts.js";
import {
  type CandidateMemory,
  type DedupDecision,
  type DedupResult,
  type ExtractionStats,
  type MemoryCategory,
  ALWAYS_MERGE_CATEGORIES,
  MERGE_SUPPORTED_CATEGORIES,
  MEMORY_CATEGORIES,
  normalizeCategory,
} from "./memory-categories.js";
import { isNoise } from "./noise-filter.js";
import { buildSmartMetadata, parseSmartMetadata, stringifySmartMetadata } from "./smart-metadata.js";

// ============================================================================
// Constants
// ============================================================================

const SIMILARITY_THRESHOLD = 0.7;
const MAX_SIMILAR_FOR_PROMPT = 3;
const MAX_MEMORIES_PER_EXTRACTION = 5;
const VALID_DECISIONS = new Set<string>(["create", "merge", "skip"]);

// ============================================================================
// Smart Extractor
// ============================================================================

export interface SmartExtractorConfig {
  /** User identifier for extraction prompt. */
  user?: string;
  /** Minimum conversation messages before extraction triggers. */
  extractMinMessages?: number;
  /** Maximum characters of conversation text to process. */
  extractMaxChars?: number;
  /** Default scope for new memories. */
  defaultScope?: string;
  /** Logger function. */
  log?: (msg: string) => void;
}

export interface ExtractPersistOptions {
  /** Target scope for newly created memories. */
  scope?: string;
  /** Scopes visible to the current agent for dedup/merge. */
  scopeFilter?: string[];
}

export class SmartExtractor {
  private log: (msg: string) => void;

  constructor(
    private store: MemoryStore,
    private embedder: Embedder,
    private llm: LlmClient,
    private config: SmartExtractorConfig = {},
  ) {
    this.log = config.log ?? ((msg: string) => console.log(msg));
  }

  // --------------------------------------------------------------------------
  // Main entry point
  // --------------------------------------------------------------------------

  /**
   * Extract memories from a conversation text and persist them.
   * Returns extraction statistics.
   */
  async extractAndPersist(
    conversationText: string,
    sessionKey: string = "unknown",
    options: ExtractPersistOptions = {},
  ): Promise<ExtractionStats> {
    const stats: ExtractionStats = { created: 0, merged: 0, skipped: 0 };
    const targetScope = options.scope ?? this.config.defaultScope ?? "global";
    const scopeFilter =
      options.scopeFilter && options.scopeFilter.length > 0
        ? options.scopeFilter
        : [targetScope];

    // Step 1: LLM extraction
    const candidates = await this.extractCandidates(conversationText);

    if (candidates.length === 0) {
      this.log("memory-pro: smart-extractor: no memories extracted");
      return stats;
    }

    this.log(
      `memory-pro: smart-extractor: extracted ${candidates.length} candidate(s)`,
    );

    // Step 2: Process each candidate through dedup pipeline
    for (const candidate of candidates.slice(0, MAX_MEMORIES_PER_EXTRACTION)) {
      try {
        await this.processCandidate(
          candidate,
          sessionKey,
          stats,
          targetScope,
          scopeFilter,
        );
      } catch (err) {
        this.log(
          `memory-pro: smart-extractor: failed to process candidate [${candidate.category}]: ${String(err)}`,
        );
      }
    }

    return stats;
  }

  // --------------------------------------------------------------------------
  // Step 1: LLM Extraction
  // --------------------------------------------------------------------------

  /**
   * Call LLM to extract candidate memories from conversation text.
   */
  private async extractCandidates(
    conversationText: string,
  ): Promise<CandidateMemory[]> {
    const maxChars = this.config.extractMaxChars ?? 8000;
    const truncated =
      conversationText.length > maxChars
        ? conversationText.slice(-maxChars)
        : conversationText;

    const user = this.config.user ?? "User";
    const prompt = buildExtractionPrompt(truncated, user);

    const result = await this.llm.completeJson<{
      memories: Array<{
        category: string;
        abstract: string;
        overview: string;
        content: string;
      }>;
    }>(prompt);

    if (!result?.memories || !Array.isArray(result.memories)) {
      return [];
    }

    // Validate and normalize candidates
    const candidates: CandidateMemory[] = [];
    for (const raw of result.memories) {
      const category = normalizeCategory(raw.category ?? "");
      if (!category) continue;

      const abstract = (raw.abstract ?? "").trim();
      const overview = (raw.overview ?? "").trim();
      const content = (raw.content ?? "").trim();

      // Skip empty or noise
      if (!abstract || abstract.length < 5) continue;
      if (isNoise(abstract)) continue;

      candidates.push({ category, abstract, overview, content });
    }

    return candidates;
  }

  // --------------------------------------------------------------------------
  // Step 2: Dedup + Persist
  // --------------------------------------------------------------------------

  /**
   * Process a single candidate memory: dedup → merge/create → store
   */
  private async processCandidate(
    candidate: CandidateMemory,
    sessionKey: string,
    stats: ExtractionStats,
    targetScope: string,
    scopeFilter: string[],
  ): Promise<void> {
    // Profile always merges (skip dedup)
    if (ALWAYS_MERGE_CATEGORIES.has(candidate.category)) {
      await this.handleProfileMerge(
        candidate,
        sessionKey,
        targetScope,
        scopeFilter,
      );
      stats.merged++;
      return;
    }

    // Embed the candidate for vector dedup
    const embeddingText = `${candidate.abstract} ${candidate.content}`;
    const vector = await this.embedder.embed(embeddingText);
    if (!vector || vector.length === 0) {
      this.log("memory-pro: smart-extractor: embedding failed, storing as-is");
      await this.storeCandidate(candidate, vector || [], sessionKey, targetScope);
      stats.created++;
      return;
    }

    // Dedup pipeline
    const dedupResult = await this.deduplicate(candidate, vector, scopeFilter);

    switch (dedupResult.decision) {
      case "create":
        await this.storeCandidate(candidate, vector, sessionKey, targetScope);
        stats.created++;
        break;

      case "merge":
        if (
          dedupResult.matchId &&
          MERGE_SUPPORTED_CATEGORIES.has(candidate.category)
        ) {
          await this.handleMerge(
            candidate,
            dedupResult.matchId,
            scopeFilter,
            targetScope,
          );
          stats.merged++;
        } else {
          // Category doesn't support merge → create instead
          await this.storeCandidate(candidate, vector, sessionKey, targetScope);
          stats.created++;
        }
        break;

      case "skip":
        this.log(
          `memory-pro: smart-extractor: skipped [${candidate.category}] ${candidate.abstract.slice(0, 60)}`,
        );
        stats.skipped++;
        break;
    }
  }

  // --------------------------------------------------------------------------
  // Dedup Pipeline (vector pre-filter + LLM decision)
  // --------------------------------------------------------------------------

  /**
   * Two-stage dedup: vector similarity search → LLM decision.
   */
  private async deduplicate(
    candidate: CandidateMemory,
    candidateVector: number[],
    scopeFilter: string[],
  ): Promise<DedupResult> {
    // Stage 1: Vector pre-filter — find similar memories
    const similar = await this.store.vectorSearch(
      candidateVector,
      5,
      SIMILARITY_THRESHOLD,
      scopeFilter,
    );

    if (similar.length === 0) {
      return { decision: "create", reason: "No similar memories found" };
    }

    // Stage 2: LLM decision
    return this.llmDedupDecision(candidate, similar);
  }

  private async llmDedupDecision(
    candidate: CandidateMemory,
    similar: MemorySearchResult[],
  ): Promise<DedupResult> {
    const topSimilar = similar.slice(0, MAX_SIMILAR_FOR_PROMPT);
    const existingFormatted = topSimilar
      .map((r, i) => {
        // Extract L0 abstract from metadata if available, fallback to text
        let metaObj: Record<string, unknown> = {};
        try {
          metaObj = JSON.parse(r.entry.metadata || "{}");
        } catch { }
        const abstract = (metaObj.l0_abstract as string) || r.entry.text;
        const overview = (metaObj.l1_overview as string) || "";
        return `${i + 1}. [${(metaObj.memory_category as string) || r.entry.category}] ${abstract}\n   Overview: ${overview}\n   Score: ${r.score.toFixed(3)}`;
      })
      .join("\n");

    const prompt = buildDedupPrompt(
      candidate.abstract,
      candidate.overview,
      candidate.content,
      existingFormatted,
    );

    try {
      const data = await this.llm.completeJson<{
        decision: string;
        reason: string;
        match_index?: number;
      }>(prompt);

      if (!data) {
        this.log(
          "memory-pro: smart-extractor: dedup LLM returned unparseable response, defaulting to CREATE",
        );
        return { decision: "create", reason: "LLM response unparseable" };
      }

      const decision = (data.decision?.toLowerCase() ??
        "create") as DedupDecision;
      if (!VALID_DECISIONS.has(decision)) {
        return {
          decision: "create",
          reason: `Unknown decision: ${data.decision}`,
        };
      }

      // Resolve merge target from LLM's match_index (1-based)
      const idx = data.match_index;
      const matchEntry =
        typeof idx === "number" && idx >= 1 && idx <= topSimilar.length
          ? topSimilar[idx - 1]
          : topSimilar[0];

      return {
        decision,
        reason: data.reason ?? "",
        matchId: decision === "merge" ? matchEntry?.entry.id : undefined,
      };
    } catch (err) {
      this.log(
        `memory-pro: smart-extractor: dedup LLM failed: ${String(err)}`,
      );
      return { decision: "create", reason: `LLM failed: ${String(err)}` };
    }
  }

  // --------------------------------------------------------------------------
  // Merge Logic
  // --------------------------------------------------------------------------

  /**
   * Profile always-merge: read existing profile, merge with LLM, upsert.
   */
  private async handleProfileMerge(
    candidate: CandidateMemory,
    sessionKey: string,
    targetScope: string,
    scopeFilter: string[],
  ): Promise<void> {
    // Find existing profile memory by category
    const embeddingText = `${candidate.abstract} ${candidate.content}`;
    const vector = await this.embedder.embed(embeddingText);

    // Search for existing profile memories
    const existing = await this.store.vectorSearch(
      vector || [],
      1,
      0.3,
      scopeFilter,
    );
    const profileMatch = existing.find((r) => {
      try {
        const meta = JSON.parse(r.entry.metadata || "{}");
        return meta.memory_category === "profile";
      } catch {
        return false;
      }
    });

    if (profileMatch) {
      await this.handleMerge(
        candidate,
        profileMatch.entry.id,
        scopeFilter,
        targetScope,
      );
    } else {
      // No existing profile — create new
      await this.storeCandidate(candidate, vector || [], sessionKey, targetScope);
    }
  }

  /**
   * Merge a candidate into an existing memory using LLM.
   */
  private async handleMerge(
    candidate: CandidateMemory,
    matchId: string,
    scopeFilter: string[],
    targetScope: string,
  ): Promise<void> {
    let existingAbstract = "";
    let existingOverview = "";
    let existingContent = "";

    try {
      const existing = await this.store.getById(matchId, scopeFilter);
      if (existing) {
        const meta = parseSmartMetadata(existing.metadata, existing);
        existingAbstract = meta.l0_abstract || existing.text;
        existingOverview = meta.l1_overview || "";
        existingContent = meta.l2_content || existing.text;
      }
    } catch {
      // Fallback: store as new
      this.log(
        `memory-pro: smart-extractor: could not read existing memory ${matchId}, storing as new`,
      );
      const vector = await this.embedder.embed(
        `${candidate.abstract} ${candidate.content}`,
      );
      await this.storeCandidate(
        candidate,
        vector || [],
        "merge-fallback",
        targetScope,
      );
      return;
    }

    // Call LLM to merge
    const prompt = buildMergePrompt(
      existingAbstract,
      existingOverview,
      existingContent,
      candidate.abstract,
      candidate.overview,
      candidate.content,
      candidate.category,
    );

    const merged = await this.llm.completeJson<{
      abstract: string;
      overview: string;
      content: string;
    }>(prompt);

    if (!merged) {
      this.log("memory-pro: smart-extractor: merge LLM failed, skipping merge");
      return;
    }

    // Re-embed the merged content
    const mergedText = `${merged.abstract} ${merged.content}`;
    const newVector = await this.embedder.embed(mergedText);

    // Update existing memory via store.update()
    const existing = await this.store.getById(matchId, scopeFilter);
    const metadata = stringifySmartMetadata(
      buildSmartMetadata(existing ?? { text: merged.abstract }, {
        l0_abstract: merged.abstract,
        l1_overview: merged.overview,
        l2_content: merged.content,
        memory_category: candidate.category,
        tier: "working",
        confidence: 0.8,
      }),
    );

    await this.store.update(
      matchId,
      {
        text: merged.abstract,
        vector: newVector,
        metadata,
      },
      scopeFilter,
    );

    this.log(
      `memory-pro: smart-extractor: merged [${candidate.category}] into ${matchId.slice(0, 8)}`,
    );
  }

  // --------------------------------------------------------------------------
  // Store Helper
  // --------------------------------------------------------------------------

  /**
   * Store a candidate memory as a new entry with L0/L1/L2 metadata.
   */
  private async storeCandidate(
    candidate: CandidateMemory,
    vector: number[],
    sessionKey: string,
    targetScope: string,
  ): Promise<void> {
    // Map 6-category to existing store categories for backward compatibility
    const storeCategory = this.mapToStoreCategory(candidate.category);

    const metadata = stringifySmartMetadata(
      buildSmartMetadata(
        {
          text: candidate.abstract,
          category: this.mapToStoreCategory(candidate.category),
        },
        {
          l0_abstract: candidate.abstract,
          l1_overview: candidate.overview,
          l2_content: candidate.content,
          memory_category: candidate.category,
          tier: "working",
          access_count: 0,
          confidence: 0.7,
          source_session: sessionKey,
        },
      ),
    );

    await this.store.store({
      text: candidate.abstract, // L0 used as the searchable text
      vector,
      category: storeCategory,
      scope: targetScope,
      importance: this.getDefaultImportance(candidate.category),
      metadata,
    });

    this.log(
      `memory-pro: smart-extractor: created [${candidate.category}] ${candidate.abstract.slice(0, 60)}`,
    );
  }

  /**
   * Map 6-category to existing 5-category store type for backward compatibility.
   */
  private mapToStoreCategory(
    category: MemoryCategory,
  ): "preference" | "fact" | "decision" | "entity" | "other" {
    switch (category) {
      case "profile":
        return "fact";
      case "preferences":
        return "preference";
      case "entities":
        return "entity";
      case "events":
        return "decision";
      case "cases":
        return "fact";
      case "patterns":
        return "other";
      default:
        return "other";
    }
  }

  /**
   * Get default importance score by category.
   */
  private getDefaultImportance(category: MemoryCategory): number {
    switch (category) {
      case "profile":
        return 0.9; // Identity is very important
      case "preferences":
        return 0.8;
      case "entities":
        return 0.7;
      case "events":
        return 0.6;
      case "cases":
        return 0.8; // Problem-solution pairs are high value
      case "patterns":
        return 0.85; // Reusable processes are high value
      default:
        return 0.5;
    }
  }
}

/**
 * Memory Store with advanced features
 * - Batch operations
 * - Memory decay
 * - Similarity deduplication
 */

import * as lancedb from "vectordb";

export interface MemoryEntry {
  id: string;
  text: string;
  vector: number[];
  importance: number;
  category: "preference" | "fact" | "decision" | "entity" | "other";
  scope: string;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
}

export interface StoreConfig {
  uri: string;
  tableName?: string;
  enableDecay?: boolean;
  decayDays?: number;
  decayThreshold?: number;
  enableDeduplication?: boolean;
  dedupThreshold?: number;
}

export class MemoryStore {
  private db: lancedb.Connection;
  private table: lancedb.Table | null = null;
  private tableName: string;
  private config: Required<StoreConfig>;

  constructor(config: StoreConfig) {
    this.config = {
      uri: config.uri,
      tableName: config.tableName || "memories",
      enableDecay: config.enableDecay ?? false,
      decayDays: config.decayDays ?? 30,
      decayThreshold: config.decayThreshold ?? 0.3,
      enableDeduplication: config.enableDeduplication ?? true,
      dedupThreshold: config.dedupThreshold ?? 0.95,
    };
    this.tableName = this.config.tableName;
  }

  async init(): Promise<void> {
    this.db = await lancedb.connect(this.config.uri);
    
    try {
      this.table = await this.db.openTable(this.tableName);
    } catch {
      // Table doesn't exist, will be created on first insert
      this.table = null;
    }
  }

  private async ensureTable(vector: number[]): Promise<lancedb.Table> {
    if (!this.table) {
      const sampleEntry: MemoryEntry = {
        id: this.generateId(),
        text: "",
        vector,
        importance: 0.5,
        category: "other",
        scope: "default",
        timestamp: Date.now(),
        accessCount: 0,
        lastAccessed: Date.now(),
      };
      this.table = await this.db.createTable(this.tableName, [sampleEntry]);
      // Remove the sample entry
      await this.table.delete(`id = "${sampleEntry.id}"`);
    }
    return this.table;
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  }

  // =========================================================================
  // Core Operations
  // =========================================================================

  async store(entry: {
    text: string;
    vector: number[];
    importance: number;
    category: "preference" | "fact" | "decision" | "entity" | "other";
    scope: string;
  }): Promise<{ id: string }> {
    const table = await this.ensureTable(entry.vector);
    
    // Check for duplicates
    if (this.config.enableDeduplication) {
      const existing = await this.vectorSearch(entry.vector, 1, this.config.dedupThreshold, [entry.scope]);
      if (existing.length > 0 && existing[0].score > this.config.dedupThreshold) {
        // Update access count for existing entry
        const existingId = existing[0].entry.id;
        await this.updateAccessCount(existingId);
        return { id: existingId };
      }
    }

    const id = this.generateId();
    const now = Date.now();
    
    const fullEntry: MemoryEntry = {
      id,
      text: entry.text,
      vector: entry.vector,
      importance: entry.importance,
      category: entry.category,
      scope: entry.scope,
      timestamp: now,
      accessCount: 1,
      lastAccessed: now,
    };

    await table.add([fullEntry]);
    return { id };
  }

  // =========================================================================
  // Batch Operations
  // =========================================================================

  async storeBatch(entries: Array<{
    text: string;
    vector: number[];
    importance: number;
    category: "preference" | "fact" | "decision" | "entity" | "other";
    scope: string;
  }>): Promise<{ ids: string[]; skipped: number; duplicates: string[] }> {
    if (entries.length === 0) {
      return { ids: [], skipped: 0, duplicates: [] };
    }

    const table = await this.ensureTable(entries[0].vector);
    const ids: string[] = [];
    const duplicates: string[] = [];
    let skipped = 0;

    // Check for duplicates in batch
    const toStore: MemoryEntry[] = [];
    const seenVectors: Map<string, number[]> = new Map();

    for (const entry of entries) {
      // Check against already seen vectors in this batch
      let isDuplicate = false;
      for (const [_, v] of seenVectors) {
        const similarity = this.cosineSimilarity(entry.vector, v);
        if (similarity > this.config.dedupThreshold) {
          isDuplicate = true;
          break;
        }
      }

      if (isDuplicate) {
        skipped++;
        continue;
      }

      // Check against database
      if (this.config.enableDeduplication) {
        const existing = await this.vectorSearch(entry.vector, 1, this.config.dedupThreshold, [entry.scope]);
        if (existing.length > 0 && existing[0].score > this.config.dedupThreshold) {
          duplicates.push(existing[0].entry.id);
          skipped++;
          continue;
        }
      }

      const id = this.generateId();
      const now = Date.now();
      
      toStore.push({
        id,
        text: entry.text,
        vector: entry.vector,
        importance: entry.importance,
        category: entry.category,
        scope: entry.scope,
        timestamp: now,
        accessCount: 1,
        lastAccessed: now,
      });
      
      ids.push(id);
      seenVectors.set(id, entry.vector);
    }

    if (toStore.length > 0) {
      await table.add(toStore);
    }

    return { ids, skipped, duplicates };
  }

  // =========================================================================
  // Memory Decay
  // =========================================================================

  async decayOldMemories(params?: {
    decayDays?: number;
    threshold?: number;
    dryRun?: boolean;
  }): Promise<{ removed: number; kept: number; details: Array<{ id: string; reason: string }> }> {
    const decayDays = params?.decayDays ?? this.config.decayDays;
    const threshold = params?.threshold ?? this.config.decayThreshold;
    const dryRun = params?.dryRun ?? false;

    if (!this.table) {
      return { removed: 0, kept: 0, details: [] };
    }

    const cutoffTime = Date.now() - decayDays * 24 * 60 * 60 * 1000;
    const entries = await this.table.query().toArray() as MemoryEntry[];
    
    const removed: string[] = [];
    const kept: string[] = [];
    const details: Array<{ id: string; reason: string }> = [];

    for (const entry of entries) {
      // Calculate decay score based on:
      // - Age (older = lower score)
      // - Access count (more accessed = higher score)
      // - Importance (higher importance = higher score)
      
      const ageInDays = (Date.now() - entry.timestamp) / (24 * 60 * 60 * 1000);
      const ageFactor = Math.max(0, 1 - ageInDays / decayDays);
      const accessFactor = Math.min(1, entry.accessCount / 10);
      const importanceFactor = entry.importance;
      
      const decayScore = (ageFactor * 0.4) + (accessFactor * 0.3) + (importanceFactor * 0.3);
      
      if (decayScore < threshold) {
        removed.push(entry.id);
        details.push({
          id: entry.id,
          reason: `Decay score ${decayScore.toFixed(2)} < ${threshold}`,
        });
      } else {
        kept.push(entry.id);
      }
    }

    if (!dryRun && removed.length > 0) {
      // Delete in batches
      const batchSize = 100;
      for (let i = 0; i < removed.length; i += batchSize) {
        const batch = removed.slice(i, i + batchSize);
        const ids = batch.map(id => `"${id}"`).join(", ");
        await this.table.delete(`id IN (${ids})`);
      }
    }

    return {
      removed: dryRun ? 0 : removed.length,
      kept: kept.length,
      details: dryRun ? details.slice(0, 10) : [],
    };
  }

  // =========================================================================
  // Similarity Deduplication
  // =========================================================================

  async deduplicateBySimilarity(params?: {
    threshold?: number;
    scope?: string;
    dryRun?: boolean;
  }): Promise<{ removed: number; pairs: Array<{ kept: string; removed: string; similarity: number }> }> {
    const threshold = params?.threshold ?? this.config.dedupThreshold;
    const dryRun = params?.dryRun ?? false;

    if (!this.table) {
      return { removed: 0, pairs: [] };
    }

    let entries = await this.table.query().toArray() as MemoryEntry[];
    
    if (params?.scope) {
      entries = entries.filter(e => e.scope === params.scope);
    }

    // Sort by importance (higher first) and timestamp (newer first)
    entries.sort((a, b) => {
      if (b.importance !== a.importance) return b.importance - a.importance;
      return b.timestamp - a.timestamp;
    });

    const toRemove: Set<string> = new Set();
    const pairs: Array<{ kept: string; removed: string; similarity: number }> = [];

    for (let i = 0; i < entries.length; i++) {
      if (toRemove.has(entries[i].id)) continue;

      for (let j = i + 1; j < entries.length; j++) {
        if (toRemove.has(entries[j].id)) continue;

        const similarity = this.cosineSimilarity(entries[i].vector, entries[j].vector);
        
        if (similarity > threshold) {
          toRemove.add(entries[j].id);
          pairs.push({
            kept: entries[i].id,
            removed: entries[j].id,
            similarity,
          });
        }
      }
    }

    if (!dryRun && toRemove.size > 0) {
      const ids = Array.from(toRemove).map(id => `"${id}"`).join(", ");
      await this.table.delete(`id IN (${ids})`);
    }

    return {
      removed: dryRun ? 0 : toRemove.size,
      pairs: pairs.slice(0, 10),
    };
  }

  // =========================================================================
  // Search Operations
  // =========================================================================

  async vectorSearch(
    vector: number[],
    limit: number,
    minScore: number,
    scopeFilter?: string[]
  ): Promise<Array<{ entry: MemoryEntry; score: number }>> {
    if (!this.table) {
      return [];
    }

    let query = this.table.vectorSearch(vector).limit(limit * 2);
    
    const results = await query.toArray() as Array<MemoryEntry & { _distance?: number }>;
    
    const filtered: Array<{ entry: MemoryEntry; score: number }> = [];
    
    for (const result of results) {
      if (scopeFilter && !scopeFilter.includes(result.scope)) {
        continue;
      }

      const score = 1 - (result._distance || 0);
      if (score >= minScore) {
        const { _distance, ...entry } = result;
        filtered.push({ entry, score });
      }
      
      if (filtered.length >= limit) break;
    }

    // Update access counts
    for (const { entry } of filtered) {
      await this.updateAccessCount(entry.id);
    }

    return filtered;
  }

  private async updateAccessCount(id: string): Promise<void> {
    // Note: LanceDB doesn't support efficient updates
    // This is a simplified implementation
    // In production, you might want to use a separate access tracking table
  }

  // =========================================================================
  // Utilities
  // =========================================================================

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    if (normA === 0 || normB === 0) return 0;
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async count(): Promise<number> {
    if (!this.table) return 0;
    return await this.table.countRows();
  }

  async getStats(): Promise<{
    total: number;
    byCategory: Record<string, number>;
    byScope: Record<string, number>;
    avgImportance: number;
    avgAccessCount: number;
  }> {
    if (!this.table) {
      return { total: 0, byCategory: {}, byScope: {}, avgImportance: 0, avgAccessCount: 0 };
    }

    const entries = await this.table.query().toArray() as MemoryEntry[];
    
    const byCategory: Record<string, number> = {};
    const byScope: Record<string, number> = {};
    let totalImportance = 0;
    let totalAccessCount = 0;

    for (const entry of entries) {
      byCategory[entry.category] = (byCategory[entry.category] || 0) + 1;
      byScope[entry.scope] = (byScope[entry.scope] || 0) + 1;
      totalImportance += entry.importance;
      totalAccessCount += entry.accessCount;
    }

    return {
      total: entries.length,
      byCategory,
      byScope,
      avgImportance: entries.length > 0 ? totalImportance / entries.length : 0,
      avgAccessCount: entries.length > 0 ? totalAccessCount / entries.length : 0,
    };
  }

  async deleteByIds(ids: string[]): Promise<number> {
    if (!this.table || ids.length === 0) return 0;
    
    const idList = ids.map(id => `"${id}"`).join(", ");
    await this.table.delete(`id IN (${idList})`);
    return ids.length;
  }

  async clear(scope?: string): Promise<number> {
    if (!this.table) return 0;
    
    if (scope) {
      await this.table.delete(`scope = "${scope}"`);
    } else {
      // Clear all
      const count = await this.table.countRows();
      await this.table.delete("id != ''");
      return count;
    }
    
    return 0;
  }
}

/**
 * Memory LanceDB Pro - Enhanced with batch operations, decay, and deduplication
 * 
 * New Features:
 * - storeBatch(): Batch store multiple entries
 * - decayOldMemories(): Automatic memory decay
 * - deduplicateBySimilarity(): Remove duplicate memories
 * - getStats(): Memory statistics
 * - embedPassages()/embedQueries(): Batch embedding
 * 
 * @version 1.1.0
 */

import type { PluginAPI } from "openclaw/plugin-sdk/core";
import type { MemoryPluginAPI, MemoryEntry, MemoryScopeManager, MemoryRetriever, MemoryStore, MemoryEmbedder } from "openclaw/plugin-sdk/memory";
import * as lancedb from "vectordb";

// ... (保持原有代码结构，添加新功能)

// 在 MemoryStore 类中添加以下方法：

/**
 * Batch store multiple entries at once.
 * More efficient than individual store() calls.
 */
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
      if (similarity > 0.95) {
        isDuplicate = true;
        break;
      }
    }

    if (isDuplicate) {
      skipped++;
      continue;
    }

    // Check against database
    const existing = await this.vectorSearch(entry.vector, 1, 0.95, [entry.scope]);
    if (existing.length > 0 && existing[0].score > 0.95) {
      duplicates.push(existing[0].entry.id);
      skipped++;
      continue;
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
    await this.table.add(toStore);
  }

  return { ids, skipped, duplicates };
}

/**
 * Decay old memories based on age, access count, and importance.
 */
async decayOldMemories(params?: {
  decayDays?: number;
  threshold?: number;
  dryRun?: boolean;
}): Promise<{ removed: number; kept: number; details: Array<{ id: string; reason: string }> }> {
  const decayDays = params?.decayDays ?? 30;
  const threshold = params?.threshold ?? 0.3;
  const dryRun = params?.dryRun ?? false;

  const cutoffTime = Date.now() - decayDays * 24 * 60 * 60 * 1000;
  const entries = await this.table.query().toArray();
  
  const removed: string[] = [];
  const kept: string[] = [];
  const details: Array<{ id: string; reason: string }> = [];

  for (const entry of entries) {
    const ageInDays = (Date.now() - entry.timestamp) / (24 * 60 * 60 * 1000);
    const ageFactor = Math.max(0, 1 - ageInDays / decayDays);
    const accessFactor = Math.min(1, (entry.accessCount || 0) / 10);
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
    for (const id of removed) {
      await this.table.delete(`id = "${id}"`);
    }
  }

  return {
    removed: dryRun ? 0 : removed.length,
    kept: kept.length,
    details: dryRun ? details.slice(0, 10) : [],
  };
}

/**
 * Remove duplicate memories by similarity.
 */
async deduplicateBySimilarity(params?: {
  threshold?: number;
  scope?: string;
  dryRun?: boolean;
}): Promise<{ removed: number; pairs: Array<{ kept: string; removed: string; similarity: number }> }> {
  const threshold = params?.threshold ?? 0.95;
  const dryRun = params?.dryRun ?? false;

  let entries = await this.table.query().toArray();
  
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
    for (const id of toRemove) {
      await this.table.delete(`id = "${id}"`);
    }
  }

  return {
    removed: dryRun ? 0 : toRemove.size,
    pairs: pairs.slice(0, 10),
  };
}

/**
 * Get memory statistics.
 */
async getStats(): Promise<{
  total: number;
  byCategory: Record<string, number>;
  byScope: Record<string, number>;
  avgImportance: number;
  avgAccessCount: number;
}> {
  const entries = await this.table.query().toArray();
  
  const byCategory: Record<string, number> = {};
  const byScope: Record<string, number> = {};
  let totalImportance = 0;
  let totalAccessCount = 0;

  for (const entry of entries) {
    byCategory[entry.category] = (byCategory[entry.category] || 0) + 1;
    byScope[entry.scope] = (byScope[entry.scope] || 0) + 1;
    totalImportance += entry.importance;
    totalAccessCount += entry.accessCount || 0;
  }

  return {
    total: entries.length,
    byCategory,
    byScope,
    avgImportance: entries.length > 0 ? totalImportance / entries.length : 0,
    avgAccessCount: entries.length > 0 ? totalAccessCount / entries.length : 0,
  };
}

// 在 MemoryEmbedder 类中添加批量嵌入方法：

/**
 * Batch embed multiple texts at once.
 */
async embedPassages(texts: string[]): Promise<number[][]> {
  const results: number[][] = [];
  
  // Process in batches of 20 to avoid rate limits
  for (let i = 0; i < texts.length; i += 20) {
    const batch = texts.slice(i, i + 20);
    // Use provider-specific batch embedding
    const batchResults = await this.embedBatchWithProvider(batch);
    results.push(...batchResults);
  }
  
  return results;
}

async embedQueries(texts: string[]): Promise<number[][]> {
  return this.embedPassages(texts);
}

// 工具方法

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

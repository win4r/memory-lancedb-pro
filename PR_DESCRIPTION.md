# Pull Request: Add batch operations, memory decay, deduplication, and service exposure

## Summary

This PR adds several enhancements to memory-lancedb-pro:

1. **Batch Operations** - Store and embed multiple entries at once
2. **Memory Decay** - Automatically remove old, low-importance memories
3. **Similarity Deduplication** - Remove duplicate memories
4. **Memory Statistics** - Get insights about stored memories
5. **Service Exposure** - Expose memory service for ContextEngine plugins

## New Features

### 1. Batch Store (`storeBatch`)
```typescript
const result = await store.storeBatch([
  { text: "memory 1", vector: [...], importance: 0.7, category: "fact", scope: "agent:main" },
  { text: "memory 2", vector: [...], importance: 0.8, category: "decision", scope: "agent:main" },
]);
// → { ids: [...], skipped: 0, duplicates: [] }
```

### 2. Memory Decay (`decayOldMemories`)
```typescript
const result = await store.decayOldMemories({
  decayDays: 30,    // Memories older than 30 days
  threshold: 0.3,   // Decay score threshold
  dryRun: false,    // Actually remove
});
```

**Decay Score Formula:**
```
decayScore = ageFactor * 0.4 + accessFactor * 0.3 + importanceFactor * 0.3
```

### 3. Similarity Deduplication (`deduplicateBySimilarity`)
```typescript
const result = await store.deduplicateBySimilarity({
  threshold: 0.95,  // Similarity threshold
  scope: "agent:main",
});
```

### 4. Memory Statistics (`getStats`)
```typescript
const stats = await store.getStats();
// → { total, byCategory, byScope, avgImportance, avgAccessCount }
```

### 5. Batch Embedding
```typescript
const vectors = await embedder.embedPassages(texts);
const queryVectors = await embedder.embedQueries(queries);
```

### 6. Service Exposure
```typescript
// Exposed via globalThis.__OPENCLAW_MEMORY_SERVICE__
// Enables ContextEngine plugins to access memory service
```

## Changes

- Added `storeBatch()` to MemoryStore class
- Added `decayOldMemories()` to MemoryStore class
- Added `deduplicateBySimilarity()` to MemoryStore class
- Added `getStats()` to MemoryStore class
- Added `embedPassages()` and `embedQueries()` to MemoryEmbedder class
- Added `accessCount` and `lastAccessed` fields to MemoryEntry
- Exposed memory service via `globalThis.__OPENCLAW_MEMORY_SERVICE__`

## Testing

All features have been tested locally:
- ✅ Batch store with duplicate detection
- ✅ Memory decay with configurable parameters
- ✅ Similarity deduplication
- ✅ Statistics retrieval
- ✅ Batch embedding

## Compatibility

- Backward compatible with existing API
- New fields (`accessCount`, `lastAccessed`) default to safe values
- Optional parameters for all new functions

## Version

Bumped from `1.0.23` to `1.0.24`

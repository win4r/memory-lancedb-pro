# Reflection Recall Implementation Research Notes

## Verified Current Code Paths

### Generic Auto-Recall

- Hook registration: `index.ts` around the existing `if (config.autoRecall === true)` block.
- Current output: `<relevant-memories>`.
- Current retrieval source: `retriever.retrieve({ query, limit, scopeFilter, source: "auto-recall" })`.
- Current session dedupe state:
  - `recallHistory: Map<string, Map<string, number>>`
  - `turnCounter: Map<string, number>`

### Reflection inheritance injection

- Hook registration: `index.ts` in the integrated `memoryReflection` section.
- Current output: `<inherited-rules>`.
- Current source path:
  - `loadAgentReflectionSlices()` in `index.ts`
  - `loadAgentReflectionSlicesFromEntries()` in `src/reflection-store.ts`
- Current ranking behavior:
  - per-entry reflection logistic scoring via `computeReflectionScore()` in `src/reflection-ranking.ts`
  - normalized-key aggregation by `normalizeReflectionLineForAggregation()`
  - current aggregation sums scores for duplicate normalized keys
- Current fixed injection takes top 6 invariants.

### Reflection storage

- Reflection item persistence:
  - `storeReflectionToLanceDB()` in `src/reflection-store.ts`
  - reflection item metadata defaults in `src/reflection-item-store.ts`
- Reflection items are stored as `category: "reflection"`.
- Reflection item writes do not use similarity-based duplicate blocking.

## Architectural Implications

1. Dynamic Reflection-Recall should not be implemented by reusing the generic retriever unchanged.
   - Reflection ranking today depends on reflection-specific metadata (`storedAt`, `decayMidpointDays`, `decayK`, `baseWeight`, `quality`, `itemKind`).
2. Fixed mode should remain outside the shared dynamic engine.
   - It is a compatibility mode, not a query-aware recall flow.
3. A shared dynamic recall engine is still beneficial.
   - Shared parts: prompt gating, result trimming, session repeated-injection suppression, output block assembly, optional per-key capping helper.
   - Non-shared parts: candidate loading and primary scoring.
4. The current reflection aggregation should be refined.
   - Present code sums all normalized-key scores within the candidate pool.
   - New design should limit each normalized key to the most recent 10 entries before aggregation.
5. Auto-Recall should stop mixing reflection rows into `<relevant-memories>` when dual-channel mode is enabled.

## Proposed Module Boundaries

### New shared module

`src/recall-engine.ts`

Suggested responsibilities:
- prompt gating wrapper for dynamic recall
- per-session repeated injection suppression helper
- normalized-key recent-entry limiter helper
- block assembly helper (`<relevant-memories>` / `<inherited-rules>`)
- generic orchestration for dynamic recall providers

### Reflection-specific dynamic module

`src/reflection-recall.ts`

Suggested responsibilities:
- load reflection item entries from store scope set
- filter by `itemKind`
- apply reflection scoring and aggregation
- enforce `maxAgeMs` and `maxEntriesPerKey`
- return ranked reflection recall rows

### Generic auto-recall provider changes

Either:
- extend `src/retriever.ts` with post-retrieval per-key/time-window limiting helpers, or
- add a small post-processing adapter in the shared engine for memory retrieval results.

## Data/Behavior Compatibility Notes

- Keep `<inherited-rules>` block text stable enough that existing prompt instructions do not break.
- Update human-facing docs and config schema labels to refer to Reflection-Recall as the mechanism name.
- Continue to allow `memoryReflection.injectMode = inheritance+derived`; only the inheritance side changes mode semantics.
- `derived-focus` handoff note generation for `/new` / `/reset` remains separate from Reflection-Recall.

## Risks

1. Over-sharing code may blur fixed vs dynamic semantics.
2. Adding config fields without careful defaults may change existing installs unexpectedly.
3. Reflection recall dual-mode tests must be explicit or fixed behavior may silently regress.
4. Auto-Recall post-processing must not break current hybrid retrieval ranking guarantees more than intended.

## Recommendation

Implement in small slices:
1. config + shared engine scaffold
2. reflection dynamic mode
3. auto-recall enhancements
4. docs + tests

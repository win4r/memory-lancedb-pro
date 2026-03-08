# Technical Documentation

## Scope

This scope adds a compatibility-preserving Reflection-Recall layer to `memory-lancedb-pro`.

## Terminology

- **Auto-Recall**: generic memory retrieval channel that injects `<relevant-memories>`.
- **Reflection-Recall**: reflection-specific rule retrieval channel that injects `<inherited-rules>`.
- **Fixed Reflection-Recall**: current behavior; inject stable reflection invariants without query-aware dynamic selection.
- **Dynamic Reflection-Recall**: new behavior; query-gated reflection retrieval with independent top-k and aggregation controls.

## High-level design

### Fixed mode

- Trigger: `before_agent_start`
- Data source: ranked reflection invariants from LanceDB-backed reflection item rows
- Output tag: `<inherited-rules>`
- Compatibility goal: preserve current behavior when `memoryReflection.recall.mode` is unset or `fixed`

### Dynamic mode

- Trigger: `before_agent_start`
- Data source: reflection item rows loaded from scope-filtered LanceDB entries
- Output tag: `<inherited-rules>`
- Ranking: reflection logistic scoring + normalized-key aggregation using only the most recent N rows per key
- Default top-k: 6
- Session dedupe: reflection-specific cooldown map

### Generic Auto-Recall

- Trigger: `before_agent_start`
- Data source: hybrid retriever
- Output tag: `<relevant-memories>`
- Enhancement: category allowlist/exclude support plus optional time-window and per-key recent-entry controls

## Shared dynamic recall engine

A shared engine should orchestrate dynamic recall channels while keeping candidate loading pluggable.

Shared responsibilities:
- prompt gating / skip logic
- session turn bookkeeping
- repeated-injection suppression
- per-key limiting helper
- output block assembly

Channel-specific responsibilities:
- memory candidate loading and scoring
- reflection candidate loading and scoring

## Config shape

### Reflection-Recall

```json
{
  "memoryReflection": {
    "recall": {
      "mode": "fixed",
      "topK": 6,
      "includeKinds": ["invariant"],
      "maxAgeDays": 45,
      "maxEntriesPerKey": 10,
      "minRepeated": 2,
      "minScore": 0.18,
      "minPromptLength": 8
    }
  }
}
```

### Auto-Recall additions

```json
{
  "autoRecallTopK": 3,
  "autoRecallCategories": ["preference", "fact", "decision", "entity", "other"],
  "autoRecallExcludeReflection": true,
  "autoRecallMaxAgeDays": 30,
  "autoRecallMaxEntriesPerKey": 10
}
```

## Decision points

1. Keep fixed mode outside the shared dynamic engine.
2. Keep `<inherited-rules>` output tag for backward compatibility.
3. Make dynamic Reflection-Recall top-k independent from generic Auto-Recall.
4. Limit normalized-key aggregation to recent entries to reduce stale vote stacking.

## Test focus

- fixed reflection compatibility
- dynamic reflection top-k independence
- time-window filtering
- per-key recent-entry cap of 10
- reflection exclusion from generic auto-recall

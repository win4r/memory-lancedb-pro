# Reflection Recall Contracts

## Context

The repository currently supports:

- generic Auto-Recall via `before_agent_start` and `<relevant-memories>`
- reflection inheritance injection via `before_agent_start` and `<inherited-rules>`
- reset/new reflection handoff generation via `command:new` / `command:reset`

The requested change is to evolve reflection inheritance into a more compatible **Reflection-Recall** feature without breaking current fixed behavior.

## Goals

1. Introduce the Reflection-Recall concept while preserving existing prompt tag compatibility.
2. Support `memoryReflection.recall.mode = "fixed" | "dynamic"`.
3. Keep current fixed inheritance behavior as the default/compatibility path.
4. Allow dynamic Reflection-Recall to inject `<inherited-rules>` independently from generic Auto-Recall.
5. Ensure dynamic Reflection-Recall has its own top-k budget, cooldown history, and ranking path.
6. Add time-window filtering and per-key recent-entry cap (`10`) to reflection aggregation.
7. Improve Auto-Recall so it can exclude reflection rows and apply similar per-key/time-window controls.
8. Route both dynamic channels through a shared public recall orchestration helper where possible.

## Non-goals

1. Do not redesign reflection generation, storage schema versioning, or session reset flow beyond what is required for recall.
2. Do not remove `<inherited-rules>` output tag in this change.
3. Do not migrate historical LanceDB rows.
4. Do not replace reflection-specific scoring with generic retriever-only scoring.

## Required Behavior

### Reflection-Recall fixed mode

- When `memoryReflection.recall.mode` is absent or `fixed`, behavior must stay aligned with current inheritance injection.
- `<inherited-rules>` remains sourced from ranked reflection invariants loaded from LanceDB-backed reflection items.
- Auto-Recall being off must not disable fixed reflection inheritance.

### Reflection-Recall dynamic mode

- Reflection-Recall runs on `before_agent_start`.
- It produces `<inherited-rules>` using reflection item candidates only.
- It must not consume the generic Auto-Recall top-k budget.
- It must rank by reflection score and return the top 6 rows by default.
- Per normalized key, only the most recent 10 entries may contribute to aggregation.
- Items outside the configured time window must not contribute.
- Session-level repeated injection suppression must use a reflection-specific history map independent of generic Auto-Recall.

### Generic Auto-Recall

- Generic Auto-Recall must remain responsible for `<relevant-memories>`.
- It must support excluding `reflection` category rows.
- It should gain configurable time-window and per-key recent-entry limiting.
- Reflection-Recall and Auto-Recall may both run in the same session when enabled, and each must keep an independent result count.

## Interface / Config Contract

Additive config only; preserve backward compatibility.

### New reflection config

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

### New auto-recall config (additive)

```json
{
  "autoRecall": true,
  "autoRecallMinLength": 8,
  "autoRecallMinRepeated": 3,
  "autoRecallTopK": 3,
  "autoRecallCategories": ["preference", "fact", "decision", "entity", "other"],
  "autoRecallExcludeReflection": true,
  "autoRecallMaxAgeDays": 30,
  "autoRecallMaxEntriesPerKey": 10
}
```

## Invariants

1. Fixed reflection inheritance remains available without enabling generic Auto-Recall.
2. Dynamic Reflection-Recall and generic Auto-Recall use separate top-k accounting.
3. Reflection-specific scoring semantics remain reflection-owned.
4. Shared orchestration must not force fixed mode through the dynamic path.
5. Existing tests covering fixed `<inherited-rules>` behavior must remain valid or be updated only where config explicitly selects `dynamic`.

## Verification Contract

Implementation is acceptable only if:

1. Config parsing preserves current defaults.
2. Fixed reflection mode reproduces current `<inherited-rules>` behavior.
3. Dynamic Reflection-Recall respects time window and per-key recent-entry caps.
4. Dynamic Reflection-Recall top-k is independent from generic Auto-Recall top-k.
5. Auto-Recall can exclude `reflection` rows.
6. Tests cover fixed mode, dynamic mode, per-key cap, and dual-channel coexistence.

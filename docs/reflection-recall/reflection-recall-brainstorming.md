# Reflection Recall Brainstorming

## Problem

`memory-lancedb-pro` currently has two unrelated pre-start injection paths:

1. `autoRecall` injects `<relevant-memories>` from generic memory retrieval.
2. `memoryReflection.injectMode` injects `<inherited-rules>` via a fixed reflection slice load.

This makes reflection guidance either always-on (fixed inheritance) or absent. It cannot behave like a low-frequency dynamic recall channel with an independent budget.

## Goals

- Introduce **Reflection-Recall** as the mechanism name for reflection-based rule injection.
- Preserve current behavior when Auto-Recall is disabled and reflection recall mode remains `fixed`.
- Support a `dynamic` Reflection-Recall mode with an independent top-k budget.
- Keep `<inherited-rules>` tag output for prompt compatibility.
- Share common recall orchestration between Auto-Recall and Reflection-Recall where practical.
- Add time-window and per-key recent-entry caps to reduce stale score stacking.

## Options

### Option A — Keep separate implementations

- Leave Auto-Recall and reflection injection as two fully separate code paths.
- Add dynamic reflection logic only to the reflection path.

Pros:
- Lowest short-term risk.
- Minimal refactor.

Cons:
- Duplicates skip, cooldown, grouping, formatting, and ranking glue.
- Harder to keep behavior aligned.

### Option B — Shared recall engine + separate candidate providers

- Extract a reusable recall orchestration layer.
- Auto-Recall uses generic retriever-backed candidate loading.
- Reflection-Recall uses reflection-item-backed candidate loading.
- Fixed reflection mode remains a thin compatibility path outside the dynamic engine.

Pros:
- Best balance of compatibility and maintainability.
- Shared cooldown / top-k / formatting logic.
- Reflection-specific scoring stays isolated.

Cons:
- Moderate refactor touching config parsing and tests.

### Option C — Force reflection into generic retriever only

- Treat reflection items as ordinary memories and let Auto-Recall retrieve them.

Pros:
- Fewer code paths.

Cons:
- Loses reflection-specific scoring semantics.
- Makes independent top-k and prompt blocks awkward.
- Weak compatibility story for existing fixed inheritance mode.

## Recommendation

Choose **Option B**.

- Keep `fixed` reflection injection for compatibility.
- Add `dynamic` Reflection-Recall powered by a shared recall engine.
- Preserve `<inherited-rules>` output tag while renaming the mechanism to Reflection-Recall in config/docs/logging.
- Add per-key recent-entry caps and time windows to both dynamic channels.

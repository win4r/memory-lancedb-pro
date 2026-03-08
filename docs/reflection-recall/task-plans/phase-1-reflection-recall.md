# Phase 1 — Config and naming scaffold

## Goal

Introduce Reflection-Recall terminology and additive config parsing without breaking existing behavior.

## Tasks

- Add config types for `memoryReflection.recall.*`.
- Add config types for Auto-Recall enhancements (`topK`, category filters, age window, per-key cap).
- Preserve reflection fixed mode as the default.
- Update logging strings where helpful to distinguish Auto-Recall from Reflection-Recall.

## DoD

- Existing configs continue to behave the same.
- New config fields parse safely with defaults.
- No existing tests regress due to default changes.

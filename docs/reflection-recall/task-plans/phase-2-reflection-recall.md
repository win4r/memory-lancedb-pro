# Phase 2 — Shared dynamic recall engine

## Goal

Extract shared orchestration for dynamic recall channels without forcing fixed reflection mode through the same path.

## Tasks

- Create a shared recall engine module.
- Move reusable logic for prompt gating, repeated-injection suppression, and block assembly into the engine.
- Keep provider-specific candidate loading separate.

## DoD

- Auto-Recall can call the shared engine.
- Reflection-Recall dynamic mode can call the shared engine.
- Fixed reflection mode still uses its compatibility path.

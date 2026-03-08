# Phase 3 — Reflection-Recall dynamic mode + Auto-Recall enhancements

## Goal

Add the requested dynamic reflection behavior and align Auto-Recall with new filtering/capping controls.

## Tasks

- Implement reflection dynamic candidate loading/ranking.
- Enforce reflection time window and per-key recent-entry cap (`10`).
- Return reflection top 6 independently from generic Auto-Recall results.
- Exclude reflection rows from generic Auto-Recall when configured.
- Add per-key/time-window post-processing for generic Auto-Recall.

## DoD

- Reflection dynamic mode injects `<inherited-rules>` from dynamic results.
- Fixed mode still behaves compatibly.
- Auto-Recall and Reflection-Recall budgets are independent.

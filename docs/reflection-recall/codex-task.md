Implement the Reflection-Recall plan described in the docs under `docs/reflection-recall/`.

Required reading before changes:
- docs/reflection-recall/reflection-recall-contracts.md
- docs/reflection-recall/reflection-recall-implementation-research-notes.md
- docs/reflection-recall/technical-documentation.md
- docs/reflection-recall/task-plans/4phases-checklist.md
- docs/reflection-recall/task-plans/phase-1-reflection-recall.md
- docs/reflection-recall/task-plans/phase-2-reflection-recall.md
- docs/reflection-recall/task-plans/phase-3-reflection-recall.md
- docs/reflection-recall/task-plans/phase-4-reflection-recall.md

Implementation requirements:
1. Introduce Reflection-Recall terminology while keeping `<inherited-rules>` output tag compatible.
2. Add `memoryReflection.recall.mode = fixed|dynamic` with fixed as the backward-compatible default.
3. Keep fixed reflection inheritance behavior when reflection recall mode is fixed, regardless of generic Auto-Recall state.
4. Add a shared dynamic recall orchestration helper that Auto-Recall and dynamic Reflection-Recall can both call.
5. Keep fixed reflection mode outside the shared dynamic recall path.
6. Implement dynamic Reflection-Recall with independent top-k budgeting (default 6) and independent session repeat suppression.
7. Reflection dynamic aggregation must apply a time window and per normalized key cap of the most recent 10 entries.
8. Improve generic Auto-Recall so it can exclude `reflection` category rows and apply similar time-window / per-key recent-entry controls.
9. Update config parsing and `openclaw.plugin.json` schema/help entries.
10. Update README.md and README_CN.md to document Reflection-Recall fixed/dynamic behavior and new config fields.
11. Add or update tests covering:
   - fixed mode compatibility
   - dynamic Reflection-Recall behavior
   - top-k independence between `<relevant-memories>` and `<inherited-rules>`
   - reflection exclusion from generic Auto-Recall
   - reflection per-key recent-entry cap
12. Run `npm test` and report changed files + verification.

Constraints:
- Work only in this verify worktree.
- Do not modify global Codex config.
- Keep persistent docs in English.
- Prefer small, coherent commits in the working tree; do not commit unless explicitly asked.

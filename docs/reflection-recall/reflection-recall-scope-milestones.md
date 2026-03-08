# Reflection Recall Scope Milestones

## Milestone 1 — Compatibility framing

- Add Reflection-Recall terminology to docs/config comments/logging.
- Add config parsing for `memoryReflection.recall.mode` and related fields.
- Preserve current defaults (`fixed` behavior for reflection inheritance).

## Milestone 2 — Shared dynamic recall engine

- Introduce a reusable orchestration helper for dynamic recall channels.
- Support independent history maps, block tags, top-k, and prompt gating.
- Keep fixed reflection mode outside the shared dynamic path.

## Milestone 3 — Dynamic Reflection-Recall

- Add reflection-specific dynamic candidate loading and ranking.
- Enforce time-window filtering.
- Enforce per normalized key cap of recent 10 entries.
- Inject `<inherited-rules>` from dynamic results when reflection recall mode is `dynamic`.

## Milestone 4 — Auto-Recall improvements

- Add category exclusion / allowlist support.
- Exclude reflection rows by default when requested.
- Add recent-per-key cap and time-window controls.
- Ensure `<relevant-memories>` and `<inherited-rules>` budgets remain independent.

## Milestone 5 — Verification and docs

- Update `README.md`, `README_CN.md`, and `openclaw.plugin.json` schema/help text.
- Add or update tests for fixed mode, dynamic mode, dual-channel coexistence, and per-key cap behavior.
- Run repository test suite and report residual risks.

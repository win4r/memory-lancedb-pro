# Central Task Board — memory-lancedb-pro

## Status legend
- TODO
- IN_PROGRESS
- BLOCKED
- DONE

## Project goal
Implement an incremental, architecture-compatible upgrade path for agent-scoped memory management and later historical-memory upgrade support.

## Track A — Project review & architecture control
- [DONE] Establish git-managed development clone under workspace
- [DONE] Create feature branch `feat/agent-scoped-memory-upgrade-flow`
- [DONE] Record project review and minimal-change design notes
- [TODO] Confirm exact phase boundaries before parallel coding

## Track B — Engineering safety rails
- [DONE] Add local verification script `scripts/verify-before-submit.sh`
- [DONE] Add git hook entrypoint under `.githooks/pre-commit`
- [TODO] Enable repo-local `core.hooksPath=.githooks`
- [TODO] Smoke test hook behavior with a real commit in this dev repo

## Track C — Phase 1 feature work
### C1. Per-agent default isolation
- [TODO] Add config switch for automatic agent-scoped defaults
- [TODO] Implement runtime default scope behavior with backward compatibility
- [TODO] Add/extend tests for scope resolution and accessible scopes

### C2. Initialization / upgrade scaffolding
- [TODO] Add first-run marker / initialization summary behavior
- [TODO] Detect presence of candidate historical memory files/workspaces
- [TODO] Add tests for first-run behavior and non-blocking logging

## Track D — Phase 2 preparation
- [TODO] Define Markdown import command surface
- [TODO] Define mdMirror-log parser boundary
- [TODO] Define freeform Markdown extraction boundary

## Multi-agent execution rules
1. Main agent owns architecture understanding and task decomposition.
2. Each coding worker must use an isolated git worktree or isolated non-overlapping working directory.
3. Do not assign overlapping mutable paths to concurrent workers.
4. Require stage reports: understanding -> tests -> implementation -> verification.
5. No code is presented upward until tests pass locally.

## Proposed parallel execution plan
### Worker 1
- Scope: `src/scopes.ts` + related tests
- Goal: auto agent-scoped default behavior

### Worker 2
- Scope: `index.ts` first-run scaffolding + related tests
- Goal: initialization summary + upgrade detection scaffolding

### Main agent
- Owns: architecture guardrails, task board, integration review, final merge, verification

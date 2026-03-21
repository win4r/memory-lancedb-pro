# Phase 2 Worker Decomposition — memory-lancedb-pro

## Purpose
Translate the Phase 2 main-agent design into concrete, non-overlapping worker assignments that can be executed in isolated git worktrees.

This document assumes the following requirements are fixed:
- non-destructive midpoint adoption
- non-residual / reversible exit
- original OpenClaw Markdown + SQLite systems remain valid throughout
- while the plugin is enabled, **Markdown and SQLite continuity must both be preserved**, not just LanceDB state
- agent guidance should prefer `memory-lancedb-pro` retrieval once enabled

---

## Phase 2 implementation waves

## Wave 1 — Read-only / preview-first foundation
No destructive import, no broad sync writes yet.

### Worker P2-W1 — Upgrade Scan CLI
**Goal:** implement `memory-pro upgrade-scan`

**Scope:**
- `cli.ts`
- `src/upgrade-planner.ts` (new)
- read-only source reporting tests

**Responsibilities:**
- enumerate workspace Markdown sources
- enumerate SQLite stores
- report discovery mode / agent mapping / ambiguity warnings
- output human-friendly summary and machine-readable structure

**Non-goals:**
- no import
- no sync/backfill writes
- no retrieval preference updates

---

### Worker P2-W2 — Markdown Preview Parser
**Goal:** implement preview-only Markdown upgrade flow

**Scope:**
- `src/md-import.ts` (new)
- Markdown parser tests
- CLI dry-run surfaces if needed in coordination with W1

**Responsibilities:**
- parse `MEMORY.md`
- parse `memory/YYYY-MM-DD.md`
- classify candidate memory units
- label likely durable vs noisy
- support dry-run preview output

**Non-goals:**
- no real import write yet
- no SQLite parsing yet
- no runtime sync changes yet

---

### Worker P2-W3 — SQLite Preview Reader
**Goal:** implement preview-only SQLite upgrade inspection

**Scope:**
- `src/sqlite-import.ts` (new)
- SQLite preview tests

**Responsibilities:**
- inspect `~/.openclaw/memory/*.sqlite`
- report schema/rows or importable memory candidates
- identify overlap / duplication risk with workspace Markdown
- define when SQLite contributes unique information

**Non-goals:**
- no direct SQLite writes yet
- no final import writes yet
- no runtime sync changes yet

---

## Wave 2 — Controlled import
Only starts after Wave 1 preview tooling is reviewed.

### Worker P2-W4 — Real Markdown Import
**Goal:** implement controlled Markdown import into LanceDB

**Scope:**
- `src/md-import.ts`
- import application path
- dedupe/supersede tests

**Responsibilities:**
- turn approved Markdown candidates into LanceDB writes
- map to `agent:<id>` / `global`
- produce import report

**Non-goals:**
- no runtime sync/backfill yet
- no SQLite write continuity yet

---

### Worker P2-W5 — Real SQLite Import
**Goal:** implement controlled SQLite import after preview semantics are validated

**Scope:**
- `src/sqlite-import.ts`
- import decision tests

**Responsibilities:**
- import only unique/needed legacy SQLite content
- avoid duplication where Markdown already covers the same memory
- produce import report

**Non-goals:**
- no runtime SQLite continuity writes yet

---

## Wave 3 — Runtime coexistence and reversibility
This is where the stronger compatibility requirements become real implementation work.

### Worker P2-W6 — Compatibility Write Path (Markdown continuity)
**Goal:** ensure durable newly accepted memories continue to reach a Markdown-compatible layer while plugin is enabled

**Scope:**
- future `src/legacy-sync.ts`
- mdMirror/backfill helpers
- sync behavior tests

**Responsibilities:**
- define/implement durable-memory mirror/backfill to Markdown-compatible output
- target a dedicated per-agent workspace parallel subtree such as `memory/plugins/memory-lancedb-pro/` rather than mixing plugin output into human-authored daily logs
- provision and maintain a small `README.md` / `STATEMENT.md` explaining the subtree's purpose as a compatibility/reversibility layer
- keep the first runtime write-path minimal: `README.md` + dated Markdown files, without extra derived subpaths unless later needed
- avoid noisy over-write of transient material

---

### Worker P2-W7 — Compatibility Write Path (SQLite continuity)
**Goal:** ensure SQLite-backed legacy continuity remains live during plugin-enabled runtime

**Scope:**
- future `src/sqlite-sync.ts` or equivalent
- compatibility tests

**Responsibilities:**
- keep old SQLite-backed path from going stale while plugin is enabled
- make reversibility real for A→B period memories
- align with old-system expectations without brittle unsafe mutation strategy

**Critical acceptance requirement:**
- this worker exists because SQLite continuity is required, not optional

---

### Worker P2-W8 — Disable/Uninstall Reversibility Flow
**Goal:** provide report/export/backfill mechanisms for clean exit

**Scope:**
- CLI commands such as `export-legacy` / backfill helpers
- disable/uninstall workflow docs/tests

**Responsibilities:**
- ensure durable memories are not trapped only in LanceDB
- support explicit reversibility checks before exit

---

## Wave 4 — Retrieval preference & docs/skill layer

### Worker P2-W9 — Skill/docs retrieval preference
**Goal:** update skill/docs so agents prefer `memory-lancedb-pro` retrieval when enabled

**Scope:**
- skill docs
- README/reference updates
- retrieval precedence guidance

**Responsibilities:**
- tell agents to prefer LanceDB recall/search
- keep Markdown / SQLite described as compatibility/fallback/upgrade sources

---

## Main-agent responsibilities across all waves
The main agent retains ownership of:
- acceptance criteria
- import safety rules
- sync strategy decisions
- worker boundaries
- integration review
- final verification

The main agent should also decide when a later worker is allowed to start:
- Wave 2 only after Wave 1 previews are reviewed
- Wave 3 only after import behavior is understood
- Wave 4 can overlap late, but not before retrieval preference semantics are stable

---

## Recommended immediate next worker candidates
If implementation resumes soon, the safest first Phase 2 workers are:
1. **P2-W1 Upgrade Scan CLI**
2. **P2-W2 Markdown Preview Parser**
3. **P2-W3 SQLite Preview Reader**

These three can be scoped into separate worktrees with minimal overlap if the main agent prepares shared interfaces first.

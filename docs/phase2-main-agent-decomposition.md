# Phase 2 Main-Agent Decomposition — memory-lancedb-pro

## Purpose
Turn the high-level Phase 2 goals into an execution-ready main-agent plan before assigning any new coding workers.

Phase 2 is no longer just about detecting old memory sources. It is about making `memory-lancedb-pro` a **preferred memory management layer** that remains:
- non-destructive when enabled mid-stream,
- compatible with OpenClaw’s original memory systems,
- reversible when disabled/uninstalled,
- practical for both existing agents and future agents.

---

## Top-level acceptance criteria
All Phase 2 work should be reviewed against these product rules:

1. **Halfway-adoption non-destructive principle**
   - Enabling the plugin mid-stream must not make agents effectively lose access to prior OpenClaw memories.

2. **Post-uninstall non-residual principle**
   - Disabling/uninstalling the plugin must not permanently trap users inside the plugin memory layer.

3. **Enter/exit anytime first principle**
   - OpenClaw’s original Markdown / SQLite memory systems must remain valid and usable throughout enable/disable transitions.

4. **Preferred-management principle**
   - Once enabled, `memory-lancedb-pro` should become the preferred management/retrieval layer, while old systems continue as compatibility/fallback sources.

---

## Phase 2 architecture boundaries
### Keep
- one global plugin/backend instance
- per-agent separation via scopes (`agent:<id>`)
- existing LanceDB storage architecture
- existing `memory-pro` CLI surface, extended additively
- existing Markdown and SQLite systems as compatibility sources

### Do not do
- do not replace OpenClaw’s native Markdown/SQLite systems wholesale
- do not require one plugin instance per agent
- do not make adoption irreversible
- do not silently import everything without user confirmation and preview

---

## Main-agent decomposition

## Track P2-A — Source model and import boundaries
### Goal
Define exactly what can be upgraded into LanceDB and what should remain only as source/reference material.

### Decisions to make
1. **Markdown source classes**
   - `MEMORY.md`
   - `memory/YYYY-MM-DD.md`
   - possible mdMirror-generated files
2. **SQLite source classes**
   - what data can be read from `~/.openclaw/memory/*.sqlite`
   - whether SQLite should be treated as primary import source, fallback source, or metadata-only source
3. **Import unit**
   - paragraph?
   - bullet?
   - fact/event extraction unit?
4. **Agent assignment rule**
   - map imported material to `agent:<id>` or `global`
5. **Value filter rule**
   - what is worth importing vs. leaving only in legacy systems

### Expected output
- a source matrix
- a memory-unit definition
- a scope-mapping rule
- import safety rules

---

## Track P2-B — Runtime coexistence / sync strategy
### Goal
Prevent A→B-period memories from existing only in LanceDB while also avoiding excessive duplication.

### Strategies to evaluate
1. **Dual-write on accepted memory writes**
   - new memories written to LanceDB and a compatibility mirror
2. **Periodic/export-based backfill**
   - write to LanceDB first, then export back to legacy layer on demand / disable / uninstall
3. **Hybrid strategy**
   - only high-value accepted memories dual-write
   - lower-value or transient memories stay in LanceDB only

### Questions to resolve
- what exactly is the "legacy write target"?
  - preferred current direction: per-agent workspace subtree under `memory/plugin-memory-pro/`
  - current frozen minimum target: `README.md` + `YYYY-MM-DD.md`
  - do not introduce extra derived subpaths unless runtime behavior proves they are needed
- how to avoid duplicate growth and conflicting histories?
- which memories are durable enough to merit backward sync?
- what happens when plugin is disabled but old systems continue alone?

### Expected output
- recommended sync strategy
- write-path definition
- rollback / disable behavior definition

---

## Track P2-C — Retrieval preference and fallback behavior
### Goal
Make agents prefer `memory-lancedb-pro` retrieval without losing the ability to fall back safely.

### Questions to resolve
1. Should all agent guidance prefer `memory-lancedb-pro` once enabled?
2. When should legacy retrieval still be consulted?
3. Is fallback automatic or manual?
4. How do we keep the user experience simple while avoiding hidden divergence?

### Expected output
- preferred retrieval order
- fallback policy
- skill/documentation update requirements

---

## Track P2-D — User workflow and reversibility UX
### Goal
Define how users see, control, and reverse memory upgrades.

### Workflow stages
1. Plugin enabled
2. Candidate legacy sources detected
3. User is informed that an upgrade path exists
4. User can preview what would be imported
5. User confirms import/upgrade
6. User can later disable/uninstall plugin without losing the ability to use old systems

### Questions to resolve
- how much should be auto-discovered vs. explicitly selected?
- should imports be dry-run by default?
- what audit/report output should be produced?
- how should disable/uninstall export/backfill behave?

### Expected output
- user-facing upgrade flow
- disable/uninstall flow
- preview/reporting expectations

---

## Track P2-E — CLI and module decomposition
### Goal
Translate Phase 2 into additive code surfaces.

### Likely modules
- `src/md-import.ts`
- `src/sqlite-import.ts`
- `src/upgrade-planner.ts`
- `src/legacy-sync.ts` (only if dual-write/backfill design is approved)

### Likely CLI surface
- `memory-pro import-md ...`
- `memory-pro import-sqlite ...`
- `memory-pro upgrade-scan ...`
- `memory-pro upgrade-preview ...`
- `memory-pro export-legacy ...` (only if reversible sync/export is chosen)

### Expected output
- CLI command map
- module responsibility map
- phased implementation order

---

## Recommended implementation order
### Step 1 — Design freeze
Main agent finalizes:
- import unit definition
- source matrix
- scope mapping
- coexistence strategy
- reversibility strategy

### Step 2 — Preview-first tooling
Implement read-only / dry-run tools before real import:
- upgrade scan
- upgrade preview
- source reporting

### Step 3 — Controlled import
Implement safe import for:
- Markdown first
- SQLite second (after source semantics are validated)

### Step 4 — Runtime coexistence
Implement the chosen sync/backfill strategy.

### Step 5 — Skill/docs preference layer
Update skill/docs so agents prefer `memory-lancedb-pro` retrieval while preserving fallback compatibility.

---

## Suggested future worker split (not started yet)
### Worker A — Markdown import planner / parser
- scope: `src/md-import.ts`, parser tests

### Worker B — SQLite upgrade planner / reader
- scope: `src/sqlite-import.ts`, source semantics tests

### Worker C — Retrieval preference + docs/skill updates
- scope: skill/docs, retrieval precedence notes

### Main agent
- owns coexistence strategy, reversibility model, integration, and final acceptance review

---

## Main-agent immediate next tasks
1. Define the **legacy source matrix** (Markdown vs SQLite, source of truth, importability, risk)
2. Decide the **Phase 2 sync strategy** (dual-write vs export/backfill vs hybrid)
3. Define the **preview-first CLI plan**
4. Only after that, spawn narrowly scoped workers for Phase 2 implementation

# Graphiti Deep Integration Specification

## Scope

This document defines the implementation plan to deeply integrate Graphiti into `memory-lancedb-pro` with three layers:

- LanceDB as raw memory system of record.
- Graphiti as structured entity/relation/inference layer.
- Shared workspace markdown files as materialized views.

Managed workspace files:

- `~/.openclaw/workspace/USER.md`
- `~/.openclaw/workspace/AGENTS.md`
- `~/.openclaw/workspace/IDENTITY.md`
- `~/.openclaw/workspace/MEMORY.md`
- `~/.openclaw/workspace/SOUL.md`
- `~/.openclaw/workspace/HEARTBEAT.md`
- `~/.openclaw/workspace/TOOLS.md`

## Guardrails

- Conservative mode is mandatory.
- Plugin only edits marker-scoped managed sections.
- Plugin never rewrites whole markdown files.
- Inferred knowledge is never treated as asserted truth by default.

Example marker block:

```md
<!-- memory-lancedb-pro:begin USER -->
... managed content ...
<!-- memory-lancedb-pro:end USER -->
```

## Trust Model

Every graph/doc artifact must be labeled as one of:

- `asserted`
- `inferred`
- `candidate`
- `rejected`

Promotion policy:

- `USER.md`, `AGENTS.md`, `IDENTITY.md`, `SOUL.md`: only `asserted` or repeated high-confidence `inferred`.
- `MEMORY.md`, `HEARTBEAT.md`: may include `candidate`/`inferred` with explicit labels.
- `TOOLS.md`: verified tool behavior only.

## Data Model Requirements

Unified provenance metadata across LanceDB/Graphiti/docs:

- `sourceType`
- `sourceId`
- `runId`
- `agentId`
- `sessionKey`
- `scope`
- `contentHash`
- `supportedBy`
- `confidence`
- `assertionKind`
- `syncStatus`
- `lastSyncedAt`
- `updatedAt`

## Workstreams

### W0: Foundations

- Align Graphiti default config values across runtime/schema/docs.
- Add Graphiti auth config support.
- Implement MCP initialize/session/auth flow in runtime client.
- Remove parser drift and duplicate config fields.

### W1: Graph Sync Pipeline

- Centralize all Graphiti writes through a sync service.
- Cover create/update/delete consistency.
- Add backfill/resync flow and sync status metadata.

### W2: Graph-aware Reflection

- Build graph context pack before reflection generation.
- Extend reflection output schema with graph sections.
- Parse and store graph candidates from reflection.

### W3: Inference Pipeline

- Periodically infer relations/results from LanceDB + graph neighborhood.
- Persist as `candidate`/`inferred` with evidence and confidence.
- Add contradiction detection.

### W4: Workspace Docs Materializer

- Build a single writer for all managed markdown blocks.
- Support event-triggered and scheduled refresh.
- Keep user-authored content outside markers untouched.

### W5: Ops + CLI + Observability

- Add CLI commands: doctor/sync/backfill/infer/docs-refresh.
- Add metrics: sync lag, promotion count, contradiction count, auth failures.

### W6: Testing + Hardening

- Unit tests for MCP/auth/session and promotion policy.
- Integration tests for sync + reflection + docs materialization.
- Regression tests for existing memory and reflection behavior.

## Implementation Checklist

1. Foundation
   - [x] Graphiti runtime/schema/doc defaults aligned.
   - [x] MCP client supports initialize, session id, auth headers.
   - [x] Config parser cleanup merged.
2. Sync
   - [x] `memory_store` routed through graph sync service.
   - [x] `memory_update` and `memory_forget` graph consistency strategy implemented.
   - [x] Backfill and resync command path implemented.
3. Reflection
   - [x] Graph context assembled before reflection generation.
   - [x] Reflection parser handles graph sections and promotion candidates.
4. Inference
   - [x] Inference job emits labeled outputs with provenance.
   - [x] Contradictions block high-trust promotion.
5. Docs
   - [x] Marker-scoped materializer updates shared workspace files.
   - [x] High-trust files only accept conservative promotions.
6. Operations
   - [x] CLI graph/docs operational commands available.
   - [x] Doctor reports endpoint/auth/capability/sync lag status.
7. Quality
   - [x] Unit/integration/regression tests passing.

## Extended Implementation (Phase 0-5)

### Phase 0: Foundations
- Fixed config defaults drift (base URL, parser duplicates)
- Upgraded Graphiti MCP client with `initialize`/`notifications/initialized`/session/auth headers
- Added `GraphitiAuthConfig` to schema

### Phase 1: Graph Sync Pipeline
- Created centralized `GraphitiSyncService` (`src/graphiti/sync.ts`)
- Wired `memory_store`/`memory_update`/`memory_forget` through sync service
- Auto-capture and reflection-mapped memory also flow through sync service

### Phase 2: Graph-aware Reflection
- Added `buildGraphReflectionContext()` (`src/graphiti/reflection.ts`)
- Injects graph context into reflection generation
- Persists graph-inferred candidates with provenance metadata

### Phase 3: Inference Layer
- Created scheduled graph inference job (`src/graphiti/inference.ts`)
- Runs independently of reflection (startup + interval)
- Config supports `includeScopes`/`excludeScopes` whitelist/denylist

### Phase 4: Workspace Docs Materializer
- New `WorkspaceDocsMaterializer` (`src/workspace-docs.ts`)
- Marker-scoped block updates (USER.md, AGENTS.md, IDENTITY.md, MEMORY.md, SOUL.md, HEARTBEAT.md, TOOLS.md)
- Integrated with scheduled refresh and reflection-triggered refresh

### Phase 5: Promotion Policy
- Created `promotion-policy.ts` with trust levels (`asserted`/`inferred`/`candidate`/`rejected`)
- Contradiction detection (positive/negative polarity)
- Queue reason statistics
- Manual approval/rejection workflow

### CLI Commands Added
- `memory-pro graph-doctor` — inspect sync metadata health
- `memory-pro graph-infer --once --dry-run --scope --include-scopes --exclude-scopes` — manual trigger
- `memory-pro graph-sync --mode backfill|resync --dry-run --scope --limit`
- `memory-pro graph-backfill`, `memory-pro graph-resync` — shortcuts
- `memory-pro graph-import --mode recall|list --scope --query --limit --dry-run` — reverse import from Graphiti
- `memory-pro promotion-queue --scope --limit --json`
- `memory-pro promotion-approve <id> --target`
- `memory-pro promotion-reject <id> --reason`
- `memory-pro docs-refresh --workspace --reason`

## File-by-File Change Map

- `index.ts`
  - parse new graphiti auth/sync/inference/workspace docs config.
  - initialize sync/inference/materializer services.
  - wire reflection hook integration points.
- `src/graphiti/types.ts`
  - define auth and trust/provenance/sync-state types.
- `src/graphiti/mcp.ts`
  - MCP initialize/session/auth/capability handling.
- `src/graphiti/bridge.ts`
  - capability-aware graph operations and recall adapters.
- `src/tools.ts`
  - route memory create/update/delete through graph sync path.
- `src/reflection-slices.ts`
  - parse graph-aware reflection sections and promotion candidates.
- `src/reflection-store.ts`
  - persist graph-related reflection artifacts.
- `src/self-improvement-files.ts`
  - reuse/extend safe queued file writing for docs materializer.
- `src/store.ts`
  - support metadata and incremental sync helpers.
- `cli.ts`
  - add `memory-pro graph doctor|sync|backfill|infer` and `docs refresh`.
- `openclaw.plugin.json`
  - schema + ui hints for auth/sync/inference/docs settings.
- `README.md`
  - document deep integration behavior and conservative mode.
- `test/*`
  - unit/integration coverage for MCP, sync, reflection, docs materialization.

## Milestone Acceptance Test Plan

### M0 Foundation

- MCP handshake test includes `initialize` + `notifications/initialized`.
- Session id header is reused on subsequent calls.
- Auth header behavior verified.

### M1 Sync

- Create/update/delete memory flows produce expected graph sync status.
- Backfill processes historical LanceDB entries incrementally.

### M2 Reflection

- Reflection receives bounded graph context.
- Graph sections are parsed and persisted.

### M3 Inference

- Inference outputs include confidence + evidence.
- Contradiction handling prevents unsafe promotion.

### M4 Docs Materialization

- Managed marker blocks update deterministically.
- User-authored text outside markers remains unchanged.

### M5 Ops

- CLI doctor/sync/backfill/infer/docs-refresh flows pass smoke checks.

### M6 Regression

- Existing memory, self-improvement, and reflection behavior remains intact.

## Initial Delivery Slice

Recommended first production slice:

- Foundation (MCP auth/session + config parity).
- Graph sync create/update/delete baseline.
- `MEMORY.md` and `HEARTBEAT.md` materialization.
- CLI doctor + docs refresh.

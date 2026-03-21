# Project Review — Agent-Scoped Memory Upgrade

## Goal
Enhance `memory-lancedb-pro` without reworking its architecture:
- keep LanceDB + scopes + mdMirror + CLI + tools as the foundation
- add per-agent default memory isolation
- add a safe upgrade path for existing Markdown memories later
- keep behavior backward compatible unless explicitly enabled

## Current Runtime / Development State
- Installed runtime plugin path: `~/.openclaw/extensions/memory-lancedb-pro`
- Git-managed development clone: `~/.openclaw/workspace/dev/memory-lancedb-pro`
- Active development branch: `feat/agent-scoped-memory-upgrade-flow`
- Current live embedding setup on host: local Ollama `bge-m3`

## Architecture Snapshot
### Core files
- `index.ts` — plugin entrypoint, lifecycle, auto-capture/recall, mdMirror wiring
- `src/store.ts` — LanceDB persistence and search
- `src/retriever.ts` — hybrid retrieval, ranking, rerank plumbing
- `src/embedder.ts` — embedding provider abstraction (OpenAI-compatible)
- `src/scopes.ts` — scope definitions and access model
- `src/tools.ts` — agent-facing memory tools
- `cli.ts` — operational CLI surface (`memory-pro ...`)
- `src/migrate.ts` — legacy LanceDB migration utilities

## What already supports our needs
### Per-agent isolation foundations already exist
- `agent:<id>` scope naming is already supported
- agent access control is already modeled in `scopes.agentAccess`
- default scope selection already flows through scope manager

### Migration foundations partially exist
- legacy LanceDB migration already has its own module
- mdMirror already gives a Markdown dual-write outlet
- CLI surface already exists, so future import/upgrade commands can be additive

## What is missing for our target workflow
1. **Automatic per-agent default isolation**
   - today isolation is possible, but still too config/manual driven
   - new/unknown agents can still fall back to `global`

2. **First-run / first-enable upgrade guidance**
   - plugin does not yet guide users when historical `MEMORY.md` / daily memory files exist
   - there is no structured “detect -> preview -> confirm -> import” workflow

3. **Markdown -> LanceDB ingest path**
   - current `mdMirror` is outbound only
   - there is no inward upgrade path for old memory files yet

## Constraints for implementation
- do not redesign the storage model
- do not replace scopes with a new permission model
- do not make migration mandatory or automatic without user confirmation
- keep defaults backward compatible
- prefer additive CLI/config behavior over invasive runtime changes

## Development policy for this repo
- all edits happen in the git-managed dev clone, never directly inside installed runtime paths
- use git branches / worktrees for isolated parallel work
- main agent owns architecture understanding and task decomposition
- coding workers receive scoped tasks, not open-ended repo exploration

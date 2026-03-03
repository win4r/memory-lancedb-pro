# memory-lancedb-pro — Architecture & Module Reference

> Generated from codebase exploration. Version 1.0.23.

## Table of Contents

1. [Overview](#overview)
2. [Repository Layout](#repository-layout)
3. [Data Flow](#data-flow)
4. [Root Entry Points](#root-entry-points)
   - [index.ts — Plugin Core](#indexts--plugin-core)
   - [cli.ts — CLI Layer](#clits--cli-layer)
5. [Core Modules (`src/`)](#core-modules-src)
   - [store.ts — LanceDB Storage](#storerts--lancedb-storage)
   - [embedder.ts — Embedding Abstraction](#embedderrs--embedding-abstraction)
   - [retriever.ts — Hybrid Retrieval Pipeline](#retrieverrs--hybrid-retrieval-pipeline)
   - [tools.ts — Agent Tool Definitions](#toolsrs--agent-tool-definitions)
   - [scopes.ts — Multi-Scope Access Control](#scopesrs--multi-scope-access-control)
   - [adaptive-retrieval.ts — Query Skip Detection](#adaptive-retrievalrs--query-skip-detection)
   - [noise-filter.ts — Content Quality Filter](#noise-filterrs--content-quality-filter)
   - [chunker.ts — Long-Context Chunking](#chunkerrs--long-context-chunking)
   - [migrate.ts — Legacy Migration](#migraterts--legacy-migration)
6. [Supporting Files](#supporting-files)
   - [openclaw.plugin.json — Plugin Manifest](#openclawpluginjson--plugin-manifest)
7. [Scripts](#scripts)
   - [scripts/jsonl_distill.py](#scriptsjsonl_distillpy)
   - [scripts/smoke-openclaw.sh](#scriptssmoke-openclawsh)
8. [Examples](#examples)
   - [examples/new-session-distill](#examplesnew-session-distill)
9. [Skills](#skills)
   - [skills/lesson/SKILL.md](#skillslessonskillmd)
10. [Tests](#tests)
    - [test/cli-smoke.mjs](#testcli-smokemjs)
11. [Key Architectural Patterns](#key-architectural-patterns)

---

## Overview

`memory-lancedb-pro` is an **OpenClaw plugin** that gives AI agents persistent, long-term memory backed by [LanceDB](https://lancedb.github.io/lancedb/) — an embedded vector database that runs entirely on-disk without a separate server.

**Core capabilities:**

| Capability | Description |
|---|---|
| Hybrid retrieval | Combines vector similarity search with BM25 full-text search |
| Cross-encoder reranking | Optional HTTP reranking call (Jina, Voyage, Pinecone, SiliconFlow) |
| Multi-scope isolation | Isolates memories per agent, project, user, or globally |
| Auto-capture | Extracts notable content from agent conversations after each session |
| Auto-recall | Injects relevant memories as context before an agent starts |
| Long-context chunking | Splits oversized documents before embedding |
| CLI management | Full `openclaw memory-pro` CLI for search, import/export, stats, migration |

---

## Repository Layout

```
memory-lancedb-pro/
├── index.ts                    # Plugin entry point — wires everything together
├── cli.ts                      # All `openclaw memory-pro` CLI subcommands
├── package.json                # ES Module, dependencies, test script
├── openclaw.plugin.json        # JSON Schema manifest for plugin configuration UI
├── CHANGELOG.md
├── README.md / README_CN.md
│
├── src/                        # Core library modules
│   ├── store.ts                # LanceDB storage layer
│   ├── embedder.ts             # OpenAI-compatible embedding client + LRU cache
│   ├── retriever.ts            # Hybrid retrieval pipeline (fuse → rerank → score → MMR)
│   ├── tools.ts                # MCP-style tool registrations for agents
│   ├── scopes.ts               # Scope definitions and access control
│   ├── adaptive-retrieval.ts   # Decides whether to skip retrieval for trivial queries
│   ├── noise-filter.ts         # Filters boilerplate/denial text from results
│   ├── chunker.ts              # Splits long documents for embedding
│   └── migrate.ts              # Migrates data from legacy `memory-lancedb` plugin
│
├── docs/
│   ├── ARCHITECTURE.md         # This file
│   └── long-context-chunking.md
│
├── scripts/
│   ├── jsonl_distill.py        # Incremental JSONL session reader for distillation pipelines
│   └── smoke-openclaw.sh       # On-host preflight validation script
│
├── examples/
│   └── new-session-distill/    # Full async distillation pipeline (hook → queue → worker)
│       ├── hook/               # OpenClaw hook that enqueues tasks on `/new`
│       └── worker/             # Gemini Map-Reduce worker + systemd unit
│
├── skills/
│   └── lesson/SKILL.md         # `/lesson` skill for storing technical pitfalls
│
└── test/
    └── cli-smoke.mjs           # Smoke test for CLI registration and reembed
```

---

## Data Flow

### Write Path (storing a memory)

```
User message
    │
    ▼
shouldCapture(text)          ← regex-based trigger detection (EN/CZ/CJK)
    │ yes
    ▼
detectCategory(text)         ← classifies: preference | fact | decision | entity | other
    │
    ▼
isNoise(text)                ← rejects boilerplate / denial phrases
    │ not noise
    ▼
Embedder.embedPassage(text)  ← OpenAI-compatible API + LRU cache; auto-chunks on ctx error
    │
    ▼
MemoryStore.vectorSearch()   ← duplicate check (cosine > 0.98 → skip)
    │ not duplicate
    ▼
MemoryStore.store(entry)     ← writes { id, text, vector, category, scope, importance, timestamp, metadata }
    │
    ▼
LanceDB table.add()          ← on-disk storage
```

### Read Path (retrieving memories)

```
Query string
    │
    ▼
adaptive-retrieval: shouldSkipRetrieval()   ← skip greetings, commands, short text, etc.
    │ proceed
    ▼
Embedder.embedQuery(query)                  ← task-aware embedding (query vs. passage)
    │
    ▼
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
MemoryStore.vectorSearch()          MemoryStore.bm25Search()   ← parallel
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
           fuseResults()            ← vector-base score + 15% BM25 bonus; ghost-entry check
                   │
                   ▼
       filter by minScore (0.3)
                   │
                   ▼
           rerankResults()          ← cross-encoder HTTP call (5s timeout) → blend 60/40
                   │ fallback: cosine similarity
                   ▼
       applyRecencyBoost()          ← additive exponential decay bonus for newer entries
                   │
                   ▼
       applyImportanceWeight()      ← multiplicative: 0.7 + 0.3 * importance
                   │
                   ▼
       applyLengthNormalization()   ← logarithmic penalty for very long entries
                   │
                   ▼
       applyTimeDecay()             ← multiplicative age penalty, floor 0.5x
                   │
                   ▼
       filter by hardMinScore (0.35)
                   │
                   ▼
           filterNoise()            ← removes denial/boilerplate entries
                   │
                   ▼
       applyMMRDiversity()          ← cosine sim > 0.85 → demote near-duplicates
                   │
                   ▼
           top-k results
```

---

## Root Entry Points

### `index.ts` — Plugin Core

The main entry point exported as an OpenClaw plugin. Every other module is instantiated and wired here.

**Key exports:**

| Export | Description |
|---|---|
| `register(api)` | Called by OpenClaw to initialize the plugin |
| `shouldCapture(text)` | Determines if user text is worth storing |
| `detectCategory(text)` | Classifies text into a memory category |

**What `register(api)` does:**

1. **Resolves config** — Reads `PluginConfig` from OpenClaw settings; substitutes `${ENV_VAR}` placeholders via `resolveEnvVars()`.
2. **Instantiates modules** — Creates `MemoryStore`, `Embedder`, `MemoryRetriever`, `MemoryScopeManager`, `MemoryMigrator`.
3. **Registers agent tools** — Calls `registerAllMemoryTools()` from `tools.ts`.
4. **Registers CLI** — Calls `createMemoryCLI()` from `cli.ts` and passes it to `api.registerCli()`.
5. **Registers lifecycle hooks:**

   | Hook | Behavior | Default |
   |---|---|---|
   | `before_agent_start` | Auto-recall: reads query, retrieves relevant memories, injects `<relevant-memories>` block | OFF |
   | `agent_end` | Auto-capture: extracts LLM-generated content from session JSONL, detects notable text, stores memories | ON |
   | `command:new` | Session memory: on `/new`, optionally summarizes the conversation before resetting | OFF |

6. **Registers service** — Provides `start`/`stop` lifecycle. On startup: validates storage path, tests embedding API, creates FTS index, runs migration check. Starts a daily auto-backup (JSONL export, 7-day retention).

**Session file handling:**

`readSessionMessages()`, `readSessionContentWithResetFallback()`, and `findPreviousSessionFile()` parse OpenClaw's JSONL session logs. They handle file rotation caused by `/new` (OpenClaw renames the active file to `.reset.<timestamp>`).

---

### `cli.ts` — CLI Layer

Implements all `openclaw memory-pro <subcommand>` commands using the `commander` library.

**Subcommands:**

| Command | Description |
|---|---|
| `version` | Print plugin version |
| `list` | List memories; supports `--scope`, `--category`, `--limit`, `--offset`, `--json` |
| `search <query>` | Hybrid retrieval search; labels results with source (vector/BM25/reranked) |
| `stats` | Count totals, breakdowns by scope/category, retrieval mode |
| `delete <id>` | Delete by full UUID or 8-char hex prefix |
| `delete-bulk` | Bulk delete; requires `--scope` or `--before` as safety gate; supports `--dry-run` |
| `export` | Export memories to JSON (vectors stripped) |
| `import <file>` | Import from JSON; detects duplicates; supports `--dry-run` |
| `reembed` | Re-embed an existing LanceDB into a new DB under a different model |
| `migrate check/run/verify` | Migration from legacy `memory-lancedb` plugin |

**`reembed` safety**: Refuses in-place re-embedding (source and destination cannot be the same path).

**Exports:**

| Export | Description |
|---|---|
| `registerMemoryCLI(program, context)` | Attaches all subcommands to a Commander program |
| `createMemoryCLI(context)` | Factory returning the function OpenClaw expects from `api.registerCli` |

---

## Core Modules (`src/`)

### `store.ts` — LanceDB Storage

The only module that directly imports `@lancedb/lancedb`. All other modules go through this abstraction.

**Types:**

```typescript
interface MemoryEntry {
  id: string;           // UUID v4
  text: string;
  vector: number[];
  category: string;     // preference | fact | decision | entity | other
  scope: string;        // e.g. "global", "agent:main"
  importance: number;   // 0.0–1.0
  timestamp: string;    // ISO 8601
  metadata: string;     // JSON string for extensible fields
}

interface MemorySearchResult {
  entry: MemoryEntry;
  score: number;        // 0.0–1.0 normalized similarity
}
```

**`MemoryStore` class:**

| Method | Description |
|---|---|
| `ensureInitialized()` | Lazy init with promise deduplication. Opens or creates `"memories"` table. Validates vector dimensions. Creates FTS index (graceful fallback if unavailable). |
| `store(entry)` | Generates UUID + timestamp, calls `table.add()`. |
| `importEntry(entry)` | Preserves original `id`/`timestamp`. Used for migration/reembed. |
| `hasId(id)` | Point lookup. Used to detect ghost entries in BM25 results (FTS index can outlive deleted rows). |
| `vectorSearch(vector, limit, minScore, scopeFilter)` | Over-fetches by 10× then filters. Converts L2 distance → similarity: `1/(1+distance)`. |
| `bm25Search(query, limit, scopeFilter)` | Uses `table.search(query, "fts")`. Normalizes BM25 score via sigmoid: `1/(1+exp(-score/5))`. Returns `[]` if FTS unavailable. |
| `delete(id, scopeFilter)` | Supports full UUID and 8-char prefix. Throws on ambiguous prefix. |
| `update(id, updates, scopeFilter)` | LanceDB has no in-place update — implements delete + re-add. Preserves original timestamp. |
| `list(...)` | Fetches all, sorts by timestamp descending, slices. Does not return vectors. |
| `stats(scopeFilter)` | Aggregate counts by scope and category. |
| `bulkDelete(scopeFilter, beforeTimestamp)` | Requires at least one filter for safety. |
| `hasFtsSupport` (getter) | Whether BM25 is available. |

**Standalone exports:**

- `loadLanceDB()` — Lazy singleton import with error wrapping (also used by `migrate.ts` and `cli.ts`).
- `validateStoragePath(dbPath)` — Resolves symlinks, creates directories, checks write permissions. Throws descriptive errors with fix instructions.

**Scope filtering:** Applied at both the LanceDB SQL `WHERE` clause level and re-checked in application code. `scope IS NULL` entries are included for backward compatibility with pre-scope data.

---

### `embedder.ts` — Embedding Abstraction

Wraps the OpenAI SDK for any OpenAI-compatible embedding API.

**Internal `EmbeddingCache`:**
- LRU cache with 256-entry maximum and 30-minute TTL.
- SHA-256 keyed by `task + text`.
- Tracks hit/miss stats accessible via `Embedder.cacheStats`.

**`EMBEDDING_DIMENSIONS`:** Static lookup table of known vector dimensions per model (OpenAI, Gemini, Jina v5, BAAI/bge-m3, common local models).

**`Embedder` class:**

| Method | Description |
|---|---|
| `embedQuery(text)` | Embeds using `taskQuery` task type (distinguishes query vs. passage for providers like Jina). |
| `embedPassage(text)` | Embeds using `taskPassage` task type. |
| `embed(text)` / `embedBatch(texts)` | Backward-compatible aliases for passage embedding. |
| `embedSingle(text, task)` | Core method. Checks cache first. On context-length error, falls back to `smartChunk()` → embeds chunks in parallel → mean embedding → caches. |
| `embedMany(texts, task)` | Batches valid texts. Same chunk fallback on context-length error. |
| `test()` | Validates API connectivity with a "test" string. |
| `cacheStats` | Returns `{ hits, misses, size }`. |

`buildPayload()` constructs provider-specific request bodies, passing `task`, `normalized`, and `dimensions` only when configured.

**Exported helpers:**
- `getVectorDimensions(model, overrideDims)` — Used by `index.ts` to determine `vectorDim` for `MemoryStore`.

---

### `retriever.ts` — Hybrid Retrieval Pipeline

Orchestrates the entire read path from a raw query string to a ranked list of `RetrievalResult` objects.

**`RetrievalConfig` defaults:**

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"hybrid"` | `"hybrid"` (vector + BM25) or `"vector"` only |
| `vectorWeight` | `0.7` | Weight for vector score in fusion |
| `bm25Weight` | `0.3` | Weight for BM25 score in fusion |
| `minScore` | `0.3` | Pre-rerank score filter |
| `rerank` | `"cross-encoder"` | `"cross-encoder"`, `"lightweight"` (cosine), or `"none"` |
| `candidatePoolSize` | `20` | Number of candidates before reranking |
| `recencyHalfLifeDays` | `14` | Half-life for recency bonus |
| `recencyWeight` | `0.10` | Max additive bonus from recency |
| `filterNoise` | `true` | Apply noise filter after scoring |
| `lengthNormAnchor` | `500` | Characters at which length penalty activates |
| `hardMinScore` | `0.35` | Post-rerank score filter |
| `timeDecayHalfLifeDays` | `60` | Half-life for time decay penalty |

**`RetrievalResult`** extends `MemorySearchResult` with a `sources` object:
```typescript
sources: { vector: boolean; bm25: boolean; fused: boolean; reranked: boolean }
```

**Reranker provider adapters:**

| Provider | Auth header | Response field |
|---|---|---|
| jina / siliconflow | `Authorization: Bearer` | `results[].relevance_score` |
| voyage | `Authorization: Bearer` | `data[].relevance_score` |
| pinecone | `Api-Key` | `data[].score` |

Cross-encoder calls use a 5-second `AbortController` timeout and fall back to cosine similarity (`"lightweight"`) on failure.

**Result fusion (`fuseResults`):**
- Vector result score is the base.
- A BM25 hit adds a 15% bonus on top.
- BM25-only results use raw BM25 score but are validated via `hasId()` to skip ghost entries (rows deleted from the table whose FTS index entry lingers).

**Rerank blending:**
- Final score = `0.6 × cross-encoder score + 0.4 × fused score`.
- Candidates not returned by the reranker are penalized 20%.

---

### `tools.ts` — Agent Tool Definitions

Registers MCP-style tools with the OpenClaw API using [TypeBox](https://github.com/sinclairzx81/typebox) JSON schemas.

**Core tools (always registered):**

| Tool | Parameters | Description |
|---|---|---|
| `memory_recall` | `query`, `limit` (1–20, default 5), `scope`, `category` | Hybrid search. Returns formatted results with score % and source labels. Enforces scope access control. |
| `memory_store` | `text`, `importance` (0–1), `category`, `scope` | Embeds and stores. Pre-checks: noise filter, scope access, near-duplicate detection (cosine > 0.98 → skip). |
| `memory_forget` | `query` or `memoryId`, `scope` | Delete by ID (direct), or by query (retrieves top-5; auto-deletes if single confident match, otherwise lists candidates for selection). |
| `memory_update` | `memoryId`, `text`, `importance`, `category` | Updates a memory in place, preserving original timestamp. Accepts non-UUID `memoryId` resolved via search query. Re-embeds if `text` changes. |

**Management tools (gated by `enableManagementTools` config):**

| Tool | Parameters | Description |
|---|---|---|
| `memory_stats` | _(none)_ | Returns counts by scope and category, retrieval mode, FTS availability. |
| `memory_list` | `limit` (max 50), `scope`, `category`, `offset` | Paginated memory listing with timestamps. |

`resolveAgentId(api)` extracts the calling agent's identity at tool invocation time, enabling per-agent scope enforcement.

---

### `scopes.ts` — Multi-Scope Access Control

Manages memory isolation between agents and contexts.

**Built-in scope patterns:**

| Pattern | Example | Description |
|---|---|---|
| `global` | `global` | Shared across all agents |
| `agent:<id>` | `agent:main` | Private to a specific agent |
| `project:<id>` | `project:my-app` | Project-wide scope |
| `user:<id>` | `user:oscar` | User-scoped |
| `custom:<name>` | `custom:research` | Named custom scope |

**`MemoryScopeManager` class:**

| Method | Description |
|---|---|
| `getAccessibleScopes(agentId)` | Returns explicit access list if configured, otherwise `["global"]` + `agent:<id>`. |
| `getDefaultScope(agentId)` | Returns agent scope if accessible, otherwise `global`. |
| `isAccessible(scope, agentId)` | Checks the accessible scope list. |
| `validateScope(scope)` | Validates against defined scopes and built-in patterns. |
| `addScopeDefinition` / `removeScopeDefinition` | Manage scope registry; `global` cannot be removed. |
| `setAgentAccess` / `removeAgentAccess` | Configure per-agent access control lists. |
| `exportConfig` / `importConfig` | Serialization for persistence. |
| `getStats()` | Scope counts by type. |

**Exported factory helpers:** `createScopeManager`, `createAgentScope`, `createCustomScope`, `createProjectScope`, `createUserScope`, `parseScopeId`, `isScopeAccessible`, `filterScopesForAgent`.

---

### `adaptive-retrieval.ts` — Query Skip Detection

Decides whether a query is trivial enough to skip retrieval entirely, saving embedding API calls and avoiding noise injection.

**`normalizeQuery(query)`** — Strips OpenClaw-injected prefixes before pattern matching:
- `[cron:<id> <name>]` headers
- Timestamp headers `[Mon 2026-03-02 04:21 GMT+8]`
- `Conversation info (untrusted metadata):` blocks

**`shouldSkipRetrieval(query, minLength?)` — returns `true` to skip:**

| Condition | Logic |
|---|---|
| Force-retrieve patterns | Memory-related words, "last time", "my name" — checked first, even before length |
| Too short | Under 5 characters |
| Skip patterns | Greetings, slash commands, shell commands, affirmations, continuation prompts, pure emoji, `HEARTBEAT`, system messages, single-word pings |
| Length threshold | CJK: 6 chars; Latin: 15 chars |
| Contains `?` / `？` | Always proceed to retrieval |

---

### `noise-filter.ts` — Content Quality Filter

Filters low-quality or unhelpful text from both stored memories and retrieval results.

**Three pattern categories:**

| Category | Examples |
|---|---|
| `DENIAL_PATTERNS` | "I don't have information about", "I don't recall", "I have no memory" |
| `META_QUESTION_PATTERNS` | "Do you remember", "Did I tell you", "Can you recall" |
| `BOILERPLATE_PATTERNS` | Greetings, "fresh session", `HEARTBEAT` |

**Exports:**

| Export | Description |
|---|---|
| `isNoise(text, options)` | Returns `true` if text matches any pattern. Used in `memory_store` and retriever. |
| `filterNoise<T>(items, getText, options)` | Generic array filter using a text extractor function. |

---

### `chunker.ts` — Long-Context Chunking

Splits documents that exceed embedding model context limits. The embedder uses this lazily — only triggered after a context-length error from the provider.

**`EMBEDDING_CONTEXT_LIMITS`:** Known token limits per model (OpenAI, Jina, local models, etc.).

**`ChunkerConfig` defaults:**

| Parameter | Default | Description |
|---|---|---|
| `maxChunkSize` | `4000` | Max characters per chunk |
| `overlapSize` | `200` | Character overlap between consecutive chunks |
| `minChunkSize` | `200` | Minimum chunk size to avoid tiny fragments |
| `semanticSplit` | `true` | Prefer sentence boundaries when splitting |
| `maxLinesPerChunk` | `50` | Hard line count limit per chunk |

**`chunkDocument(text, config)` — core algorithm:**

1. If remaining text ≤ `maxChunkSize`, take the rest.
2. Find a split point by scanning backward from `maxEnd` to `minEnd`:
   - Respect `maxLinesPerChunk` first (scan for Nth newline).
   - If `semanticSplit`, prefer sentence boundaries (`.!?。！？`).
   - Fall back to newline, then any whitespace, then hard cut.
3. Advance position with overlap: `nextPos = end - overlapSize`.
4. Guard against infinite loops with a max-iteration safety check.

**`smartChunk(text, embedderModel)`** — Adapts config to the model's known context limit:
- `maxChunkSize = 70%` of context limit
- `overlapSize = 5%` of context limit
- `minChunkSize = 10%` of context limit

---

### `migrate.ts` — Legacy Migration

Migrates data from the old `memory-lancedb` plugin to `memory-lancedb-pro`.

**`MemoryMigrator` class:**

| Method | Description |
|---|---|
| `migrate(options)` | Finds source DB, loads legacy entries, converts schema, imports via `MemoryStore.importEntry()`. Stores migration provenance in `metadata`: `{ migratedFrom, originalId, originalCreatedAt }`. |
| `checkMigrationNeeded()` | Checks default paths (`~/.openclaw/memory/lancedb`, `~/.claude/memory/lancedb`) for existing LanceDB files. |
| `verifyMigration()` | Compares source vs. target row count. |

**Legacy schema differences:**
- Old: `createdAt` field, no `metadata` field.
- New: `timestamp` field, `metadata` as JSON string.

**Standalone export:** `checkForLegacyData()` — async function for CLI usage.

---

## Supporting Files

### `openclaw.plugin.json` — Plugin Manifest

Full JSON Schema for configuration validation and plugin UI generation.

**Notable features:**
- Defines all `PluginConfig` properties with types, defaults, valid ranges, and descriptions.
- Uses `"advanced": true` flags for progressive disclosure in the plugin UI.
- Marks `embedding.apiKey` and `retrieval.rerankApiKey` as `"sensitive": true` so they are redacted in logs and UI.

---

## Scripts

### `scripts/jsonl_distill.py`

A standalone Python utility for incrementally reading OpenClaw session JSONL files to feed a downstream distillation agent. Does **not** call any LLM or write to LanceDB directly.

**Subcommands:**

| Command | Description |
|---|---|
| `init` | Sets cursor to current EOF for all sessions (start tracking from now). |
| `run` | Reads new content from session tails since last cursor. Handles file rotation/truncation via inode+size checks. Filters noise, caps at 30 messages/agent. Creates a batch JSON file. |
| `commit <batch-file>` | Advances committed byte offsets; deletes the batch file. |

**Safety features:**
- Excludes the `memory-distiller` agent to prevent self-ingestion loops.
- Supports an agent allowlist via `OPENCLAW_JSONL_DISTILL_ALLOWED_AGENT_IDS` env var.

---

### `scripts/smoke-openclaw.sh`

Shell script for on-host preflight validation of a real OpenClaw installation. Runs the following CLI subcommands in sequence and checks for errors:

`version` → `stats` → `list` → `search` → `export` → `import --dry-run` → `delete --help` → `delete-bulk --dry-run` → `migrate check` → `reembed --dry-run`

---

## Examples

### `examples/new-session-distill`

A complete, production-ready async pipeline for distilling session knowledge into memories on every `/new` command.

**Architecture:**

```
/new command
    │
    ▼
Hook (fast, no LLM)
    │  writes task JSON to inbox/
    ▼
Filesystem Queue (inbox/ → processing/ → done/|error/)
    │
    ▼
Worker (async, Gemini)
    │
    ├── Map phase: stream JSONL in 12k-char chunks
    │             extract structured lessons via Gemini
    │
    ├── Reduce phase: deduplicate, score quality,
    │                filter, cap at 20 lessons
    │
    └── Import: `openclaw memory-pro import --scope agent:main`
                [optional: Telegram notification]
```

**Hook** (`hook/enqueue-lesson-extract/handler.ts`):
- Fires on `command:new` events.
- Finds the session JSONL (handles `.reset.*` rotation).
- Writes a task JSON with SHA1 ID (`sessionKey + sessionId + file + timestamp`).

**Worker** (`worker/lesson-extract-worker.mjs`):
- Uses `fs.watch` for inbox polling.
- **Map phase**: Streams JSONL in 12k-char chunks with 10-message overlap. For each chunk, calls Gemini with a structured extraction prompt requesting `{ category, importance, text, evidence, tags }`.
- **Reduce phase**: Normalizes text for deduplication, scores by quality factors (pitfall format, evidence count, importance score, known tech terms), filters by minimum quality, caps at 20 final lessons.
- **Reliability**: Moves task files through `inbox/ → processing/ → done/|error/` for fault tolerance.

**Deployment:** A systemd user-level service unit (`worker/systemd/lesson-extract-worker.service`) is provided for Linux.

---

## Skills

### `skills/lesson/SKILL.md`

An OpenClaw skill triggered by `/lesson`. Instructs the agent to extract and store two layers of knowledge from recent context:

1. **Technical layer** — Format: `Pitfall / Cause / Fix / Prevention` (category: `fact`, importance ≥ 0.8).
2. **Principle layer** — Format: `Decision principle / Trigger / Action` (category: `decision`, importance ≥ 0.85).

After storing, the skill verifies with `memory_recall` using anchor keywords and reports what was stored.

---

## Tests

### `test/cli-smoke.mjs`

A minimal integration smoke test run via `node test/cli-smoke.mjs` (uses `jiti` for TypeScript execution without pre-compilation).

**What it tests:**

1. Creates a temp LanceDB with one 4-dimensional test row (small vector for speed).
2. Constructs minimal stub context objects.
3. Registers `createMemoryCLI` with a Commander instance.
4. Verifies `memory-pro version` doesn't throw.
5. Verifies `memory-pro reembed --source-db ... --limit 1 --batch-size 999 --dry-run` — regression test confirming `clampInt` correctly clamps batch size to valid range.
6. Cleans up the temp directory.

---

## Key Architectural Patterns

### Lazy Initialization with Promise Deduplication

`MemoryStore.ensureInitialized()` uses a single `Promise` stored as an instance variable. Concurrent callers all await the same promise, preventing duplicate table creation on startup.

### Graceful BM25 Degradation

If LanceDB's FTS index is unavailable (older versions or write-protected paths), the retriever silently falls back to vector-only mode. The `hasFtsSupport` getter propagates this signal from `MemoryStore` through to `MemoryRetriever` and the `memory_stats` tool.

### Ghost Entry Protection

LanceDB's FTS index can retain references to rows that have been deleted from the main table. `bm25Search` results are validated via `hasId()` (a point lookup) before being included in fusion, preventing ghost entries from influencing results.

### Task-Aware Embeddings

`Embedder` exposes both `embedQuery()` and `embedPassage()` methods using provider task types. This supports models like Jina that encode queries and documents differently for improved retrieval accuracy.

### Scope Threading

Every read and write operation carries a `scopeFilter: string[]` parameter. Scope enforcement happens at two levels: the LanceDB SQL `WHERE` clause and a redundant application-layer check. Null-scope legacy entries are treated as `"global"` for backward compatibility.

### Inverse Update Pattern

LanceDB does not support in-place row updates. `MemoryStore.update()` implements update as delete + re-add, explicitly preserving the original `timestamp` to maintain sort order in listings.

### Provider-Agnostic Reranking

`retriever.ts` normalizes four cross-encoder API formats (Jina, Voyage, Pinecone, SiliconFlow) through `buildRerankRequest` / `parseRerankResponse` adapters, with a 5-second `AbortController` timeout and cosine similarity fallback.

# Phase 2 Compatibility Subtree Freeze

## Purpose
Freeze the preferred Markdown compatibility target for `memory-lancedb-pro` without over-designing future structure.

This document intentionally keeps the contract small:
- a per-agent workspace subtree,
- a required README / statement,
- and dated Markdown files.

It does **not** freeze extra derived paths that were not part of the original agreed requirement.

---

## Frozen target

For each agent workspace, the compatibility Markdown output should live under:

```text
memory/plugin-memory-pro/
```

That subtree is **plugin-managed compatibility state**, not the human-authored primary daily log.

### Frozen directory layout

```text
<agent-workspace>/
  MEMORY.md
  memory/
    YYYY-MM-DD.md                  # human-authored daily log
    plugin-memory-pro/
      README.md                    # required explanatory file
      YYYY-MM-DD.md                # plugin-managed compatibility daily file
```

---

## Why this layout

### 1. Keep human daily logs clean
Do **not** write plugin-generated material directly into:

```text
memory/YYYY-MM-DD.md
```

Those files remain the human-authored / agent-authored daily memory surface.

### 2. Stay inside the native memory tree
Keeping the subtree under `memory/` preserves compatibility with OpenClaw's legacy Markdown + SQLite indexing expectations.

### 3. Keep the first implementation understandable
The current goal is:
- a dedicated subtree,
- a clear README,
- append-only dated Markdown files.

Do **not** freeze extra derived subpaths until runtime behavior actually needs them.

### 4. Make plugin side effects explainable
`README.md` is required so a later user can tell why these files exist and what role they play after enable/disable cycles.

---

## Semantics by path

## `README.md`
Required. Must explain:
- this subtree exists because `memory-lancedb-pro` was enabled
- files here are compatibility / reversibility artifacts
- top-level `memory/YYYY-MM-DD.md` remains the human-authored daily log
- deleting this subtree may reduce legacy continuity after plugin disable/uninstall

## `YYYY-MM-DD.md`
Append-only compatibility daily log.

Use this for:
- human-readable chronological trace of plugin-managed durable memory activity
- compatibility continuity during plugin-enabled runtime
- helping ensure A→B memories are not trapped only in LanceDB

This file is the currently frozen minimum target.

---

## Required README template

```md
# plugin-memory-pro compatibility subtree

This directory was created because `memory-lancedb-pro` was enabled for this agent workspace.

## What this directory is
- A compatibility / reversibility projection of plugin-managed durable memory.
- A bridge that helps OpenClaw's original Markdown / SQLite memory systems remain usable.
- Not the primary human-authored daily log.

## What the files mean
- `YYYY-MM-DD.md` files contain plugin-managed compatibility memory written during active plugin use.

## Important note
The top-level file `memory/YYYY-MM-DD.md` remains the normal human-authored / agent-authored daily memory log.
Files under `memory/plugin-memory-pro/` exist so that enabling and later disabling the plugin is non-destructive and reversible.
```

---

## Guardrails

1. Do not silently mix plugin output into top-level daily logs.
2. Do not over-freeze internal subpaths that are not yet required by the agreed design.
3. Do not require users to understand LanceDB internals to recover from plugin removal.
4. Keep the subtree per-agent workspace local so compatibility remains explainable.
5. Preserve legacy SQLite continuity alongside this subtree; the subtree does not replace the SQLite requirement.

---

## Phase 2 implementation note

This document freezes the **minimum directory contract**.
It does **not** force future deeper structure such as per-memory canonical files.
If later runtime behavior truly needs more structure, that should be introduced by a separate design update rather than being assumed now.

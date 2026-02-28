# Changelog

## 1.0.12

- Fix: ghost memories stuck in autoRecall after deletion (#15). BM25-only results from stale FTS index are now validated via `store.hasId()` before inclusion in fused results. Removed the BM25-only floor score of 0.5 that allowed deleted entries to survive `hardMinScore` filtering.
- Fix: HEARTBEAT pattern now matches anywhere in the prompt (not just at start), preventing autoRecall from triggering on prefixed HEARTBEAT messages.
- Add: `autoRecallMinLength` config option to set a custom minimum prompt length for autoRecall (default: 15 chars English, 6 CJK). Prompts shorter than this threshold are skipped.
- Add: `ping`, `pong`, `test`, `debug` added to skip patterns in adaptive retrieval.

## 1.0.11

- Change: set `autoRecall` default to `false` to avoid the model echoing injected `<relevant-memories>` blocks.

## 1.0.10

- Fix: avoid blocking OpenClaw gateway startup on external network calls by running startup self-checks in the background with timeouts.

## 1.0.9

- Change: update default `retrieval.rerankModel` to `jina-reranker-v3` (still fully configurable).

## 1.0.8

- Add: JSONL distill extractor supports optional agent allowlist via env var `OPENCLAW_JSONL_DISTILL_ALLOWED_AGENT_IDS` (default off / compatible).

## 1.0.7

- Fix: resolve `agentId` from hook context (`ctx?.agentId`) for `before_agent_start` and `agent_end`, restoring per-agent scope isolation when using multi-agent setups.

## 1.0.6

- Fix: auto-recall injection now correctly skips cron prompts wrapped as `[cron:...] run ...` (reduces token usage for cron jobs).
- Fix: JSONL distill extractor filters more transcript/system noise (BOOT.md, HEARTBEAT, CLAUDE_CODE_DONE, queued blocks) to avoid polluting distillation batches.

## 1.0.5

- Add: optional JSONL session distillation workflow (incremental cursor + batch format) via `scripts/jsonl_distill.py`.
- Docs: document the JSONL distiller setup in README (EN) and README_CN (ZH).

## 1.0.4

- Fix: `embedding.dimensions` is now parsed robustly (number / numeric string / env-var string), so it properly overrides hardcoded model dims (fixes Ollama `nomic-embed-text` dimension mismatch).

## 1.0.3

- Fix: `memory-pro reembed` no longer crashes (missing `clampInt` helper).

## 1.0.2

- Fix: pass through `embedding.dimensions` to the OpenAI-compatible `/embeddings` request payload when explicitly configured.
- Chore: unify plugin version fields (`openclaw.plugin.json` now matches `package.json`).

## 1.0.1

- Fix: CLI command namespace updated to `memory-pro`.

## 1.0.0

- Initial npm release.

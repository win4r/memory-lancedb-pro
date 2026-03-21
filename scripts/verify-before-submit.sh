#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[verify] running full test suite for memory-lancedb-pro"
npm test

echo "[verify] success"

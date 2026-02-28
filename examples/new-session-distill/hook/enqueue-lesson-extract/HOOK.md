---
name: enqueue-lesson-extract
description: "Enqueue a lesson-extraction task on /new (async Map-Reduce → LanceDB Pro)"
metadata:
  {
    "openclaw": {
      "emoji": "🧾",
      "events": ["command:new"]
    }
  }
---

# Enqueue Lesson Extract Hook

Writes a small JSON task file to a queue directory when `/new` is issued.

This is intentionally fast and non-blocking. A separate systemd worker consumes tasks and:
- reads the session JSONL transcript
- runs Map-Reduce extraction with Gemini Flash
- writes high-signal, deduped lessons into LanceDB Pro
- sends a notification (optional)

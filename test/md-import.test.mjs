/**
 * Tests for src/md-import.ts — P2-W2 Markdown preview parser / dry-run foundation.
 *
 * Scope: parse MEMORY.md and memory/YYYY-MM-DD.md content strings into candidate
 * memory units, classify durable vs noisy, produce dry-run preview output.
 * No file I/O, no LanceDB, no SQLite, no real import writes.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import path from "node:path";
import jitiFactory from "jiti";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const {
  parseMemoryMd,
  parseDailyMd,
  classifyCandidates,
  previewMd,
} = jiti("../src/md-import.ts");

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const MEMORY_MD_SIMPLE = `
# User Preferences

- Prefers concise diffs over verbose ones
- Uses dark mode in all editors

# Project Rules

- Always write tests before implementation
- No hardcoded secrets in source code
`;

const MEMORY_MD_MIXED = `
---
name: example
---

# Identity

I am a senior TypeScript engineer.

## Tools

- Prefers VSCode
- Uses zsh

# Transient notes

Meeting at 3pm today about sprint planning.
Check deploy at end of day.
`;

const MEMORY_MD_EMPTY = ``;

const MEMORY_MD_HEADINGS_ONLY = `
# Section A

## Subsection

# Section B
`;

const DAILY_MD_SIMPLE = `
- Decided to use LanceDB for vector storage
- Fixed authentication bug in OAuth flow
- Follow up: review PR #42 tomorrow
- Discovered that chunker handles CJK text correctly
`;

const DAILY_MD_MIXED = `
# 2026-03-20

- Committed phase 2 design docs
- Long meeting about deployment — not durable

Ran tests. All passing.

- TODO: check alignment with P2-W1 on shared interfaces
`;

const DAILY_MD_NOISY = `
- ok
- done
- yes
- x
`;

// ---------------------------------------------------------------------------
// parseMemoryMd
// ---------------------------------------------------------------------------

describe("parseMemoryMd", () => {
  it("extracts bullet items from MEMORY.md content", () => {
    const candidates = parseMemoryMd(MEMORY_MD_SIMPLE);

    assert.ok(Array.isArray(candidates), "should return an array");
    assert.ok(candidates.length > 0, "should find at least one candidate");

    const texts = candidates.map((c) => c.text);
    assert.ok(
      texts.some((t) => t.includes("concise diffs")),
      "should extract bullet about diffs"
    );
    assert.ok(
      texts.some((t) => t.includes("dark mode")),
      "should extract bullet about dark mode"
    );
    assert.ok(
      texts.some((t) => t.includes("tests before implementation")),
      "should extract bullet about TDD"
    );
  });

  it("attaches heading context to each candidate", () => {
    const candidates = parseMemoryMd(MEMORY_MD_SIMPLE);

    const diffEntry = candidates.find((c) => c.text.includes("concise diffs"));
    assert.ok(diffEntry, "entry about diffs must exist");
    assert.ok(
      diffEntry.headingContext,
      "should have a headingContext field"
    );
    assert.match(diffEntry.headingContext, /User Preferences/);
  });

  it("returns empty array for empty content", () => {
    const candidates = parseMemoryMd(MEMORY_MD_EMPTY);
    assert.deepEqual(candidates, []);
  });

  it("returns empty array when content has headings but no bullet items or paragraphs", () => {
    const candidates = parseMemoryMd(MEMORY_MD_HEADINGS_ONLY);
    assert.deepEqual(candidates, []);
  });

  it("skips YAML frontmatter", () => {
    const candidates = parseMemoryMd(MEMORY_MD_MIXED);
    const texts = candidates.map((c) => c.text);
    assert.ok(
      !texts.some((t) => t.includes("name: example")),
      "should not include frontmatter lines as candidates"
    );
  });

  it("sets sourceType to 'memory-md' on each candidate", () => {
    const candidates = parseMemoryMd(MEMORY_MD_SIMPLE);
    for (const c of candidates) {
      assert.equal(c.sourceType, "memory-md");
    }
  });

  it("each candidate has required fields: text, sourceType, headingContext, lineNumber", () => {
    const candidates = parseMemoryMd(MEMORY_MD_SIMPLE);
    for (const c of candidates) {
      assert.ok(typeof c.text === "string" && c.text.length > 0, "text must be non-empty string");
      assert.ok(typeof c.sourceType === "string", "sourceType must be string");
      assert.ok(typeof c.headingContext === "string", "headingContext must be string");
      assert.ok(typeof c.lineNumber === "number", "lineNumber must be number");
    }
  });

  it("recognises * and + bullet markers as well as -", () => {
    const content = `
# Preferences

* Uses asterisk bullets
+ Uses plus bullets
- Uses dash bullets
`;
    const candidates = parseMemoryMd(content);
    const texts = candidates.map((c) => c.text);
    assert.ok(texts.some((t) => t.includes("asterisk bullets")), "should extract * bullets");
    assert.ok(texts.some((t) => t.includes("plus bullets")), "should extract + bullets");
    assert.ok(texts.some((t) => t.includes("dash bullets")), "should extract - bullets");
  });
});

// ---------------------------------------------------------------------------
// parseDailyMd
// ---------------------------------------------------------------------------

describe("parseDailyMd", () => {
  it("extracts bullet items from daily file content", () => {
    const candidates = parseDailyMd(DAILY_MD_SIMPLE, "2026-03-20");

    assert.ok(Array.isArray(candidates), "should return an array");
    assert.ok(candidates.length > 0, "should find at least one candidate");

    const texts = candidates.map((c) => c.text);
    assert.ok(
      texts.some((t) => t.includes("LanceDB")),
      "should extract LanceDB decision"
    );
    assert.ok(
      texts.some((t) => t.includes("authentication bug")),
      "should extract bug fix"
    );
  });

  it("attaches date to each candidate", () => {
    const candidates = parseDailyMd(DAILY_MD_SIMPLE, "2026-03-20");
    for (const c of candidates) {
      assert.equal(c.date, "2026-03-20");
    }
  });

  it("sets sourceType to 'daily-md' on each candidate", () => {
    const candidates = parseDailyMd(DAILY_MD_SIMPLE, "2026-03-20");
    for (const c of candidates) {
      assert.equal(c.sourceType, "daily-md");
    }
  });

  it("each candidate has required fields: text, sourceType, date, lineNumber", () => {
    const candidates = parseDailyMd(DAILY_MD_SIMPLE, "2026-03-20");
    for (const c of candidates) {
      assert.ok(typeof c.text === "string" && c.text.length > 0, "text must be non-empty string");
      assert.ok(typeof c.sourceType === "string", "sourceType must be string");
      assert.ok(typeof c.date === "string", "date must be string");
      assert.ok(typeof c.lineNumber === "number", "lineNumber must be number");
    }
  });

  it("returns empty array for empty content", () => {
    const candidates = parseDailyMd("", "2026-03-20");
    assert.deepEqual(candidates, []);
  });

  it("strips leading bullet marker from text", () => {
    const candidates = parseDailyMd(DAILY_MD_SIMPLE, "2026-03-20");
    for (const c of candidates) {
      assert.doesNotMatch(c.text, /^[-*+]\s/, "text should not start with bullet marker");
    }
  });
});

// ---------------------------------------------------------------------------
// classifyCandidates
// ---------------------------------------------------------------------------

describe("classifyCandidates", () => {
  it("labels durable candidates with durability=durable", () => {
    const input = [
      { text: "Prefers concise diffs over verbose ones", sourceType: "memory-md", headingContext: "User Preferences", lineNumber: 3, date: null },
      { text: "Always write tests before implementation", sourceType: "memory-md", headingContext: "Project Rules", lineNumber: 7, date: null },
    ];
    const classified = classifyCandidates(input);

    assert.ok(classified.every((c) => c.durability === "durable"), "preference/rule items should be durable");
  });

  it("labels transient/event text as noisy", () => {
    const input = [
      { text: "Meeting at 3pm today about sprint planning", sourceType: "daily-md", headingContext: "", lineNumber: 5, date: "2026-03-20" },
      { text: "Check deploy at end of day", sourceType: "daily-md", headingContext: "", lineNumber: 6, date: "2026-03-20" },
    ];
    const classified = classifyCandidates(input);

    assert.ok(classified.every((c) => c.durability === "noisy"), "time-bound/transient items should be noisy");
  });

  it("labels very short items as noisy", () => {
    const input = [
      { text: "ok", sourceType: "daily-md", headingContext: "", lineNumber: 1, date: "2026-03-20" },
      { text: "done", sourceType: "daily-md", headingContext: "", lineNumber: 2, date: "2026-03-20" },
    ];
    const classified = classifyCandidates(input);

    assert.ok(classified.every((c) => c.durability === "noisy"), "very short items should be noisy");
  });

  it("adds durability field to each candidate", () => {
    const input = [
      { text: "Prefers VSCode for all TypeScript work", sourceType: "memory-md", headingContext: "Tools", lineNumber: 2, date: null },
    ];
    const classified = classifyCandidates(input);

    assert.ok(classified[0].durability === "durable" || classified[0].durability === "noisy");
  });

  it("returns a new array (does not mutate input)", () => {
    const input = [
      { text: "Prefers concise diffs", sourceType: "memory-md", headingContext: "Preferences", lineNumber: 3, date: null },
    ];
    const original = { ...input[0] };
    classifyCandidates(input);

    assert.deepEqual(input[0], original, "input should not be mutated");
  });

  it("preserves all original candidate fields after classification", () => {
    const input = [
      { text: "Prefers concise diffs", sourceType: "memory-md", headingContext: "Preferences", lineNumber: 3, date: null },
    ];
    const classified = classifyCandidates(input);

    assert.equal(classified[0].text, input[0].text);
    assert.equal(classified[0].sourceType, input[0].sourceType);
    assert.equal(classified[0].headingContext, input[0].headingContext);
    assert.equal(classified[0].lineNumber, input[0].lineNumber);
  });

  it("daily-md items default toward noisy unless clearly durable", () => {
    const input = [
      { text: "Discovered that chunker handles CJK text correctly", sourceType: "daily-md", headingContext: "", lineNumber: 4, date: "2026-03-20" },
      { text: "Ran tests. All passing", sourceType: "daily-md", headingContext: "", lineNumber: 5, date: "2026-03-20" },
    ];
    const classified = classifyCandidates(input);
    // At least "Ran tests" should be noisy
    const ranTests = classified.find((c) => c.text.includes("Ran tests"));
    assert.equal(ranTests.durability, "noisy");
  });
});

// ---------------------------------------------------------------------------
// previewMd
// ---------------------------------------------------------------------------

describe("previewMd", () => {
  it("returns an object with candidates and skipped arrays", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });

    assert.ok(typeof result === "object" && result !== null);
    assert.ok(Array.isArray(result.candidates), "result.candidates must be array");
    assert.ok(Array.isArray(result.skipped), "result.skipped must be array");
  });

  it("result.candidates contains only durable items", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });

    for (const c of result.candidates) {
      assert.equal(c.durability, "durable", "all candidates should be durable");
    }
  });

  it("result.skipped contains only noisy items", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });

    for (const s of result.skipped) {
      assert.equal(s.durability, "noisy", "all skipped should be noisy");
    }
  });

  it("result includes summary with counts", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });

    assert.ok(typeof result.summary === "object" && result.summary !== null);
    assert.equal(typeof result.summary.totalFound, "number");
    assert.equal(typeof result.summary.durableCount, "number");
    assert.equal(typeof result.summary.noisyCount, "number");
    assert.equal(
      result.summary.totalFound,
      result.summary.durableCount + result.summary.noisyCount,
      "total must equal durable + noisy"
    );
  });

  it("summary counts match actual arrays", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });

    assert.equal(result.summary.durableCount, result.candidates.length);
    assert.equal(result.summary.noisyCount, result.skipped.length);
  });

  it("supports daily-md sourceType via options", () => {
    const result = previewMd(DAILY_MD_SIMPLE, { sourceType: "daily-md", date: "2026-03-20" });

    assert.ok(Array.isArray(result.candidates));
    assert.ok(Array.isArray(result.skipped));
    assert.equal(typeof result.summary.totalFound, "number");
  });

  it("returns empty candidates and skipped for empty content", () => {
    const result = previewMd("", { sourceType: "memory-md" });

    assert.deepEqual(result.candidates, []);
    assert.deepEqual(result.skipped, []);
    assert.equal(result.summary.totalFound, 0);
  });

  it("result includes warnings array", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });
    assert.ok(Array.isArray(result.warnings), "result.warnings must be array");
  });

  it("each candidate in result has inferredScope field", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });

    for (const c of result.candidates) {
      assert.ok(
        typeof c.inferredScope === "string",
        "each candidate should have an inferredScope string"
      );
    }
  });

  it("does not mutate input content string", () => {
    const content = MEMORY_MD_SIMPLE.slice();
    previewMd(content, { sourceType: "memory-md" });
    assert.equal(content, MEMORY_MD_SIMPLE);
  });

  it("scope override is applied to all candidates", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md", scope: "agent:custom" });

    for (const c of result.candidates) {
      assert.equal(c.inferredScope, "agent:custom", "scope override must propagate to all candidates");
    }
  });

  it("memory-md candidates default to inferredScope 'global'", () => {
    const result = previewMd(MEMORY_MD_SIMPLE, { sourceType: "memory-md" });

    for (const c of result.candidates) {
      assert.equal(c.inferredScope, "global");
    }
  });

  it("daily-md candidates default to inferredScope 'agent:main'", () => {
    const result = previewMd(DAILY_MD_SIMPLE, { sourceType: "daily-md", date: "2026-03-20" });

    // all PreviewCandidates (candidates + skipped) should default to agent:main
    const all = [...result.candidates, ...result.skipped];
    for (const c of all) {
      assert.equal(c.inferredScope, "agent:main");
    }
  });

  it("emits a warning when daily-md source has no date provided", () => {
    const result = previewMd(DAILY_MD_SIMPLE, { sourceType: "daily-md" });

    assert.ok(
      result.warnings.some((w) => w.toLowerCase().includes("date")),
      "should warn when no date is provided for daily-md"
    );
  });

  it("emits a warning when content is non-empty but yields no parseable candidates", () => {
    const result = previewMd(MEMORY_MD_HEADINGS_ONLY, { sourceType: "memory-md" });

    assert.ok(
      result.warnings.some((w) => w.toLowerCase().includes("no parseable candidates")),
      "should warn when no candidates found in non-empty content"
    );
  });
});

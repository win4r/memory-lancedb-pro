/**
 * Test: memory_update routes through supersede for temporal-versioned categories.
 *
 * Validates the fix for the memory_update bypass identified in PR #183 review:
 * when text changes on a preferences/entities record, the update must create a
 * new superseding record and invalidate the old one, rather than mutating in place.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import Module from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";

import jitiFactory from "jiti";

process.env.NODE_PATH = [
  process.env.NODE_PATH,
  "/opt/homebrew/lib/node_modules/openclaw/node_modules",
  "/opt/homebrew/lib/node_modules",
].filter(Boolean).join(":");
Module._initPaths();

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { MemoryStore } = jiti("../src/store.ts");
const {
  appendRelation,
  buildSmartMetadata,
  deriveFactKey,
  isMemoryActiveAt,
  parseSmartMetadata,
  stringifySmartMetadata,
} = jiti("../src/smart-metadata.ts");
const { TEMPORAL_VERSIONED_CATEGORIES } = jiti("../src/memory-categories.ts");

const VECTOR_DIM = 8;

function makeVector(seed = 1) {
  const v = new Array(VECTOR_DIM).fill(1 / Math.sqrt(VECTOR_DIM));
  v[0] = seed * 0.1;
  return v;
}

/**
 * Simulate the supersede guard logic from tools.ts memory_update handler.
 * This mirrors the actual code path without requiring the full MCP tool SDK.
 */
async function simulateMemoryUpdate(store, resolvedId, text, newVector, importance, category, scopeFilter) {
  // --- Same guard logic as tools.ts ---
  if (text && newVector) {
    const existing = await store.getById(resolvedId, scopeFilter);
    if (existing) {
      const meta = parseSmartMetadata(existing.metadata, existing);
      if (TEMPORAL_VERSIONED_CATEGORIES.has(meta.memory_category)) {
        const now = Date.now();
        const factKey =
          meta.fact_key ?? deriveFactKey(meta.memory_category, text);

        const newMeta = buildSmartMetadata(
          { text, category: existing.category },
          {
            l0_abstract: text,
            l1_overview: meta.l1_overview,
            l2_content: text,
            memory_category: meta.memory_category,
            tier: meta.tier,
            access_count: 0,
            confidence: importance !== undefined ? Math.min(1, Math.max(0, importance)) : meta.confidence,
            valid_from: now,
            fact_key: factKey,
            supersedes: resolvedId,
            relations: appendRelation([], {
              type: "supersedes",
              targetId: resolvedId,
            }),
          },
        );

        const newEntry = await store.store({
          text,
          vector: newVector,
          category: category || existing.category,
          scope: existing.scope,
          importance: importance !== undefined ? importance : existing.importance,
          metadata: stringifySmartMetadata(newMeta),
        });

        const invalidatedMeta = buildSmartMetadata(existing, {
          fact_key: factKey,
          invalidated_at: now,
          superseded_by: newEntry.id,
          relations: appendRelation(meta.relations, {
            type: "superseded_by",
            targetId: newEntry.id,
          }),
        });
        await store.update(
          resolvedId,
          { metadata: stringifySmartMetadata(invalidatedMeta) },
          scopeFilter,
        );

        return { action: "superseded", oldId: resolvedId, newId: newEntry.id };
      }
    }
  }

  // Fall through: raw in-place update
  const updates = {};
  if (text) { updates.text = text; updates.vector = newVector; }
  if (importance !== undefined) updates.importance = importance;
  if (category) updates.category = category;

  const updated = await store.update(resolvedId, updates, scopeFilter);
  return { action: "updated", id: updated?.id };
}

async function runTests() {
  const workDir = mkdtempSync(path.join(tmpdir(), "update-supersede-"));
  const dbPath = path.join(workDir, "db");
  const store = new MemoryStore({ dbPath, vectorDim: VECTOR_DIM });
  const scopeFilter = ["test"];

  try {
    // ====================================================================
    // Test 1: Text change on preferences record triggers supersede
    // ====================================================================
    console.log("Test 1: text change on preferences record triggers supersede...");

    const oldText = "饮品偏好：乌龙茶";
    const oldEntry = await store.store({
      text: oldText,
      vector: makeVector(1),
      category: "preference",
      scope: "test",
      importance: 0.8,
      metadata: stringifySmartMetadata(
        buildSmartMetadata(
          { text: oldText, category: "preference", importance: 0.8 },
          {
            l0_abstract: oldText,
            l1_overview: "- 喜欢乌龙茶",
            l2_content: oldText,
            memory_category: "preferences",
            tier: "working",
            confidence: 0.8,
          },
        ),
      ),
    });

    const newText = "饮品偏好：咖啡";
    const result1 = await simulateMemoryUpdate(
      store, oldEntry.id, newText, makeVector(2), undefined, undefined, scopeFilter,
    );

    assert.equal(result1.action, "superseded", "should trigger supersede");
    assert.ok(result1.newId, "should return new record ID");
    assert.equal(result1.oldId, oldEntry.id, "should reference old record");

    // Verify old record is invalidated
    const oldAfter = await store.getById(oldEntry.id, scopeFilter);
    assert.ok(oldAfter, "old record should still exist");
    assert.equal(oldAfter.text, oldText, "old record text should be unchanged");
    const oldMeta = parseSmartMetadata(oldAfter.metadata, oldAfter);
    assert.ok(oldMeta.invalidated_at, "old record should have invalidated_at");
    assert.equal(oldMeta.superseded_by, result1.newId, "old record should point to new");
    assert.equal(isMemoryActiveAt(oldMeta), false, "old record should be inactive");

    // Verify new record has supersede chain
    const newAfter = await store.getById(result1.newId, scopeFilter);
    assert.ok(newAfter, "new record should exist");
    assert.equal(newAfter.text, newText, "new record should have updated text");
    const newMeta = parseSmartMetadata(newAfter.metadata, newAfter);
    assert.equal(newMeta.supersedes, oldEntry.id, "new record should link to old");
    assert.ok(newMeta.valid_from, "new record should have valid_from");
    assert.equal(isMemoryActiveAt(newMeta), true, "new record should be active");
    assert.equal(newMeta.fact_key, oldMeta.fact_key, "fact_key should match");

    console.log("  ✅ text change on preferences creates supersede chain");

    // ====================================================================
    // Test 2: Metadata-only change on preferences does NOT trigger supersede
    // ====================================================================
    console.log("\nTest 2: metadata-only change on preferences updates in-place...");

    const prefEntry = await store.store({
      text: "编辑器偏好：VS Code",
      vector: makeVector(3),
      category: "preference",
      scope: "test",
      importance: 0.5,
      metadata: stringifySmartMetadata(
        buildSmartMetadata(
          { text: "编辑器偏好：VS Code", category: "preference", importance: 0.5 },
          {
            l0_abstract: "编辑器偏好：VS Code",
            l1_overview: "- VS Code",
            l2_content: "编辑器偏好：VS Code",
            memory_category: "preferences",
            tier: "working",
            confidence: 0.5,
          },
        ),
      ),
    });

    const result2 = await simulateMemoryUpdate(
      store, prefEntry.id, undefined, undefined, 0.9, undefined, scopeFilter,
    );

    assert.equal(result2.action, "updated", "should do in-place update");
    assert.equal(result2.id, prefEntry.id, "should update same record");

    const prefAfter = await store.getById(prefEntry.id, scopeFilter);
    assert.equal(prefAfter.importance, 0.9, "importance should be updated");
    const prefMeta = parseSmartMetadata(prefAfter.metadata, prefAfter);
    assert.ok(!prefMeta.invalidated_at, "should NOT be invalidated");

    console.log("  ✅ metadata-only change updates in-place without supersede");

    // ====================================================================
    // Test 3: Text change on non-temporal category updates in-place
    // ====================================================================
    console.log("\nTest 3: text change on non-temporal category updates in-place...");

    const eventEntry = await store.store({
      text: "参加了2026年技术大会",
      vector: makeVector(4),
      category: "fact",
      scope: "test",
      importance: 0.6,
      metadata: stringifySmartMetadata(
        buildSmartMetadata(
          { text: "参加了2026年技术大会", category: "fact", importance: 0.6 },
          {
            l0_abstract: "参加了2026年技术大会",
            l1_overview: "- 技术大会",
            l2_content: "参加了2026年技术大会",
            memory_category: "cases",
            tier: "working",
            confidence: 0.6,
          },
        ),
      ),
    });

    const newEventText = "参加了2026年AI技术峰会";
    const result3 = await simulateMemoryUpdate(
      store, eventEntry.id, newEventText, makeVector(5), undefined, undefined, scopeFilter,
    );

    assert.equal(result3.action, "updated", "should do in-place update for non-temporal");
    assert.equal(result3.id, eventEntry.id, "should update same record");

    const eventAfter = await store.getById(eventEntry.id, scopeFilter);
    assert.equal(eventAfter.text, newEventText, "text should be updated in-place");

    console.log("  ✅ non-temporal category text change updates in-place");

    // ====================================================================
    // Test 4: Text change on entities record also triggers supersede
    // ====================================================================
    console.log("\nTest 4: text change on entities record triggers supersede...");

    const entityEntry = await store.store({
      text: "Project Alpha: status active",
      vector: makeVector(6),
      category: "entity",
      scope: "test",
      importance: 0.7,
      metadata: stringifySmartMetadata(
        buildSmartMetadata(
          { text: "Project Alpha: status active", category: "entity", importance: 0.7 },
          {
            l0_abstract: "Project Alpha: status active",
            l1_overview: "- Project Alpha is active",
            l2_content: "Project Alpha: status active",
            memory_category: "entities",
            tier: "working",
            confidence: 0.7,
          },
        ),
      ),
    });

    const newEntityText = "Project Alpha: status paused";
    const result4 = await simulateMemoryUpdate(
      store, entityEntry.id, newEntityText, makeVector(7), undefined, undefined, scopeFilter,
    );

    assert.equal(result4.action, "superseded", "entities should trigger supersede too");
    assert.ok(result4.newId, "should have new record");

    const entityOld = await store.getById(entityEntry.id, scopeFilter);
    const entityOldMeta = parseSmartMetadata(entityOld.metadata, entityOld);
    assert.ok(entityOldMeta.invalidated_at, "old entity should be invalidated");
    assert.equal(isMemoryActiveAt(entityOldMeta), false);

    const entityNew = await store.getById(result4.newId, scopeFilter);
    const entityNewMeta = parseSmartMetadata(entityNew.metadata, entityNew);
    assert.equal(entityNewMeta.supersedes, entityEntry.id);
    assert.equal(isMemoryActiveAt(entityNewMeta), true);

    console.log("  ✅ entities text change creates supersede chain");

    console.log("\n=== All memory_update supersede tests passed! ===");
  } finally {
    rmSync(workDir, { recursive: true, force: true });
  }
}

await runTests();

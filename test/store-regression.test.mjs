import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { MemoryStore } = jiti("../src/store.ts");
const TEST_ID = "11111111-1111-1111-1111-111111111111";

function makeEntry(overrides = {}) {
  return {
    id: TEST_ID,
    text: "original memory",
    vector: [1, 0, 0, 0],
    category: "fact",
    scope: "global",
    importance: 0.7,
    timestamp: 1700000000000,
    metadata: "{}",
    ...overrides,
  };
}

function createMockQueryTable(state, hooks = {}) {
  return {
    query() {
      let whereClause = "";
      return {
        where(clause) {
          whereClause = clause;
          return this;
        },
        select() {
          return this;
        },
        limit() {
          return this;
        },
        async toArray() {
          hooks.onQuery?.(whereClause);
          const idMatch = /id = '([^']+)'/.exec(whereClause);
          if (idMatch) {
            const id = idMatch[1].replace(/''/g, "'");
            return state.has(id) ? [{ ...state.get(id) }] : [];
          }
          return [...state.values()].map((row) => ({ ...row }));
        },
      };
    },
    async delete(whereClause) {
      hooks.onDelete?.(whereClause);
      const idMatch = /id = '([^']+)'/.exec(whereClause);
      if (idMatch) {
        const id = idMatch[1].replace(/''/g, "'");
        state.delete(id);
      }
    },
    async add(entries) {
      const [entry] = entries;
      if (hooks.onAdd) {
        await hooks.onAdd(entry);
      }
      state.set(entry.id, { ...entry });
    },
    async listIndices() {
      return [];
    },
  };
}

function makeStore(table) {
  const store = new MemoryStore({ dbPath: "/tmp/memory-lancedb-pro-test", vectorDim: 4 });
  store.table = table;
  return store;
}

describe("MemoryStore regressions", () => {
  it("uses cosine distance for vector search", async () => {
    let receivedDistanceType = null;
    let receivedWhereClause = null;
    let receivedLimit = null;

    const chain = {
      distanceType(value) {
        receivedDistanceType = value;
        return this;
      },
      limit(value) {
        receivedLimit = value;
        return this;
      },
      where(value) {
        receivedWhereClause = value;
        return this;
      },
      async toArray() {
        return [makeEntry({ _distance: 0.1 })];
      },
    };

    const table = {
      vectorSearch(vector) {
        assert.deepEqual(vector, [1, 0, 0, 0]);
        return chain;
      },
    };

    const store = makeStore(table);
    const results = await store.vectorSearch([1, 0, 0, 0], 1, 0.1, ["global"]);

    assert.equal(receivedDistanceType, "cosine");
    assert.equal(receivedLimit, 10);
    assert.match(receivedWhereClause, /scope = 'global'/);
    assert.equal(results.length, 1);
  });

  it("restores the latest available record when update add fails after delete", async () => {
    const state = new Map([[TEST_ID, makeEntry()]]);
    let addCount = 0;

    const table = createMockQueryTable(state, {
      onAdd(entry) {
        addCount += 1;
        if (addCount === 1) {
          throw new Error(`simulated add failure for ${entry.text}`);
        }
      },
    });

    const store = makeStore(table);

    await assert.rejects(
      store.update(TEST_ID, { text: "updated memory" }),
      /latest available record restored/,
    );

    assert.equal(addCount, 2, "expected one failed write and one rollback write");
    assert.equal(state.get(TEST_ID)?.text, "original memory");
  });

  it("serializes concurrent updates to avoid stale rollback races", async () => {
    const state = new Map([[TEST_ID, makeEntry()]]);
    let deleteCount = 0;
    let firstAddRelease;
    let firstAddPending = true;
    let resolveFirstDelete;
    let resolveFirstAddBlocked;
    const firstDeleteSeen = new Promise((resolve) => {
      resolveFirstDelete = resolve;
    });
    const firstAddBlocked = new Promise((resolve) => {
      resolveFirstAddBlocked = resolve;
    });

    const table = createMockQueryTable(state, {
      onDelete() {
        deleteCount += 1;
        if (deleteCount === 1) {
          resolveFirstDelete();
        }
      },
      async onAdd(entry) {
        if (firstAddPending) {
          firstAddPending = false;
          resolveFirstAddBlocked();
          await new Promise((resolve) => {
            firstAddRelease = resolve;
          });
        }
        state.set(entry.id, { ...entry });
      },
    });

    const store = makeStore(table);
    const first = store.update(TEST_ID, { text: "first update" });
    const second = store.update(TEST_ID, { text: "second update" });

    await firstDeleteSeen;
    assert.equal(deleteCount, 1, "second update should wait for the first serialized update");

    await firstAddBlocked;
    firstAddRelease();
    const [, secondResult] = await Promise.all([first, second]);

    assert.equal(deleteCount, 2);
    assert.equal(secondResult?.text, "second update");
    assert.equal(state.get(TEST_ID)?.text, "second update");
  });

  it("reports FTS status and can rebuild the index", async () => {
    let droppedIndex = null;
    let forcedRebuild = false;

    const table = {
      async listIndices() {
        return [{ name: "memories_text_fts_idx", indexType: "FTS", columns: ["text"] }];
      },
      async dropIndex(name) {
        droppedIndex = name;
      },
    };

    const store = makeStore(table);
    store.ftsSupported = true;
    store.ftsIndexCreated = false;
    store.lastFtsError = "previous failure";
    store.createFtsIndex = async (_table, force = false) => {
      forcedRebuild = force;
    };

    assert.deepEqual(store.getFtsStatus(), {
      available: false,
      supported: true,
      indexExists: false,
      lastError: "previous failure",
    });

    const result = await store.rebuildFtsIndex();
    assert.deepEqual(result, { success: true });
    assert.equal(droppedIndex, "memories_text_fts_idx");
    assert.equal(forcedRebuild, true);
    assert.equal(store.ftsIndexCreated, true);
    assert.equal(store.lastFtsError, null);
  });
});

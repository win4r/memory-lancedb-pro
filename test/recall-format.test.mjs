import assert from "node:assert/strict";
import jitiFactory from "jiti";
const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { buildRecallPrependContext, containsRecallInjection, RECALL_BLOCK_START, RECALL_BLOCK_END } = jiti("../src/recall-format.ts");

const sample = [{ entry: { category: "fact", scope: "global", text: "Use Gemini embeddings for LanceDB memory." }, score: 0.87, sources: { bm25: true } }];

const plain = buildRecallPrependContext(sample, "plain");
assert.match(plain, /Internal memory recall for background context only/i);
assert.ok(plain.includes(RECALL_BLOCK_START), plain);
assert.ok(plain.includes(RECALL_BLOCK_END), plain);
assert.doesNotMatch(plain, /<relevant-memories>/i);
assert.equal(containsRecallInjection(plain), true);

const xml = buildRecallPrependContext(sample, "xml");
assert.match(xml, /<relevant-memories>/i);
assert.equal(containsRecallInjection(xml), true);

console.log("OK: recall format helpers verified");
